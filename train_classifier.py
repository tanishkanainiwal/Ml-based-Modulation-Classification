"""
"""
import os, glob, re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CFG = dict(
    dataset_root = "dataset",           # folder produced by MATLAB script
    n_samples    = 1024,                # IQ samples per frame
    n_frames_per_file = 500,            # must match MATLAB N_FRAMES
    batch_size   = 128,
    epochs       = 25,
    lr           = 1e-3,
    weight_decay = 1e-4,
    seed         = 42,
    device       = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir     = "results",
)

CLASSES = ["DSB-TC","FM","AM","PM",
           "BPSK","QPSK","8PSK",
           "16QAM","64QAM",
           "ASK","FSK","MSK"]

SNR_RANGE = list(range(-10, 25, 5))   # -10,-5,0,...,20  (7 levels)

torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
os.makedirs(CFG["save_dir"], exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  1.  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_file(path, n_samples, n_frames):
    """Read a GNU-Radio-format .dat file → (N_frames, n_samples, 2) float32."""
    raw = np.fromfile(path, dtype=np.float32)
    expected = n_frames * n_samples * 2
    if raw.size < expected:
        # pad with zeros if file is short
        raw = np.pad(raw, (0, expected - raw.size))
    raw = raw[:expected].reshape(n_frames, n_samples, 2)
    return raw                                              # (N, 1024, 2)

def build_dataset(root, classes, snr_range, n_samples, n_frames):
    """
    Returns:
        X  : (total_frames, n_samples, 2)   IQ data
        y  : (total_frames,)                integer class labels
        snr: (total_frames,)                SNR label (dB)
    """
    X_list, y_list, snr_list = [], [], []
    le = LabelEncoder().fit(classes)

    for cls in classes:
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Missing class folder: {cls_dir}")
            continue
        for snr_db in snr_range:
            tag = f"{snr_db:+03d}dB"
            pattern = os.path.join(cls_dir, f"*{tag}.dat")
            files = glob.glob(pattern)
            if not files:
                print(f"[WARN] No file for {cls} | SNR={tag}")
                continue
            data = load_file(files[0], n_samples, n_frames)
            X_list.append(data)
            y_list.append(np.full(len(data), le.transform([cls])[0], dtype=np.int64))
            snr_list.append(np.full(len(data), snr_db, dtype=np.int32))

    X   = np.concatenate(X_list,   axis=0)
    y   = np.concatenate(y_list,   axis=0)
    snr = np.concatenate(snr_list, axis=0)
    return X, y, snr, le

# ──────────────────────────────────────────────────────────────────────────────
#  2.  PYTORCH DATASET
# ──────────────────────────────────────────────────────────────────────────────
class IQDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, 1024, 2)  →  transpose to (N, 2, 1024) for Conv1d
        self.X = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ──────────────────────────────────────────────────────────────────────────────
#  3.  NOVEL MODEL ARCHITECTURE
#      "IQ-Net": Residual-CNN  →  Bidirectional LSTM  →  Multi-Head Attention
# ──────────────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """1-D Residual Block with squeeze-excitation channel attention."""
    def __init__(self, ch, kernel=7):
        super().__init__()
        pad = (kernel-1)//2
        self.conv = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(ch),
        )
        # Squeeze-Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch, ch//4), nn.ReLU(),
            nn.Linear(ch//4, ch), nn.Sigmoid(),
        )
        self.act = nn.GELU()

    def forward(self, x):
        h  = self.conv(x)
        se = self.se(h).unsqueeze(-1)
        return self.act(x + h * se)         # residual + channel-wise scaling


class MultiHeadSelfAttention(nn.Module):
    """Lightweight multi-head self-attention over time steps."""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True,
                                          dropout=0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):                   # x: (B, T, D)
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class IQNet(nn.Module):
    """
    Novel IQ-Net architecture (3 stages):
      Stage 1 — Multi-scale CNN front-end (captures spectral features)
      Stage 2 — Bidirectional LSTM       (captures temporal patterns)
      Stage 3 — Multi-Head Self-Attention (global context)
      Head    — MLP classifier
    """
    def __init__(self, n_classes=12):
        super().__init__()

        # ── Stage 1: Multi-scale CNN front-end ────────────────────────────
        # Three parallel convolution branches with different kernel sizes
        # to capture short, mid, and long-range spectral patterns
        self.branch_s = nn.Sequential(          # short (k=3)
            nn.Conv1d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.GELU())
        self.branch_m = nn.Sequential(          # mid (k=7)
            nn.Conv1d(2, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.GELU())
        self.branch_l = nn.Sequential(          # long (k=15)
            nn.Conv1d(2, 32, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.GELU())

        self.merge = nn.Sequential(             # merge 96 ch → 128 ch
            nn.Conv1d(96, 128, 1, bias=False),
            nn.BatchNorm1d(128), nn.GELU())

        # Residual blocks (with SE attention)
        self.res_blocks = nn.Sequential(
            ResBlock(128, kernel=7),
            nn.MaxPool1d(2),                    # 1024 → 512
            ResBlock(128, kernel=7),
            nn.MaxPool1d(2),                    # 512  → 256
            ResBlock(128, kernel=5),
            nn.MaxPool1d(4),                    # 256  → 64
            ResBlock(128, kernel=3),
        )

        # ── Stage 2: Bidirectional LSTM ───────────────────────────────────
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        # BiLSTM output dim = 256

        # ── Stage 3: Multi-Head Self-Attention ────────────────────────────
        self.attn = MultiHeadSelfAttention(256, n_heads=4)

        # ── Classifier Head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):                       # x: (B, 2, 1024)
        # Stage 1
        s = self.branch_s(x)
        m = self.branch_m(x)
        l = self.branch_l(x)
        x = self.merge(torch.cat([s, m, l], dim=1))   # (B, 128, 1024)
        x = self.res_blocks(x)                         # (B, 128, 64)

        # Stage 2  — LSTM expects (B, T, F)
        x = x.permute(0, 2, 1)                         # (B, 64, 128)
        x, _ = self.lstm(x)                            # (B, 64, 256)

        # Stage 3 — Self-Attention
        x = self.attn(x)                               # (B, 64, 256)

        # Global average pool → classify
        x = x.mean(dim=1)                              # (B, 256)
        return self.head(x)                            # (B, n_classes)

# ──────────────────────────────────────────────────────────────────────────────
#  4.  TRAINING
# ──────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss/total, correct/total

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss/total, correct/total

# ──────────────────────────────────────────────────────────────────────────────
#  5.  SNR-WISE ACCURACY EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def accuracy_vs_snr(model, X_all, y_all, snr_all, device, n_samples):
    model.eval()
    results = {}
    for snr_db in SNR_RANGE:
        mask = (snr_all == snr_db)
        if mask.sum() == 0:
            continue
        X_s = X_all[mask]
        y_s = y_all[mask]
        ds  = IQDataset(X_s, y_s)
        dl  = DataLoader(ds, batch_size=256, shuffle=False)
        correct, total = 0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred   = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total   += len(yb)
        results[snr_db] = correct / total
    return results

# ──────────────────────────────────────────────────────────────────────────────
#  6.  PLOTTING UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def plot_accuracy_vs_snr(snr_acc_dict, save_dir):
    snrs = sorted(snr_acc_dict.keys())
    accs = [snr_acc_dict[s]*100 for s in snrs]

    plt.figure(figsize=(8,5))
    plt.plot(snrs, accs, 'bo-', linewidth=2, markersize=7)
    for s, a in zip(snrs, accs):
        plt.annotate(f"{a:.1f}%", (s, a), textcoords="offset points",
                     xytext=(0,8), ha='center', fontsize=8)
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Classification Accuracy (%)", fontsize=12)
    plt.title("IQ-Net: Accuracy vs SNR\n(12-Class Modulation Classification)", fontsize=13)
    plt.grid(True, alpha=0.4)
    plt.ylim(0, 105)
    plt.xticks(snrs)
    plt.tight_layout()
    path = os.path.join(save_dir, "accuracy_vs_snr.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"IQ-Net: {title} (All SNRs)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")

def plot_training_curves(history, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'],   label='Val')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves"); ax1.legend(); ax1.grid(alpha=0.4)

    ax2.plot([a*100 for a in history['train_acc']], label='Train')
    ax2.plot([a*100 for a in history['val_acc']],   label='Val')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves"); ax2.legend(); ax2.grid(alpha=0.4)
    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")

# ──────────────────────────────────────────────────────────────────────────────
#  7.  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def main():
    device = CFG["device"]
    print(f"Device: {device}")

    # ── 7.1  Load dataset ────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    X, y, snr, le = build_dataset(
        CFG["dataset_root"], CLASSES, SNR_RANGE,
        CFG["n_samples"], CFG["n_frames_per_file"]
    )
    print(f"  X shape : {X.shape}   dtype: {X.dtype}")
    print(f"  Classes : {le.classes_}")
    print(f"  SNR vals: {np.unique(snr)}")

    # ── 7.2  Train / Val split (stratified, SNR-agnostic) ───────────────────
    idx_train, idx_val = train_test_split(
        np.arange(len(y)), test_size=0.2,
        stratify=y, random_state=CFG["seed"]
    )
    train_ds = IQDataset(X[idx_train], y[idx_train])
    val_ds   = IQDataset(X[idx_val],   y[idx_val])
    train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=0, pin_memory=(device=="cuda"))
    val_dl   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=0, pin_memory=(device=="cuda"))
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── 7.3  Build model ─────────────────────────────────────────────────────
    print("\n[2/5] Building IQ-Net...")
    model = IQNet(n_classes=len(CLASSES)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG["epochs"], eta_min=1e-5)

    # ── 7.4  Training loop ───────────────────────────────────────────────────
    print(f"\n[3/5] Training for {CFG['epochs']} epochs...")
    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_val_acc = 0.0

    for epoch in range(1, CFG["epochs"]+1):
        tr_loss, tr_acc = train_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model, val_dl,   criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(),
                       os.path.join(CFG["save_dir"], "best_model.pt"))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/{CFG['epochs']} | "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc*100:.1f}% | "
                  f"val_loss={va_loss:.4f} val_acc={va_acc*100:.1f}%")

    print(f"\n  Best Val Accuracy: {best_val_acc*100:.2f}%")

    # Reload best weights
    model.load_state_dict(torch.load(
        os.path.join(CFG["save_dir"], "best_model.pt"), map_location=device))

    # ── 7.5  Evaluation & Plots ──────────────────────────────────────────────
    print("\n[4/5] Evaluating accuracy vs SNR...")
    snr_acc = accuracy_vs_snr(model, X, y, snr, device, CFG["n_samples"])
    for s, a in sorted(snr_acc.items()):
        print(f"  SNR={s:+3d} dB → {a*100:.1f}%")

    print("\n[5/5] Generating plots...")
    plot_training_curves(history, CFG["save_dir"])
    plot_accuracy_vs_snr(snr_acc, CFG["save_dir"])

    # Confusion matrix on full validation set
    model.eval()
    y_true_all, y_pred_all = [], []
    val_dl2 = DataLoader(val_ds, batch_size=512, shuffle=False)
    with torch.no_grad():
        for xb, yb in val_dl2:
            pred = model(xb.to(device)).argmax(1).cpu().numpy()
            y_pred_all.extend(pred)
            y_true_all.extend(yb.numpy())

    plot_confusion_matrix(y_true_all, y_pred_all, CLASSES, CFG["save_dir"])

    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(y_true_all, y_pred_all, target_names=CLASSES))
    print(f"\nAll outputs saved to: {CFG['save_dir']}/")


if __name__ == "__main__":
    main()
