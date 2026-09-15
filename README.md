# Ml-based-Modulation-Classification

## Model Architecture
Raw IQ → Multi-scale CNN → Bi-LSTM → Self-Attention → Softmax

- Multi-scale CNN: parallel kernels (3, 5, 7) for 
  multi-resolution feature extraction
- Bi-LSTM: bidirectional temporal modelling (hidden dim: 256)
- Self-Attention: 4 heads, focuses on discriminative 
  time-steps under noise
- Total parameters: ~2.1M

## Dataset
- 12 classes: DSB-TC, FM, AM, PM, BPSK, QPSK, 
  8-PSK, ASK, FSK, MSK, 16-QAM, 64-QAM
- Generated via GNU Radio + PlutoSDR simulation
- SNR range: −10 dB to +20 dB (7 levels, step 5 dB)
- Hardware impairments: CFO + Phase Noise
- ~50K samples | Split: 70/15/15 (train/val/test)

## Results
| SNR      | Accuracy |
|----------|----------|
| −10 dB   | 87.6%    |
| 0 dB     | 93.7%    |
| +20 dB   | 95.1%    |

- BPSK: 100% (easiest — max symbol distance)
- 64-QAM: 87.6% (hardest — dense constellation)
- Main confusion: 16-QAM ↔ 64-QAM at low SNR

## Training Setup
- Framework: PyTorch 2.x
- Optimizer: Adam (lr=1e-3)
- Loss: Cross-Entropy
- Epochs: 25 | Batch size: 64
- Early stopping: patience 5
- LR scheduler: ReduceLROnPlateau

## Key Result
Self-Attention adds <2% parameters vs plain Bi-LSTM 
but improves accuracy by ~4-6% at low SNR conditions.

## Requirements
