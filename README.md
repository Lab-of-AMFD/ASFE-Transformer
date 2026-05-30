# ASFE-Transformer

Official PyTorch implementation of **ASFE-Transformer** for acoustic-based rolling bearing fault diagnosis.

> **Phoneme-inspired acoustic frame embedded lightweight transformer for rolling bearing fault diagnosis**  
> Linhao Peng, Fang Liu, Zhongliang Lv, Yongbin Liu, Min Xia  
> *Mechanical Systems and Signal Processing*, 248, 114030, 2026.

<p align="center">
  <img src="./ASFE_Transformer.png" width="95%" alt="ASFE-Transformer architecture">
</p>

## Overview

ASFE-Transformer is a lightweight Transformer-based framework designed for rolling bearing fault diagnosis using acoustic signals. Inspired by the phoneme processing paradigm in speech analysis, the model converts continuous acoustic waveforms into compact acoustic frame tokens and learns both global temporal dependencies and local transient fault patterns.

The framework contains three main components:

- **Analytic Amplitude-Phase Representation (AAPR):** transforms a real-valued acoustic signal into a dual-channel analytic representation using the Hilbert transform.
- **Acoustic Frame Embedding Module (AFEM):** segments the dual-channel acoustic representation into overlapping short-time frames and embeds each frame as a token.
- **Lightweight Transformer Encoder:** combines low-dimensional multi-head self-attention with the proposed **Swish-Depthwise Gated Linear Unit (SD-GLU)** to capture global frame dependencies and local acoustic continuity.

<p align="center">
  <img src="./Framework.png" width="80%" alt="Overall diagnostic framework">
</p>

## Highlights

- End-to-end acoustic fault diagnosis framework.
- Hilbert-transform-based amplitude-phase representation.
- Frame-level acoustic token embedding for compact sequence modeling.
- Lightweight Transformer encoder with SD-GLU.
- Suitable for resource-constrained deployment scenarios.
- Reproducible implementation for comparison with CNN and Transformer-based baselines.

## Model Architecture

The overall pipeline is:

```text
Raw acoustic signal
        ↓
AAPR: Analytic Amplitude-Phase Representation
        ↓
AFEM: Acoustic Frame Embedding Module
        ↓
Class token + position embedding
        ↓
Transformer Encoder: LMHSA + SD-GLU
        ↓
Classifier
        ↓
Fault category
```

The default input shape is:

```text
(B, 1, 1024)
```

where `B` denotes the batch size, `1` is the single acoustic channel, and `1024` is the input sample length.

## Requirements

The code has been tested with Python and PyTorch. A typical environment is:

```text
python >= 3.8
pytorch >= 1.10
numpy
scikit-learn
joblib
matplotlib
thop       # optional, for FLOPs/MACs calculation
```

You can install the main dependencies with:

```bash
pip install torch numpy scikit-learn joblib matplotlib thop
```


## Quick Start

### 1. Import the model

```python
import torch
from ASFE_Transformer import ASFE_Transformer

model = ASFE_Transformer(
    None,
    in_channel=1,
    out_channel=5,
    input_length=1024
)

x = torch.randn(4, 1, 1024)
y = model(x)

print(y.shape)  # torch.Size([4, 5])
```

### 2. Train the model

Prepare your acoustic samples as one-dimensional time-series windows. Each sample should contain `1024` points by default.

Example command:

```bash
python train.py --data_dir ./data --epochs 50 --batch_size 32 --lr 0.001
```

### 3. Test the model

```bash
python test.py --data_dir ./data --checkpoint ./results/best_model.pth
```

### 4. Noise robustness evaluation

```bash
python noise_test.py --data_dir ./data --snr all
```

## Data Preparation

The model expects acoustic time-series samples. A common preprocessing pipeline is:

1. Normalize the raw acoustic signal.
2. Segment the continuous signal using a sliding window.
3. Use a window length of `1024` points.
4. Assign the corresponding fault label to each segment.
5. Split samples into training, validation, and test sets.

Example sample format:

```text
sample: (1024,)
label : integer class index
```

During model forwarding, the input tensor should be reshaped as:

```text
(B, 1, 1024)
```

## Default Configuration

The default ASFE-Transformer configuration is:

| Component | Parameter | Value |
|---|---:|---:|
| Input | Input sample length | 1024 |
| AFEM | Frame length | 16 |
| AFEM | Hop size | 8 |
| AFEM | Number of frames | 127 |
| Transformer | Number of heads | 4 |
| Transformer | Encoder depth | 8 |
| FFN | SD-GLU kernel size | 5 |
| Classifier | Number of classes | dataset-dependent |

> Note: The number of output classes should be set according to the target dataset.

## Citation

If our work is useful to you, please cite the following paper. It is the greatest encouragement to our open-source work. Thank you very much!

```bibtex
@article{peng2026asfe_transformer,
  title   = {Phoneme-inspired acoustic frame embedded lightweight transformer for rolling bearing fault diagnosis},
  author  = {Peng, Linhao and Liu, Fang and Lv, Zhongliang and Liu, Yongbin and Xia, Min},
  journal = {Mechanical Systems and Signal Processing},
  volume  = {248},
  pages   = {114030},
  year    = {2026},
  doi     = {10.1016/j.ymssp.2026.114030},
  url     = {https://doi.org/10.1016/j.ymssp.2026.114030}
}
```

## Acknowledgement

This repository is released to facilitate reproducibility and further research on lightweight acoustic fault diagnosis. We sincerely thank all researchers who contribute to open-source intelligent fault diagnosis.

## Contact

For questions about the paper or implementation, please open an issue in this repository.

