# SAGE: A Parameter-Efficient Gated Architecture for Zero-Shot Video Deepfake Detection

[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/IEEE-Signal_Processing_Letters-blue.svg)](https://ieeexplore.ieee.org/)

Official repository for the paper **"SAGE: A Parameter-Efficient Gated Architecture for Zero-Shot Video Deepfake Detection"** (Submitted to *IEEE Signal Processing Letters*).

## Overview
This repository contains the training, ablation, and evaluation code for the **Steganalysis Arbitration and Gating Engine (SAGE)**. SAGE is a parameter-efficient video deepfake detector that dynamically fuses semantic priors from a frozen foundation model (DINOv2) with high-frequency forensic artifacts extracted via a constrained convolutional layer (BayarConv). 

By utilizing an Enhanced Disagreement Gate, SAGE optimizes only ~0.24M parameters while effectively shielding the architecture from domain-shift collapse, achieving competitive zero-shot transfer on the Celeb-DF (v2) benchmark.

## Repository Structure

The repository is organized by experiment to match the ablation studies and baselines presented in the paper:

```text
.
├── SAGE(30 frames)/                 # Core SAGE architecture (BayarConv + Gated Fusion)
│   └── sage-dv2-30frames.ipynb
├── Baselines/                       # Unified 30-frame spatiotemporal baselines
│   ├── EfficientNet/                # EfficientNet-B0 + BiLSTM
│   ├── MBNET/                       # MobileNetV2 + BiLSTM
│   ├── MSCVIT/                      # MSCViT-T + BiLSTM
│   └── RESNET50/                    # ResNet-50 + BiLSTM
├── fixed fusion/                    # Static alpha weighting ablations
│   ├── SemanticOnly/                # alpha = 1.0 (DINOv2-S Only)
│   ├── alpha = 0.75/
│   ├── alpha = 0.50/
│   ├── alpha = 0.25/
│   └── alpha = 0/                   # alpha = 0.0 (Frequency Only)
└── Naive concatenation ablations/   # Modality collapse experiments
    ├── bayar_naive/                 # BayarConv + Concatenation
    └── Freq-2(SRM+naive concat)/    # SRM + Concatenation
```

## Datasets
The models are trained on **FaceForensics++ (c23)** and evaluated for zero-shot generalization on **Celeb-DF (v2)**. 
- All videos are preprocessed using MTCNN for face extraction.
- The default temporal window is `T = 30` frames per sequence.


#### the pretrained model weights are included in the github releases section.

```
