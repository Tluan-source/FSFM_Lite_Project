# FSFM-Lite: Face Spoofing Detection with Foundation Model — Lite

<div align="center">

**A lightweight face anti-spoofing system leveraging DINOv2 vision foundation model and a novel Three Consistencies (ThreeC) module for robust live/spoof classification.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![DINOv2](https://img.shields.io/badge/Backbone-DINOv2_ViT--B%2F14-4285F4)
![Status](https://img.shields.io/badge/Status-Trained_%E2%9C%85-brightgreen)
![License](https://img.shields.io/badge/License-Academic-green)

</div>

---

## 📋 Table of Contents

- [Project Status](#-project-status)
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Usage](#usage)
- [Real-Time Webcam Demo](#-real-time-webcam-demo)
- [Model Details](#model-details)
- [References](#references)

---

## 🚦 Project Status

| Phase | Status | Details |
|-------|--------|---------|
| ✅ Dataset Preparation | **Done** | CelebA-Spoof parsed, metadata CSVs created (270K samples) |
| ✅ Face Processing | **Done** | Cropping pipeline with RetinaFace bboxes + 20% padding |
| ✅ Feature Extraction | **Done** | DINOv2 ViT-B/14 integrated, CLS + 256 patch tokens extracted |
| ✅ ThreeC Module | **Done** | Spatial / Feature / Semantic consistency branches + Fusion |
| ✅ Full Model Assembly | **Done** | End-to-end forward pass verified, ~98M parameters |
| ✅ **Model Training** | **Done** | `best_fsfm_lite.pth` saved (~460 MB) |
| ✅ **Fine-tuning** | **Done** | `best_fsfm_lite_finetuned.pth` saved (~515 MB) |
| ✅ **Real-time Demo** | **Done** | Webcam inference via `src/main.py` |
| ⏳ Evaluation & Metrics | In Progress | ACER / HTER / EER / ROC not yet computed |
| ⏳ Configuration Files | Pending | `configs/` directory still empty |

---

## Overview

**FSFM-Lite** is a face anti-spoofing (liveness detection) model designed to distinguish between **live (real)** and **spoof (fake)** face images. The system combines a powerful self-supervised vision foundation model (**DINOv2 ViT-B/14**) as a feature extractor with a custom **Three Consistencies (ThreeC) Module** that analyzes spatial, feature-level, and semantic consistency patterns to detect spoofing artifacts.

### Key Highlights

- 🧠 **Foundation Model Backbone**: Utilizes Facebook's DINOv2 ViT-B/14 pretrained model for robust feature extraction
- 🔍 **Three Consistencies Module (ThreeC)**: Novel module analyzing spatial, feature, and semantic consistency in parallel
- 📊 **CelebA-Spoof Dataset**: Trained on the large-scale CelebA-Spoof benchmark (~270K images)
- ⚡ **~98M Parameters**: Full model with DINOv2 backbone + ThreeC + Classifier Head
- 🎥 **Real-Time Demo**: Live webcam face anti-spoofing with bounding box overlay and confidence score
- 💾 **Trained Checkpoints**: Base model + fine-tuned model weights available

---

## Architecture

The FSFM-Lite model follows a three-stage pipeline:

```
Input Image (224×224×3)
        │
        ▼
┌─────────────────────────────────────┐
│        DINOv2 ViT-B/14 Backbone     │
│  (Pretrained, 12 Transformer Blocks)│
│                                     │
│  PatchEmbed: Conv2d(3→768, k=14)    │
│  → 256 patch tokens (16×16 grid)    │
│  → 1 CLS token                     │
└────────────┬────────────────────────┘
             │
     ┌───────┴───────┐
     │               │
 CLS Token     Patch Tokens
 [B, 768]     [B, 256, 768]
     │               │
     └───────┬───────┘
             │
             ▼
┌─────────────────────────────────────┐
│      ThreeC Module (3 Branches)     │
│                                     │
│  ┌───────────┐ ┌──────────────────┐ │
│  │  Spatial   │ │    Feature       │ │
│  │Consistency │ │  Consistency     │ │
│  │(Self-Attn) │ │    (MLP)         │ │
│  └─────┬─────┘ └───────┬──────────┘ │
│        │               │            │
│  ┌─────┴───────────────┴──────────┐ │
│  │     Semantic Consistency       │ │
│  │   (Cross-Attn with CLS)       │ │
│  └──────────────┬─────────────────┘ │
│                 │                    │
│        ┌────────┴────────┐          │
│        │  Fusion Block   │          │
│        │ Concat → Linear │          │
│        │ [2304] → [768]  │          │
│        └────────┬────────┘          │
└─────────────────┼───────────────────┘
                  │
          Enhanced Tokens
           [B, 256, 768]
                  │
                  ▼
         Mean Pooling (dim=1)
              [B, 768]
                  │
                  ▼
┌─────────────────────────────────────┐
│         Classifier Head             │
│  Linear(768→256) → GELU            │
│  → Dropout(0.2) → Linear(256→2)    │
└─────────────────┬───────────────────┘
                  │
                  ▼
          Logits [B, 2]
        (Live vs Spoof)
```

### ThreeC Module — Three Consistencies

| Branch | Method | Purpose |
|--------|--------|---------|
| **Spatial Consistency** | Multi-Head Self-Attention (8 heads) | Captures spatial relationships between patches to detect local inconsistencies |
| **Feature Consistency** | Feed-Forward MLP (768→3072→768) | Processes each patch independently to detect feature-level anomalies |
| **Semantic Consistency** | Cross-Attention (patches ↔ CLS token) | Relates local patch features to global image semantics for holistic analysis |
| **Fusion Block** | Concatenation + Linear Projection | Merges three consistency outputs (2304→768) into unified representation |

---

## Project Structure

```
FSFM_Lite_Project/
│
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
│
├── src/                               # Source code (reusable modules)
│   ├── main.py                        # ⭐ Real-time webcam demo (inference entry point)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dino_backbone.py           # DINOv2 ViT-B/14 wrapper
│   │   ├── threec_module.py           # ThreeC module (core novelty)
│   │   ├── classifier_head.py         # Binary classification head
│   │   └── fsfm_lite.py              # Full FSFM-Lite model assembly
│   │
│   ├── datasets/
│   │   └── celeba_spoof_dataset.py    # CelebA-Spoof Dataset & face cropping
│   │
│   ├── losses/                        # Loss functions (planned)
│   └── utils/                         # Evaluation utilities (planned)
│
├── notebooks/                         # Step-by-step development notebooks
│   ├── 01_dataset_preparation.ipynb   # Data loading, exploration & metadata creation
│   ├── 02_face_processing.ipynb       # Face cropping, transforms & dataset class
│   ├── 03_dino_feature_extraction.ipynb  # DINOv2 feature extraction demo
│   ├── 04_threec_module.ipynb         # ThreeC module design & testing
│   └── 05_fsfm_lite_model.ipynb       # Full model assembly & end-to-end test
│
├── data/
│   └── CelebA_Spoof/                  # CelebA-Spoof dataset (not tracked by git)
│       ├── Data/                      # Images organized by train/test → ID → live/spoof
│       ├── metas/                     # Label JSON files & protocols
│       └── README                     # Dataset documentation
│
├── metadata/                          # Generated metadata
│   ├── train_df.csv                   # Training set metadata (244,274 samples)
│   ├── test_df.csv                    # Test set metadata (25,758 samples)
│   └── dataset_stats.json            # Dataset statistics
│
├── configs/                           # Configuration files (planned)
├── docs/                              # Project documentation
│   ├── Architecture Design Document.docx
│   └── Kế hoạch Dự Án.docx
│
└── outputs/                           # Training outputs
    ├── checkpoints/                   # ⭐ Saved model weights
    │   ├── best_fsfm_lite.pth             (~460 MB) — base trained model
    │   └── best_fsfm_lite_finetuned.pth   (~515 MB) — fine-tuned model
    ├── figures/                       # Visualization outputs
    ├── logs/                          # Training logs
    └── reports/                       # Evaluation reports
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd FSFM_Lite_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy opencv-python matplotlib seaborn Pillow tqdm pyarrow
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0 | Deep learning framework |
| `torchvision` | ≥ 0.15 | Image transforms & utilities |
| `opencv-python` | ≥ 4.0 | Image reading, webcam capture & face detection |
| `pandas` | ≥ 1.5 | Data manipulation |
| `numpy` | ≥ 1.24 | Numerical computing |
| `matplotlib` | ≥ 3.7 | Visualization |
| `seaborn` | ≥ 0.12 | Statistical plotting |
| `Pillow` | ≥ 9.0 | Image handling |
| `tqdm` | ≥ 4.65 | Progress bars |

---

## Dataset

### CelebA-Spoof

This project uses the [CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) dataset, a large-scale face anti-spoofing benchmark.

| Split | Total Samples | Live | Spoof | Ratio (Live:Spoof) |
|-------|--------------|------|-------|---------------------|
| Train | 244,274 | 82,727 | 161,547 | ~1:1.95 |
| Test  | 25,758 | — | — | — |
| **Total** | **270,032** | — | — | — |

### Dataset Structure

```
CelebA_Spoof/
├── Data/
│   ├── train/
│   │   └── {subject_id}/
│   │       ├── live/
│   │       │   ├── 000001.jpg
│   │       │   └── 000001_BB.txt    # Bounding box (RetinaFace)
│   │       └── spoof/
│   │           ├── 000001.jpg
│   │           └── 000001_BB.txt
│   └── test/
│       └── ...
└── metas/
    └── intra_test/ (protocol1)
        ├── train_label.json
        └── test_label.json
```

### Bounding Box Format

Bounding boxes are generated by [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace) and stored in **224×224 normalized space**:

```
x  y  w  h  confidence
15 30 146 166 0.99992573
```

To convert to actual image coordinates:
```python
real_x = int(x * (image_width / 224))
real_y = int(y * (image_height / 224))
real_w = int(w * (image_width / 224))
real_h = int(h * (image_height / 224))
```

### Label Convention

- `0` — **Live** (real face)
- `1` — **Spoof** (fake/attack)

---

## Pipeline

The project is organized as a sequential pipeline across 5 notebooks:

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: Dataset Preparation (01_dataset_preparation)   │
│  Load JSON labels → Build DataFrames → Save metadata    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: Face Processing (02_face_processing)           │
│  Read bboxes → Crop faces (20% padding) → Transform     │
│  → Build CelebASpoofDataset PyTorch Dataset             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: Feature Extraction (03_dino_feature_extraction) │
│  Load DINOv2 ViT-B/14 → Extract CLS + 256 patch tokens │
│  → Feature dim: 768                                     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4: ThreeC Module (04_threec_module)               │
│  Spatial + Feature + Semantic Consistency → Fusion      │
│  Enhanced tokens: [B, 256, 768]                         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 5: Full Model (05_fsfm_lite_model)                │
│  DINOv2 → ThreeC → Mean Pooling → Classifier → Logits  │
│  Output: [B, 2] (Live vs Spoof)                         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Training & Fine-tuning  ✅                              │
│  best_fsfm_lite.pth (~460 MB)                           │
│  best_fsfm_lite_finetuned.pth (~515 MB)                 │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Real-Time Webcam Demo  ✅  (src/main.py)               │
│  OpenCV face detection → FSFM-Lite inference            │
│  → LIVE / SPOOF label + confidence score overlay        │
└─────────────────────────────────────────────────────────┘
```

---

## Usage

### Quick Start — Inference with Full Model

```python
import torch
from torchvision import transforms
from src.models.fsfm_lite import FSFMLite
from src.datasets.celeba_spoof_dataset import crop_face

# Initialize model
model = FSFMLite()

# Load trained weights
checkpoint = torch.load("outputs/checkpoints/best_fsfm_lite_finetuned.pth",
                        map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Define transforms (ImageNet normalization)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Crop face and preprocess
face = crop_face("path/to/image.jpg", "path/to/image_BB.txt")
image = transform(face).unsqueeze(0)  # [1, 3, 224, 224]

# Inference
with torch.no_grad():
    logits = model(image)                       # [1, 2]
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1).item()          # 0=Live, 1=Spoof
    conf = probs.max().item()
    print(f"Prediction: {'Spoof' if pred == 1 else 'Live'} ({conf:.2%})")
```

### Using the Dataset

```python
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from src.datasets.celeba_spoof_dataset import CelebASpoofDataset

# Load metadata
train_df = pd.read_csv("metadata/train_df.csv")

# Define transforms
face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Create dataset & dataloader
dataset = CelebASpoofDataset(train_df, transform=face_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for batch in dataloader:
    images = batch["image"]   # [32, 3, 224, 224]
    labels = batch["label"]   # [32]
    break
```

### Step-by-Step Feature Inspection

```python
model = FSFMLite()
model.eval()

with torch.no_grad():
    # 1. Extract backbone features
    cls_token, patch_tokens = model.backbone(image)
    print(f"CLS token:    {cls_token.shape}")      # [1, 768]
    print(f"Patch tokens: {patch_tokens.shape}")    # [1, 256, 768]

    # 2. Apply ThreeC module
    enhanced = model.threec(patch_tokens, cls_token)
    print(f"Enhanced:     {enhanced.shape}")        # [1, 256, 768]

    # 3. Pool and classify
    pooled = enhanced.mean(dim=1)                   # [1, 768]
    logits = model.head(pooled)                     # [1, 2]
```

---

## 🎥 Real-Time Webcam Demo

`src/main.py` provides a live demo using your system webcam. It uses **OpenCV's Haar Cascade** for face detection, then feeds each detected face through the trained FSFM-Lite model.

### Run the Demo

Make sure the trained checkpoint exists at `outputs/checkpoints/best_fsfm_lite.pth`, then:

```bash
python src/main.py
```

### How it works

```
Webcam Frame
    │
    ▼
OpenCV Haar Cascade Face Detector
    │  (scaleFactor=1.1, minNeighbors=5, minSize=80×80)
    ▼
Crop face region with 20% padding
    │
    ▼
BGR → RGB → Resize(224×224) → Normalize
    │
    ▼
FSFMLite.forward(image)
    │
    ▼
Softmax → class prediction + confidence
    │
    ▼
Draw bounding box + label on frame
    ├── 🟢 LIVE  XX.XX%   (green box)
    └── 🔴 SPOOF XX.XX%   (red box)
```

Press **`ESC`** to exit the demo.

---

## Model Details

### Saved Checkpoints

| File | Size | Description |
|------|------|-------------|
| `best_fsfm_lite.pth` | ~460 MB | Base model trained on CelebA-Spoof |
| `best_fsfm_lite_finetuned.pth` | ~515 MB | Fine-tuned version (improved performance) |

Checkpoint format:
```python
{
    "model_state_dict": { ... }   # model weights
    # may also contain: optimizer_state_dict, epoch, loss, accuracy
}
```

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| DINOv2 ViT-B/14 Backbone | ~86M |
| ThreeC Module | ~11.5M |
| Classifier Head | ~197K |
| **Total** | **~98M** |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input Size | 224 × 224 |
| Patch Size (DINOv2) | 14 × 14 |
| Embedding Dimension | 768 |
| Number of Patches | 256 (16×16) |
| Attention Heads (ThreeC) | 8 |
| MLP Expansion Ratio | 4× (768→3072) |
| Classifier Hidden Dim | 256 |
| Dropout Rate | 0.2 |
| Number of Classes | 2 (Live / Spoof) |
| Normalization | ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |
| Bbox Padding | 20% |

---

## References

- **DINOv2**: Oquab, M., et al. "DINOv2: Learning Robust Visual Features without Supervision." *arXiv:2304.07193*, 2023. [[Paper](https://arxiv.org/abs/2304.07193)] [[Code](https://github.com/facebookresearch/dinov2)]
- **CelebA-Spoof**: Zhang, Y., et al. "CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations." *ECCV 2020*. [[Paper](https://arxiv.org/abs/2007.12342)] [[Dataset](https://github.com/ZhangYuanhan-AI/CelebA-Spoof)]
- **RetinaFace**: Deng, J., et al. "RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild." *CVPR 2020*. [[Paper](https://arxiv.org/abs/1905.00641)] [[Code](https://github.com/deepinsight/insightface/tree/master/RetinaFace)]

---

<div align="center">
<i>FSFM-Lite Project — Face Anti-Spoofing with Foundation Model</i>
</div>
