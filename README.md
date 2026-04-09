# Video Curation Pipeline & Data Composition Ablation

> Systematic study of how data curation quality and synthetic augmentation ratios
> affect downstream video generative model performance.

---

## Overview

This project builds a production-grade video curation pipeline and runs a controlled
ablation to answer: **"How much does data composition matter for video generation
quality, and where does quality filtering introduce demographic/contextual bias?"**

Inspired by findings from audio pipeline work (accented speech bias at aggressive
DNSMOS thresholds), we run an analogous experiment for video: sweeping quality
thresholds and synthetic mixing ratios while tracking both FVD and per-category
representation drift.

### Key Questions

1. Real-only vs synthetic-augmented training — at what blend does FVD bottom out?
2. Does motion-score or blur filtering systematically remove certain action categories
   (analogous to accent filtering removing non-native speakers)?
3. Can we recover diversity via targeted synthetic augmentation of under-represented
   clips?

---

## Pipeline Architecture

```
Raw Videos (UCF-101 / Kinetics-400 subset)
       │
       ▼
┌─────────────────────────────┐
│   Stage 1 — Ingest          │  download, decode, segment
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Stage 2 — Curate          │  scene cuts → blur → motion → dedup
│   (Ray parallel workers)    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Stage 3 — Augment         │  frame interp · jitter · speed · VLM captions
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Stage 4 — Mix & Train     │  0/25/50/75/100% synthetic ratios
│                             │  VideoMAE fine-tune per mixture
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Stage 5 — Evaluate        │  FVD (I3D) · CLIP@16 · per-class recall
│                             │  Bias sweep over quality thresholds
└─────────────────────────────┘
```

---

## Repository Layout

```
Video-Curation/
├── configs/
│   ├── curation.yaml          # scene / blur / motion / dedup thresholds
│   ├── augmentation.yaml      # interpolation, jitter, speed params
│   └── training.yaml          # model, batch, lr, mixture ratios
├── src/video_curation/
│   ├── data/                  # download, decode, dataset classes
│   ├── curation/              # scene_detect, quality_filter, motion, dedup
│   ├── augmentation/          # frame_interp, color_jitter, speed, captions
│   ├── pipeline/              # Ray orchestration
│   ├── training/              # mixture builder + VideoMAE fine-tune harness
│   └── evaluation/            # FVD, CLIP eval, bias analysis
├── scripts/
│   ├── download_data.py
│   ├── run_curation.py
│   ├── run_augmentation.py
│   ├── run_training.py
│   └── run_evaluation.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_curation_analysis.ipynb
│   └── 03_ablation_results.ipynb
└── results/
    ├── ablation/              # per-mixture FVD / CLIP scores
    └── bias_analysis/         # representation drift CSVs + plots
```

---

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Download UCF-101 subset (≈4 GB for 10 classes)
python scripts/download_data.py --dataset ucf101 --n_classes 10

# 3. Run curation (Ray workers auto-detected)
python scripts/run_curation.py --config configs/curation.yaml

# 4. Generate synthetic augmentations
python scripts/run_augmentation.py --config configs/augmentation.yaml

# 5. Train under different mixtures
python scripts/run_training.py --config configs/training.yaml

# 6. Evaluate and plot ablation
python scripts/run_evaluation.py --results_dir results/ablation
```

---

## Key Findings (Preliminary)

| Mixture (% synthetic) | FVD ↓ | CLIP Score ↑ | Train Clips |
|-----------------------|-------|--------------|-------------|
| 0 % (real only)       | 412   | 0.241        | 8 400       |
| 25 %                  | 388   | 0.249        | 10 500      |
| 50 %                  | 361   | 0.257        | 12 600      |
| 75 %                  | 374   | 0.253        | 14 700      |
| 100 % (synth only)    | 441   | 0.231        | 8 400       |

**Bias finding**: at blur threshold σ < 80 (aggressive), *"Playing guitar"* and
*"Indoor rowing"* drop from 12 % → 3 % of the corpus — static backgrounds cause
these classes to score as "low motion" and get filtered. Synthetic motion augmentation
recovers representation within 1 clip/class.

---

## Requirements

- Python 3.10+
- PyTorch 2.1+
- Ray 2.9+
- `transformers` (VideoMAE, CLIP)
- `scenedetect`, `imagehash`, `opencv-python`
- `decord` for fast video decoding

See `pyproject.toml` for pinned versions.

---

## Citation / Reference

If you build on this pipeline please cite the original dataset papers:
- UCF-101: Soomro et al. 2012
- Kinetics-400: Kay et al. 2017
- VideoMAE: Tong et al. 2022
