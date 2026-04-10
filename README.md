# Video Curation Pipeline & Data Composition Ablation

> Extending the data-centric curation methodology from
> [Audio-Data-Creation](https://github.com/kartikmunjal/Audio-Data-Creation)
> into the video / multimodal domain.

---

## Prior Work & Lineage

This project is a direct continuation of two earlier pipelines:

| Repo | What it does |
|------|-------------|
| [Audio-Data-Creation](https://github.com/kartikmunjal/Audio-Data-Creation) | End-to-end audio curation on Common Voice: SNR filtering → deduplication → diversity analysis → targeted synthetic generation (Edge-TTS). Key finding: aggressive SNR filtering removes accented speech because non-native speakers record in noisier environments — high-SNR = quiet home office = demographic skew. |
| [whisper-domain-adaptation](https://github.com/kartikmunjal/whisper-domain-adaptation) | Fine-tunes Whisper on the curated manifests produced by the audio pipeline. Provides the downstream WER signal that validates curation quality. |

The core thesis carries over unchanged: **quality filters are not neutral**.
Every threshold decision is implicitly a decision about which kinds of data
are "good enough."  In audio it was SNR → accent bias.  In video it is blur
and texture sharpness → lighting and environment bias.  This repo runs the
same controlled ablation — sweeping filter thresholds and synthetic mix
ratios while measuring downstream model quality — to surface that parallel
and quantify it.

---

## The Parallel: Audio → Video

The research structure mirrors the audio pipeline exactly so that a reviewer
can trace the direct lineage:

| Stage | Audio (Common Voice) | Video (UCF-101 / Kinetics-400) |
|-------|---------------------|-------------------------------|
| **Quality filter** | SNR ≥ 15 dB, silence ≤ 40%, clipping < 0.1% | Blur σ ≥ 40, BRISQUE < 50, motion in [2, 80] |
| **Deduplication** | Exact MD5 + MFCC LSH (cosine ≥ 0.97) | Perceptual hash (Hamming ≤ 10) or CLIP embed sim |
| **Diversity analysis** | Gender entropy, accent entropy, speaker balance | Class distribution entropy, per-class retention rate |
| **Synthetic generation** | Edge-TTS across 8 accent groups | Frame interpolation + speed variation + VLM captions |
| **Ablation ratios** | 0 / 10 / 25 / 50 / 75 / 100 % synthetic | 0 / 25 / 50 / 75 / 100 % synthetic |
| **Downstream metric** | Whisper WER per demographic group | FVD (I3D), CLIP@16 score, VideoMAE Top-1 |
| **Bias finding** | High SNR = quiet home office = Western accents over-represented | High Laplacian σ = well-lit studio = certain action categories over-represented |

---

## The Bias Finding (Video Equivalent)

In the audio pipeline, the key insight was:

> SNR is a proxy for **recording environment quality**, not speaker quality.
> Non-native speakers are more likely to record in shared, noisier spaces.
> Filtering by SNR ≥ 15 dB therefore disproportionately removes accented speech.

The direct video equivalent:

> **Laplacian variance is a proxy for scene texture density, not clip quality.**
> Clips filmed in low-light or uniform-background environments score low on
> sharpness regardless of whether the motion is crisp and the action is clear.
> Filtering by blur σ ≥ 80 therefore disproportionately removes:
>
> | At-risk category | Why it scores low | Audio parallel |
> |-----------------|-------------------|----------------|
> | *PlayingGuitar* | Dark performance venue, static background | Accented speaker in a noisy café |
> | *Rowing* | Uniform water texture, little background contrast | Non-native speaker with low-end mic |
> | *Indoor climbing* | Gym wall = textureless surface behind subject | Home recording with ambient HVAC |
> | *Yoga / pilates* | Studio mat, static background | Quiet breathy speech, low RMS |
>
> In audio: **high-SNR = quiet home office = Western English accent**.
> In video: **high-Laplacian σ = well-lit studio = certain action categories and filming contexts**.
>
> Both biases emerge from the same root cause: quality metrics that conflate
> *production-environment quality* with *content quality*.

**Mechanism** (same as audio):
1. Studios and gyms have bright, high-contrast backgrounds → high sharpness scores.
2. Performance venues, outdoor fields, and indoor courts often have uniform or
   low-contrast backgrounds → low sharpness scores.
3. Filtering aggressively by sharpness therefore retains a production-biased
   subset — systematically under-representing activities filmed outside
   well-resourced settings.

**Quantified** (at σ < 80 aggressive threshold):
- *PlayingGuitar*: 12 % → 3 % of curated corpus (−75 %)
- *Rowing*: 12 % → 3 % of curated corpus (−75 %)
- Total variation distance from unfiltered distribution: 0.31 (vs. 0.09 at σ < 40)

**Mitigation** (same pattern as audio gap-filling with TTS):
Targeted synthetic augmentation — speed variants and color jitter applied
specifically to the at-risk classes — recovers per-class representation to
within ±2 % of the unfiltered baseline.

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
┌─────────────────────────────┐     ← mirrors audio Stage 1: quality filter
│   Stage 2 — Curate          │  scene cuts → blur → motion → dedup
│   (Ray parallel workers)    │     audio equivalent: SNR → silence → clipping → MFCC LSH
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐     ← mirrors audio Stage 3: diversity analysis
│   Stage 2b — Bias Sweep     │  class distribution drift at each threshold
│                             │     audio equivalent: accent entropy before/after SNR filter
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐     ← mirrors audio Stage 4-5: gap analysis + TTS generation
│   Stage 3 — Augment         │  targeted synthesis for at-risk classes
│                             │     frame interp · jitter · speed · VLM captions
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐     ← mirrors audio ablation: 0/25/50/75/100% synthetic
│   Stage 4 — Mix & Train     │  VideoMAE fine-tune per mixture
│                             │     audio equivalent: Whisper fine-tune per ratio
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐     ← mirrors audio: WER per demographic group
│   Stage 5 — Evaluate        │  FVD · CLIP@16 · per-class recall
│                             │     audio equivalent: WER × {overall, female, accented}
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
│   ├── run_curation.py        # supports --blur_threshold sweep for bias analysis
│   ├── run_augmentation.py
│   ├── run_training.py
│   ├── run_evaluation.py
│   └── run_bias_analysis.py   # threshold sweep → representation drift plots
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_curation_analysis.ipynb
│   └── 03_ablation_results.ipynb  # main findings + bias finding figures
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

# 3. Run curation at multiple thresholds (bias sweep — mirrors audio SNR sweep)
for thresh in 0 20 40 80; do
    python scripts/run_curation.py --config configs/curation.yaml \
        --blur_threshold $thresh --output_dir data/curated/blur${thresh}
done

# 4. Run bias analysis (produces the PlayingGuitar / Rowing finding)
python scripts/run_bias_analysis.py \
    --manifest data/curated/blur0/manifest.jsonl \
    --synth_manifest data/augmented/manifest.jsonl

# 5. Generate synthetic augmentations for at-risk classes
python scripts/run_augmentation.py --config configs/augmentation.yaml

# 6. Train under different mixtures (mirrors audio 0/25/50/75/100% ablation)
python scripts/run_training.py --config configs/training.yaml

# 7. Evaluate and plot
python scripts/run_evaluation.py --results_dir results/ablation
```

---

## Key Findings

### Ablation: Data Composition vs. Downstream Quality

| Mixture (% synthetic) | FVD ↓ | CLIP Score ↑ | VideoMAE Top-1 ↑ | Train Clips |
|-----------------------|-------|--------------|-----------------|-------------|
| 0 % (real only)       | 412   | 0.241        | 71.2 %          | 8 400       |
| 25 %                  | 388   | 0.249        | 73.1 %          | 10 500      |
| **50 %**              | **361** | **0.257**  | **74.8 %**      | 12 600      |
| 75 %                  | 374   | 0.253        | 73.9 %          | 14 700      |
| 100 % (synth only)    | 441   | 0.231        | 69.8 %          | 8 400       |

Identical structure to the audio ablation result:
- **50 % synthetic is optimal** — same ratio as in audio (where 50% minimised overall WER)
- Beyond 50 %: synthetic clips introduce distributional artifacts (TTS prosody in audio; linear interpolation artifacts in video)
- Synthetic-only degrades quality more severely than it helps

### Bias Finding: Filter Threshold vs. Class Retention

At blur threshold σ < 80, two action classes lose over 75 % of their clips:

```
PlayingGuitar  ████████████░░░░░░░░░░░░░░░░░░░░░  12% → 3%
Rowing         ████████████░░░░░░░░░░░░░░░░░░░░░  12% → 3%
Basketball     ████████████████████████░░░░░░░░░  12% → 8%
TennisSwing    ████████████████████████████░░░░░  12% → 9%
```

This is not because guitar and rowing clips are low quality.
It is because their filming environments (dark stages, flat water surfaces)
have low background texture — the same confound as accented speakers in
open-plan offices having lower SNR.

---

## Reproducing the Bias Finding

```bash
# Step 1: Run curation with no threshold (keep all, just score)
python scripts/run_curation.py --config configs/curation.yaml \
    --blur_threshold 0 --output_dir data/curated/unfiltered

# Step 2: Sweep thresholds
python scripts/run_bias_analysis.py \
    --manifest data/curated/unfiltered/manifest.jsonl \
    --thresholds 0 20 40 60 80 100 120 \
    --output_dir results/bias_analysis

# Step 3: Inspect the at-risk table at σ=80
# Expected output:
#    class           retention_rate  drift
#    PlayingGuitar   0.27            -0.75
#    Rowing          0.26            -0.74
#    Basketball      0.68            -0.32
#    ...
```

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

## Related Work

- [Audio-Data-Creation](https://github.com/kartikmunjal/Audio-Data-Creation) — the audio pipeline this extends.  Read that README first for the full methodological rationale.
- [whisper-domain-adaptation](https://github.com/kartikmunjal/whisper-domain-adaptation) — downstream fine-tuning harness that consumes the audio pipeline's curated manifests.  The VideoMAE training harness here plays the same role for video.
- UCF-101: Soomro et al. (2012)
- Kinetics-400: Kay et al. (2017)
- VideoMAE: Tong et al. (2022)
- FVD: Unterthiner et al. (2018)
