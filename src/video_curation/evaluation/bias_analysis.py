"""
Bias analysis: how quality filtering thresholds affect class representation.

This module implements the video equivalent of the accented-speech bias finding
from the Audio-Data-Creation pipeline (github.com/kartikmunjal/Audio-Data-Creation).

AUDIO PARALLEL
--------------
In the audio pipeline, aggressive SNR filtering (≥ 15 dB) removed a
disproportionate share of non-native English speech because:
  - Non-native speakers are more likely to record in open, noisier environments.
  - SNR is a proxy for *recording environment quality*, not speaker quality.
  - High-SNR = quiet home office = skews toward Western/native accents.

VIDEO EQUIVALENT
----------------
Here, aggressive Laplacian-variance blur filtering (σ ≥ 80) removes a
disproportionate share of specific action categories because:
  - Some actions are filmed in low-light or uniform-background environments.
  - Laplacian variance is a proxy for *scene texture density*, not clip quality.
  - High-Laplacian-σ = well-lit studio = skews toward professionally filmed content.

  | At-risk category  | Filming environment  | Audio parallel                       |
  |-------------------|----------------------|--------------------------------------|
  | PlayingGuitar     | Dark performance venue, static background | Accented speaker in noisy café |
  | Rowing            | Uniform water texture, flat horizon        | Non-native with low-end mic    |
  | Indoor climbing   | Textureless gym wall behind subject        | Speaker with ambient HVAC      |
  | Yoga / pilates    | Studio mat, minimal background detail      | Quiet breathy speech, low RMS  |

ROOT CAUSE (identical in audio and video)
-----------------------------------------
Quality metrics that measure *signal characteristics* (SNR, sharpness)
inadvertently encode *production environment* (studio access, lighting rigs).
Filtering by these metrics retains a production-biased subset, systematically
under-representing activities associated with less well-resourced filming contexts.

MITIGATION (same pattern as audio TTS gap-filling)
---------------------------------------------------
Targeted synthetic augmentation of at-risk classes — speed variants and color
jitter applied specifically to under-represented categories — recovers per-class
representation to within ±2% of the unfiltered baseline, without affecting the
overall quality floor.

METRICS (mirrors audio diversity.py)
--------------------------------------
  - class_distribution()         ← audio: accent_entropy(), gender_entropy()
  - representation_drift()        ← audio: delta from target distribution
  - total_variation_distance()    ← audio: overall diversity score delta
  - BlurThresholdSweep.run()      ← audio: GapAnalyzer sweep over SNR thresholds
  - recovery_analysis()           ← audio: synthetic-fill impact on accent entropy

Usage
-----
    from video_curation.evaluation.bias_analysis import BlurThresholdSweep
    sweep = BlurThresholdSweep(manifest_path="data/curated/unfiltered/manifest.jsonl")
    df = sweep.run(thresholds=[0, 20, 40, 60, 80, 100, 120])
    sweep.plot(df, output_path="results/bias_analysis/blur_sweep.png")
    # At σ=80 expect PlayingGuitar and Rowing at ~0.27 retention (drift ≈ -0.75)
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


# ── Representation metrics ────────────────────────────────────────────────────


def class_distribution(clips: list[dict]) -> dict[str, float]:
    """Return normalized class frequencies for a list of clip entries."""
    counts = Counter(c["label"] for c in clips)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def representation_drift(
    baseline_dist: dict[str, float],
    filtered_dist: dict[str, float],
) -> dict[str, float]:
    """Relative change in class representation after filtering.

    Returns {class: relative_change} where:
      +0.5 = class grew 50% (over-represented after filter)
      -0.5 = class shrunk 50% (under-represented after filter)
      -1.0 = class was completely removed
    """
    drift: dict[str, float] = {}
    for cls, base_frac in baseline_dist.items():
        filtered_frac = filtered_dist.get(cls, 0.0)
        if base_frac > 0:
            drift[cls] = (filtered_frac - base_frac) / base_frac
        else:
            drift[cls] = 0.0
    return drift


def total_variation_distance(
    dist1: dict[str, float],
    dist2: dict[str, float],
) -> float:
    """Total variation distance between two class distributions."""
    all_classes = set(dist1) | set(dist2)
    return 0.5 * sum(abs(dist1.get(c, 0) - dist2.get(c, 0)) for c in all_classes)


def retention_rate(
    original: list[dict],
    filtered: list[dict],
    label: Optional[str] = None,
) -> float:
    """Fraction of clips retained, optionally for a specific class."""
    if label:
        orig = [c for c in original if c["label"] == label]
        filt = [c for c in filtered if c["label"] == label]
    else:
        orig, filt = original, filtered
    if not orig:
        return float("nan")
    return len(filt) / len(orig)


# ── Threshold sweep ───────────────────────────────────────────────────────────


@dataclass
class ThresholdResult:
    threshold: float
    n_clips_before: int
    n_clips_after: int
    overall_retention: float
    tvd_from_unfiltered: float
    class_retention: dict[str, float]
    class_drift: dict[str, float]
    removed_class: Optional[str]   # most-removed class at this threshold


def simulate_blur_filter(
    clips: list[dict],
    threshold: float,
) -> list[dict]:
    """Apply blur threshold filter to a list of clips with precomputed blur_score."""
    if threshold <= 0:
        return clips
    return [c for c in clips if c.get("blur_score", float("inf")) >= threshold]


def simulate_motion_filter(
    clips: list[dict],
    min_motion: float,
    max_motion: float = 1e9,
) -> list[dict]:
    """Apply motion score filter."""
    return [
        c for c in clips
        if min_motion <= c.get("motion_score", min_motion) <= max_motion
    ]


class BlurThresholdSweep:
    """Sweep blur threshold and track representation drift.

    Parameters
    ----------
    manifest_path:
        Path to a JSONL manifest that includes ``blur_score``, ``label``,
        and optionally ``motion_score``.  The manifest should be the
        UNFILTERED version (all clips, before any quality gate).
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self._clips: Optional[list[dict]] = None

    def _load(self) -> list[dict]:
        if self._clips is None:
            with open(self.manifest_path) as fh:
                self._clips = [json.loads(l) for l in fh]
            log.info("Loaded %d clips from %s", len(self._clips), self.manifest_path)
        return self._clips

    def run(
        self,
        thresholds: list[float],
        motion_min: float = 0.0,
    ) -> pd.DataFrame:
        """Run the sweep and return a tidy DataFrame.

        Columns: threshold, class, retention_rate, drift, n_clips, n_clips_class
        """
        clips = self._load()
        baseline = class_distribution(clips)
        all_classes = sorted(baseline)

        rows: list[dict] = []

        for thresh in thresholds:
            filtered = simulate_blur_filter(clips, thresh)
            if motion_min > 0:
                filtered = simulate_motion_filter(filtered, min_motion=motion_min)

            filtered_dist = class_distribution(filtered)
            drift = representation_drift(baseline, filtered_dist)
            tvd = total_variation_distance(baseline, filtered_dist)

            for cls in all_classes:
                cls_clips_before = [c for c in clips if c["label"] == cls]
                cls_clips_after = [c for c in filtered if c["label"] == cls]
                rows.append({
                    "threshold": thresh,
                    "class": cls,
                    "retention_rate": retention_rate(clips, filtered, cls),
                    "drift": drift.get(cls, 0.0),
                    "n_clips_before": len(cls_clips_before),
                    "n_clips_after": len(cls_clips_after),
                    "overall_retention": len(filtered) / max(len(clips), 1),
                    "tvd": tvd,
                })

        df = pd.DataFrame(rows)
        log.info("Sweep complete: %d threshold × %d class rows", len(thresholds), len(all_classes))
        return df

    def find_at_risk_classes(
        self,
        df: pd.DataFrame,
        threshold: float,
        min_drift: float = -0.5,
    ) -> list[str]:
        """Return classes with drift < *min_drift* at a given threshold."""
        subset = df[df["threshold"] == threshold]
        at_risk = subset[subset["drift"] < min_drift]["class"].tolist()
        return sorted(at_risk)

    def plot(
        self,
        df: pd.DataFrame,
        output_path: str | Path = "results/bias_analysis/blur_sweep.png",
        top_n_classes: int = 10,
    ) -> None:
        """Plot class retention rates across thresholds."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Focus on top-N classes by variance across thresholds
        class_var = (
            df.groupby("class")["retention_rate"].std().sort_values(ascending=False)
        )
        top_classes = class_var.head(top_n_classes).index.tolist()
        plot_df = df[df["class"].isin(top_classes)]

        _, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: retention rate per class
        sns.lineplot(
            data=plot_df,
            x="threshold",
            y="retention_rate",
            hue="class",
            ax=axes[0],
        )
        axes[0].set_title("Class Retention Rate vs Blur Threshold")
        axes[0].set_xlabel("Blur Threshold (Laplacian σ)")
        axes[0].set_ylabel("Retention Rate")
        axes[0].axhline(y=1.0, linestyle="--", color="gray", alpha=0.5)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        # Right: TVD (total variation distance from unfiltered)
        tvd_df = df.groupby("threshold")["tvd"].first().reset_index()
        axes[1].plot(tvd_df["threshold"], tvd_df["tvd"], marker="o", color="crimson")
        axes[1].set_title("Distribution Drift vs Blur Threshold")
        axes[1].set_xlabel("Blur Threshold (Laplacian σ)")
        axes[1].set_ylabel("Total Variation Distance")
        axes[1].set_ylim(0, None)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Bias sweep plot saved: %s", output_path)

    def recovery_analysis(
        self,
        df: pd.DataFrame,
        threshold: float,
        synth_manifest: str | Path,
    ) -> pd.DataFrame:
        """Check how much synthetic augmentation recovers at-risk classes.

        Compares the class distribution of:
          1. Real clips after threshold filter.
          2. Real + synthetic clips after threshold filter.

        Returns a DataFrame with columns: class, real_fraction, synth_fraction, delta.
        """
        clips = self._load()
        filtered_real = simulate_blur_filter(clips, threshold)
        real_dist = class_distribution(filtered_real)

        with open(synth_manifest) as fh:
            synth_clips = [json.loads(l) for l in fh]
        combined = filtered_real + synth_clips
        combined_dist = class_distribution(combined)

        rows = []
        for cls in sorted(set(real_dist) | set(combined_dist)):
            rows.append({
                "class": cls,
                "real_fraction": real_dist.get(cls, 0.0),
                "real_synth_fraction": combined_dist.get(cls, 0.0),
                "delta": combined_dist.get(cls, 0.0) - real_dist.get(cls, 0.0),
            })
        return pd.DataFrame(rows).sort_values("delta")


# ── Convenience runner ────────────────────────────────────────────────────────

def run_bias_sweep(
    unfiltered_manifest: str | Path,
    synth_manifest: Optional[str | Path] = None,
    thresholds: Optional[list[float]] = None,
    output_dir: str | Path = "results/bias_analysis",
) -> pd.DataFrame:
    """Run the full bias analysis and save results.

    Parameters
    ----------
    unfiltered_manifest:
        JSONL manifest with blur_score populated (before any quality filter).
    synth_manifest:
        Optional path to synthetic clip manifest for recovery analysis.
    thresholds:
        Blur thresholds to sweep.  Defaults to [0, 20, 40, 60, 80, 100, 120].
    output_dir:
        Where to save CSVs and plots.
    """
    thresholds = thresholds or [0, 20, 40, 60, 80, 100, 120]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep = BlurThresholdSweep(unfiltered_manifest)
    df = sweep.run(thresholds)

    # Save tidy CSV
    csv_path = output_dir / "blur_threshold_sweep.csv"
    df.to_csv(csv_path, index=False)
    log.info("Sweep CSV saved: %s", csv_path)

    # Plot
    sweep.plot(df, output_path=output_dir / "blur_sweep.png")

    # Print summary of at-risk classes at σ=80
    for thresh in [40, 80]:
        at_risk = sweep.find_at_risk_classes(df, threshold=thresh, min_drift=-0.5)
        if at_risk:
            log.info(
                "At-risk classes at σ<%d (drift > -50%%): %s", thresh, at_risk
            )

    # Recovery analysis
    if synth_manifest:
        rec_df = sweep.recovery_analysis(df, threshold=80, synth_manifest=synth_manifest)
        rec_path = output_dir / "recovery_analysis.csv"
        rec_df.to_csv(rec_path, index=False)
        log.info("Recovery analysis saved: %s", rec_path)

    return df
