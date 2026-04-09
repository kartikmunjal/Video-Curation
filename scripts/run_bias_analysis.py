#!/usr/bin/env python3
"""
Run the quality-threshold bias sweep and produce representation drift plots.

    python scripts/run_bias_analysis.py \\
        --manifest data/raw_manifest.jsonl \\
        --synth_manifest data/augmented/manifest.jsonl \\
        --output_dir results/bias_analysis

The raw manifest must include ``blur_score`` and ``motion_score`` fields
(populated by the curation pipeline before any threshold filtering).
If you ran run_curation.py with --blur_threshold 0, the output manifest
already has scores but all clips retained.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_curation.evaluation.bias_analysis import run_bias_sweep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True,
                   help="Unfiltered manifest with blur_score populated")
    p.add_argument("--synth_manifest", default=None,
                   help="Augmented manifest for recovery analysis")
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0, 20, 40, 60, 80, 100, 120],
                   help="Blur thresholds to sweep")
    p.add_argument("--output_dir", default="results/bias_analysis")
    return p.parse_args()


def main():
    args = parse_args()
    df = run_bias_sweep(
        unfiltered_manifest=args.manifest,
        synth_manifest=args.synth_manifest,
        thresholds=args.thresholds,
        output_dir=args.output_dir,
    )
    print(f"\nSweep complete. {len(df)} rows written to {args.output_dir}/")
    print("\nSample (threshold=80, sorted by drift):")
    subset = df[df["threshold"] == 80].sort_values("drift")
    print(subset[["class", "retention_rate", "drift", "n_clips_after"]].to_string(index=False))


if __name__ == "__main__":
    main()
