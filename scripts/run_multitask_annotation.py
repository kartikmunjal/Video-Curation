"""
Stage 2c — Multitask Annotation CLI.

Enriches a curated manifest with three task signals:
  • Caption   (BLIP-2 / LLaVA)
  • Optical flow  (Farneback or RAFT-small)
  • Depth estimate  (DPT-Large)

Writes an enriched manifest_multitask.jsonl alongside the source manifest
and prints a per-class task signal quality report showing whether at-risk
classes fail differently on flow/depth than on blur.

Usage:
    # Full annotation with all three signals
    python scripts/run_multitask_annotation.py \
        --manifest data/curated/blur40/manifest.jsonl \
        --output_dir data/tasks \
        --device cuda

    # Caption-only (fast, no GPU required for BLIP-2 with --device cpu)
    python scripts/run_multitask_annotation.py \
        --manifest data/curated/blur40/manifest.jsonl \
        --output_dir data/tasks \
        --skip_flow --skip_depth --device cpu

    # Flow-only with RAFT backend
    python scripts/run_multitask_annotation.py \
        --manifest data/curated/blur40/manifest.jsonl \
        --output_dir data/tasks \
        --flow_method raft --skip_caption --skip_depth

    # Dry-run: annotate first 20 clips only
    python scripts/run_multitask_annotation.py \
        --manifest data/curated/blur40/manifest.jsonl \
        --output_dir data/tasks \
        --max_clips 20 --device cpu
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_curation.curation.multitask_annotator import (
    MultitaskAnnotator,
    MultitaskAnnotatorConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2c: Multitask clip annotation")
    p.add_argument("--manifest", required=True, help="Input curated manifest JSONL")
    p.add_argument("--output_dir", default="data/tasks", help="Root dir for task artifacts")
    p.add_argument("--output_manifest", default=None,
                   help="Output manifest path (default: <manifest_dir>/manifest_multitask.jsonl)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--caption_model", default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--flow_method", default="farneback", choices=["farneback", "raft"])
    p.add_argument("--depth_model", default="Intel/dpt-large")
    p.add_argument("--skip_caption", action="store_true")
    p.add_argument("--skip_flow", action="store_true")
    p.add_argument("--skip_depth", action="store_true")
    p.add_argument("--max_clips", type=int, default=None, help="Limit to first N clips")
    p.add_argument("--print_report", action="store_true", default=True,
                   help="Print per-class task signal quality report after annotation")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.manifest).exists():
        logger.error(f"Manifest not found: {args.manifest}")
        sys.exit(1)

    # Optionally truncate manifest for dry-run
    if args.max_clips:
        import tempfile, os
        with open(args.manifest) as f:
            lines = f.readlines()[:args.max_clips]
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.writelines(lines)
        tmp.close()
        source_manifest = tmp.name
        logger.info(f"Dry-run mode: annotating first {args.max_clips} clips")
    else:
        source_manifest = args.manifest

    cfg = MultitaskAnnotatorConfig(
        caption_model=args.caption_model,
        flow_method=args.flow_method,
        depth_model=args.depth_model,
        device=args.device,
        skip_caption=args.skip_caption,
        skip_flow=args.skip_flow,
        skip_depth=args.skip_depth,
    )

    annotator = MultitaskAnnotator(config=cfg)

    enriched = annotator.annotate_manifest(
        manifest_path=source_manifest,
        output_dir=args.output_dir,
        output_manifest=args.output_manifest,
    )

    logger.info(f"Annotated {len(enriched)} clips.")

    # Clean up temp file if used
    if args.max_clips:
        os.unlink(source_manifest)

    if args.print_report:
        try:
            report = annotator.compute_task_quality_report(enriched)
            print("\n=== Per-Class Task Signal Quality Report ===")
            print(report.to_string())
            print()
            print("Interpretation:")
            print("  blur_score:            Laplacian variance — low = at-risk for blur filter")
            print("  flow_mean_magnitude:   Mean optical flow — low = near-static clips")
            print("  flow_direction_entropy:Flow isotropy — low = coherent directional motion")
            print("  depth_variance:        Depth map variance — low = flat/uniform scene")
            print()
            print("Key question: do at-risk classes (PlayingGuitar, Rowing) have low blur_score")
            print("but non-low flow/depth signals? If so, the blur filter is the only problem.")
            print("If they also have low depth_variance, the filming environment affects all signals.")
        except Exception as e:
            logger.warning(f"Could not compute report: {e}")


if __name__ == "__main__":
    main()
