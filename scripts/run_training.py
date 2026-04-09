#!/usr/bin/env python3
"""
Build data mixture splits and fine-tune VideoMAE across all mixture ratios.

    python scripts/run_training.py --config configs/training.yaml

    # Single ratio (quick test):
    python scripts/run_training.py --config configs/training.yaml --ratio 0.5

    # Dry run: just print split stats:
    python scripts/run_training.py --config configs/training.yaml --dry_run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from video_curation.training.data_mixture import build_splits, print_mixture_stats
from video_curation.training.finetune import run_ablation, run_finetune

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to training.yaml")
    p.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Run only a single synth ratio (e.g. 0.5)",
    )
    p.add_argument("--dry_run", action="store_true",
                   help="Print split stats without training")
    p.add_argument("--skip_build", action="store_true",
                   help="Skip split building if manifests already exist")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    ratios = cfg["mixture_ratios"]
    if args.ratio is not None:
        ratios = [args.ratio]

    splits_dir = Path(cfg["splits_dir"])
    log.info("Building mixture splits → %s", splits_dir)

    if not args.skip_build:
        split_paths = build_splits(
            real_manifest=cfg["real_manifest"],
            synth_manifest=cfg["synth_manifest"],
            output_dir=splits_dir,
            ratios=ratios,
            seed=cfg["training"]["seed"],
        )
    else:
        # Reconstruct path dict from disk
        split_paths = {}
        for ratio in ratios:
            ratio_dir = splits_dir / f"ratio_{ratio:.2f}".replace(".", "p")
            split_paths[ratio] = {
                "train": ratio_dir / "train.jsonl",
                "val": ratio_dir / "val.jsonl",
                "test": ratio_dir / "test.jsonl",
            }

    print_mixture_stats(split_paths)

    if args.dry_run:
        log.info("[dry_run] Splits ready — skipping training")
        return

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    hw_cfg = cfg["hardware"]

    results = run_ablation(
        split_paths=split_paths,
        output_dir=train_cfg["output_dir"],
        model_name=model_cfg["name"],
        num_classes=model_cfg["num_classes"],
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        grad_accum=train_cfg["gradient_accumulation_steps"],
        lr=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        fp16=train_cfg["fp16"],
        seed=train_cfg["seed"],
        num_frames=cfg["clip"]["num_frames"],
        frame_size=cfg["clip"]["frame_size"],
        fps_target=cfg["clip"]["fps"],
        device=hw_cfg["device"],
    )

    log.info("\nAblation complete:")
    log.info(f"{'Ratio':>8}  {'Top-1 Acc':>10}  {'Train Clips':>12}")
    for r in sorted(results, key=lambda x: x["synth_ratio"]):
        log.info(
            f"{r['synth_ratio']:>8.2f}  {r['best_top1']:>10.4f}  {r['train_clips']:>12}"
        )


if __name__ == "__main__":
    main()
