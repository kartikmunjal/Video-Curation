#!/usr/bin/env python3
"""
Run the augmentation pipeline on a curated manifest.

    python scripts/run_augmentation.py --config configs/augmentation.yaml

    # With VLM captioning for all synthetic clips (slow):
    python scripts/run_augmentation.py \\
        --config configs/augmentation.yaml --caption

    # Override multiplier (e.g. generate 2 synth per real clip):
    python scripts/run_augmentation.py \\
        --config configs/augmentation.yaml --multiplier 2.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from video_curation.augmentation.augmentor import AugmentationPipeline
from video_curation.augmentation.caption_augmentation import augment_captions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to augmentation.yaml")
    p.add_argument("--multiplier", type=float, default=None,
                   help="Override augmentation_multiplier from config")
    p.add_argument("--caption", action="store_true",
                   help="Generate VLM captions for synthetic clips")
    p.add_argument("--dry_run", action="store_true",
                   help="Show what would be done without writing files")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    # Load curated manifest
    manifest_in = Path(cfg["manifest_in"])
    if not manifest_in.exists():
        log.error("Curated manifest not found: %s — run run_curation.py first", manifest_in)
        sys.exit(1)

    with open(manifest_in) as fh:
        real_clips = [json.loads(line) for line in fh]

    log.info("Loaded %d curated clips from %s", len(real_clips), manifest_in)

    if args.dry_run:
        mult = args.multiplier or cfg.get("augmentation_multiplier", 1.0)
        n_synth = int(len(real_clips) * mult) * 3  # ~3 augments per clip
        log.info("[dry_run] Would generate ~%d synthetic clips", n_synth)
        return

    pipeline = AugmentationPipeline(cfg)
    synth_clips = pipeline.run(real_clips, multiplier=args.multiplier)

    if args.caption and cfg.get("caption_augmentation", {}).get("enabled", True):
        cap_cfg = cfg["caption_augmentation"]
        log.info("Generating VLM captions for %d synthetic clips...", len(synth_clips))
        synth_clips = augment_captions(
            synth_clips,
            model_name=cap_cfg.get("model", "Salesforce/blip2-opt-2.7b"),
            device=cap_cfg.get("device", "cuda"),
            sample_frames=cap_cfg.get("sample_frames", 4),
            max_new_tokens=cap_cfg.get("max_new_tokens", 64),
            batch_size=cap_cfg.get("batch_size", 8),
        )

    manifest_out = Path(cfg["manifest_out"])
    pipeline.save_manifest(synth_clips, manifest_out)
    log.info("Done. Synthetic manifest: %s", manifest_out)


if __name__ == "__main__":
    main()
