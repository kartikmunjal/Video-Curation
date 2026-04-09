#!/usr/bin/env python3
"""
Download UCF-101 or Kinetics-400 subset for the ablation experiments.

Examples
--------
    # UCF-101, default 10-class subset
    python scripts/download_data.py --dataset ucf101

    # UCF-101, custom classes
    python scripts/download_data.py --dataset ucf101 \
        --classes Basketball Biking TennisSwing

    # Kinetics-400, 5 classes, 100 clips each
    python scripts/download_data.py --dataset kinetics \
        --classes "playing guitar" "swimming" "cycling" \
        --max_clips 100
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_curation.data.downloader import (
    UCF101Downloader,
    UCF101_DEFAULT_CLASSES,
    KineticsDownloader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        choices=["ucf101", "kinetics"],
        default="ucf101",
        help="Which dataset to download",
    )
    p.add_argument(
        "--root",
        default="data/raw",
        help="Root directory for raw video files",
    )
    p.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Specific class names (default: built-in 10-class subset)",
    )
    p.add_argument(
        "--split",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="UCF-101 train/test split index",
    )
    p.add_argument(
        "--max_clips",
        type=int,
        default=200,
        help="Kinetics: max clips per class",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Download parallelism",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset == "ucf101":
        classes = args.classes or UCF101_DEFAULT_CLASSES
        log.info("Downloading UCF-101 subset: %s", classes)
        dl = UCF101Downloader(
            root=Path(args.root) / "ucf101",
            classes=classes,
            split=args.split,
        )
        videos = dl.download()
        log.info("Done. %d clips ready.", len(videos))

    elif args.dataset == "kinetics":
        classes = args.classes
        if not classes:
            log.error("--classes required for kinetics download")
            sys.exit(1)
        log.info("Downloading Kinetics-400 subset: %s", classes)
        dl = KineticsDownloader(
            root=Path(args.root) / "kinetics400",
            classes=classes,
            max_clips_per_class=args.max_clips,
        )
        clips = dl.build_clip_list(split="train")
        videos = dl.download_clips(clips, num_workers=args.workers)
        log.info("Done. %d clips downloaded.", len(videos))


if __name__ == "__main__":
    main()
