"""
CLI entry point for the curation pipeline.

    python -m video_curation.pipeline.curator --config configs/curation.yaml
    # or via installed script:
    vc-curate --config configs/curation.yaml
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import yaml

from video_curation.pipeline.ray_pipeline import CurationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the video curation pipeline on a directory of clips."
    )
    p.add_argument("--config", required=True, help="Path to curation.yaml")
    p.add_argument(
        "--input_dir",
        default=None,
        help="Override input_dir from config",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Override output_dir from config",
    )
    p.add_argument(
        "--extensions",
        nargs="+",
        default=["mp4", "avi", "mov", "mkv"],
        help="Video file extensions to scan",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N clips (for testing)",
    )
    p.add_argument(
        "--blur_threshold",
        type=float,
        default=None,
        help="Override blur threshold for bias sweep",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print stats without writing output",
    )
    return p.parse_args(argv)


def collect_clips(input_dir: str, extensions: list[str]) -> list[str]:
    paths: list[str] = []
    for ext in extensions:
        paths.extend(glob.glob(f"{input_dir}/**/*.{ext}", recursive=True))
        paths.extend(glob.glob(f"{input_dir}/**/*.{ext.upper()}", recursive=True))
    return sorted(set(paths))


def main(argv=None) -> None:
    args = parse_args(argv)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    if args.input_dir:
        cfg["input_dir"] = args.input_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.blur_threshold is not None:
        cfg.setdefault("blur_filter", {})["threshold"] = args.blur_threshold

    input_dir = cfg["input_dir"]
    log.info("Scanning for clips in: %s", input_dir)
    clips = collect_clips(input_dir, args.extensions)

    if not clips:
        log.error("No video files found in %s", input_dir)
        sys.exit(1)

    if args.limit:
        clips = clips[: args.limit]

    log.info("Found %d clips to process", len(clips))

    if args.dry_run:
        log.info("[dry_run] Would process %d clips with config: %s", len(clips), cfg)
        return

    pipeline = CurationPipeline(cfg)
    manifest = pipeline.run(clips)

    log.info("Pipeline stats: %s", pipeline.stats)

    manifest_path = cfg.get("manifest_path", "data/curated/manifest.jsonl")
    pipeline.save_manifest(manifest_path)
    log.info("Done. Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
