#!/usr/bin/env python3
"""
Export a curated manifest in the format expected by Video-Generation.

Video-Curation produces JSONL manifests; Video-Generation's VideoDataset
expects either a directory + captions.json or a WebVid-style TSV.
This script bridges the two.

It also implements the data-composition handoff: you specify which mixture
ratio produced the best results (default: the 50% synthetic optimal mix),
and it exports that split's training data for use in generative model training.

Usage
-----
    # Export the 50% synthetic training split for Video-Generation
    python scripts/export_for_generation.py \\
        --manifest data/splits/ratio_0p50/train.jsonl \\
        --output_dir /path/to/Video-Generation/data/curated_50pct

    # Export real-only baseline
    python scripts/export_for_generation.py \\
        --manifest data/splits/ratio_0p00/train.jsonl \\
        --output_dir /path/to/Video-Generation/data/curated_real_only

    # Write Video-Generation curation_ablation.yaml automatically
    python scripts/export_for_generation.py \\
        --all_splits data/splits \\
        --output_dir /path/to/Video-Generation/data \\
        --write_ablation_config /path/to/Video-Generation/configs/curation_ablation.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Format converters ─────────────────────────────────────────────────────────


def manifest_to_captions_json(
    entries: list[dict],
    video_output_dir: Path,
    symlink: bool = True,
) -> dict[str, str]:
    """Convert JSONL entries to Video-Generation's {stem: caption} format.

    Optionally symlinks video files into *video_output_dir* so the
    Video-Generation repo can reference them without copying.
    """
    video_output_dir.mkdir(parents=True, exist_ok=True)
    captions: dict[str, str] = {}

    for entry in entries:
        src_path = Path(entry["path"])
        if not src_path.exists():
            log.debug("Skipping missing clip: %s", src_path)
            continue

        stem = src_path.stem
        dst_path = video_output_dir / src_path.name

        # Avoid name collisions from different class directories
        if dst_path.exists() and str(dst_path.resolve()) != str(src_path.resolve()):
            stem = f"{entry.get('label', 'unknown')}_{stem}"
            dst_path = video_output_dir / f"{stem}.mp4"

        if not dst_path.exists():
            if symlink:
                try:
                    dst_path.symlink_to(src_path.resolve())
                except OSError:
                    shutil.copy2(str(src_path), str(dst_path))
            else:
                shutil.copy2(str(src_path), str(dst_path))

        # Use caption if available; fall back to label
        caption = entry.get("caption") or entry.get("label", "")
        captions[stem] = caption

    return captions


def manifest_to_jsonl_native(entries: list[dict], out_path: Path) -> None:
    """Write a Video-Curation-native JSONL manifest that VideoDataset can consume
    directly once JSONL mode is added to VideoDataset (see video_dataset.py).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    log.info("Native JSONL manifest written: %s (%d clips)", out_path, len(entries))


# ── Ablation config writer ────────────────────────────────────────────────────


_ABLATION_CONFIG_TEMPLATE = """\
# ──────────────────────────────────────────────────────────────────────────────
# Video-Generation: curation ablation configuration
# Produced by: Video-Curation/scripts/export_for_generation.py
#
# Connects to: https://github.com/kartikmunjal/Video-Curation
# Consumes manifests from: Video-Curation data/splits/
# ──────────────────────────────────────────────────────────────────────────────

# Reference to the curation pipeline that produced these splits
curation_pipeline:
  repo: "https://github.com/kartikmunjal/Video-Curation"
  dataset: "UCF-101 (10-class subset)"
  blur_threshold: 40
  dedup_method: "phash+lancedb"
  augmentation: "frame_interpolation + color_jitter + speed_variation"
  bias_finding: >
    At blur threshold σ < 80, PlayingGuitar and Rowing drop from 12% to 3%
    of the corpus (uniform backgrounds score low on Laplacian variance).
    Synthetic augmentation recovers representation.

# Ablation: train CogVideoX-2B LoRA at each mixture ratio and compare FVD
mixture_ablation:
  ratios: {ratios}
  data_dirs:
{data_dirs}
  optimal_ratio: 0.50   # from Video-Curation ablation: lowest FVD at 50% synthetic

# Model config (inherits from cogvideox_lora.yaml)
model:
  pretrained_model_name_or_path: "THUDM/CogVideoX-2b"
  dtype: "bfloat16"

lora:
  r: 16
  alpha: 32
  target_modules: [to_q, to_k, to_v, to_out.0, ff.net.0.proj, ff.net.2]

training:
  num_epochs: 3
  batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4
  output_dir: "checkpoints/curation_ablation"
  seed: 42

evaluation:
  metrics: [fvd, clip_score, motion_smoothness, temporal_consistency]
  eval_prompts_path: "data/eval_prompts.txt"
  num_videos_per_prompt: 4
"""


def write_ablation_config(
    ratio_dirs: dict[float, Path],
    output_path: Path,
) -> None:
    ratios_str = "[" + ", ".join(f"{r:.2f}" for r in sorted(ratio_dirs)) + "]"
    data_dirs_str = "\n".join(
        f"    - ratio: {r:.2f}\n      video_dir: \"{d}/videos\"\n      captions: \"{d}/captions.json\""
        for r, d in sorted(ratio_dirs.items())
    )
    config = _ABLATION_CONFIG_TEMPLATE.format(
        ratios=ratios_str,
        data_dirs=data_dirs_str,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(config)
    log.info("Ablation config written: %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", help="Single JSONL manifest to export")
    g.add_argument("--all_splits", help="Directory of ratio_* split dirs (exports all)")

    p.add_argument("--output_dir", required=True,
                   help="Destination directory for Video-Generation data")
    p.add_argument("--symlink", action="store_true", default=True,
                   help="Symlink videos instead of copying (default: True)")
    p.add_argument("--write_ablation_config",
                   help="Path to write curation_ablation.yaml for Video-Generation")
    p.add_argument("--split", default="train", help="Which split to export (default: train)")
    return p.parse_args()


def _export_single(manifest_path: Path, output_dir: Path, symlink: bool) -> None:
    with open(manifest_path) as fh:
        entries = [json.loads(l) for l in fh]

    video_dir = output_dir / "videos"
    captions = manifest_to_captions_json(entries, video_dir, symlink=symlink)

    captions_path = output_dir / "captions.json"
    with open(captions_path, "w") as fh:
        json.dump(captions, fh, indent=2)

    # Also write native JSONL for direct consumption
    manifest_to_jsonl_native(entries, output_dir / "manifest.jsonl")

    log.info(
        "Exported %d clips → %s\n  captions.json: %s\n  manifest.jsonl: %s",
        len(captions), output_dir, captions_path, output_dir / "manifest.jsonl",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.manifest:
        _export_single(Path(args.manifest), output_dir, args.symlink)

    elif args.all_splits:
        splits_dir = Path(args.all_splits)
        ratio_dirs: dict[float, Path] = {}

        for ratio_dir in sorted(splits_dir.glob("ratio_*")):
            ratio_str = ratio_dir.name.replace("ratio_", "").replace("p", ".")
            try:
                ratio = float(ratio_str)
            except ValueError:
                continue
            manifest = ratio_dir / f"{args.split}.jsonl"
            if not manifest.exists():
                log.warning("No %s.jsonl in %s", args.split, ratio_dir)
                continue
            out = output_dir / f"ratio_{ratio:.2f}".replace(".", "p")
            _export_single(manifest, out, args.symlink)
            ratio_dirs[ratio] = out

        if args.write_ablation_config and ratio_dirs:
            write_ablation_config(ratio_dirs, Path(args.write_ablation_config))

    log.info("Export complete. Point Video-Generation VideoDataset at %s", output_dir)


if __name__ == "__main__":
    main()
