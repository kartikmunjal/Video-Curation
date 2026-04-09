"""
Ray-parallel augmentation orchestrator.

Reads the curated manifest and produces a synthetic manifest by applying
all enabled augmentation stages in parallel.  Each real clip generates
``augmentation_multiplier`` synthetic clips.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)


def make_aug_worker():
    """Create a Ray remote function that applies all augmentations to one clip."""
    import ray

    @ray.remote(num_cpus=1)
    def aug_worker(
        entry: dict,
        output_dir: str,
        interp_cfg: dict,
        jitter_cfg: dict,
        speed_cfg: dict,
        seed: int,
    ) -> list[dict]:
        """Apply augmentations to one clip; return list of synthetic clip entries."""
        from video_curation.augmentation.frame_interpolation import apply_frame_interpolation
        from video_curation.augmentation.color_jitter import jitter_clip
        from video_curation.augmentation.speed_variation import generate_speed_variants

        src_path = Path(entry["path"])
        out_dir = Path(output_dir) / entry.get("label", "unknown")
        out_dir.mkdir(parents=True, exist_ok=True)

        synthetic: list[dict] = []

        def _make_entry(path: str, aug_type: str) -> dict:
            e = dict(entry)
            e["path"] = str(path)
            e["is_synthetic"] = True
            e["source_path"] = entry["path"]
            e["caption"] = None  # will be filled by captioner
            return e

        rng = random.Random(seed)

        # 1. Frame interpolation
        if interp_cfg.get("enabled", True):
            out_path = out_dir / (src_path.stem + "_interp.mp4")
            result = apply_frame_interpolation(
                src_path, out_path,
                method=interp_cfg.get("method", "linear"),
                multiplier=interp_cfg.get("target_fps_multiplier", 2),
            )
            if result:
                synthetic.append(_make_entry(str(result), "interp"))

        # 2. Color jitter
        if jitter_cfg.get("enabled", True) and rng.random() < jitter_cfg.get("apply_prob", 0.8):
            out_path = out_dir / (src_path.stem + "_jitter.mp4")
            result = jitter_clip(
                src_path, out_path,
                brightness=jitter_cfg.get("brightness", 0.3),
                contrast=jitter_cfg.get("contrast", 0.3),
                saturation=jitter_cfg.get("saturation", 0.3),
                hue=jitter_cfg.get("hue", 0.1),
                apply_prob=1.0,  # already decided above
                seed=seed,
            )
            if result:
                synthetic.append(_make_entry(str(result), "jitter"))

        # 3. Speed variation
        if speed_cfg.get("enabled", True) and rng.random() < speed_cfg.get("apply_prob", 0.5):
            variants = generate_speed_variants(
                src_path,
                out_dir,
                factors=speed_cfg.get("factors", [0.75, 1.25]),
                method=speed_cfg.get("method", "frame_sampling"),
            )
            for v in variants:
                synthetic.append(_make_entry(str(v), "speed"))

        return synthetic

    return aug_worker


class AugmentationPipeline:
    """Ray-parallel augmentation pipeline.

    Parameters
    ----------
    config:
        Parsed augmentation.yaml dict.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config

    def run(
        self,
        manifest_entries: list[dict],
        multiplier: Optional[float] = None,
    ) -> list[dict]:
        """Generate synthetic augmentations for all clips in *manifest_entries*.

        Returns a NEW list of synthetic-only manifest entries (does not
        include the originals).
        """
        import ray
        from tqdm import tqdm

        if not ray.is_initialized():
            ray_cfg = self.cfg.get("ray", {})
            ray.init(
                num_cpus=ray_cfg.get("num_cpus"),
                num_gpus=ray_cfg.get("num_gpus", 0),
                ignore_reinit_error=True,
                logging_level=logging.WARNING,
            )

        mult = multiplier or self.cfg.get("augmentation_multiplier", 1.0)
        output_dir = self.cfg["output_dir"]
        interp_cfg = self.cfg.get("frame_interpolation", {})
        jitter_cfg = self.cfg.get("color_jitter", {})
        speed_cfg = self.cfg.get("speed_variation", {})
        batch_size = self.cfg.get("ray", {}).get("batch_size", 32)

        AugWorker = make_aug_worker()

        synthetic_manifest: list[dict] = []
        rng = random.Random(42)

        # Sample clips according to multiplier
        n_clips = len(manifest_entries)
        n_to_aug = int(n_clips * mult)
        clips_to_aug = rng.choices(manifest_entries, k=n_to_aug)

        log.info(
            "Augmenting %d clips (multiplier=%.2f, total synth target=%d)",
            n_clips, mult, n_to_aug,
        )

        for batch_start in range(0, len(clips_to_aug), batch_size):
            batch = clips_to_aug[batch_start: batch_start + batch_size]
            futures = [
                AugWorker.remote(
                    entry=e,
                    output_dir=output_dir,
                    interp_cfg=interp_cfg,
                    jitter_cfg=jitter_cfg,
                    speed_cfg=speed_cfg,
                    seed=rng.randint(0, 2**31),
                )
                for e in batch
            ]
            results = ray.get(futures)
            for synth_list in results:
                synthetic_manifest.extend(synth_list)

        log.info("Generated %d synthetic clips", len(synthetic_manifest))
        return synthetic_manifest

    def save_manifest(self, entries: list[dict], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")
        log.info("Augmented manifest saved: %s (%d clips)", path, len(entries))
