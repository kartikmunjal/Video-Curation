"""
Ray-parallel curation pipeline.

Wraps the four sequential curation stages (scene detection, quality filter,
motion score, dedup) into a Ray DAG so they run across all available CPUs.

Architecture
------------
              ┌──────────────────────────────────┐
  clip_path → │  SceneWorker (Ray remote fn)      │ → trimmed clip path + SceneInfo
              └──────────────────────────────────┘
                           │
              ┌──────────────────────────────────┐
              │  QualityWorker (Ray remote fn)    │ → QualityScore
              └──────────────────────────────────┘
                           │
              ┌──────────────────────────────────┐
              │  MotionWorker (Ray remote fn)     │ → MotionScore
              └──────────────────────────────────┘
                           │
              ┌──────────────────────────────────┐
              │  DedupActor  (Ray Actor)          │ → DedupResult
              │  (serialised index, single actor) │
              └──────────────────────────────────┘
                           │
                    ClipMeta manifest

Usage
-----
    from video_curation.pipeline.ray_pipeline import CurationPipeline
    import yaml

    with open("configs/curation.yaml") as f:
        cfg = yaml.safe_load(f)

    pipeline = CurationPipeline(cfg)
    manifest = pipeline.run(clip_paths)
    pipeline.save_manifest("data/curated/manifest.jsonl")
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ── Ray remote functions ──────────────────────────────────────────────────────

def _init_ray(num_cpus: Optional[int] = None, num_gpus: int = 0) -> None:
    import ray
    if not ray.is_initialized():
        ray.init(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            ignore_reinit_error=True,
            logging_level=logging.WARNING,
        )
        log.info("Ray initialised: %s", ray.cluster_resources())


def make_scene_worker():
    import ray
    from video_curation.curation.scene_detect import analyze_clip, trim_to_dominant_scene

    @ray.remote(num_cpus=1)
    def scene_worker(
        path: str,
        output_dir: str,
        detector: str,
        threshold: float,
        min_scene_len_frames: int,
        min_scene_duration: float,
        keep_longest: bool,
    ) -> Optional[dict]:
        """Detect scenes and optionally trim to dominant scene."""
        info = analyze_clip(path, detector=detector, threshold=threshold,
                            min_scene_len_frames=min_scene_len_frames)
        if not keep_longest:
            return {"path": path, "scene_info": asdict(info)}

        out_name = Path(path).stem + "_scene.mp4"
        out_path = Path(output_dir) / Path(path).parent.name / out_name
        trimmed = trim_to_dominant_scene(
            path, out_path, info=info, min_duration_sec=min_scene_duration
        )
        if trimmed is None:
            return None  # too short, reject
        return {"path": str(trimmed), "scene_info": asdict(info)}

    return scene_worker


def make_quality_worker():
    import ray
    from video_curation.curation.quality_filter import score_clip

    @ray.remote(num_cpus=1)
    def quality_worker(
        path: str,
        blur_method: str,
        blur_threshold: float,
        quality_method: str,
        quality_threshold: float,
        sample_frames: int,
        clip_score_agg: str,
    ) -> dict:
        score = score_clip(
            path,
            blur_method=blur_method,
            quality_method=quality_method,
            sample_frames=sample_frames,
            quality_sample_frames=max(2, sample_frames // 2),
            clip_score_agg=clip_score_agg,
            blur_threshold=blur_threshold,
            quality_threshold=quality_threshold,
        )
        return asdict(score)

    return quality_worker


def make_motion_worker():
    import ray
    from video_curation.curation.motion_score import score_clip

    @ray.remote(num_cpus=1)
    def motion_worker(
        path: str,
        method: str,
        sample_pairs: int,
        min_motion: float,
        max_motion: float,
    ) -> dict:
        score = score_clip(
            path,
            method=method,
            sample_pairs=sample_pairs,
            min_motion=min_motion,
            max_motion=max_motion,
        )
        return asdict(score)

    return motion_worker


def make_dedup_actor():
    import ray
    from video_curation.curation.deduplication import PHashIndex, CLIPEmbedIndex

    @ray.remote(num_cpus=1)
    class DedupActor:
        """Stateful dedup actor — single instance, serialised index."""

        def __init__(
            self,
            method: str,
            hash_fn: str,
            hash_size: int,
            hamming_threshold: int,
            sample_frames: int,
            embed_model: str,
            embed_sim_threshold: float,
        ) -> None:
            if method in ("phash", "dhash", "ahash"):
                self._index = PHashIndex(
                    hash_fn=hash_fn,
                    hash_size=hash_size,
                    hamming_threshold=hamming_threshold,
                    sample_frames=sample_frames,
                )
            else:
                self._index = CLIPEmbedIndex(
                    model_name=embed_model,
                    sim_threshold=embed_sim_threshold,
                    sample_frames=sample_frames,
                )
            self._method = method

        def check_and_add(self, path: str) -> dict:
            from dataclasses import asdict as _asdict
            result = self._index.query(path)
            if not result.is_duplicate:
                self._index.add(path)
            return _asdict(result)

        def size(self) -> int:
            return len(self._index)

        def save(self, path: str) -> None:
            self._index.save(path)

    return DedupActor


# ── Main Pipeline class ────────────────────────────────────────────────────────


class CurationPipeline:
    """End-to-end curation pipeline using Ray for parallelism.

    Parameters
    ----------
    config:
        Parsed YAML config dict (from configs/curation.yaml).
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self._manifest: list[dict] = []

    def run(
        self,
        clip_paths: list[str],
        label_map: Optional[dict[str, str]] = None,
    ) -> list[dict]:
        """Run the full curation pipeline on *clip_paths*.

        Parameters
        ----------
        clip_paths:
            List of raw video file paths.
        label_map:
            Optional {path: label} dict for recording ground-truth labels.

        Returns
        -------
        list[dict]
            Manifest entries for clips that passed all filters.
        """
        import ray
        from tqdm import tqdm

        ray_cfg = self.cfg.get("ray", {})
        _init_ray(
            num_cpus=ray_cfg.get("num_cpus"),
            num_gpus=ray_cfg.get("num_gpus", 0),
        )

        output_dir = self.cfg.get("output_dir", "data/curated")
        batch_size: int = ray_cfg.get("batch_size", 64)

        scene_cfg = self.cfg.get("scene_detect", {})
        quality_cfg = self.cfg.get("blur_filter", {})
        q2_cfg = self.cfg.get("quality_filter", {})
        motion_cfg = self.cfg.get("motion_filter", {})
        dedup_cfg = self.cfg.get("dedup", {})

        # Instantiate workers / actors
        SceneWorker = make_scene_worker()
        QualityWorker = make_quality_worker()
        MotionWorker = make_motion_worker()
        DedupActor = make_dedup_actor()

        dedup_actor = DedupActor.remote(
            method=dedup_cfg.get("method", "phash"),
            hash_fn=dedup_cfg.get("method", "phash"),
            hash_size=dedup_cfg.get("hash_size", 8),
            hamming_threshold=dedup_cfg.get("hamming_threshold", 10),
            sample_frames=dedup_cfg.get("sample_frames", 4),
            embed_model=dedup_cfg.get("embed_model", "openai/clip-vit-base-patch32"),
            embed_sim_threshold=dedup_cfg.get("embed_sim_threshold", 0.97),
        )

        manifest: list[dict] = []
        t0 = time.time()

        for batch_start in range(0, len(clip_paths), batch_size):
            batch = clip_paths[batch_start: batch_start + batch_size]
            log.info(
                "Processing batch %d/%d (%d clips)",
                batch_start // batch_size + 1,
                (len(clip_paths) - 1) // batch_size + 1,
                len(batch),
            )

            # ── Stage 1: Scene detection (parallel) ──────────────────────────
            scene_futures = [
                SceneWorker.remote(
                    path=p,
                    output_dir=output_dir,
                    detector=scene_cfg.get("detector", "content"),
                    threshold=scene_cfg.get("threshold", 27.0),
                    min_scene_len_frames=scene_cfg.get("min_scene_len_frames", 15),
                    min_scene_duration=scene_cfg.get("min_scene_len_sec", 0.5),
                    keep_longest=scene_cfg.get("keep_longest_scene", True),
                )
                for p in batch
            ]
            scene_results = ray.get(scene_futures)  # list[Optional[dict]]
            scene_passed = [r for r in scene_results if r is not None]
            log.info("After scene filter: %d/%d", len(scene_passed), len(batch))

            if not scene_passed:
                continue

            trimmed_paths = [r["path"] for r in scene_passed]

            # ── Stage 2: Quality filter (parallel) ───────────────────────────
            quality_futures = [
                QualityWorker.remote(
                    path=p,
                    blur_method=quality_cfg.get("method", "laplacian_var"),
                    blur_threshold=quality_cfg.get("threshold", 40.0),
                    quality_method=q2_cfg.get("method", "brisque"),
                    quality_threshold=q2_cfg.get("threshold", 50.0),
                    sample_frames=quality_cfg.get("sample_frames", 8),
                    clip_score_agg=quality_cfg.get("clip_score_agg", "mean"),
                )
                for p in trimmed_paths
            ]
            quality_results = ray.get(quality_futures)

            q_passed_pairs = [
                (p, s, q)
                for p, s, q in zip(trimmed_paths, scene_passed, quality_results)
                if q["passed"]
            ]
            log.info(
                "After quality filter: %d/%d", len(q_passed_pairs), len(trimmed_paths)
            )
            if not q_passed_pairs:
                continue

            qpaths = [t[0] for t in q_passed_pairs]

            # ── Stage 3: Motion filter (parallel) ────────────────────────────
            motion_futures = [
                MotionWorker.remote(
                    path=p,
                    method=motion_cfg.get("method", "farneback"),
                    sample_pairs=motion_cfg.get("sample_pairs", 6),
                    min_motion=motion_cfg.get("min_motion", 2.0),
                    max_motion=motion_cfg.get("max_motion", 80.0),
                )
                for p in qpaths
            ]
            motion_results = ray.get(motion_futures)

            m_passed = [
                (p, s, q, m)
                for (p, s, q), m in zip(q_passed_pairs, motion_results)
                if m["passed"]
            ]
            log.info(
                "After motion filter: %d/%d", len(m_passed), len(q_passed_pairs)
            )
            if not m_passed:
                continue

            # ── Stage 4: Dedup (sequential through actor — must maintain state)
            for path, scene_r, quality_r, motion_r in m_passed:
                dedup_result = ray.get(dedup_actor.check_and_add.remote(path))
                if dedup_result["is_duplicate"]:
                    continue

                # Build manifest entry
                label = (label_map or {}).get(path, Path(path).parent.name)
                entry = {
                    "path": path,
                    "label": label,
                    "label_idx": -1,  # filled in post-processing
                    "split": "train",
                    "is_synthetic": False,
                    "duration_sec": scene_r["scene_info"].get(
                        "dominant_scene_duration_sec", 0.0
                    ),
                    "fps": 0.0,  # filled from video metadata
                    "width": 0,
                    "height": 0,
                    "blur_score": quality_r["blur_score"],
                    "motion_score": motion_r["mean_flow_magnitude"],
                    "quality_score": quality_r["quality_score"],
                    "caption": None,
                    "source_path": None,
                    "phash": dedup_result.get("hash_str"),
                }
                manifest.append(entry)

        # Assign label indices
        labels = sorted({e["label"] for e in manifest})
        label2idx = {l: i for i, l in enumerate(labels)}
        for entry in manifest:
            entry["label_idx"] = label2idx[entry["label"]]

        elapsed = time.time() - t0
        n_total = len(clip_paths)
        n_kept = len(manifest)
        log.info(
            "Curation complete: %d/%d clips kept (%.1f%%) in %.1fs",
            n_kept, n_total, 100 * n_kept / max(n_total, 1), elapsed,
        )

        self._manifest = manifest
        return manifest

    def save_manifest(self, path: str | Path) -> None:
        """Write manifest to a JSONL file."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for entry in self._manifest:
                fh.write(json.dumps(entry) + "\n")
        log.info("Manifest saved: %s (%d clips)", path, len(self._manifest))

    @property
    def stats(self) -> dict:
        if not self._manifest:
            return {}
        labels = [e["label"] for e in self._manifest]
        from collections import Counter
        return {
            "total_clips": len(self._manifest),
            "n_classes": len(set(labels)),
            "class_distribution": dict(Counter(labels)),
            "mean_blur": sum(e["blur_score"] for e in self._manifest) / len(self._manifest),
            "mean_motion": sum(e["motion_score"] for e in self._manifest) / len(self._manifest),
        }
