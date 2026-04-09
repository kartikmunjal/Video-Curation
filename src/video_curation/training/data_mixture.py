"""
Data mixture builder for ablation experiments.

Constructs train/val/test splits at each synthetic ratio
[0.0, 0.25, 0.50, 0.75, 1.0] and writes split manifests to disk.

The splits are stratified by class to ensure equal class representation
regardless of mixture ratio.  A fixed seed ensures all experiments are
comparable (same real clips in every mixture).
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_RATIOS = [0.0, 0.25, 0.50, 0.75, 1.0]

# Default split fractions
TRAIN_FRAC = 0.75
VAL_FRAC = 0.10
TEST_FRAC = 0.15


def _stratified_split(
    clips: list[dict],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified train/val/test split preserving class proportions."""
    rng = random.Random(seed)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for c in clips:
        by_class[c["label"]].append(c)

    train, val, test = [], [], []
    for cls_clips in by_class.values():
        shuffled = list(cls_clips)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train: n_train + n_val])
        test.extend(shuffled[n_train + n_val:])

    rng.shuffle(train)
    return train, val, test


def _mix_splits(
    real_train: list[dict],
    synth_all: list[dict],
    ratio: float,
    seed: int = 42,
) -> list[dict]:
    """Build a training set with *ratio* fraction of synthetic clips.

    The number of total clips is always ``len(real_train) / (1 - ratio)``
    (synthetic clips added on top of the full real set).

    Special cases:
      ratio == 0.0 → real only
      ratio == 1.0 → synthetic only (sample same N as real set)
    """
    rng = random.Random(seed)

    if ratio <= 0.0:
        return real_train

    if ratio >= 1.0:
        n = len(real_train)
        return rng.choices(synth_all, k=n) if len(synth_all) >= n else synth_all

    n_synth = round(len(real_train) * ratio / (1 - ratio))
    n_synth = min(n_synth, len(synth_all))
    synth_sample = rng.sample(synth_all, n_synth)
    mixed = real_train + synth_sample
    rng.shuffle(mixed)
    return mixed


def build_splits(
    real_manifest: str | Path,
    synth_manifest: str | Path,
    output_dir: str | Path,
    ratios: list[float] = DEFAULT_RATIOS,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = 42,
) -> dict[float, dict[str, Path]]:
    """Build per-ratio train/val/test manifests.

    Returns
    -------
    {ratio: {"train": Path, "val": Path, "test": Path}}
    """
    real_manifest, synth_manifest = Path(real_manifest), Path(synth_manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _load(p: Path) -> list[dict]:
        if not p.exists():
            log.warning("Manifest not found: %s", p)
            return []
        with open(p) as fh:
            return [json.loads(l) for l in fh]

    real_clips = _load(real_manifest)
    synth_clips = _load(synth_manifest)

    if not real_clips:
        raise ValueError(f"Real manifest is empty or missing: {real_manifest}")

    # Fixed split of real clips (shared across all ratios)
    real_train, real_val, real_test = _stratified_split(
        real_clips, train_frac=train_frac, val_frac=val_frac, seed=seed
    )
    # Mark splits
    for c in real_train: c["split"] = "train"
    for c in real_val:   c["split"] = "val"
    for c in real_test:  c["split"] = "test"

    log.info(
        "Real splits: train=%d, val=%d, test=%d",
        len(real_train), len(real_val), len(real_test),
    )

    paths: dict[float, dict[str, Path]] = {}

    for ratio in ratios:
        ratio_dir = output_dir / f"ratio_{ratio:.2f}".replace(".", "p")
        ratio_dir.mkdir(exist_ok=True)

        train_mixed = _mix_splits(real_train, synth_clips, ratio, seed)
        for c in train_mixed: c["split"] = "train"

        split_paths: dict[str, Path] = {}
        for subset, data in [("train", train_mixed), ("val", real_val), ("test", real_test)]:
            p = ratio_dir / f"{subset}.jsonl"
            with open(p, "w") as fh:
                for entry in data:
                    fh.write(json.dumps(entry) + "\n")
            split_paths[subset] = p
            log.info("  ratio=%.2f %s: %d clips → %s", ratio, subset, len(data), p)

        paths[ratio] = split_paths

    return paths


def print_mixture_stats(paths: dict[float, dict[str, Path]]) -> None:
    """Print a table of mixture stats for quick inspection."""
    header = f"{'Ratio':>8}  {'Train':>8}  {'Val':>6}  {'Test':>6}  {'Synth %':>8}"
    print(header)
    print("-" * len(header))
    for ratio, split_paths in sorted(paths.items()):
        def _count(p):
            if not p.exists(): return 0
            with open(p) as fh: return sum(1 for _ in fh)
        def _synth_pct(p):
            if not p.exists(): return 0
            entries = [json.loads(l) for l in open(p)]
            if not entries: return 0
            return 100 * sum(1 for e in entries if e.get("is_synthetic")) / len(entries)

        n_train = _count(split_paths["train"])
        n_val = _count(split_paths.get("val", Path("_")))
        n_test = _count(split_paths.get("test", Path("_")))
        pct = _synth_pct(split_paths["train"])
        print(f"{ratio:>8.2f}  {n_train:>8}  {n_val:>6}  {n_test:>6}  {pct:>7.1f}%")
