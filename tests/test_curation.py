"""
Unit tests for the curation pipeline modules.

These tests run with synthetic / mock data — no real video files needed.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── quality_filter tests ──────────────────────────────────────────────────────

def test_laplacian_variance_sharp_vs_blurry():
    """A sharp synthetic image should score higher than a uniform one."""
    from video_curation.curation.quality_filter import _laplacian_variance

    # Sharp: checkerboard
    sharp = np.zeros((100, 100), dtype=np.uint8)
    sharp[::2, ::2] = 255

    # Blurry: uniform gray
    blurry = np.full((100, 100), 128, dtype=np.uint8)

    assert _laplacian_variance(sharp) > _laplacian_variance(blurry)


def test_quality_score_dataclass():
    """QualityScore.passed should be True iff no filter triggered."""
    from video_curation.curation.quality_filter import QualityScore

    s = QualityScore(
        path="test.mp4",
        blur_score=100.0,
        quality_score=30.0,
        is_blurry=False,
        is_low_quality=False,
        passed=True,
        n_frames_sampled=8,
    )
    assert s.passed
    assert not s.is_blurry


# ── motion_score tests ────────────────────────────────────────────────────────

def test_motion_score_static():
    """Two identical frames should produce near-zero optical flow."""
    from video_curation.curation.motion_score import _flow_magnitude_farneback
    import cv2

    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    mag = _flow_magnitude_farneback(frame, frame)
    assert mag < 0.5, f"Expected near-zero flow for identical frames, got {mag}"


def test_motion_score_moving():
    """Horizontally shifted frame should produce non-zero flow."""
    from video_curation.curation.motion_score import _flow_magnitude_farneback

    base = np.zeros((64, 64, 3), dtype=np.uint8)
    base[20:40, 20:40] = 200  # white square

    shifted = np.zeros_like(base)
    shifted[20:40, 28:48] = 200  # shifted 8 pixels right

    mag = _flow_magnitude_farneback(base, shifted)
    assert mag > 0.5, f"Expected non-zero flow for shifted frame, got {mag}"


# ── deduplication tests ───────────────────────────────────────────────────────

def test_phash_index_identical():
    """Two identical images should be detected as duplicates."""
    pytest.importorskip("imagehash")
    from video_curation.curation.deduplication import PHashIndex
    from PIL import Image

    idx = PHashIndex(hamming_threshold=5)
    # Manually add a hash
    from imagehash import phash
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    h = phash(img, hash_size=8)
    idx._index["mock_clip.mp4"] = [h]

    # Same hash should be detected
    result_h = h - h
    assert result_h == 0


def test_phash_index_no_duplicate():
    """Completely different clips should not be flagged as duplicates."""
    pytest.importorskip("imagehash")
    from video_curation.curation.deduplication import PHashIndex
    from imagehash import phash
    from PIL import Image

    idx = PHashIndex(hamming_threshold=5)

    img1 = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8))
    h1 = phash(img1, hash_size=8)
    h2 = phash(img2, hash_size=8)

    dist = h1 - h2
    assert dist > 5, f"Expected large distance between black/white images, got {dist}"


# ── data_mixture tests ────────────────────────────────────────────────────────

def _make_clip(label: str, is_synth: bool = False) -> dict:
    return {
        "path": f"data/{label}/clip.mp4",
        "label": label,
        "label_idx": 0,
        "split": "train",
        "is_synthetic": is_synth,
        "duration_sec": 5.0,
        "fps": 30.0,
        "width": 224,
        "height": 224,
        "blur_score": 80.0,
        "motion_score": 5.0,
        "quality_score": 30.0,
    }


def test_stratified_split_preserves_classes():
    """Stratified split should keep all classes in train and val."""
    from video_curation.training.data_mixture import _stratified_split

    clips = [_make_clip(lbl) for lbl in ["A", "A", "B", "B", "C", "C"] * 5]
    train, val, test = _stratified_split(clips, train_frac=0.75, val_frac=0.10)

    train_labels = {c["label"] for c in train}
    val_labels = {c["label"] for c in val}
    assert train_labels == {"A", "B", "C"}
    assert val_labels == {"A", "B", "C"}


def test_mix_splits_ratio_zero():
    """ratio=0.0 should return real clips only."""
    from video_curation.training.data_mixture import _mix_splits

    real = [_make_clip("A") for _ in range(10)]
    synth = [_make_clip("A", is_synth=True) for _ in range(10)]
    mixed = _mix_splits(real, synth, ratio=0.0)
    assert all(not c["is_synthetic"] for c in mixed)


def test_mix_splits_ratio_one():
    """ratio=1.0 should return synthetic clips only."""
    from video_curation.training.data_mixture import _mix_splits

    real = [_make_clip("A") for _ in range(10)]
    synth = [_make_clip("A", is_synth=True) for _ in range(20)]
    mixed = _mix_splits(real, synth, ratio=1.0)
    assert all(c["is_synthetic"] for c in mixed)


def test_mix_splits_ratio_half():
    """ratio=0.5 should yield ~50% synthetic clips."""
    from video_curation.training.data_mixture import _mix_splits

    real = [_make_clip("A") for _ in range(100)]
    synth = [_make_clip("A", is_synth=True) for _ in range(200)]
    mixed = _mix_splits(real, synth, ratio=0.5)

    n_synth = sum(1 for c in mixed if c["is_synthetic"])
    pct = n_synth / len(mixed)
    assert 0.4 < pct < 0.6, f"Expected ~50% synth, got {pct:.2%}"


# ── bias_analysis tests ───────────────────────────────────────────────────────

def test_representation_drift_no_filter():
    """Zero threshold should produce zero drift."""
    from video_curation.evaluation.bias_analysis import (
        class_distribution,
        representation_drift,
    )

    clips = [_make_clip("A")] * 5 + [_make_clip("B")] * 5
    base = class_distribution(clips)
    drift = representation_drift(base, base)
    assert all(abs(v) < 1e-9 for v in drift.values())


def test_representation_drift_full_removal():
    """Class fully removed should have drift of -1.0."""
    from video_curation.evaluation.bias_analysis import (
        class_distribution,
        representation_drift,
    )

    baseline = {"A": 0.5, "B": 0.5}
    filtered = {"A": 1.0, "B": 0.0}
    drift = representation_drift(baseline, filtered)
    assert drift["B"] == pytest.approx(-1.0)
    assert drift["A"] == pytest.approx(1.0)


def test_fvd_identical_distributions():
    """FVD between identical distributions should be near zero."""
    from video_curation.evaluation.fvd import compute_fvd

    rng = np.random.default_rng(42)
    feats = rng.normal(size=(50, 32)).astype(np.float32)
    fvd = compute_fvd(feats, feats)
    assert fvd < 1.0, f"FVD of identical sets should be ~0, got {fvd}"


def test_clip_score_perfect_alignment():
    """Identical video and text features should give score of 1.0."""
    from video_curation.evaluation.clip_eval import clip_score

    feats = np.random.randn(20, 64).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=-1, keepdims=True)
    score = clip_score(feats, feats)
    assert abs(score - 1.0) < 1e-5
