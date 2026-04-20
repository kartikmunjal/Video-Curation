"""
Per-clip quality scoring: blur detection + image quality assessment (BRISQUE/NIQE).

The blur score is the primary signal used in the bias ablation sweep.
Threshold recommendations:
    σ < 20  — very aggressive (removes many low-texture clips → bias risk)
    σ < 40  — moderate  (recommended default)
    σ < 80  — relaxed
    disabled — no blur filtering

BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator):
    Score 0-100. Higher = worse. Threshold ~50 is a reasonable cut.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal test envs
    cv2 = None

log = logging.getLogger(__name__)


@dataclass
class QualityScore:
    path: str
    blur_score: float           # Laplacian variance — higher = sharper
    quality_score: float        # BRISQUE score — lower = better quality
    is_blurry: bool             # True if blur_score < threshold
    is_low_quality: bool        # True if quality_score > threshold
    passed: bool                # True if clip passes ALL quality filters
    n_frames_sampled: int


# ── Blur detection ────────────────────────────────────────────────────────────


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("OpenCV is required for video-file decoding")


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        return frame
    if cv2 is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # BGR luminance approximation used when OpenCV is not installed.
    return (0.114 * frame[..., 0] + 0.587 * frame[..., 1] + 0.299 * frame[..., 2]).astype(
        np.float32
    )


def _laplacian_variance(frame: np.ndarray) -> float:
    """Compute Laplacian variance as sharpness proxy.

    Higher = sharper.  Typical values: <100 blurry, >300 sharp.
    """
    gray = _to_gray(frame).astype(np.float64)
    if cv2 is not None:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    return float(lap.var())


def _fft_energy(frame: np.ndarray, cutoff: float = 0.3) -> float:
    """High-frequency energy in FFT domain — alternative sharpness metric."""
    gray = _to_gray(frame)
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    # Mask out low-frequency center
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * cutoff)
    y, x = np.ogrid[:h, :w]
    mask = (y - cy) ** 2 + (x - cx) ** 2 > r ** 2
    return float(magnitude[mask].mean())


# ── BRISQUE ───────────────────────────────────────────────────────────────────


def _brisque_score(frame: np.ndarray) -> float:
    """Compute BRISQUE quality score for a single frame.

    Uses scikit-image's implementation; falls back to a simpler
    signal-to-noise proxy if unavailable.
    """
    try:
        from skimage.metrics import mean_squared_error  # noqa
        from skimage.restoration import estimate_sigma

        gray = _to_gray(frame)
        noise_sigma = estimate_sigma(gray, average_sigmas=True)
        # Normalise to 0-100 range (higher σ → worse quality)
        return float(min(100.0, noise_sigma * 200))
    except Exception:
        pass

    # Ultra-light fallback: pixel std (high std ≈ richer content ≈ lower "score")
    gray = _to_gray(frame)
    std = float(gray.std())
    # Invert: low std → high score (bad)
    return float(max(0.0, 100.0 - std))


def _niqe_proxy(frame: np.ndarray) -> float:
    """Lightweight NIQE-like proxy using local variance statistics."""
    _require_cv2()
    gray = _to_gray(frame)
    gray_f = gray.astype(np.float32)
    # Local mean and variance via sliding window
    mu = cv2.GaussianBlur(gray_f, (7, 7), 7.0 / 6)
    mu2 = mu * mu
    sigma = cv2.GaussianBlur(gray_f * gray_f, (7, 7), 7.0 / 6)
    sigma = np.sqrt(np.maximum(sigma - mu2, 0))
    # High local variance = richer structure = lower "naturalness distance"
    mean_sigma = float(sigma.mean())
    return max(0.0, 100.0 - mean_sigma * 2)


# ── Frame sampling ─────────────────────────────────────────────────────────────


def _sample_frames(
    path: str,
    n_frames: int = 8,
) -> list[np.ndarray]:
    """Sample *n_frames* uniformly from *path* using OpenCV."""
    _require_cv2()
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return []

    indices = [int(i * total / n_frames) for i in range(n_frames)]
    frames: list[np.ndarray] = []
    prev_idx = -1

    for idx in indices:
        if idx != prev_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        prev_idx = idx

    cap.release()
    return frames


# ── Public API ─────────────────────────────────────────────────────────────────


def score_clip(
    path: str,
    blur_method: str = "laplacian_var",
    quality_method: str = "brisque",
    sample_frames: int = 8,
    quality_sample_frames: int = 4,
    clip_score_agg: str = "mean",
    blur_threshold: float = 40.0,
    quality_threshold: float = 50.0,
) -> QualityScore:
    """Compute blur and quality scores for a video clip.

    Parameters
    ----------
    path:
        Path to the video file.
    blur_method:
        ``"laplacian_var"`` (recommended) or ``"fft_energy"``.
    quality_method:
        ``"brisque"``, ``"niqe"``, or ``"combined"`` (mean of both).
    sample_frames:
        Number of frames to sample for blur scoring.
    quality_sample_frames:
        Number of frames to sample for BRISQUE/NIQE (slower).
    clip_score_agg:
        How to aggregate per-frame scores into a clip score: ``"mean"``,
        ``"min"``, or ``"median"``.
    blur_threshold:
        Clips with blur_score < threshold are marked blurry.
    quality_threshold:
        Clips with quality_score > threshold are marked low-quality.
    """
    frames = _sample_frames(path, n_frames=max(sample_frames, quality_sample_frames))
    if not frames:
        log.warning("Could not read any frames from %s", path)
        return QualityScore(
            path=path,
            blur_score=0.0,
            quality_score=100.0,
            is_blurry=True,
            is_low_quality=True,
            passed=False,
            n_frames_sampled=0,
        )

    # Blur scores (all sampled frames)
    blur_frames = frames[:sample_frames]
    if blur_method == "laplacian_var":
        blur_values = [_laplacian_variance(f) for f in blur_frames]
    elif blur_method == "fft_energy":
        blur_values = [_fft_energy(f) for f in blur_frames]
    else:
        raise ValueError(f"Unknown blur_method: {blur_method}")

    # Quality scores (fewer frames — BRISQUE is slow)
    q_frames = frames[:quality_sample_frames]
    if quality_method == "brisque":
        q_values = [_brisque_score(f) for f in q_frames]
    elif quality_method == "niqe":
        q_values = [_niqe_proxy(f) for f in q_frames]
    elif quality_method == "combined":
        q_values = [
            (_brisque_score(f) + _niqe_proxy(f)) / 2.0 for f in q_frames
        ]
    else:
        raise ValueError(f"Unknown quality_method: {quality_method}")

    def _agg(vals: list[float]) -> float:
        if clip_score_agg == "mean":
            return float(np.mean(vals))
        if clip_score_agg == "min":
            return float(np.min(vals))
        if clip_score_agg == "median":
            return float(np.median(vals))
        raise ValueError(f"Unknown agg: {clip_score_agg}")

    blur_score = _agg(blur_values)
    quality_score = _agg(q_values)
    is_blurry = blur_threshold > 0 and blur_score < blur_threshold
    is_low_q = quality_score > quality_threshold

    return QualityScore(
        path=path,
        blur_score=blur_score,
        quality_score=quality_score,
        is_blurry=is_blurry,
        is_low_quality=is_low_q,
        passed=not is_blurry and not is_low_q,
        n_frames_sampled=len(frames),
    )


def filter_clips(
    paths: list[str],
    blur_threshold: float = 40.0,
    quality_threshold: float = 50.0,
    sample_frames: int = 8,
    blur_method: str = "laplacian_var",
) -> tuple[list[str], list[QualityScore]]:
    """Filter a list of clip paths, returning (passed, scores)."""
    scores = [
        score_clip(
            p,
            blur_method=blur_method,
            sample_frames=sample_frames,
            blur_threshold=blur_threshold,
            quality_threshold=quality_threshold,
        )
        for p in paths
    ]
    passed = [p for p, s in zip(paths, scores) if s.passed]
    n_rejected = len(paths) - len(passed)
    log.info(
        "Quality filter: %d/%d passed (blur<%.0f, quality<%.0f), %d rejected",
        len(passed),
        len(paths),
        blur_threshold,
        quality_threshold,
        n_rejected,
    )
    return passed, scores
