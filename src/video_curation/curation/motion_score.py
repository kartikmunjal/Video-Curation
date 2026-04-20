"""
Motion scoring for video clips using optical flow.

Two modes:
  - farneback:   Dense optical flow (CPU, fast, no extra deps)
  - raft:        RAFT-Small (GPU, accurate, requires torchvision >= 0.15)

Clips below min_motion are "static" (low camera/subject movement).
Clips above max_motion are "shaky" or full of compression artifacts.

The motion score is also used in the bias analysis: low-texture action classes
(e.g., yoga, meditation) have naturally low motion and get over-filtered at
aggressive min_motion thresholds.
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
class MotionScore:
    path: str
    mean_flow_magnitude: float   # average optical flow magnitude across sampled pairs
    max_flow_magnitude: float    # 95th percentile
    motion_uniformity: float     # std of per-frame scores (low = uniform motion)
    is_static: bool              # True if below min_motion
    is_shaky: bool               # True if above max_motion
    passed: bool
    n_pairs_sampled: int


# ── Farneback ─────────────────────────────────────────────────────────────────


_FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("OpenCV is required for video-file decoding and Farneback flow")


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        return frame.astype(np.float32)
    if cv2 is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return (0.114 * frame[..., 0] + 0.587 * frame[..., 1] + 0.299 * frame[..., 2]).astype(
        np.float32
    )


def _flow_magnitude_farneback(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute mean optical flow magnitude between two BGR frames."""
    g1 = _to_gray(frame1)
    g2 = _to_gray(frame2)
    if cv2 is None:
        return float(np.mean(np.abs(g2.astype(np.float32) - g1.astype(np.float32))) / 10.0)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, **_FARNEBACK_PARAMS)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(mag.mean())


# ── RAFT (optional GPU path) ──────────────────────────────────────────────────


def _flow_magnitude_raft(
    frame1: np.ndarray,
    frame2: np.ndarray,
    model=None,
    device: str = "cuda",
) -> float:
    """Compute mean optical flow using RAFT-Small model."""
    import torch
    import torchvision.transforms.functional as TF

    if model is None:
        raise ValueError("Pass a pre-loaded RAFT model to avoid reload overhead")

    def _to_tensor(bgr: np.ndarray) -> "torch.Tensor":
        _require_cv2()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        return t.to(device)

    with torch.no_grad():
        flow_low, flow_up = model(_to_tensor(frame1), _to_tensor(frame2), iters=12)

    flow = flow_up[0].cpu().numpy()  # (2, H, W)
    mag = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
    return float(mag.mean())


def _load_raft(device: str = "cuda"):
    """Load RAFT-Small from torchvision."""
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights).to(device).eval()
    return model


# ── Frame sampling ─────────────────────────────────────────────────────────────


def _sample_frame_pairs(
    path: str, n_pairs: int = 6
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sample n_pairs of consecutive frame pairs for optical flow."""
    _require_cv2()
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return []

    # Sample start indices uniformly, take consecutive pairs
    step = max(1, (total - 1) // n_pairs)
    starts = [i * step for i in range(n_pairs)]

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for start in starts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret1, f1 = cap.read()
        ret2, f2 = cap.read()
        if ret1 and ret2:
            pairs.append((f1, f2))

    cap.release()
    return pairs


# ── Public API ─────────────────────────────────────────────────────────────────


def score_clip(
    path: str,
    method: str = "farneback",
    sample_pairs: int = 6,
    min_motion: float = 2.0,
    max_motion: float = 80.0,
    raft_model=None,
    raft_device: str = "cuda",
) -> MotionScore:
    """Compute motion score for a single clip.

    Parameters
    ----------
    path:
        Video file path.
    method:
        ``"farneback"`` (default) or ``"raft"``.
    sample_pairs:
        Number of frame-pair samples.
    min_motion:
        Minimum mean flow magnitude; below = static clip.
    max_motion:
        Maximum mean flow magnitude; above = shaky/artifactual.
    raft_model:
        Pre-loaded RAFT model (only needed for method="raft").
    raft_device:
        Device string for RAFT inference.
    """
    pairs = _sample_frame_pairs(path, n_pairs=sample_pairs)
    if not pairs:
        log.warning("No frame pairs extracted from %s", path)
        return MotionScore(
            path=path,
            mean_flow_magnitude=0.0,
            max_flow_magnitude=0.0,
            motion_uniformity=0.0,
            is_static=True,
            is_shaky=False,
            passed=False,
            n_pairs_sampled=0,
        )

    if method == "farneback":
        mags = [_flow_magnitude_farneback(f1, f2) for f1, f2 in pairs]
    elif method == "raft":
        if raft_model is None:
            raft_model = _load_raft(raft_device)
        mags = [
            _flow_magnitude_raft(f1, f2, model=raft_model, device=raft_device)
            for f1, f2 in pairs
        ]
    else:
        raise ValueError(f"Unknown motion method: {method}")

    mean_mag = float(np.mean(mags))
    max_mag = float(np.percentile(mags, 95))
    uniformity = float(np.std(mags))

    is_static = mean_mag < min_motion
    is_shaky = mean_mag > max_motion

    return MotionScore(
        path=path,
        mean_flow_magnitude=mean_mag,
        max_flow_magnitude=max_mag,
        motion_uniformity=uniformity,
        is_static=is_static,
        is_shaky=is_shaky,
        passed=not is_static and not is_shaky,
        n_pairs_sampled=len(pairs),
    )


def score_clips_batch(
    paths: list[str],
    method: str = "farneback",
    sample_pairs: int = 6,
    min_motion: float = 2.0,
    max_motion: float = 80.0,
) -> list[MotionScore]:
    """Score a list of clips sequentially (use Ray wrapper for parallelism)."""
    scores: list[MotionScore] = []
    raft_model = None
    if method == "raft":
        raft_model = _load_raft()

    for path in paths:
        try:
            s = score_clip(
                path,
                method=method,
                sample_pairs=sample_pairs,
                min_motion=min_motion,
                max_motion=max_motion,
                raft_model=raft_model,
            )
        except Exception as exc:
            log.warning("Motion scoring failed for %s: %s", path, exc)
            s = MotionScore(
                path=path,
                mean_flow_magnitude=0.0,
                max_flow_magnitude=0.0,
                motion_uniformity=0.0,
                is_static=True,
                is_shaky=False,
                passed=False,
                n_pairs_sampled=0,
            )
        scores.append(s)

    n_passed = sum(1 for s in scores if s.passed)
    log.info(
        "Motion filter: %d/%d passed (%.1f < mag < %.1f)",
        n_passed, len(scores), min_motion, max_motion,
    )
    return scores
