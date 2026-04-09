"""
Temporal speed augmentation: slow-motion and fast-motion variants.

Two methods:
  frame_sampling  — drop or duplicate frames to change effective speed (fast, no quality loss)
  ffmpeg          — use ffmpeg setpts filter for frame-accurate resampling

Slow-motion (factor < 1.0): clips appear to move more slowly, increasing temporal
resolution. Useful for fine-grained action recognition training.

Fast-motion (factor > 1.0): clips appear faster. Good for diversifying temporal
scale in the training distribution.

Speed variation is one of the most effective temporal augmentations for
video generative models — it teaches the model about temporal scale invariance.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Speed factors used in the ablation; intentionally asymmetric
DEFAULT_FACTORS = [0.75, 1.25]


def _speed_via_frame_sampling(
    input_path: str | Path,
    output_path: str | Path,
    factor: float,
) -> Optional[Path]:
    """Change speed by uniformly sampling or repeating frames.

    factor < 1.0  → slow motion (more frames, original fps)
    factor > 1.0  → fast motion (fewer frames, original fps)
    """
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list[np.ndarray] = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    if not frames:
        return None

    n_orig = len(frames)

    if factor > 1.0:
        # Fast motion: subsample frames
        n_out = max(1, int(n_orig / factor))
        indices = [int(i * n_orig / n_out) for i in range(n_out)]
        out_frames = [frames[i] for i in indices]
    else:
        # Slow motion: repeat frames to stretch time
        n_out = int(n_orig / factor)
        out_frames = []
        for i in range(n_out):
            src_idx = min(int(i * factor), n_orig - 1)
            out_frames.append(frames[src_idx])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for frame in out_frames:
        writer.write(frame)
    writer.release()

    if not output_path.exists():
        return None
    return output_path


def _speed_via_ffmpeg(
    input_path: str | Path,
    output_path: str | Path,
    factor: float,
) -> Optional[Path]:
    """Change speed using ffmpeg's setpts and atempo filters.

    More accurate than frame sampling for fractional factors.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # setpts factor is the inverse of speed factor
    pts_factor = 1.0 / factor

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(input_path),
        "-vf", f"setpts={pts_factor:.4f}*PTS",
        "-an",  # drop audio
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as exc:
        log.warning("ffmpeg speed change failed: %s", exc.stderr.decode()[:200])
        # Fall back to frame sampling
        return _speed_via_frame_sampling(input_path, output_path, factor)


def apply_speed_variation(
    input_path: str | Path,
    output_path: str | Path,
    factor: float,
    method: str = "frame_sampling",
) -> Optional[Path]:
    """Apply speed augmentation to a clip.

    Parameters
    ----------
    input_path, output_path:
        Source and destination paths.
    factor:
        Speed multiplier.  0.75 = 25% slower, 1.25 = 25% faster.
    method:
        ``"frame_sampling"`` (default) or ``"ffmpeg"``.
    """
    if abs(factor - 1.0) < 1e-3:
        return None  # no-op

    if method == "ffmpeg":
        return _speed_via_ffmpeg(input_path, output_path, factor)
    return _speed_via_frame_sampling(input_path, output_path, factor)


def generate_speed_variants(
    input_path: str | Path,
    output_dir: str | Path,
    factors: list[float] = DEFAULT_FACTORS,
    method: str = "frame_sampling",
) -> list[Path]:
    """Generate all speed variants for a clip.

    Returns a list of (output_path, factor) for each successful variant.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []
    for factor in factors:
        suffix = f"_speed{factor:.2f}".replace(".", "p")
        out_name = input_path.stem + suffix + input_path.suffix
        out_path = output_dir / out_name
        result = apply_speed_variation(input_path, out_path, factor, method)
        if result:
            results.append(result)

    return results
