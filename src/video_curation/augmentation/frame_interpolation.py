"""
Frame interpolation augmentation.

Increases temporal density of a clip by inserting synthetic intermediate frames.

Two modes:
  linear  — simple pixel-level averaging between adjacent frames (fast, CPU)
  rife    — RIFE v4 neural interpolation (GPU, much higher quality)

Output is always an mp4 at the upsampled frame rate.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _interp_linear(f1: np.ndarray, f2: np.ndarray, t: float = 0.5) -> np.ndarray:
    """Linear blend of two BGR frames at temporal position t ∈ (0,1)."""
    return (f1.astype(np.float32) * (1 - t) + f2.astype(np.float32) * t).astype(np.uint8)


def interpolate_linear(
    input_path: str | Path,
    output_path: str | Path,
    multiplier: int = 2,
) -> Optional[Path]:
    """Double (or more) the frame rate via linear interpolation.

    Parameters
    ----------
    input_path:
        Source video.
    output_path:
        Destination mp4.
    multiplier:
        Frame rate multiplier (2 = 2x fps, 3 = 3x, etc.).
    """
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_fps = fps * multiplier
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (w, h))

    prev_frame: Optional[np.ndarray] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if prev_frame is not None:
            for i in range(1, multiplier):
                t = i / multiplier
                interp = _interp_linear(prev_frame, frame, t)
                writer.write(interp)
        writer.write(frame)
        prev_frame = frame

    cap.release()
    writer.release()

    if not output_path.exists() or output_path.stat().st_size == 0:
        log.warning("Frame interpolation produced empty file: %s", output_path)
        return None

    return output_path


def interpolate_rife(
    input_path: str | Path,
    output_path: str | Path,
    multiplier: int = 2,
    rife_model: str = "rife-v4.6",
) -> Optional[Path]:
    """RIFE neural frame interpolation via the inference CLI.

    Requires the ``rife-ncnn-vulkan`` or ``ECCV2022-RIFE`` CLI to be installed.
    Falls back to linear interpolation on failure.

    Parameters
    ----------
    input_path, output_path:
        Source and destination video paths.
    multiplier:
        2 = 2x fps, 4 = 4x fps.
    rife_model:
        Model name passed to the RIFE CLI.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        frames_in = Path(tmp) / "frames_in"
        frames_out = Path(tmp) / "frames_out"
        frames_in.mkdir()
        frames_out.mkdir()

        # Extract frames
        extract_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_path),
            str(frames_in / "%06d.png"),
        ]
        try:
            subprocess.run(extract_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            log.warning("ffmpeg frame extraction failed: %s", exc)
            return interpolate_linear(input_path, output_path, multiplier)

        # Run RIFE
        rife_cmd = [
            "rife-ncnn-vulkan",
            "-i", str(frames_in),
            "-o", str(frames_out),
            "-m", rife_model,
            "-n", str(multiplier),
        ]
        try:
            subprocess.run(rife_cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            log.info("RIFE CLI unavailable, falling back to linear interpolation")
            return interpolate_linear(input_path, output_path, multiplier)

        # Get original fps
        cap = cv2.VideoCapture(str(input_path))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # Reassemble
        assemble_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-r", str(orig_fps * multiplier),
            "-i", str(frames_out / "%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        try:
            subprocess.run(assemble_cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as exc:
            log.warning("ffmpeg reassemble failed: %s", exc)
            return None


def apply_frame_interpolation(
    input_path: str | Path,
    output_path: str | Path,
    method: str = "linear",
    multiplier: int = 2,
    **kwargs,
) -> Optional[Path]:
    """Dispatch to the right interpolation method."""
    if method == "linear":
        return interpolate_linear(input_path, output_path, multiplier)
    if method == "rife":
        return interpolate_rife(input_path, output_path, multiplier, **kwargs)
    raise ValueError(f"Unknown interpolation method: {method}")
