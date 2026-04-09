"""
Per-clip color jitter augmentation applied consistently across all frames.

Applies a single random parameter draw per clip (not per frame) to preserve
temporal coherence — important for video generation tasks.

Implemented in pure OpenCV + NumPy to avoid torchvision import at augment time.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── Low-level frame transforms ────────────────────────────────────────────────


def _adjust_brightness(frame: np.ndarray, factor: float) -> np.ndarray:
    """Multiply pixel values by factor (clamped to [0, 255])."""
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _adjust_contrast(frame: np.ndarray, factor: float) -> np.ndarray:
    """Scale contrast around the mean pixel value."""
    mean = frame.mean()
    out = np.clip((frame.astype(np.float32) - mean) * factor + mean, 0, 255)
    return out.astype(np.uint8)


def _adjust_saturation(frame: np.ndarray, factor: float) -> np.ndarray:
    """Adjust saturation in HSV space."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _adjust_hue(frame: np.ndarray, delta: float) -> np.ndarray:
    """Shift hue by *delta* degrees (–180 to +180)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + int(delta)) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ── Clip-level transform ──────────────────────────────────────────────────────


class ClipColorJitter:
    """Temporally-coherent color jitter for video clips.

    Parameters
    ----------
    brightness, contrast, saturation:
        Maximum fractional deviation from 1.0 (e.g. 0.3 → range [0.7, 1.3]).
    hue:
        Maximum absolute hue shift in [0, 0.5] mapped to [0, 90] degrees.
    apply_prob:
        Probability of applying any augmentation to a given clip.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
        hue: float = 0.1,
        apply_prob: float = 0.8,
        seed: Optional[int] = None,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.apply_prob = apply_prob
        self._rng = random.Random(seed)

    def sample_params(self) -> dict:
        """Draw one set of random parameters for a clip."""
        return {
            "brightness_f": self._rng.uniform(
                1 - self.brightness, 1 + self.brightness
            ),
            "contrast_f": self._rng.uniform(
                1 - self.contrast, 1 + self.contrast
            ),
            "saturation_f": self._rng.uniform(
                1 - self.saturation, 1 + self.saturation
            ),
            "hue_delta": self._rng.uniform(
                -self.hue * 180, self.hue * 180
            ),
        }

    def apply_to_frame(self, frame: np.ndarray, params: dict) -> np.ndarray:
        frame = _adjust_brightness(frame, params["brightness_f"])
        frame = _adjust_contrast(frame, params["contrast_f"])
        frame = _adjust_saturation(frame, params["saturation_f"])
        if abs(params["hue_delta"]) > 1.0:
            frame = _adjust_hue(frame, params["hue_delta"])
        return frame

    def __call__(
        self, frames: list[np.ndarray], params: Optional[dict] = None
    ) -> list[np.ndarray]:
        """Apply consistent jitter across all frames in a clip."""
        if self._rng.random() > self.apply_prob:
            return frames
        p = params or self.sample_params()
        return [self.apply_to_frame(f, p) for f in frames]


# ── File-level API ─────────────────────────────────────────────────────────────


def jitter_clip(
    input_path: str | Path,
    output_path: str | Path,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
    hue: float = 0.1,
    apply_prob: float = 0.8,
    seed: Optional[int] = None,
) -> Optional[Path]:
    """Apply color jitter to all frames of a clip and write to *output_path*.

    Returns the output path on success, None if the input cannot be read.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        log.warning("Cannot open %s", input_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read all frames first (needed to apply consistent params)
    raw_frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    if not raw_frames:
        log.warning("Empty video: %s", input_path)
        return None

    jitter = ClipColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        apply_prob=apply_prob,
        seed=seed,
    )
    augmented = jitter(raw_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for frame in augmented:
        writer.write(frame)
    writer.release()

    return output_path
