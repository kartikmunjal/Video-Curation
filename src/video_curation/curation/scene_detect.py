"""
Scene cut detection for raw video clips.

Wraps PySceneDetect and provides a simple API to:
  1. Split a long video into shot-level segments.
  2. Score an existing clip for scene-change density (used as a quality signal).
  3. Keep only the dominant (longest) scene per clip.

Dependencies: scenedetect[opencv]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

try:
    from scenedetect import (
        ContentDetector,
        ThresholdDetector,
        AdaptiveDetector,
        SceneManager,
        open_video,
    )
    from scenedetect.scene_manager import save_images
    _HAS_SCENEDETECT = True
except ImportError:
    _HAS_SCENEDETECT = False
    log.warning("scenedetect not installed — scene detection disabled")


@dataclass
class SceneInfo:
    """Result of scene analysis for a single clip."""

    path: str
    n_scenes: int
    dominant_scene_start_sec: float
    dominant_scene_end_sec: float
    dominant_scene_duration_sec: float
    scene_change_rate: float  # cuts per second


def _get_detector(detector_name: str, threshold: float):
    if not _HAS_SCENEDETECT:
        raise RuntimeError("scenedetect is not installed")
    detectors = {
        "content": ContentDetector(threshold=threshold),
        "threshold": ThresholdDetector(threshold=threshold),
        "adaptive": AdaptiveDetector(),
    }
    if detector_name not in detectors:
        raise ValueError(f"Unknown detector: {detector_name}. Choose from {list(detectors)}")
    return detectors[detector_name]


def detect_scenes(
    video_path: str | Path,
    detector: str = "content",
    threshold: float = 27.0,
    min_scene_len_frames: int = 15,
) -> list[tuple[float, float]]:
    """Return a list of (start_sec, end_sec) tuples for each detected scene.

    Parameters
    ----------
    video_path:
        Path to the video file.
    detector:
        Detection algorithm: ``"content"``, ``"threshold"``, or ``"adaptive"``.
    threshold:
        Detection sensitivity (lower = more sensitive).
    min_scene_len_frames:
        Scenes shorter than this are merged with neighbours.
    """
    if not _HAS_SCENEDETECT:
        # Fallback: treat entire clip as one scene
        return [(0.0, _get_video_duration(str(video_path)))]

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(_get_detector(detector, threshold))

    try:
        scene_manager.detect_scenes(video, show_progress=False)
    except Exception as exc:
        log.warning("Scene detection failed for %s: %s", video_path, exc)
        dur = _get_video_duration(str(video_path))
        return [(0.0, dur)]

    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        dur = _get_video_duration(str(video_path))
        return [(0.0, dur)]

    return [
        (scene[0].get_seconds(), scene[1].get_seconds())
        for scene in scene_list
    ]


def analyze_clip(
    video_path: str | Path,
    detector: str = "content",
    threshold: float = 27.0,
    min_scene_len_frames: int = 15,
) -> SceneInfo:
    """Analyse a clip and return scene metadata."""
    scenes = detect_scenes(video_path, detector, threshold, min_scene_len_frames)

    if not scenes:
        dur = _get_video_duration(str(video_path))
        return SceneInfo(
            path=str(video_path),
            n_scenes=1,
            dominant_scene_start_sec=0.0,
            dominant_scene_end_sec=dur,
            dominant_scene_duration_sec=dur,
            scene_change_rate=0.0,
        )

    # Find dominant (longest) scene
    durations = [e - s for s, e in scenes]
    best_idx = max(range(len(durations)), key=lambda i: durations[i])
    start, end = scenes[best_idx]
    total_dur = scenes[-1][1] - scenes[0][0]
    change_rate = (len(scenes) - 1) / max(total_dur, 1e-6)

    return SceneInfo(
        path=str(video_path),
        n_scenes=len(scenes),
        dominant_scene_start_sec=start,
        dominant_scene_end_sec=end,
        dominant_scene_duration_sec=durations[best_idx],
        scene_change_rate=change_rate,
    )


def trim_to_dominant_scene(
    video_path: str | Path,
    output_path: str | Path,
    info: Optional[SceneInfo] = None,
    detector: str = "content",
    threshold: float = 27.0,
    min_duration_sec: float = 0.5,
) -> Optional[Path]:
    """Trim clip to its dominant scene using ffmpeg.

    Returns the output path on success, None if the scene is too short.
    """
    import subprocess

    if info is None:
        info = analyze_clip(video_path, detector, threshold)

    if info.dominant_scene_duration_sec < min_duration_sec:
        log.debug(
            "Dominant scene too short (%.2fs < %.2fs): %s",
            info.dominant_scene_duration_sec,
            min_duration_sec,
            video_path,
        )
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(info.dominant_scene_start_sec),
        "-to", str(info.dominant_scene_end_sec),
        "-i", str(video_path),
        "-c", "copy",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as exc:
        log.warning("ffmpeg trim failed for %s: %s", video_path, exc.stderr.decode())
        return None


def _get_video_duration(path: str) -> float:
    """Get video duration in seconds via OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return n_frames / fps
    except Exception:
        return 10.0  # safe default
