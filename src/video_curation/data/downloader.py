"""
Dataset downloaders for UCF-101 and Kinetics-400 (subset).

Usage
-----
    from video_curation.data.downloader import UCF101Downloader, KineticsDownloader

    dl = UCF101Downloader(root="data/raw/ucf101")
    dl.download(classes=["Basketball", "BenchPress", "Biking"])
"""

from __future__ import annotations

import hashlib
import logging
import os
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

# ── UCF-101 ───────────────────────────────────────────────────────────────────

UCF101_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
UCF101_ANNO_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"

# Fallback mirror (more reliable for automated downloads)
UCF101_MIRROR = (
    "https://storage.googleapis.com/thumos14_files/UCF101_videos.tar.gz"
)

# 10-class subset used in our ablation (balanced, diverse action types)
UCF101_DEFAULT_CLASSES = [
    "Basketball",
    "BenchPress",
    "Biking",
    "GolfSwing",
    "HorseRiding",
    "PlayingGuitar",
    "PullUps",
    "Rowing",
    "TennisSwing",
    "WalkingWithDog",
]

# ── Kinetics-400 ──────────────────────────────────────────────────────────────

KINETICS_BASE_URL = "https://s3.amazonaws.com/kinetics/400"
KINETICS_ANNO_URL = "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz"


@dataclass
class DownloadProgress:
    total_bytes: int = 0
    downloaded_bytes: int = 0
    n_files: int = 0
    n_done: int = 0


def _download_file(
    url: str,
    dest: Path,
    chunk_size: int = 1 << 16,
    expected_md5: Optional[str] = None,
) -> Path:
    """Stream-download *url* to *dest*, show tqdm progress, verify MD5."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.debug("Already exists, skipping: %s", dest)
        return dest

    log.info("Downloading %s → %s", url, dest)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    md5 = hashlib.md5()

    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fh.write(chunk)
            md5.update(chunk)
            bar.update(len(chunk))

    if expected_md5 and md5.hexdigest() != expected_md5:
        dest.unlink()
        raise ValueError(f"MD5 mismatch for {dest}: expected {expected_md5}")

    return dest


def _extract_archive(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s → %s", archive, dest)
    if archive.suffix == ".zip" or archive.suffixes[-2:] == [".tar", ".zip"]:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Unknown archive format: {archive}")


# ── UCF-101 Downloader ────────────────────────────────────────────────────────


class UCF101Downloader:
    """Download and organise UCF-101 clips.

    The dataset is structured as::

        root/
          <ClassName>/
            v_<ClassName>_g<group>_c<clip>.avi

    Parameters
    ----------
    root:
        Directory where raw clips will be stored.
    classes:
        Subset of action classes to download.  ``None`` downloads all 101.
    split:
        Which official train/test split to use (1, 2, or 3).
    """

    def __init__(
        self,
        root: str | Path = "data/raw/ucf101",
        classes: Optional[list[str]] = None,
        split: int = 1,
    ) -> None:
        self.root = Path(root)
        self.classes = classes or UCF101_DEFAULT_CLASSES
        self.split = split
        self._anno_dir = self.root / "annotations"

    # ------------------------------------------------------------------
    def download(self, skip_existing: bool = True) -> list[Path]:
        """Download UCF-101 clips for the configured class subset.

        Returns a list of all downloaded ``.avi`` paths.
        """
        self.root.mkdir(parents=True, exist_ok=True)

        # Download annotation splits first (small, always useful)
        self._download_annotations()

        videos: list[Path] = []
        for cls in self.classes:
            cls_dir = self.root / cls
            if skip_existing and cls_dir.exists() and any(cls_dir.glob("*.avi")):
                log.info("Class already present, skipping: %s", cls)
                videos.extend(cls_dir.glob("*.avi"))
                continue

            cls_videos = self._download_class(cls)
            videos.extend(cls_videos)

        log.info("Total clips available: %d", len(videos))
        return videos

    def _download_annotations(self) -> None:
        anno_zip = self.root / "splits.zip"
        if not self._anno_dir.exists():
            _download_file(UCF101_ANNO_URL, anno_zip)
            _extract_archive(anno_zip, self._anno_dir)

    def _download_class(self, cls_name: str) -> list[Path]:
        """Download a single class archive and return clip paths."""
        # UCF-101 individual class archives are not publicly hosted;
        # in practice you download the full dataset and filter.
        # This method handles the post-extraction filtering.
        cls_dir = self.root / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        # If full archive already extracted:
        clips = list(cls_dir.glob("*.avi"))
        if clips:
            return clips

        log.warning(
            "Class %s not found in %s. "
            "Download the full UCF-101 archive from %s, extract to %s, "
            "then re-run.",
            cls_name,
            self.root,
            UCF101_URL,
            self.root,
        )
        return []

    def get_split_files(self, subset: str = "train") -> list[Path]:
        """Return file paths for the official train/test split.

        Parameters
        ----------
        subset: ``"train"`` or ``"test"``
        """
        split_file = (
            self._anno_dir
            / f"ucfTrainTestlist"
            / f"{subset}list{self.split:02d}.txt"
        )
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                "Run download() first."
            )
        paths: list[Path] = []
        with open(split_file) as fh:
            for line in fh:
                rel = line.strip().split()[0]
                cls_name = rel.split("/")[0]
                if cls_name in self.classes:
                    paths.append(self.root / rel)
        return paths


# ── Kinetics-400 Downloader ───────────────────────────────────────────────────


class KineticsDownloader:
    """Download a class-balanced subset of Kinetics-400.

    Uses the official CSV annotations to identify YouTube IDs, then
    downloads with ``yt-dlp`` (must be installed separately).

    Parameters
    ----------
    root:
        Target directory.
    classes:
        List of action class names to include.
    max_clips_per_class:
        Cap per class (useful for ablation-scale experiments).
    resolution:
        Target resolution string for yt-dlp (e.g. ``"360p"``).
    """

    def __init__(
        self,
        root: str | Path = "data/raw/kinetics400",
        classes: Optional[list[str]] = None,
        max_clips_per_class: int = 200,
        resolution: str = "360p",
    ) -> None:
        self.root = Path(root)
        self.classes = classes
        self.max_clips_per_class = max_clips_per_class
        self.resolution = resolution

    def download_annotations(self) -> Path:
        anno_tar = self.root / "kinetics400_annotations.tar.gz"
        if not anno_tar.exists():
            _download_file(KINETICS_ANNO_URL, anno_tar)
        anno_dir = self.root / "annotations"
        if not anno_dir.exists():
            _extract_archive(anno_tar, anno_dir)
        return anno_dir

    def build_clip_list(self, split: str = "train") -> list[dict]:
        """Parse annotation CSV and return clip metadata dicts."""
        import csv

        anno_dir = self.download_annotations()
        csv_path = anno_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Kinetics annotation CSV not found: {csv_path}")

        clips: list[dict] = []
        class_counts: dict[str, int] = {}

        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                label = row["label"]
                if self.classes and label not in self.classes:
                    continue
                if class_counts.get(label, 0) >= self.max_clips_per_class:
                    continue
                clips.append(
                    {
                        "youtube_id": row["youtube_id"],
                        "time_start": int(row["time_start"]),
                        "time_end": int(row["time_end"]),
                        "label": label,
                        "split": split,
                    }
                )
                class_counts[label] = class_counts.get(label, 0) + 1

        log.info("Found %d clips across %d classes", len(clips), len(class_counts))
        return clips

    def download_clips(self, clips: list[dict], num_workers: int = 4) -> list[Path]:
        """Download clips using yt-dlp.  Returns paths to downloaded files."""
        try:
            import yt_dlp  # noqa: F401
        except ImportError:
            raise ImportError(
                "yt-dlp is required for Kinetics download. "
                "Install with: pip install yt-dlp"
            )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        downloaded: list[Path] = []
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(self._dl_one, c): c for c in clips}
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Downloading"
            ):
                path = fut.result()
                if path:
                    downloaded.append(path)
        return downloaded

    def _dl_one(self, clip: dict) -> Optional[Path]:
        import yt_dlp

        out_dir = self.root / clip["label"]
        out_dir.mkdir(parents=True, exist_ok=True)
        vid_id = clip["youtube_id"]
        out_path = out_dir / f"{vid_id}_{clip['time_start']:06d}.mp4"
        if out_path.exists():
            return out_path

        url = f"https://www.youtube.com/watch?v={vid_id}"
        ydl_opts = {
            "format": f"bestvideo[height<={self.resolution[:-1]}]+bestaudio/best",
            "outtmpl": str(out_path.with_suffix("")),
            "download_ranges": yt_dlp.utils.download_range_func(
                [], [[clip["time_start"], clip["time_end"]]]
            ),
            "force_keyframes_at_cuts": True,
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return out_path
        except Exception as exc:
            log.debug("Failed to download %s: %s", vid_id, exc)
            return None
