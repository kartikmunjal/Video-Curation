"""
Video clip deduplication via perceptual hashing or CLIP embeddings.

Two strategies:

phash / dhash
    Compute perceptual hashes for a few key frames; a clip is a near-duplicate
    if ANY frame hash is within *hamming_threshold* bits of a stored hash.
    Fast (CPU-only), works well for identical-source duplicates.

clip_embed
    Encode clips with CLIP and cluster by cosine similarity.  Catches
    semantic near-duplicates (same action, different camera angle).
    Slower but more powerful.

Both strategies maintain an in-memory index and can be serialised to disk
so that incremental dedup over large datasets doesn't reprocess old clips.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

try:
    import imagehash
    from PIL import Image

    _HAS_IMAGEHASH = True
except ImportError:
    _HAS_IMAGEHASH = False
    log.warning("imagehash not installed — falling back to MD5 hashing")


@dataclass
class DedupResult:
    path: str
    is_duplicate: bool
    duplicate_of: Optional[str]   # path of the canonical clip
    hash_str: Optional[str]       # hex hash string (phash/dhash)
    hamming_distance: Optional[int]


# ── Frame sampling helper ─────────────────────────────────────────────────────


def _sample_frames_cv2(path: str, n: int = 4) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total / n) for i in range(n)] if total >= n else list(range(total))
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


# ── Perceptual hash index ─────────────────────────────────────────────────────


class PHashIndex:
    """In-memory perceptual hash index for fast near-duplicate detection.

    Parameters
    ----------
    hash_fn:
        One of ``"phash"``, ``"dhash"``, ``"ahash"``, ``"whash"``.
    hash_size:
        Hash size (number of bits = hash_size²).
    hamming_threshold:
        Maximum Hamming distance to consider two clips duplicates.
    sample_frames:
        Frames per clip to hash.
    """

    def __init__(
        self,
        hash_fn: str = "phash",
        hash_size: int = 8,
        hamming_threshold: int = 10,
        sample_frames: int = 4,
    ) -> None:
        if not _HAS_IMAGEHASH:
            raise RuntimeError("imagehash is required for PHashIndex")
        self.hash_fn = hash_fn
        self.hash_size = hash_size
        self.hamming_threshold = hamming_threshold
        self.sample_frames = sample_frames

        self._fn = {
            "phash": imagehash.phash,
            "dhash": imagehash.dhash,
            "ahash": imagehash.average_hash,
            "whash": imagehash.whash,
        }[hash_fn]

        # {clip_path: list[ImageHash]}
        self._index: dict[str, list] = {}

    def _hash_frames(self, path: str) -> list:
        frames = _sample_frames_cv2(path, self.sample_frames)
        hashes = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            hashes.append(self._fn(pil_img, hash_size=self.hash_size))
        return hashes

    def query(self, path: str) -> DedupResult:
        """Check if *path* is a near-duplicate of any indexed clip."""
        hashes = self._hash_frames(path)
        if not hashes:
            return DedupResult(path=path, is_duplicate=False, duplicate_of=None,
                               hash_str=None, hamming_distance=None)

        for stored_path, stored_hashes in self._index.items():
            for h_new in hashes:
                for h_old in stored_hashes:
                    dist = h_new - h_old
                    if dist <= self.hamming_threshold:
                        return DedupResult(
                            path=path,
                            is_duplicate=True,
                            duplicate_of=stored_path,
                            hash_str=str(hashes[0]),
                            hamming_distance=dist,
                        )

        return DedupResult(
            path=path,
            is_duplicate=False,
            duplicate_of=None,
            hash_str=str(hashes[0]),
            hamming_distance=None,
        )

    def add(self, path: str, hashes: Optional[list] = None) -> None:
        """Add a clip to the index (call after confirming it's NOT a duplicate)."""
        if hashes is None:
            hashes = self._hash_frames(path)
        self._index[path] = hashes

    def save(self, dest: str | Path) -> None:
        with open(dest, "wb") as fh:
            pickle.dump(
                {"hash_fn": self.hash_fn, "hash_size": self.hash_size,
                 "hamming_threshold": self.hamming_threshold,
                 "index": self._index},
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> "PHashIndex":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        obj = cls(
            hash_fn=state["hash_fn"],
            hash_size=state["hash_size"],
            hamming_threshold=state["hamming_threshold"],
        )
        obj._index = state["index"]
        return obj

    def __len__(self) -> int:
        return len(self._index)


# ── CLIP embedding index ───────────────────────────────────────────────────────


class CLIPEmbedIndex:
    """Near-duplicate detection using CLIP video embeddings.

    Embeds each clip as the mean of CLIP frame embeddings, then
    removes clips with cosine similarity > *threshold* to any stored clip.

    Parameters
    ----------
    model_name:
        HuggingFace CLIP model ID.
    sim_threshold:
        Cosine similarity threshold for duplicate detection.
    sample_frames:
        Frames per clip.
    device:
        PyTorch device.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        sim_threshold: float = 0.97,
        sample_frames: int = 4,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.sim_threshold = sim_threshold
        self.sample_frames = sample_frames
        self.device = device

        self._model = None
        self._processor = None

        # {path: embedding_vector}
        self._embeddings: dict[str, np.ndarray] = {}

    def _load_model(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    def embed_clip(self, path: str) -> np.ndarray:
        """Return mean CLIP embedding for a clip."""
        import torch
        from PIL import Image

        if self._model is None:
            self._load_model()

        frames = _sample_frames_cv2(path, self.sample_frames)
        if not frames:
            return np.zeros(512, dtype=np.float32)

        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            for f in frames
        ]
        inputs = self._processor(images=pil_frames, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.mean(dim=0).cpu().numpy()

    def query(self, path: str) -> DedupResult:
        emb = self.embed_clip(path)
        for stored_path, stored_emb in self._embeddings.items():
            sim = float(np.dot(emb, stored_emb))
            if sim >= self.sim_threshold:
                return DedupResult(
                    path=path,
                    is_duplicate=True,
                    duplicate_of=stored_path,
                    hash_str=None,
                    hamming_distance=None,
                )
        return DedupResult(
            path=path, is_duplicate=False, duplicate_of=None,
            hash_str=None, hamming_distance=None,
        )

    def add(self, path: str, embedding: Optional[np.ndarray] = None) -> None:
        if embedding is None:
            embedding = self.embed_clip(path)
        self._embeddings[path] = embedding

    def save(self, dest: str | Path) -> None:
        np.savez(dest, **{k.replace("/", "_"): v for k, v in self._embeddings.items()})

    def __len__(self) -> int:
        return len(self._embeddings)


# ── Convenience dedup pass ────────────────────────────────────────────────────


def dedup_clips(
    paths: list[str],
    method: str = "phash",
    hash_fn: str = "phash",
    hash_size: int = 8,
    hamming_threshold: int = 10,
    sample_frames: int = 4,
    clip_embed_model: str = "openai/clip-vit-base-patch32",
    embed_sim_threshold: float = 0.97,
    device: str = "cpu",
    index_save_path: Optional[str] = None,
) -> tuple[list[str], list[DedupResult]]:
    """Deduplicate a list of clips, returning (unique_paths, all_results).

    Processes clips in order; first occurrence of any near-duplicate is kept.
    """
    if method in ("phash", "dhash", "ahash"):
        index = PHashIndex(
            hash_fn=hash_fn,
            hash_size=hash_size,
            hamming_threshold=hamming_threshold,
            sample_frames=sample_frames,
        )
    elif method == "clip_embed":
        index = CLIPEmbedIndex(
            model_name=clip_embed_model,
            sim_threshold=embed_sim_threshold,
            sample_frames=sample_frames,
            device=device,
        )
    else:
        raise ValueError(f"Unknown dedup method: {method}")

    unique_paths: list[str] = []
    results: list[DedupResult] = []

    for path in paths:
        try:
            result = index.query(path)
        except Exception as exc:
            log.warning("Dedup failed for %s: %s", path, exc)
            result = DedupResult(path=path, is_duplicate=False, duplicate_of=None,
                                 hash_str=None, hamming_distance=None)

        results.append(result)
        if not result.is_duplicate:
            index.add(path)
            unique_paths.append(path)

    n_dup = len(paths) - len(unique_paths)
    log.info(
        "Dedup: %d/%d unique (removed %d duplicates, method=%s)",
        len(unique_paths), len(paths), n_dup, method,
    )

    if index_save_path:
        index.save(index_save_path)
        log.info("Saved dedup index to %s", index_save_path)

    return unique_paths, results
