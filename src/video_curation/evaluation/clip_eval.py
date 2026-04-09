"""
CLIP-based video evaluation metrics.

Computes:
  1. CLIP@16 score — mean cosine similarity between video embeddings and
     caption embeddings (text–visual alignment).  Higher = better.
  2. CLIP retrieval R@1 / R@5 / R@10 — fraction of clips whose GT caption
     is the top-1/5/10 most similar text in the batch.
  3. Intra-class CLIP distance — how compact each action class is in CLIP space.
     Used in the bias analysis to detect representation collapse after filtering.

These metrics complement FVD: while FVD captures distributional realism,
CLIP scores capture semantic content alignment, crucial for generative models
conditioned on text prompts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

CLIP_MODEL_DEFAULT = "openai/clip-vit-base-patch32"


# ── CLIP encoder ─────────────────────────────────────────────────────────────


class CLIPVideoEncoder:
    """Encode video clips and captions with CLIP.

    Parameters
    ----------
    model_name:
        HuggingFace CLIP model ID.
    device:
        PyTorch device.
    sample_frames:
        Frames to average per clip.
    batch_size:
        Clips per forward pass.
    """

    def __init__(
        self,
        model_name: str = CLIP_MODEL_DEFAULT,
        device: str = "cuda",
        sample_frames: int = 16,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.sample_frames = sample_frames
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def _load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        log.info("Loading CLIP: %s", self.model_name)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()

    def encode_frames(self, frames_batch: list[list[np.ndarray]]) -> np.ndarray:
        """Encode a batch of clips as mean CLIP frame embeddings.

        Parameters
        ----------
        frames_batch:
            List of clips; each clip is a list of (H, W, 3) RGB uint8 arrays.

        Returns (B, D) normalised feature matrix.
        """
        import torch
        from PIL import Image

        if self._model is None:
            self._load()

        all_pil: list = []
        clip_lengths: list[int] = []
        for clip_frames in frames_batch:
            clip_lengths.append(len(clip_frames))
            all_pil.extend([Image.fromarray(f) for f in clip_frames])

        inputs = self._processor(images=all_pil, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        feats_np = feats.cpu().float().numpy()

        # Average per clip
        clip_feats = []
        idx = 0
        for length in clip_lengths:
            clip_mean = feats_np[idx: idx + length].mean(axis=0)
            clip_mean /= (np.linalg.norm(clip_mean) + 1e-8)
            clip_feats.append(clip_mean)
            idx += length

        return np.stack(clip_feats)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode text captions into CLIP feature vectors. Returns (N, D)."""
        import torch

        if self._model is None:
            self._load()

        all_feats: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            inputs = self._processor(
                text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self._model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().float().numpy())

        return np.concatenate(all_feats, axis=0)


# ── Metrics ───────────────────────────────────────────────────────────────────


def clip_score(
    video_feats: np.ndarray,
    text_feats: np.ndarray,
) -> float:
    """Mean diagonal cosine similarity (CLIP@16 score).

    video_feats: (N, D), text_feats: (N, D) — paired video/caption embeddings.
    """
    assert video_feats.shape == text_feats.shape
    sims = (video_feats * text_feats).sum(axis=-1)  # (N,)
    return float(sims.mean())


def retrieval_recall(
    query_feats: np.ndarray,
    gallery_feats: np.ndarray,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Video→text retrieval recall@K.

    Query i should retrieve gallery i as the top-K match.
    """
    sim_matrix = query_feats @ gallery_feats.T  # (N, N)
    N = sim_matrix.shape[0]
    recalls: dict[str, float] = {}
    for k in k_values:
        topk = np.argsort(sim_matrix, axis=-1)[:, -k:]
        correct = sum(1 for i in range(N) if i in topk[i])
        recalls[f"R@{k}"] = correct / N
    return recalls


def intra_class_compactness(
    feats: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    """Mean intra-class cosine distance for each class.

    Lower = more compact representation (clips within a class are more similar).
    Used to detect representation collapse after aggressive filtering.
    """
    unique_labels = sorted(set(labels))
    compactness: dict[str, float] = {}

    for lbl in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == lbl]
        if len(idxs) < 2:
            compactness[lbl] = float("nan")
            continue
        class_feats = feats[idxs]  # (K, D)
        # Cosine similarity matrix
        sim = class_feats @ class_feats.T  # (K, K)
        # Mean of upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
        mean_sim = float(sim[mask].mean())
        compactness[lbl] = mean_sim

    return compactness


# ── End-to-end evaluation ─────────────────────────────────────────────────────


def evaluate_clip(
    manifest_path: str | Path,
    model_name: str = CLIP_MODEL_DEFAULT,
    device: str = "cuda",
    sample_frames: int = 16,
    batch_size: int = 32,
    max_clips: Optional[int] = None,
) -> dict:
    """Compute CLIP score and retrieval metrics for a manifest.

    Parameters
    ----------
    manifest_path:
        JSONL clip manifest with ``path``, ``caption``, and ``label`` fields.

    Returns
    -------
    dict: clip_score, R@1, R@5, R@10, intra_class_compactness (per-class)
    """
    import json
    import cv2

    with open(manifest_path) as fh:
        entries = [json.loads(l) for l in fh]

    if max_clips:
        entries = entries[:max_clips]

    entries_with_cap = [e for e in entries if e.get("caption")]
    if not entries_with_cap:
        log.warning("No captions found in %s — skipping CLIP score", manifest_path)
        return {"clip_score": float("nan"), "n_clips": len(entries)}

    encoder = CLIPVideoEncoder(
        model_name=model_name, device=device,
        sample_frames=sample_frames, batch_size=batch_size,
    )

    # Sample frames for each clip
    def _sample(path: str, n: int) -> list[np.ndarray]:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = [int(i * total / n) for i in range(n)] if total >= n else list(range(total))
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if ret:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    log.info("Encoding %d clips with CLIP...", len(entries_with_cap))
    all_clip_frames = [_sample(e["path"], sample_frames) for e in entries_with_cap]
    all_texts = [e["caption"] for e in entries_with_cap]

    video_feats = encoder.encode_frames(all_clip_frames)    # (N, D)
    text_feats = encoder.encode_texts(all_texts)             # (N, D)

    score = clip_score(video_feats, text_feats)
    recalls = retrieval_recall(video_feats, text_feats)
    labels = [e["label"] for e in entries_with_cap]
    compactness = intra_class_compactness(video_feats, labels)

    log.info("CLIP score: %.4f", score)
    for k, v in recalls.items():
        log.info("  %s: %.4f", k, v)

    return {
        "clip_score": score,
        "n_clips": len(entries_with_cap),
        **recalls,
        "intra_class_compactness": compactness,
    }
