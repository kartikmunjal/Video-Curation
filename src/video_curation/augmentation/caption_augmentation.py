"""
Caption augmentation via a Vision-Language Model (VLM).

Generates natural-language descriptions for each clip by:
  1. Sampling N frames from the clip.
  2. Running them through a lightweight VLM (BLIP-2 by default).
  3. Combining frame captions into a clip-level description.

Also supports caption paraphrasing via an LLM to increase linguistic diversity.

Rationale for caption augmentation in the ablation:
  Video generative models fine-tuned with text conditioning benefit from
  richer, more diverse captions.  Synthetic captions for augmented clips
  (speed variants, jittered clips) ensure the text–visual grounding is
  maintained even for distribution-shifted inputs.

Models used (light-weight defaults):
  - Salesforce/blip2-opt-2.7b: ~2.7B params, runs on consumer GPU
  - Fallback: Salesforce/blip-image-captioning-base (~224M params, CPU-OK)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_BLIP2_DEFAULT = "Salesforce/blip2-opt-2.7b"
_BLIP_LIGHT = "Salesforce/blip-image-captioning-base"


# ── Frame sampling ─────────────────────────────────────────────────────────────


def _sample_frames(path: str, n: int = 4) -> list[np.ndarray]:
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


# ── BLIP-2 captioner ──────────────────────────────────────────────────────────


class VLMCaptioner:
    """Generate captions for video clips using a Vision-Language Model.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Defaults to BLIP-2 (2.7B).
    device:
        Torch device string.
    sample_frames:
        Number of frames sampled per clip.
    max_new_tokens:
        Maximum caption length.
    batch_size:
        Number of frames processed per forward pass.
    prompt:
        Optional prompt prefix.
    """

    def __init__(
        self,
        model_name: str = _BLIP2_DEFAULT,
        device: str = "cuda",
        sample_frames: int = 4,
        max_new_tokens: int = 64,
        batch_size: int = 8,
        prompt: str = "",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.sample_frames = sample_frames
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.prompt = prompt

        self._processor = None
        self._model = None

    def _load(self) -> None:
        """Lazy-load model on first call."""
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        from transformers import BlipForConditionalGeneration, BlipProcessor

        log.info("Loading VLM: %s on %s", self.model_name, self.device)

        if "blip2" in self.model_name.lower():
            self._processor = Blip2Processor.from_pretrained(self.model_name)
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                load_in_8bit=False,  # set True to halve memory at slight quality cost
                device_map=self.device,
            )
        else:
            # Fall back to BLIP-1 (much lighter, no 8-bit needed)
            self._processor = BlipProcessor.from_pretrained(self.model_name)
            self._model = BlipForConditionalGeneration.from_pretrained(
                self.model_name
            ).to(self.device)

        self._model.eval()
        log.info("VLM loaded")

    def caption_frames(self, frames: list[np.ndarray]) -> list[str]:
        """Caption a list of RGB numpy frames."""
        import torch
        from PIL import Image

        if self._model is None:
            self._load()

        captions: list[str] = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i: i + self.batch_size]
            pil_imgs = [Image.fromarray(f) for f in batch]

            inputs = self._processor(
                images=pil_imgs,
                text=[self.prompt] * len(pil_imgs) if self.prompt else None,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                out_ids = self._model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens
                )

            texts = self._processor.batch_decode(out_ids, skip_special_tokens=True)
            captions.extend([t.strip() for t in texts])

        return captions

    def caption_clip(self, path: str) -> str:
        """Generate a single unified caption for a video clip."""
        frames = _sample_frames(path, self.sample_frames)
        if not frames:
            return ""

        per_frame_caps = self.caption_frames(frames)

        # Deduplicate and join unique descriptive phrases
        seen: set[str] = set()
        unique: list[str] = []
        for cap in per_frame_caps:
            normalized = cap.lower().strip().rstrip(".")
            if normalized not in seen:
                seen.add(normalized)
                unique.append(cap)

        # Use the most common caption (mode) as the clip description
        if len(unique) == 1:
            return unique[0]

        # Majority vote: pick the longest unique caption (usually most descriptive)
        return max(unique, key=len)

    def caption_batch(self, paths: list[str]) -> dict[str, str]:
        """Caption multiple clips, returning {path: caption}."""
        results: dict[str, str] = {}
        for path in paths:
            try:
                results[path] = self.caption_clip(path)
            except Exception as exc:
                log.warning("Caption failed for %s: %s", path, exc)
                results[path] = ""
        return results


# ── Caption paraphraser ───────────────────────────────────────────────────────


class CaptionParaphraser:
    """Paraphrase existing captions via a text LLM to increase diversity.

    Parameters
    ----------
    model_name:
        HuggingFace text model (instruction-tuned).
    n_paraphrases:
        Number of paraphrase variants to generate per caption.
    device:
        Torch device.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        n_paraphrases: int = 2,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.n_paraphrases = n_paraphrases
        self.device = device
        self._pipe = None

    def _load(self) -> None:
        import transformers

        self._pipe = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            device_map=self.device,
            torch_dtype="auto",
        )

    def paraphrase(self, caption: str) -> list[str]:
        """Return *n_paraphrases* rewrites of *caption*."""
        if self._pipe is None:
            self._load()

        prompt = (
            f"Rewrite the following video caption {self.n_paraphrases} times "
            "in different words, keeping the same meaning. "
            "Output only the captions, one per line.\n\n"
            f"Caption: {caption}\n\nRewritten captions:"
        )
        out = self._pipe(
            prompt,
            max_new_tokens=128,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
        )
        generated = out[0]["generated_text"][len(prompt):].strip()
        lines = [l.strip().lstrip("1234567890.-) ") for l in generated.split("\n")]
        lines = [l for l in lines if l]
        return lines[: self.n_paraphrases]


# ── Convenience function ──────────────────────────────────────────────────────


def augment_captions(
    manifest_entries: list[dict],
    model_name: str = _BLIP2_DEFAULT,
    device: str = "cuda",
    sample_frames: int = 4,
    max_new_tokens: int = 64,
    batch_size: int = 8,
    only_missing: bool = True,
) -> list[dict]:
    """Add or replace captions in manifest entries.

    Parameters
    ----------
    manifest_entries:
        List of clip metadata dicts (from load_manifest).
    only_missing:
        If True, only generate captions for entries where ``caption`` is None
        or empty.

    Returns
    -------
    Updated manifest entries (captions filled in-place).
    """
    captioner = VLMCaptioner(
        model_name=model_name,
        device=device,
        sample_frames=sample_frames,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    to_caption = [
        e for e in manifest_entries
        if not only_missing or not e.get("caption")
    ]
    log.info("Generating captions for %d clips", len(to_caption))

    paths = [e["path"] for e in to_caption]
    captions = captioner.caption_batch(paths)

    for entry in to_caption:
        entry["caption"] = captions.get(entry["path"], "")

    return manifest_entries
