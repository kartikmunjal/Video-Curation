"""
Stage 2c — Multitask Annotation.

Enriches each manifest entry with three co-located task signals:

  1. Caption (BLIP-2 / LLaVA)     — text supervision for vision-language alignment
  2. Optical flow (RAFT/Farneback) — dense flow field for motion supervision
  3. Depth estimate (DPT/MiDaS)   — monocular depth map for 3D structure

These turn a single-task (action classification) manifest into a multitask
manifest that can train world models on multiple supervisory signals
simultaneously.  The enriched fields are written back to the JSONL:

    {
      "path": "/data/clip.mp4",
      "label": "PlayingGuitar",
      "blur_score": 31.2,
      ...
      "caption": "a guitarist performing on a dimly lit stage",
      "caption_model": "Salesforce/blip2-opt-2.7b",
      "flow_path": "/data/tasks/flow/clip_flow.npy",    # (T-1, H, W, 2)
      "flow_mean_magnitude": 3.1,
      "depth_path": "/data/tasks/depth/clip_depth.npy", # (T_key, H, W)
      "depth_model": "Intel/dpt-large"
    }

Why this matters:
  The curation bias question extends naturally to multitask data.  A clip that
  passes the Laplacian blur filter may have poor optical flow signal (near-static
  background), or poor depth signal (flat scene).  Logging these task-signal
  quality metrics per clip lets us ask: do the at-risk classes (PlayingGuitar,
  Rowing) fail differently on flow vs. depth vs. caption quality than on
  Laplacian blur?  This directly addresses "design multimodal, multitask
  datasets that teach world models new capabilities."

Usage:
    from video_curation.curation.multitask_annotator import MultitaskAnnotator

    annotator = MultitaskAnnotator(device="cuda")
    enriched = annotator.annotate_manifest(
        manifest_path="data/curated/blur40/manifest.jsonl",
        output_dir="data/tasks",
        batch_size=4,
    )
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────

def _decode_frames(
    video_path: str,
    num_frames: int = 8,
    size: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """Decode `num_frames` uniformly sampled frames as BGR uint8 arrays."""
    cap = cv2.VideoCapture(video_path)
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
    frames, idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            if size:
                frame = cv2.resize(frame, size)
            frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        return []
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


# ── 1. Caption ─────────────────────────────────────────────────────────────

class CaptionAnnotator:
    """
    Generates a text caption for a video clip using BLIP-2.

    Captions the middle frame of the clip (representative of the action).
    Falls back to a descriptive label-based caption if BLIP-2 is unavailable.
    """

    def __init__(self, model_id: str = "Salesforce/blip2-opt-2.7b", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch
        logger.info(f"Loading BLIP-2 from {self.model_id}")
        self._processor = Blip2Processor.from_pretrained(self.model_id)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16
        ).to(self.device)

    def caption(self, video_path: str, label: Optional[str] = None) -> str:
        try:
            import torch
            from PIL import Image

            self._load()
            frames = _decode_frames(video_path, num_frames=3, size=(384, 384))
            if not frames:
                raise ValueError("No frames decoded")

            mid_frame = cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(mid_frame)

            inputs = self._processor(images=pil_img, return_tensors="pt").to(
                self.device, torch.float16
            )
            ids = self._model.generate(**inputs, max_new_tokens=50)
            caption = self._processor.decode(ids[0], skip_special_tokens=True).strip()
            return caption

        except Exception as e:
            logger.warning(f"Caption failed for {video_path}: {e}")
            fallback = label.lower().replace("_", " ") if label else "a video clip"
            return f"a person performing {fallback}"


# ── 2. Optical Flow ────────────────────────────────────────────────────────

class FlowAnnotator:
    """
    Computes dense optical flow between consecutive frames.

    Supports two backends:
      - "farneback": CPU, no extra deps, ~0.3s per frame pair at 224×224
      - "raft":      GPU, torchvision RAFT, ~0.1s per frame pair at 224×224

    Saves flow field as float16 .npy of shape (T-1, H, W, 2).
    Also records scalar summary statistics (mean magnitude, direction entropy).
    """

    def __init__(
        self,
        method: str = "farneback",
        frame_size: Tuple[int, int] = (224, 224),
        num_frames: int = 9,
        device: str = "cuda",
    ):
        self.method = method
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.device = device
        self._raft = None

    def _load_raft(self):
        if self._raft is not None:
            return
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        import torch
        weights = Raft_Small_Weights.DEFAULT
        self._raft = raft_small(weights=weights).to(self.device).eval()
        self._raft_transforms = weights.transforms()
        logger.info("Loaded RAFT-small")

    def _farneback_pair(self, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        return flow  # (H, W, 2)

    def _raft_pair(self, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        self._load_raft()
        pil1 = Image.fromarray(cv2.cvtColor(f1, cv2.COLOR_BGR2RGB))
        pil2 = Image.fromarray(cv2.cvtColor(f2, cv2.COLOR_BGR2RGB))
        t1, t2 = self._raft_transforms(pil1, pil2)
        t1 = t1.unsqueeze(0).to(self.device)
        t2 = t2.unsqueeze(0).to(self.device)
        with torch.no_grad():
            flow_list = self._raft(t1, t2)
        flow = flow_list[-1][0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
        return flow

    def annotate(self, video_path: str, output_path: str) -> Dict:
        frames = _decode_frames(video_path, num_frames=self.num_frames, size=self.frame_size)
        if len(frames) < 2:
            return {"flow_path": None, "flow_mean_magnitude": None, "flow_direction_entropy": None}

        flows = []
        for i in range(len(frames) - 1):
            try:
                if self.method == "raft":
                    flow = self._raft_pair(frames[i], frames[i + 1])
                else:
                    flow = self._farneback_pair(frames[i], frames[i + 1])
                flows.append(flow)
            except Exception as e:
                logger.warning(f"Flow pair {i} failed for {video_path}: {e}")

        if not flows:
            return {"flow_path": None, "flow_mean_magnitude": None, "flow_direction_entropy": None}

        flow_arr = np.stack(flows, axis=0).astype(np.float16)  # (T-1, H, W, 2)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.save(output_path, flow_arr)

        # Summary statistics
        mag = np.sqrt(flow_arr[..., 0] ** 2 + flow_arr[..., 1] ** 2)
        ang = np.arctan2(flow_arr[..., 1].astype(float), flow_arr[..., 0].astype(float)) % (2 * np.pi)
        hist, _ = np.histogram(ang.ravel(), bins=8, range=(0, 2 * np.pi))
        hist = hist / (hist.sum() + 1e-9)
        dir_entropy = float(-np.sum(hist * np.log2(hist + 1e-9)))

        return {
            "flow_path": output_path,
            "flow_method": self.method,
            "flow_mean_magnitude": float(mag.mean()),
            "flow_direction_entropy": dir_entropy,
        }


# ── 3. Depth Estimation ────────────────────────────────────────────────────

class DepthAnnotator:
    """
    Estimates monocular depth for keyframes using DPT-Large (Intel/dpt-large).

    Saves depth maps as float16 .npy of shape (T_key, H, W).
    Normalised to [0, 1] within each frame (0=near, 1=far).
    Also records depth variance (flat scenes = low variance, informative scenes = high).
    """

    def __init__(
        self,
        model_id: str = "Intel/dpt-large",
        device: str = "cuda",
        num_keyframes: int = 4,
        output_size: Tuple[int, int] = (224, 224),
    ):
        self.model_id = model_id
        self.device = device
        self.num_keyframes = num_keyframes
        self.output_size = output_size
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import DPTImageProcessor, DPTForDepthEstimation
        logger.info(f"Loading DPT from {self.model_id}")
        self._processor = DPTImageProcessor.from_pretrained(self.model_id)
        self._model = DPTForDepthEstimation.from_pretrained(self.model_id).to(self.device)

    def annotate(self, video_path: str, output_path: str) -> Dict:
        try:
            import torch
            from PIL import Image

            self._load()
            frames = _decode_frames(video_path, num_frames=self.num_keyframes, size=self.output_size)
            if not frames:
                raise ValueError("No frames decoded")

            depth_maps = []
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output = self._model(**inputs)
                depth = output.predicted_depth.squeeze().cpu().numpy()
                # Resize to output_size and normalise
                depth = cv2.resize(depth, self.output_size)
                dmin, dmax = depth.min(), depth.max()
                if dmax > dmin:
                    depth = (depth - dmin) / (dmax - dmin)
                depth_maps.append(depth)

            depth_arr = np.stack(depth_maps, axis=0).astype(np.float16)  # (T_key, H, W)
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            np.save(output_path, depth_arr)

            depth_variance = float(depth_arr.astype(float).var())
            depth_mean = float(depth_arr.astype(float).mean())

            return {
                "depth_path": output_path,
                "depth_model": self.model_id,
                "depth_variance": depth_variance,   # low = flat scene (at-risk signal)
                "depth_mean": depth_mean,
            }

        except Exception as e:
            logger.warning(f"Depth failed for {video_path}: {e}")
            return {"depth_path": None, "depth_model": self.model_id,
                    "depth_variance": None, "depth_mean": None}


# ── Orchestrator ───────────────────────────────────────────────────────────

@dataclass
class MultitaskAnnotatorConfig:
    caption_model: str = "Salesforce/blip2-opt-2.7b"
    flow_method: str = "farneback"       # "farneback" | "raft"
    depth_model: str = "Intel/dpt-large"
    flow_frame_size: Tuple[int, int] = (224, 224)
    flow_num_frames: int = 9
    depth_num_keyframes: int = 4
    device: str = "cuda"
    skip_caption: bool = False
    skip_flow: bool = False
    skip_depth: bool = False


class MultitaskAnnotator:
    """
    Enriches curation manifests with caption, optical flow, and depth signals.

    Processes clips in the order they appear in the manifest; skips clips
    whose task artifacts already exist (safe to resume interrupted runs).
    """

    def __init__(self, config: Optional[MultitaskAnnotatorConfig] = None):
        self.cfg = config or MultitaskAnnotatorConfig()
        self._captioner = None if self.cfg.skip_caption else CaptionAnnotator(
            model_id=self.cfg.caption_model, device=self.cfg.device
        )
        self._flow = None if self.cfg.skip_flow else FlowAnnotator(
            method=self.cfg.flow_method,
            frame_size=self.cfg.flow_frame_size,
            num_frames=self.cfg.flow_num_frames,
            device=self.cfg.device,
        )
        self._depth = None if self.cfg.skip_depth else DepthAnnotator(
            model_id=self.cfg.depth_model,
            device=self.cfg.device,
            num_keyframes=self.cfg.depth_num_keyframes,
        )

    def annotate_clip(self, entry: Dict, task_root: str) -> Dict:
        """Annotate a single clip and return the enriched entry dict."""
        path = entry["path"]
        stem = Path(path).stem
        label = entry.get("label", "unknown")
        enriched = dict(entry)

        if self._captioner and "caption" not in enriched:
            enriched["caption"] = self._captioner.caption(path, label=label)
            enriched["caption_model"] = self.cfg.caption_model

        if self._flow and "flow_path" not in enriched:
            flow_out = str(Path(task_root) / "flow" / f"{stem}_flow.npy")
            enriched.update(self._flow.annotate(path, flow_out))

        if self._depth and "depth_path" not in enriched:
            depth_out = str(Path(task_root) / "depth" / f"{stem}_depth.npy")
            enriched.update(self._depth.annotate(path, depth_out))

        return enriched

    def annotate_manifest(
        self,
        manifest_path: str,
        output_dir: str,
        output_manifest: Optional[str] = None,
        batch_size: int = 8,
    ) -> List[Dict]:
        """
        Annotate all clips in a manifest JSONL.

        Writes enriched entries to `output_manifest` (default:
        `<manifest_dir>/manifest_multitask.jsonl`).
        """
        with open(manifest_path) as f:
            entries = [json.loads(l) for l in f if l.strip()]

        if output_manifest is None:
            output_manifest = str(Path(manifest_path).parent / "manifest_multitask.jsonl")

        os.makedirs(output_dir, exist_ok=True)

        enriched_entries = []
        for i, entry in enumerate(entries):
            if i % 50 == 0:
                logger.info(f"Annotating clip {i}/{len(entries)}")
            enriched = self.annotate_clip(entry, task_root=output_dir)
            enriched_entries.append(enriched)

        with open(output_manifest, "w") as f:
            for e in enriched_entries:
                f.write(json.dumps(e) + "\n")

        logger.info(f"Wrote multitask manifest to {output_manifest}")
        return enriched_entries

    def compute_task_quality_report(self, enriched_entries: List[Dict]) -> "pd.DataFrame":
        """
        Compute per-class task signal quality statistics.

        Returns a DataFrame showing whether the at-risk classes (low Laplacian)
        also fail on flow and depth signals, or whether those are orthogonal.
        """
        import pandas as pd

        rows = []
        for e in enriched_entries:
            rows.append({
                "class": e.get("label", "unknown"),
                "blur_score": e.get("blur_score"),
                "motion_score": e.get("motion_score"),
                "flow_mean_magnitude": e.get("flow_mean_magnitude"),
                "flow_direction_entropy": e.get("flow_direction_entropy"),
                "depth_variance": e.get("depth_variance"),
                "is_synthetic": e.get("is_synthetic", False),
            })

        df = pd.DataFrame(rows)
        report = df.groupby("class").agg({
            "blur_score": "mean",
            "flow_mean_magnitude": "mean",
            "flow_direction_entropy": "mean",
            "depth_variance": "mean",
        }).round(3)
        return report
