"""
Generative synthetic clip synthesis for underrepresented classes.

Uses a pre-trained text-to-video diffusion model (CogVideoX-2B, or a
LoRA fine-tuned variant produced by the Video-Generation pipeline) to
generate entirely new clips for classes identified as underrepresented
by the bias analysis.

This closes the loop between Video-Curation and Video-Generation:
  Video-Curation bias analysis  →  at-risk class list + captions
  Video-Generation fine-tuned model  →  generates new clips
  Video-Curation  →  curates generated clips and mixes back in

Why model-driven synthesis over transformation augmentation:
  Transformation augmentation (speed, jitter, color) preserves the
  original filming environment — a guitar clip from a dark stage stays
  a dark-stage clip.  Generative synthesis can produce PlayingGuitar
  clips filmed under a variety of lighting, background, and camera
  conditions, introducing genuine appearance diversity that the
  transformation pipeline cannot create.

Usage:
    from video_curation.augmentation.generative_synthesis import GenerativeSynthesizer

    synth = GenerativeSynthesizer(
        model_path="THUDM/CogVideoX-2b",          # or path to LoRA checkpoint
        lora_path="../Video-Generation/checkpoints/lora_r16_round3",  # optional
        device="cuda",
    )

    entries = synth.generate_for_at_risk_classes(
        bias_results_csv="results/bias_analysis/blur_threshold_sweep.csv",
        threshold=80,
        retention_cutoff=0.5,          # classes below 50% retention are "at-risk"
        n_clips_per_class=50,
        output_dir="data/synthetic_generative",
    )
    # entries: list of manifest dicts ready to concat with real manifest
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Prompt templates by class ──────────────────────────────────────────────
# Deliberately varied to maximise appearance diversity in the generated set.
# Each template has multiple variants so n_clips_per_class generates distinct
# clips rather than copies.

_PROMPT_TEMPLATES: Dict[str, List[str]] = {
    "PlayingGuitar": [
        "a musician playing acoustic guitar on a dimly lit stage, warm spotlight",
        "close-up of hands strumming a guitar, outdoor park, natural daylight",
        "a guitarist performing at an open-air concert, crowd in the background",
        "a person playing electric guitar in a rehearsal room, neon lights",
        "an elderly man playing classical guitar on a wooden chair, sunlit room",
        "a street musician playing guitar on a city sidewalk, passersby",
        "two musicians playing guitar and singing together on a small stage",
        "a guitarist playing fingerstyle on a dock by the water at sunset",
    ],
    "Rowing": [
        "two athletes rowing a sculling boat on a calm river, morning mist",
        "a rowing team of eight in a racing shell on a lake, blue sky",
        "a single sculler rowing on open water, reflections visible",
        "a canoe being paddled down a forested river, overhead shot",
        "competitive rowing race, multiple boats, referee boat alongside",
        "a person rowing a wooden rowboat on a mountain lake at dusk",
        "indoor rowing machine athlete, gym setting, effort visible",
        "dragon boat racing, colorful boat, team paddling in unison",
    ],
    "IndoorClimbing": [
        "a climber ascending an indoor bouldering wall, colourful holds",
        "a rock climber on an overhanging gym wall, chalk dust visible",
        "a beginner learning to climb on a low training wall, instructor nearby",
    ],
    "Yoga": [
        "a person performing yoga on a mat in a bright studio, morning light",
        "outdoor yoga session on a rooftop, city skyline in background",
        "a yoga class in a wood-floored studio, instructor demonstrating pose",
    ],
}

_DEFAULT_TEMPLATES = [
    "a person performing {class_name} in an outdoor setting, daytime",
    "close view of {class_name} activity, varied background",
    "a professional demonstrating {class_name}, studio setting",
]


@dataclass
class GenerationConfig:
    num_frames: int = 49       # CogVideoX default: 49 frames ≈ 6s at 8fps
    height: int = 480
    width: int = 720
    guidance_scale: float = 6.0
    num_inference_steps: int = 50
    fps: int = 8
    seed: Optional[int] = None


@dataclass
class SynthesisResult:
    path: str
    caption: str
    class_label: str
    model_path: str
    lora_path: Optional[str]
    generation_config: Dict
    is_synthetic: bool = True
    synthesis_method: str = "generative_cogvideox"


class GenerativeSynthesizer:
    """
    Generates synthetic video clips for underrepresented classes using
    CogVideoX-2B text-to-video diffusion.

    The model can be:
      - Base CogVideoX-2B (THUDM/CogVideoX-2b) — general purpose
      - LoRA fine-tuned on domain clips (from Video-Generation pipeline) —
        domain-adapted, better appearance match

    Generated clips pass through the same curation filters as real clips
    before being added to the training manifest.
    """

    def __init__(
        self,
        model_path: str = "THUDM/CogVideoX-2b",
        lora_path: Optional[str] = None,
        device: str = "cuda",
        config: Optional[GenerationConfig] = None,
        dtype_str: str = "float16",
    ):
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device
        self.config = config or GenerationConfig()
        self._pipe = None
        self._dtype_str = dtype_str

    def _load_pipeline(self):
        """Lazy-load the diffusion pipeline (avoids import at module level)."""
        if self._pipe is not None:
            return

        import torch
        from diffusers import CogVideoXPipeline

        dtype = torch.float16 if self._dtype_str == "float16" else torch.bfloat16

        logger.info(f"Loading CogVideoX from {self.model_path}")
        pipe = CogVideoXPipeline.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
        )

        if self.lora_path and Path(self.lora_path).exists():
            logger.info(f"Loading LoRA weights from {self.lora_path}")
            pipe.load_lora_weights(self.lora_path)

        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        self._pipe = pipe

    def generate_clip(
        self,
        prompt: str,
        output_path: str,
        seed: Optional[int] = None,
    ) -> bool:
        """
        Generate a single video clip from a text prompt.

        Returns True on success, False on failure (logged but not raised
        so a batch run can continue past individual failures).
        """
        import torch

        try:
            self._load_pipeline()

            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            elif self.config.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(self.config.seed)

            result = self._pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=self.config.num_inference_steps,
                num_frames=self.config.num_frames,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
                height=self.config.height,
                width=self.config.width,
            )

            # Export frames to mp4
            from diffusers.utils import export_to_video
            frames = result.frames[0]
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            export_to_video(frames, output_path, fps=self.config.fps)
            logger.info(f"Generated: {output_path}")
            return True

        except Exception as e:
            logger.warning(f"Generation failed for prompt '{prompt[:60]}...': {e}")
            return False

    def generate_for_class(
        self,
        class_name: str,
        n_clips: int,
        output_dir: str,
        prompts: Optional[List[str]] = None,
    ) -> List[SynthesisResult]:
        """
        Generate `n_clips` synthetic clips for a given class.

        Prompts are cycled from the class-specific template list (or the
        provided `prompts` list) so each clip has a distinct appearance context.
        """
        templates = prompts or _PROMPT_TEMPLATES.get(class_name)
        if not templates:
            templates = [
                t.replace("{class_name}", class_name.lower().replace("_", " "))
                for t in _DEFAULT_TEMPLATES
            ]

        output_dir = Path(output_dir) / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i in range(n_clips):
            prompt = templates[i % len(templates)]
            clip_name = f"{class_name}_{i:04d}.mp4"
            out_path = str(output_dir / clip_name)

            success = self.generate_clip(
                prompt=prompt,
                output_path=out_path,
                seed=i,  # deterministic per-clip seed for reproducibility
            )

            if success:
                results.append(SynthesisResult(
                    path=out_path,
                    caption=prompt,
                    class_label=class_name,
                    model_path=self.model_path,
                    lora_path=self.lora_path,
                    generation_config={
                        "num_frames": self.config.num_frames,
                        "height": self.config.height,
                        "width": self.config.width,
                        "guidance_scale": self.config.guidance_scale,
                        "num_inference_steps": self.config.num_inference_steps,
                        "seed": i,
                    },
                ))

        logger.info(f"{class_name}: generated {len(results)}/{n_clips} clips")
        return results

    def generate_for_at_risk_classes(
        self,
        bias_results_csv: str,
        threshold: float = 80.0,
        retention_cutoff: float = 0.5,
        n_clips_per_class: int = 50,
        output_dir: str = "data/synthetic_generative",
        write_manifest: bool = True,
    ) -> List[Dict]:
        """
        Load the bias analysis sweep, identify at-risk classes at the given
        blur threshold, and generate `n_clips_per_class` synthetic clips for
        each.

        At-risk = classes with retention_rate < `retention_cutoff` at the
        specified threshold.

        Returns a list of manifest-compatible dicts (same schema as the
        curation pipeline's JSONL output).
        """
        import pandas as pd

        df = pd.read_csv(bias_results_csv)
        at_threshold = df[df["threshold"] == threshold]
        at_risk = at_threshold[at_threshold["retention_rate"] < retention_cutoff]["class"].tolist()

        logger.info(f"At-risk classes at σ<{threshold}: {at_risk}")

        all_results: List[SynthesisResult] = []
        for cls in at_risk:
            results = self.generate_for_class(
                class_name=cls,
                n_clips=n_clips_per_class,
                output_dir=output_dir,
            )
            all_results.extend(results)

        manifest_entries = [
            {
                "path": r.path,
                "caption": r.caption,
                "label": r.class_label,
                "is_synthetic": True,
                "synthesis_method": r.synthesis_method,
                "model_path": r.model_path,
                "lora_path": r.lora_path,
                "generation_config": r.generation_config,
                # Curation scores will be filled in by run_curation.py
                "blur_score": None,
                "motion_score": None,
                "quality_score": None,
            }
            for r in all_results
        ]

        if write_manifest:
            manifest_path = Path(output_dir) / "manifest_generative.jsonl"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as f:
                for entry in manifest_entries:
                    f.write(json.dumps(entry) + "\n")
            logger.info(f"Wrote {len(manifest_entries)} entries to {manifest_path}")

        return manifest_entries


def build_generative_prompts_from_captions(
    manifest_path: str,
    class_name: str,
    n_prompts: int = 20,
    diversity_prefix: str = "cinematic video of",
) -> List[str]:
    """
    Extract diverse prompts from the existing manifest's VLM-generated captions
    for a given class, then prepend a cinematic prefix to guide generation.

    This produces prompts grounded in the actual appearance distribution of
    the class rather than hand-written templates.
    """
    with open(manifest_path) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    class_entries = [e for e in entries if e.get("label") == class_name and e.get("caption")]
    captions = [e["caption"] for e in class_entries[:n_prompts]]

    return [f"{diversity_prefix} {cap}" for cap in captions]
