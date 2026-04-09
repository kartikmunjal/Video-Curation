"""
VideoMAE fine-tuning harness for the data composition ablation.

For each mixture ratio, fine-tunes MCG-NJU/videomae-base on the
corresponding train split and saves the best checkpoint.

Uses HuggingFace Trainer for simplicity; replace with custom loop if
gradient checkpointing or FSDP is needed at larger scale.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


# ── Collate function ──────────────────────────────────────────────────────────

def videomae_collate(batch: list[dict]) -> dict:
    """Stack pixel_values and labels from a list of dataset items."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


# ── Metric computation ────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Top-1 and Top-5 accuracy for HuggingFace Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    top1 = (predictions == labels).mean()

    # Top-5
    top5_preds = np.argsort(logits, axis=-1)[:, -5:]
    top5 = np.array([labels[i] in top5_preds[i] for i in range(len(labels))]).mean()

    return {"top1_accuracy": top1, "top5_accuracy": top5}


# ── Model setup ───────────────────────────────────────────────────────────────

def build_model(model_name: str, num_classes: int, dropout: float = 0.1):
    """Load VideoMAE with a classification head."""
    from transformers import VideoMAEForVideoClassification, VideoMAEConfig

    log.info("Loading %s (num_classes=%d)", model_name, num_classes)
    try:
        model = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    except Exception as exc:
        log.warning("Could not load from hub (%s), loading from config", exc)
        config = VideoMAEConfig(num_labels=num_classes)
        model = VideoMAEForVideoClassification(config)

    # Apply dropout to the classifier head
    if hasattr(model.classifier, "dropout"):
        model.classifier.dropout.p = dropout

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("Model loaded: %.1fM parameters", n_params)
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def run_finetune(
    train_manifest: str | Path,
    val_manifest: str | Path,
    model_name: str,
    num_classes: int,
    output_dir: str | Path,
    synth_ratio: float,
    epochs: int = 20,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 5e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    fp16: bool = True,
    seed: int = 42,
    num_frames: int = 16,
    frame_size: int = 224,
    fps_target: float = 8.0,
    device: str = "cuda",
    early_stopping_patience: int = 5,
) -> dict:
    """Fine-tune VideoMAE on a single mixture split.

    Returns
    -------
    dict with keys: best_top1, best_top5, checkpoint_path, synth_ratio
    """
    from transformers import TrainingArguments, Trainer
    from transformers.trainer_utils import set_seed

    from video_curation.data.dataset import VideoClipDataset

    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets — the synth_ratio is already baked into the manifest
    train_ds = VideoClipDataset(
        manifest_path=train_manifest,
        split="train",
        synth_ratio=0.0,  # ratio already applied via data_mixture.py
        num_frames=num_frames,
        fps_target=fps_target,
        frame_size=frame_size,
    )
    val_ds = VideoClipDataset(
        manifest_path=val_manifest,
        split="val",
        synth_ratio=0.0,
        num_frames=num_frames,
        fps_target=fps_target,
        frame_size=frame_size,
    )

    model = build_model(model_name, num_classes)

    run_name = f"videomae_ratio{synth_ratio:.2f}".replace(".", "p")

    training_args = TrainingArguments(
        output_dir=str(output_dir / run_name),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        fp16=fp16 and device == "cuda",
        dataloader_num_workers=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_top1_accuracy",
        greater_is_better=True,
        logging_steps=50,
        run_name=run_name,
        seed=seed,
        report_to="none",  # disable wandb/tensorboard by default
        label_names=["labels"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=videomae_collate,
        compute_metrics=compute_metrics,
    )

    try:
        log.info("Starting training: ratio=%.2f, train=%d, val=%d",
                 synth_ratio, len(train_ds), len(val_ds))
        train_result = trainer.train()
        trainer.save_model()

        eval_results = trainer.evaluate()
        best_top1 = eval_results.get("eval_top1_accuracy", 0.0)
        best_top5 = eval_results.get("eval_top5_accuracy", 0.0)

    except Exception as exc:
        log.error("Training failed for ratio=%.2f: %s", synth_ratio, exc)
        best_top1, best_top5 = 0.0, 0.0

    result = {
        "synth_ratio": synth_ratio,
        "best_top1": best_top1,
        "best_top5": best_top5,
        "train_clips": len(train_ds),
        "val_clips": len(val_ds),
        "checkpoint_path": str(output_dir / run_name),
    }

    # Save per-run result
    with open(output_dir / run_name / "result.json", "w") as fh:
        json.dump(result, fh, indent=2)

    log.info(
        "Finished ratio=%.2f: top1=%.4f, top5=%.4f",
        synth_ratio, best_top1, best_top5,
    )
    return result


def run_ablation(
    split_paths: dict[float, dict[str, Path]],
    output_dir: str | Path,
    model_name: str = "MCG-NJU/videomae-base",
    num_classes: int = 10,
    **train_kwargs,
) -> list[dict]:
    """Run fine-tuning across all mixture ratios.

    Parameters
    ----------
    split_paths:
        Output of ``data_mixture.build_splits()``.
    output_dir:
        Root directory for checkpoints and per-ratio results.
    """
    output_dir = Path(output_dir)
    results: list[dict] = []

    for ratio, paths in sorted(split_paths.items()):
        log.info("=" * 60)
        log.info("Starting ablation run: synth_ratio=%.2f", ratio)
        result = run_finetune(
            train_manifest=paths["train"],
            val_manifest=paths["val"],
            model_name=model_name,
            num_classes=num_classes,
            output_dir=output_dir,
            synth_ratio=ratio,
            **train_kwargs,
        )
        results.append(result)

    # Aggregate results table
    table_path = output_dir / "ablation_results.json"
    with open(table_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info("Ablation results saved to %s", table_path)

    return results
