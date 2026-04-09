"""
Evaluation runner: compute FVD + CLIP metrics for all ablation runs.

CLI entry point:
    vc-eval --results_dir results/ablation --splits_dir data/splits
    python -m video_curation.evaluation.runner --results_dir ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def evaluate_all_ratios(
    splits_dir: str | Path,
    results_dir: str | Path,
    real_test_manifest: str | Path,
    i3d_model: str = "google/i3d-kinetics-400",
    clip_model: str = "openai/clip-vit-base-patch32",
    device: str = "cuda",
    num_frames: int = 16,
    frame_size: int = 224,
    max_clips: Optional[int] = None,
) -> list[dict]:
    """Evaluate FVD and CLIP metrics for each ratio sub-directory.

    Looks for checkpoints in results_dir/ratio_X_XX/ and synthesised clips
    via the model (or uses the synth manifests directly for distribution metrics).
    """
    from video_curation.evaluation.fvd import evaluate_fvd
    from video_curation.evaluation.clip_eval import evaluate_clip

    splits_dir = Path(splits_dir)
    results_dir = Path(results_dir)
    eval_output = results_dir / "eval_results.json"

    ratio_dirs = sorted(splits_dir.glob("ratio_*"))
    if not ratio_dirs:
        log.error("No ratio directories found in %s", splits_dir)
        return []

    all_results: list[dict] = []

    for ratio_dir in ratio_dirs:
        ratio_str = ratio_dir.name.replace("ratio_", "").replace("p", ".")
        try:
            ratio = float(ratio_str)
        except ValueError:
            continue

        train_manifest = ratio_dir / "train.jsonl"
        test_manifest = ratio_dir / "test.jsonl"

        if not train_manifest.exists():
            log.warning("Missing train manifest for ratio=%s", ratio)
            continue

        log.info("Evaluating ratio=%.2f", ratio)
        result: dict = {"ratio": ratio}

        # CLIP score on test set (real only)
        if test_manifest.exists():
            try:
                clip_metrics = evaluate_clip(
                    test_manifest,
                    model_name=clip_model,
                    device=device,
                    sample_frames=num_frames,
                    max_clips=max_clips,
                )
                result.update({f"test_{k}": v for k, v in clip_metrics.items()
                                if not isinstance(v, dict)})
                result["test_compactness"] = clip_metrics.get("intra_class_compactness", {})
            except Exception as exc:
                log.warning("CLIP eval failed for ratio=%.2f: %s", ratio, exc)
                result["test_clip_score"] = float("nan")

        # FVD between train split real clips and synth clips
        # (proxy: compare test real vs. train synth portion)
        try:
            synth_entries = _filter_manifest(train_manifest, is_synthetic=True)
            real_entries = _filter_manifest(test_manifest or train_manifest, is_synthetic=False)

            if synth_entries and real_entries:
                real_tmp = _write_tmp_manifest(results_dir / "_tmp_real.jsonl", real_entries)
                synth_tmp = _write_tmp_manifest(results_dir / "_tmp_synth.jsonl", synth_entries)

                fvd_result = evaluate_fvd(
                    real_tmp, synth_tmp,
                    i3d_model_name=i3d_model,
                    device=device,
                    num_frames=num_frames,
                    frame_size=frame_size,
                    max_clips=max_clips,
                )
                result.update({f"fvd_{k}": v for k, v in fvd_result.items()})
            else:
                result["fvd_fvd"] = float("nan")
                log.info("  Skipping FVD (no synthetic clips at ratio=%.2f)", ratio)
        except Exception as exc:
            log.warning("FVD eval failed for ratio=%.2f: %s", ratio, exc)
            result["fvd_fvd"] = float("nan")

        # Load training accuracy from saved result.json if available
        run_name = f"videomae_ratio{ratio:.2f}".replace(".", "p")
        run_result_path = results_dir / run_name / "result.json"
        if run_result_path.exists():
            with open(run_result_path) as fh:
                run_result = json.load(fh)
            result["best_top1"] = run_result.get("best_top1", float("nan"))
            result["best_top5"] = run_result.get("best_top5", float("nan"))
            result["train_clips"] = run_result.get("train_clips", 0)
        else:
            result["best_top1"] = float("nan")
            result["best_top5"] = float("nan")

        all_results.append(result)
        log.info("  ratio=%.2f | FVD=%.1f | CLIP=%.4f | Top-1=%.4f",
                 ratio,
                 result.get("fvd_fvd", float("nan")),
                 result.get("test_clip_score", float("nan")),
                 result.get("best_top1", float("nan")))

    # Save aggregate
    with open(eval_output, "w") as fh:
        json.dump(all_results, fh, indent=2)
    log.info("Evaluation results saved: %s", eval_output)

    return all_results


def _filter_manifest(path: Path, is_synthetic: bool) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(l) for l in fh if json.loads(l).get("is_synthetic") == is_synthetic]


def _write_tmp_manifest(path: Path, entries: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    return path


def print_results_table(results: list[dict]) -> None:
    header = f"{'Ratio':>8}  {'FVD':>8}  {'CLIP':>8}  {'Top-1':>8}  {'Train N':>8}"
    print("\n" + header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x["ratio"]):
        print(
            f"{r['ratio']:>8.2f}  "
            f"{r.get('fvd_fvd', float('nan')):>8.1f}  "
            f"{r.get('test_clip_score', float('nan')):>8.4f}  "
            f"{r.get('best_top1', float('nan')):>8.4f}  "
            f"{r.get('train_clips', 0):>8}"
        )


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", required=True, help="Ablation results directory")
    p.add_argument("--splits_dir", default="data/splits", help="Mixture splits directory")
    p.add_argument("--real_test", default=None, help="Override real test manifest")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_clips", type=int, default=None,
                   help="Cap clips per set (for fast testing)")
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    results = evaluate_all_ratios(
        splits_dir=args.splits_dir,
        results_dir=args.results_dir,
        real_test_manifest=args.real_test or Path(args.splits_dir) / "ratio_0p00" / "test.jsonl",
        device=args.device,
        max_clips=args.max_clips,
    )
    print_results_table(results)


if __name__ == "__main__":
    main()
