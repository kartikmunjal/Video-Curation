#!/usr/bin/env python3
"""
Run FVD + CLIP evaluation across all ablation mixture ratios.

    python scripts/run_evaluation.py --results_dir results/ablation

    # Fast mode (cap at 200 clips per set):
    python scripts/run_evaluation.py \\
        --results_dir results/ablation --max_clips 200
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_curation.evaluation.runner import main

if __name__ == "__main__":
    main()
