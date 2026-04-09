#!/usr/bin/env python3
"""
Convenience wrapper around the Ray curation pipeline.

For the bias ablation, run multiple times with different blur thresholds:

    python scripts/run_curation.py --config configs/curation.yaml \\
        --blur_threshold 20 --output_dir data/curated/blur20

    python scripts/run_curation.py --config configs/curation.yaml \\
        --blur_threshold 40 --output_dir data/curated/blur40

    python scripts/run_curation.py --config configs/curation.yaml \\
        --blur_threshold 80 --output_dir data/curated/blur80
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_curation.pipeline.curator import main

if __name__ == "__main__":
    main()
