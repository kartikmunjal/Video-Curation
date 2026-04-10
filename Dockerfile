# Video-Curation pipeline container
#
# Builds a self-contained image for the Ray curation workers.
# Designed to run as a Ray worker node on Kubernetes (KubeRay) or locally.
#
# Build:
#   docker build -t video-curation:latest .
#
# Run locally (single-worker smoke test):
#   docker run --rm -v $(pwd)/data:/app/data video-curation:latest \
#       python scripts/run_curation.py --config configs/curation.yaml \
#       --blur_threshold 40 --output_dir data/curated/blur40
#
# On Kubernetes: see deploy/ray-cluster.yaml

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies for OpenCV + video decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[ray]" \
    && pip install --no-cache-dir \
        ray[default]==2.9.3 \
        lancedb>=0.6.0 \
        pyarrow>=14.0.0 \
        decord==0.6.0

# Application code
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# Ray head node starts the dashboard on 8265; workers listen on 6379
EXPOSE 8265 6379

ENV PYTHONPATH=/app/src
