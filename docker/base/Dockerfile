# =============================================================================
# docker/base/Dockerfile
# Base image — PyTorch + CUDA + all Python dependencies
# Built once and pushed to Artifact Registry as :base
# Rebuilt only when requirements/base.txt changes
# =============================================================================

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to maximise layer caching
COPY requirements/base.txt requirements/base.txt

# Install all dependencies — torch is already provided by the base image
# so we skip the --extra-index-url torch wheel and install the rest
RUN pip install --no-cache-dir --upgrade pip && \
    grep -v "^--extra-index-url\|^torch\|^torchvision" requirements/base.txt \
    | pip install --no-cache-dir -r /dev/stdin
