# =============================================================================
# api/Dockerfile
# BBC News Topic Classifier — FastAPI serving container
#
# Builds on the shared base image (PyTorch + transformers already installed).
# Only adds the lightweight FastAPI / Uvicorn layer on top.
#
# Build args
# ----------
# BASE_IMAGE  Source base image (default: dev Artifact Registry)
#
# Environment variables (set at deploy time)
# ------------------------------------------
# MODEL_GCS_URI   gs://bucket/models/bert-bbc-finetuned/
# GCP_PROJECT     cs-cdwp-data-dev2188
# =============================================================================

ARG BASE_IMAGE=us-central1-docker.pkg.dev/cs-cdwp-data-dev2188/news-topic-classifier/base:latest

FROM ${BASE_IMAGE}

WORKDIR /app

# FastAPI + Uvicorn (torch / transformers already in base)
COPY requirements/api.txt requirements/api.txt
RUN pip install --no-cache-dir -r requirements/api.txt

# Project source
COPY pyproject.toml setup.cfg ./
COPY news_topic_classifier/ news_topic_classifier/
COPY api/ api/

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
