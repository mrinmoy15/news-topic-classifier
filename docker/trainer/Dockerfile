# =============================================================================
# docker/trainer/Dockerfile
# Trainer image — extends base with project source code
# Rebuilt on every code change
# =============================================================================

ARG BASE_IMAGE=us-central1-docker.pkg.dev/cs-cdwp-data-dev2188/news-topic-classifier/base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy project source and install as a package
COPY pyproject.toml setup.cfg ./
COPY news_topic_classifier/ news_topic_classifier/
COPY pipelines/ pipelines/
COPY conf/ conf/

RUN pip install --no-cache-dir -e .

# Vertex AI sets AIP_MODEL_DIR, AIP_TRAINING_DATA_URI, etc. at runtime
ENV PYTHONUNBUFFERED=1
