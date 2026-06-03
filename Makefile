# =============================================================================
# Makefile
# News Topic Classifier — CCDS v2 + GCP
# Usage: make <target>
# =============================================================================

.PHONY: help install install-dev install-api install-dashboard install-test install-lint \
        test test-cov integration-test integration-test-full \
        register-model batch-predict \
        docker-build-base docker-build-trainer docker-build-api docker-build docker-run \
        docker-push-base docker-push-trainer docker-push-api \
        docker-test-extract docker-test-preprocess docker-test-train docker-test-predict docker-test-report docker-test-all

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

LOCAL_TAG := local

REGISTRY  := us-central1-docker.pkg.dev/cs-cdwp-data-dev2188/news-topic-classifier

# -----------------------------------------------------------------------------
# DEFAULT
# -----------------------------------------------------------------------------

help:
	@echo ""
	@echo "News Topic Classifier — Available Commands"
	@echo "==========================================="
	@echo ""
	@echo "  Installation"
	@echo "  ------------"
	@echo "  make install              Install base dependencies only"
	@echo "  make install-dev          Install base + dev (notebooks, viz)"
	@echo "  make install-api          Install base + api (fastapi, uvicorn)"
	@echo "  make install-dashboard    Install base + dashboard (streamlit)"
	@echo "  make install-test         Install base + test (pytest)"
	@echo "  make install-lint         Install lint tools (ruff, mypy)"
	@echo ""
	@echo "  Docker (build)"
	@echo "  --------------"
	@echo "  make docker-build         Build base + trainer images locally"
	@echo "  make docker-build-base    Build base image only"
	@echo "  make docker-build-trainer Build trainer image only"
	@echo "  make docker-run           Run trainer container interactively"
	@echo ""
	@echo "  Docker (smoke tests)"
	@echo "  --------------------"
	@echo "  make docker-test-extract    Test BQ extraction inside container"
	@echo "  make docker-test-preprocess Test preprocessing inside container"
	@echo "  make docker-test-train      Test BERT training inside container"
	@echo "  make docker-test-predict    Test inference inside container"
	@echo "  make docker-test-all        Run all smoke tests in sequence"
	@echo ""
	@echo "  Docker (push to Artifact Registry)"
	@echo "  -----------------------------------"
	@echo "  make docker-push-base       Push base image to dev Artifact Registry"
	@echo "  make docker-push-trainer    Push trainer image to dev Artifact Registry"
	@echo "  make docker-push-api        Push api image to dev Artifact Registry"
	@echo "  Note: promotion dev->prd is handled by CI/CD (promote.yml)"
	@echo ""
	@echo "  Inference"
	@echo "  ---------"
	@echo "  make batch-predict ENV=dev          Run batch inference for today's partition"
	@echo "  make batch-predict ENV=dev DAY=5    Run for a specific day partition (0-29)"
	@echo ""

# -----------------------------------------------------------------------------
# INSTALLATION
# -----------------------------------------------------------------------------

install:
	uv pip install -e .
	uv pip install -r requirements/base.txt

install-dev:
	uv pip install -e .
	uv pip install -r requirements/base.txt -r requirements/dev.txt

install-api:
	uv pip install -e .
	uv pip install -r requirements/base.txt -r requirements/api.txt

install-dashboard:
	uv pip install -e .
	uv pip install -r requirements/base.txt -r requirements/dashboard.txt

install-test:
	uv pip install -e .
	uv pip install -r requirements/base.txt -r requirements/test.txt

install-lint:
	uv pip install -r requirements/lint.txt

# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------

test:
	pytest tests/unit/ -v

test-cov:
	pytest tests/unit/ --cov=news_topic_classifier --cov-report=term-missing

# Requires GCP credentials (ADC or WIF).
# Set INTEGRATION_TESTS=true before calling:
#   PowerShell : $env:INTEGRATION_TESTS="true"; make integration-test
#   bash/Linux : INTEGRATION_TESTS=true make integration-test
# ---------------------------------------------------------------------------
# INTEGRATION TESTS
# ---------------------------------------------------------------------------

integration-test:
	pytest tests/integration/ -m "integration and not slow" -v --no-cov

integration-test-full:
	pytest tests/integration/ -m integration -v --no-cov

# -----------------------------------------------------------------------------
# DOCKER — BUILD (local Docker Desktop)
# -----------------------------------------------------------------------------

docker-build-base:
	docker compose build base

docker-build-trainer:
	docker compose build trainer

docker-build: docker-build-base docker-build-trainer

docker-run:
	docker compose run --rm trainer bash

# -----------------------------------------------------------------------------
# DOCKER — SMOKE TESTS (run core modules inside the trainer container)
# -----------------------------------------------------------------------------

docker-test-extract:
	docker compose run --rm trainer python -m news_topic_classifier.dataset

# Usage: make docker-test-preprocess RAW_GCS_URI=gs://bucket/data/raw/.../bbc_news.parquet
docker-test-preprocess:
	docker compose run --rm trainer python -m news_topic_classifier.features \
		data.raw_gcs_uri=$(RAW_GCS_URI)

# Usage: make docker-test-train GCS_SPLITS_DIR=gs://bucket/data/processed/2026-.../
docker-test-train:
	docker compose run --rm trainer python -m news_topic_classifier.modeling.train \
		data.gcs_splits_dir=$(GCS_SPLITS_DIR)

# Usage: make docker-test-predict GCS_SPLITS_DIR=gs://bucket/data/processed/2026-.../
docker-test-predict:
	docker compose run --rm trainer python -m news_topic_classifier.modeling.predict \
		data.gcs_splits_dir=$(GCS_SPLITS_DIR)

# Usage: make docker-test-report RUN_ID=<mlflow-run-id> GCS_PREDICTIONS_URI=gs://...
docker-test-report:
	docker compose run --rm trainer python -m news_topic_classifier.modeling.report \
		+report.run_id=$(RUN_ID) \
		+report.gcs_predictions_uri=$(GCS_PREDICTIONS_URI)

docker-test-all: docker-test-extract docker-test-preprocess docker-test-train docker-test-predict docker-test-report

# -----------------------------------------------------------------------------
# DOCKER — PUSH TO ARTIFACT REGISTRY
# -----------------------------------------------------------------------------

docker-push-base: docker-build-base
	docker tag news-topic-classifier/base:$(LOCAL_TAG) $(REGISTRY)/base:latest
	docker push $(REGISTRY)/base:latest

docker-push-trainer: docker-build-trainer
	docker tag news-topic-classifier/trainer:$(LOCAL_TAG) $(REGISTRY)/trainer:latest
	docker push $(REGISTRY)/trainer:latest

docker-build-api:
	docker compose build api

docker-push-api: docker-build-api
	docker tag news-topic-classifier/api:$(LOCAL_TAG) $(REGISTRY)/api:latest
	docker push $(REGISTRY)/api:latest

# -----------------------------------------------------------------------------
# INFERENCE
# -----------------------------------------------------------------------------

# Usage: make batch-predict ENV=dev
#        make batch-predict ENV=dev DAY=5
ENV ?= dev
DAY ?=

batch-predict:
	python scripts/batch_predict.py --environment $(ENV) $(if $(DAY),--day $(DAY),)

