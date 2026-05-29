# =============================================================================
# Makefile
# News Topic Classifier — CCDS v2 + GCP
# Usage: make <target>
# =============================================================================

.PHONY: help install install-dev install-api install-dashboard install-test install-lint \
        docker-build-base docker-build-trainer docker-build docker-run docker-push-base docker-push-trainer

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

REGISTRY  := us-central1-docker.pkg.dev/cs-cdwp-data-dev2188/news-topic-classifier
LOCAL_TAG := local

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
	@echo "  make install           Install base dependencies only"
	@echo "  make install-dev       Install base + dev (notebooks, viz)"
	@echo "  make install-api       Install base + api (fastapi, uvicorn)"
	@echo "  make install-dashboard Install base + dashboard (streamlit)"
	@echo "  make install-test      Install base + test (pytest)"
	@echo "  make install-lint      Install lint tools (ruff, mypy)"
	@echo ""
	@echo "  Docker (local)"
	@echo "  --------------"
	@echo "  make docker-build      Build base + trainer images locally"
	@echo "  make docker-build-base Build base image only"
	@echo "  make docker-build-trainer Build trainer image only"
	@echo "  make docker-run        Run trainer container interactively"
	@echo ""
	@echo "  Docker (push to Artifact Registry)"
	@echo "  -----------------------------------"
	@echo "  make docker-push-base    Push base image to Artifact Registry"
	@echo "  make docker-push-trainer Push trainer image to Artifact Registry"
	@echo ""

# -----------------------------------------------------------------------------
# INSTALLATION
# -----------------------------------------------------------------------------

# Register package as editable + install core dependencies
install:
	uv pip install -e .
	uv pip install -r requirements/base.txt

# Local development — notebooks, EDA, visualisation
install-dev:
	uv pip install -e .
	uv pip install -r requirements/base.txt -r requirements/dev.txt

# Cloud Run API service
install-api:
	uv pip install -e .
	uv pip install -r requirements/base.txt -r requirements/api.txt

# Cloud Run dashboard service
install-dashboard:
	uv pip install -e .
	uv pip install -r requirements/base.txt -r requirements/dashboard.txt

# CI testing
install-test:
	uv pip install -e .
	uv pip install -r requirements/base.txt -r requirements/test.txt

# Linting + formatting (no base needed)
install-lint:
	uv pip install -r requirements/lint.txt

# -----------------------------------------------------------------------------
# DOCKER — LOCAL (Docker Desktop)
# -----------------------------------------------------------------------------

# Build base image locally
docker-build-base:
	docker compose build base

# Build trainer image locally (uses local base)
docker-build-trainer:
	docker compose build trainer

# Build both images locally
docker-build: docker-build-base docker-build-trainer

# Run trainer container interactively (mounts GCP credentials)
docker-run:
	docker compose run --rm trainer bash

# -----------------------------------------------------------------------------
# DOCKER — PUSH TO ARTIFACT REGISTRY
# -----------------------------------------------------------------------------

# Tag and push base image to Artifact Registry
docker-push-base: docker-build-base
	docker tag news-topic-classifier/base:$(LOCAL_TAG) $(REGISTRY)/base:latest
	docker push $(REGISTRY)/base:latest

# Tag and push trainer image to Artifact Registry (uses remote base)
docker-push-trainer: docker-build-trainer
	docker tag news-topic-classifier/trainer:$(LOCAL_TAG) $(REGISTRY)/trainer:latest
	docker push $(REGISTRY)/trainer:latest