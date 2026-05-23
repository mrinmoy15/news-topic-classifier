# =============================================================================
# Makefile
# News Topic Classifier — CCDS v2 + GCP
# Usage: make <target>
# =============================================================================

.PHONY: help install install-dev install-api install-dashboard install-test install-lint

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