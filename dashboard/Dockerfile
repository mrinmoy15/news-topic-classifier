# =============================================================================
# dashboard/Dockerfile
# BBC News Topic Classifier — Streamlit monitoring dashboard
#
# Lightweight Python image — no PyTorch needed (calls API via HTTP).
#
# Environment variables (set at deploy time)
# ------------------------------------------
# API_URL       http://your-api-service-url
# GCP_PROJECT   cs-cdwp-data-dev2188
# BQ_DATASET    DATA_SCNCE_DEV_DATA
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY dashboard/requirements.txt requirements/dashboard.txt
RUN pip install --no-cache-dir -r requirements/dashboard.txt

COPY pyproject.toml setup.cfg ./
COPY news_topic_classifier/ news_topic_classifier/
COPY dashboard/ dashboard/

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
