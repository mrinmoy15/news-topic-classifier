"""
BBC News Topic Classifier — Monitoring Dashboard

Tabs
----
1. Live Inference    — paste text, get real-time prediction from the API
2. Daily Accuracy    — line chart of daily accuracy over time
3. Label Distribution — stacked bar chart of predictions per day
4. Confusion Matrix  — heatmap of true vs predicted labels

Environment variables
---------------------
API_URL       Base URL of the FastAPI serving container (default: localhost:8080)
GCP_PROJECT   GCP project ID for BigQuery
BQ_DATASET    BigQuery dataset containing the predictions table
"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google.cloud import bigquery

from dashboard.bq_queries import (
    confusion_data,
    daily_accuracy,
    label_distribution,
    summary_stats,
)

# ─── Configuration ────────────────────────────────────────────────────────────

API_URL     = os.environ.get("API_URL",     "http://localhost:8080")
GCP_PROJECT = os.environ.get("GCP_PROJECT", "cs-cdwp-data-dev2188")
BQ_DATASET  = os.environ.get("BQ_DATASET",  "DATA_SCNCE_DEV_DATA")

LABEL_COLOURS = {
    "business":      "#636EFA",
    "entertainment": "#EF553B",
    "politics":      "#00CC96",
    "sport":         "#AB63FA",
    "tech":          "#FFA15A",
}

LABELS = ["business", "entertainment", "politics", "sport", "tech"]

st.set_page_config(
    page_title="BBC News Classifier",
    page_icon="📰",
    layout="wide",
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource
def _bq_client() -> bigquery.Client:
    return bigquery.Client(project=GCP_PROJECT)


@st.cache_data(ttl=300)
def _query(sql: str) -> pd.DataFrame:
    try:
        return _bq_client().query(sql).to_dataframe()
    except Exception as e:
        st.error(f"BigQuery error: {e}")
        return pd.DataFrame()


def _call_api(text: str) -> dict | None:
    try:
        resp = httpx.post(
            f"{API_URL}/predict",
            json={"instances": [{"text": text}]},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["predictions"][0]
    except httpx.ConnectError:
        st.error(f"Cannot reach API at {API_URL}. Is the serving container running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📰 BBC News Classifier")
    st.caption(f"**Project:** `{GCP_PROJECT}`")
    st.caption(f"**Dataset:** `{BQ_DATASET}`")
    st.divider()

    stats_df = _query(summary_stats(GCP_PROJECT, BQ_DATASET))
    if not stats_df.empty and int(stats_df.iloc[0]["total_predictions"]) > 0:
        row = stats_df.iloc[0]
        st.metric("Total Predictions", int(row["total_predictions"]))
        st.metric("Overall Accuracy",  f"{float(row['overall_accuracy']):.1%}")
        st.metric("Avg Confidence",    f"{float(row['avg_confidence']):.1%}")
        st.caption(f"First run: {row['first_run']}")
        st.caption(f"Latest run: {row['latest_run']}")
    else:
        st.info("No batch predictions yet.")

    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_live, tab_accuracy, tab_dist, tab_confusion = st.tabs([
    "🔴 Live Inference",
    "📈 Daily Accuracy",
    "📊 Label Distribution",
    "🔀 Confusion Matrix",
])

# ─── Tab 1: Live Inference ────────────────────────────────────────────────────

with tab_live:
    st.header("Live Inference")
    st.caption(f"Calls the production API at `{API_URL}` in real time.")

    text_input = st.text_area(
        "Paste a news article or headline:",
        height=160,
        placeholder=(
            "e.g. Apple reported record quarterly earnings on Thursday, "
            "driven by strong iPhone sales and growth in services revenue..."
        ),
    )

    if st.button("Classify", type="primary", disabled=not text_input.strip()):
        with st.spinner("Calling model API..."):
            result = _call_api(text_input.strip())

        if result:
            label  = result["label"]
            conf   = result["confidence"]
            scores = result["scores"]

            col_metric, col_chart = st.columns([1, 2])

            with col_metric:
                st.metric("Predicted Topic", label.upper())
                st.metric("Confidence", f"{conf:.1%}")

            with col_chart:
                score_df = (
                    pd.DataFrame({"label": list(scores.keys()), "score": list(scores.values())})
                    .sort_values("score", ascending=True)
                )
                fig = px.bar(
                    score_df, x="score", y="label", orientation="h",
                    color="label", color_discrete_map=LABEL_COLOURS,
                    title="Class Probability Scores",
                    labels={"score": "Probability", "label": "Category"},
                )
                fig.update_layout(showlegend=False, height=280, margin=dict(l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

# ─── Tab 2: Daily Accuracy ────────────────────────────────────────────────────

with tab_accuracy:
    st.header("Daily Accuracy")
    df = _query(daily_accuracy(GCP_PROJECT, BQ_DATASET))

    if df.empty:
        st.info("No batch predictions found. Run `scripts/batch_predict.py` first.")
    else:
        avg_acc = df["accuracy"].mean()

        fig = px.line(
            df, x="prediction_date", y="accuracy",
            markers=True,
            title="Daily Classification Accuracy",
            labels={"accuracy": "Accuracy", "prediction_date": "Date"},
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig.add_hline(
            y=avg_acc, line_dash="dash",
            annotation_text=f"avg {avg_acc:.1%}",
            annotation_position="top right",
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Latest Accuracy", f"{df['accuracy'].iloc[-1]:.1%}")
        c2.metric("Average Accuracy", f"{avg_acc:.1%}")
        c3.metric("Days Tracked", len(df))

# ─── Tab 3: Label Distribution ────────────────────────────────────────────────

with tab_dist:
    st.header("Label Distribution Over Time")
    df = _query(label_distribution(GCP_PROJECT, BQ_DATASET))

    if df.empty:
        st.info("No predictions yet.")
    else:
        fig = px.bar(
            df, x="prediction_date", y="count", color="predicted_label",
            color_discrete_map=LABEL_COLOURS,
            title="Daily Prediction Counts by Category",
            labels={
                "count": "Articles",
                "prediction_date": "Date",
                "predicted_label": "Category",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

# ─── Tab 4: Confusion Matrix ──────────────────────────────────────────────────

with tab_confusion:
    st.header("Confusion Matrix (all time)")
    df = _query(confusion_data(GCP_PROJECT, BQ_DATASET))

    if df.empty:
        st.info("No predictions yet.")
    else:
        pivot = (
            df.pivot_table(
                index="true_label", columns="predicted_label",
                values="count", fill_value=0,
            )
            .reindex(index=LABELS, columns=LABELS, fill_value=0)
        )

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values.tolist(),
            x=LABELS,
            y=LABELS,
            colorscale="Blues",
            text=pivot.values.tolist(),
            texttemplate="%{text}",
            showscale=True,
        ))
        fig.update_layout(
            title="True Label vs Predicted Label",
            xaxis_title="Predicted",
            yaxis_title="True",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
