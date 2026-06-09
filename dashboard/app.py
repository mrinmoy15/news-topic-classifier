"""
BBC News Topic Classifier — Monitoring Dashboard

Tabs
----
1. Live Inference      — paste text, get real-time prediction from the API
2. Daily Accuracy      — line chart of daily accuracy over time
3. Label Distribution  — stacked bar chart of predictions per day
4. Confusion Matrix    — heatmap of true vs predicted labels
5. Per-Class Metrics   — precision / recall / F1 / support per category
6. LLM Evaluation      — Gemini-as-judge on a random sample; compare with BERT

Environment variables
---------------------
API_URL       Base URL of the FastAPI serving container (default: localhost:8080)
GCP_PROJECT   GCP project ID for BigQuery and Vertex AI
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
    llm_eval_sample,
    per_class_metrics,
    performance_trend,
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


@st.cache_data(ttl=3300)
def _api_id_token(audience: str) -> str:
    """Fetch an OIDC ID token for the Cloud Run API audience.

    Cached for 55 minutes — tokens expire after 60 minutes so this ensures
    a fresh token is always available without hitting the token endpoint on
    every request.  Works on Cloud Run (metadata server) and locally via
    gcloud ADC or a service account key.
    """
    import google.auth.transport.requests
    import google.oauth2.id_token

    auth_req = google.auth.transport.requests.Request()
    return google.oauth2.id_token.fetch_id_token(auth_req, audience)


def _auth_headers() -> dict:
    """Return Authorization header when calling an authenticated Cloud Run endpoint."""
    if ".run.app" in API_URL:
        return {"Authorization": f"Bearer {_api_id_token(API_URL)}"}
    return {}


def _call_api(text: str) -> dict | None:
    try:
        resp = httpx.post(
            f"{API_URL}/predict",
            json={"instances": [{"text": text}]},
            headers=_auth_headers(),
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


def _gemini_classify(text: str) -> str:
    """Classify a single article with Gemini 1.5 Flash (LLM-as-judge)."""
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=GCP_PROJECT, location="us-central1")
        model = GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Classify the BBC News article below into exactly ONE category.\n"
            "Categories: business, entertainment, politics, sport, tech\n\n"
            f"Article:\n{text[:800]}\n\n"
            "Reply with ONLY the category name, nothing else."
        )
        response = model.generate_content(prompt)
        label = response.text.strip().lower().split()[0]
        return label if label in set(LABELS) else "unknown"
    except Exception as e:
        return f"error: {e}"


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

tab_live, tab_accuracy, tab_dist, tab_confusion, tab_metrics, tab_llm = st.tabs([
    "🔴 Live Inference",
    "📈 Daily Accuracy",
    "📊 Label Distribution",
    "🔀 Confusion Matrix",
    "📐 Per-Class Metrics",
    "🤖 LLM Evaluation",
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
        st.info("No batch predictions found. Run the inference pipeline first.")
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

# ─── Tab 5: Per-Class Metrics ─────────────────────────────────────────────────

with tab_metrics:
    st.header("Per-Class Metrics")
    st.caption("Precision, recall, and F1 computed from all batch predictions in BigQuery.")

    df = _query(per_class_metrics(GCP_PROJECT, BQ_DATASET))

    if df.empty:
        st.info("No predictions yet.")
    else:
        # Summary table
        display_df = df[["label", "support", "precision", "recall", "f1"]].copy()
        display_df.columns = ["Category", "Support", "Precision", "Recall", "F1"]

        # Append macro averages row
        macro = pd.DataFrame([{
            "Category": "**macro avg**",
            "Support":   int(display_df["Support"].sum()),
            "Precision": round(display_df["Precision"].mean(), 4),
            "Recall":    round(display_df["Recall"].mean(), 4),
            "F1":        round(display_df["F1"].mean(), 4),
        }])
        display_df = pd.concat([display_df, macro], ignore_index=True)

        st.dataframe(
            display_df.style.format({
                "Precision": "{:.1%}",
                "Recall":    "{:.1%}",
                "F1":        "{:.1%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Bar chart — precision / recall / F1 per class
        chart_df = df[["label", "precision", "recall", "f1"]].melt(
            id_vars="label",
            value_vars=["precision", "recall", "f1"],
            var_name="metric",
            value_name="score",
        )
        fig = px.bar(
            chart_df,
            x="label", y="score", color="metric",
            barmode="group",
            title="Precision / Recall / F1 per Category",
            labels={"label": "Category", "score": "Score", "metric": "Metric"},
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        # Confidence + accuracy drift
        trend_df = _query(performance_trend(GCP_PROJECT, BQ_DATASET))
        if not trend_df.empty:
            st.subheader("Model Drift — Accuracy & Confidence Over Time")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=trend_df["prediction_date"], y=trend_df["accuracy"],
                mode="lines+markers", name="Accuracy",
                line=dict(color="#636EFA"),
            ))
            fig2.add_trace(go.Scatter(
                x=trend_df["prediction_date"], y=trend_df["avg_confidence"],
                mode="lines+markers", name="Avg Confidence",
                line=dict(color="#FFA15A", dash="dot"),
            ))
            fig2.update_yaxes(range=[0, 1], tickformat=".0%")
            fig2.update_layout(
                title="Daily Accuracy vs Average Confidence",
                xaxis_title="Date",
                yaxis_title="Score",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig2, use_container_width=True)

# ─── Tab 6: LLM Evaluation ───────────────────────────────────────────────────

with tab_llm:
    st.header("LLM Evaluation (Gemini-as-Judge)")
    st.caption(
        "Samples recent predictions, asks Gemini 1.5 Flash to independently classify "
        "each article, then compares BERT vs Gemini accuracy and agreement rate. "
        "This acts as a lightweight automated quality check for model drift."
    )

    col_n, col_btn = st.columns([2, 1])
    with col_n:
        sample_n = st.slider("Sample size", min_value=5, max_value=30, value=15, step=5)
    with col_btn:
        st.write("")
        run_eval = st.button("▶ Run Evaluation", type="primary")

    if run_eval:
        sample_df = _query(llm_eval_sample(GCP_PROJECT, BQ_DATASET, n=sample_n))

        if sample_df.empty:
            st.info("No predictions available to sample from.")
        else:
            progress = st.progress(0, text="Calling Gemini...")
            gemini_labels = []

            for i, row in sample_df.iterrows():
                gemini_label = _gemini_classify(row["body"])
                gemini_labels.append(gemini_label)
                progress.progress((len(gemini_labels)) / len(sample_df), text=f"Classified {len(gemini_labels)}/{len(sample_df)} articles...")

            progress.empty()

            sample_df = sample_df.copy()
            sample_df["gemini_label"] = gemini_labels
            sample_df["bert_correct"]   = sample_df["predicted_label"] == sample_df["true_label"]
            sample_df["gemini_correct"] = sample_df["gemini_label"]    == sample_df["true_label"]
            sample_df["agreement"]      = sample_df["predicted_label"] == sample_df["gemini_label"]

            # Summary metrics
            bert_acc    = sample_df["bert_correct"].mean()
            gemini_acc  = sample_df["gemini_correct"].mean()
            agree_rate  = sample_df["agreement"].mean()

            m1, m2, m3 = st.columns(3)
            m1.metric("BERT Accuracy",      f"{bert_acc:.1%}",   help="BERT predictions vs true labels")
            m2.metric("Gemini Accuracy",    f"{gemini_acc:.1%}", help="Gemini predictions vs true labels")
            m3.metric("BERT-Gemini Agreement", f"{agree_rate:.1%}", help="Fraction where BERT and Gemini agree")

            st.divider()

            # Detailed comparison table
            show_df = sample_df[["title", "true_label", "predicted_label", "gemini_label",
                                  "confidence", "bert_correct", "gemini_correct"]].copy()
            show_df.columns = ["Title", "True", "BERT", "Gemini", "Conf", "BERT ✓", "Gemini ✓"]
            show_df["Title"] = show_df["Title"].str[:80]
            show_df["Conf"]  = show_df["Conf"].apply(lambda x: f"{x:.1%}")

            st.dataframe(
                show_df.style.apply(
                    lambda col: ["background-color: #d4edda" if v else "background-color: #f8d7da" for v in col]
                    if col.name in ("BERT ✓", "Gemini ✓") else [""] * len(col),
                    axis=0,
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Disagreement breakdown
            disagreements = sample_df[~sample_df["agreement"]]
            if not disagreements.empty:
                st.subheader(f"Disagreements ({len(disagreements)} articles)")
                st.caption("Cases where BERT and Gemini predicted different categories.")
                dis_df = disagreements[["title", "true_label", "predicted_label", "gemini_label", "confidence"]].copy()
                dis_df.columns = ["Title", "True", "BERT", "Gemini", "Conf"]
                dis_df["Title"] = dis_df["Title"].str[:80]
                dis_df["Conf"]  = dis_df["Conf"].apply(lambda x: f"{x:.1%}")
                st.dataframe(dis_df, use_container_width=True, hide_index=True)
