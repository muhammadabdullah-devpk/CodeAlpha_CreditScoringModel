"""
app.py — Production Flask application for CreditAI Credit Scoring System.
Author: Muhammad Abdullah
"""
from __future__ import annotations

import io
import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)

from src.config import (
    BATCH_OUTPUT_FILE,
    FORM_FEATURE_ORDER,
    METADATA_FILE,
    METRICS_FILE,
    REPORTS_DIR,
    SAMPLE_FILE,
)
from src.predict import predict_batch, predict_one

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "creditai-secret-2024-x9k2m"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── In-memory state ───────────────────────────────────────────────────────────
prediction_history: deque = deque(maxlen=100)
_last_batch_df: pd.DataFrame | None = None

CATEGORICAL_FIELDS = {"home_ownership", "education_level", "employment_type", "loan_purpose"}


# ── Helper loaders ────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Could not load {path}: {exc}")
    return {}


def load_metrics() -> dict:
    return _load_json(METRICS_FILE)


def load_feature_importance() -> dict:
    return _load_json(REPORTS_DIR / "feature_importance.json")


def load_metadata() -> dict:
    return _load_json(METADATA_FILE)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Landing page with project overview and model metrics."""
    return render_template("index.html", metrics=load_metrics(), metadata=load_metadata())


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Single-applicant prediction form and result page."""
    result = None
    form_data = {}
    error = None

    if request.method == "POST":
        try:
            payload: dict = {}
            for field in FORM_FEATURE_ORDER:
                value = request.form.get(field, "")
                form_data[field] = value
                if field in CATEGORICAL_FIELDS:
                    payload[field] = value
                else:
                    payload[field] = float(value)

            result = predict_one(payload)

            # Store in prediction history
            prediction_history.appendleft(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "credit_score": result["credit_score"],
                    "decision": result["decision"],
                    "risk_band": result["risk_band"],
                    "approval_probability": round(
                        float(result["approval_probability"]) * 100, 1
                    ),
                    "confidence": result.get("confidence", "—"),
                }
            )

            logger.info(
                f"Prediction  score={result['credit_score']}  "
                f"decision={result['decision']}  "
                f"prob={result['approval_probability']:.3f}"
            )

        except ValueError as exc:
            error = f"Invalid input value: {exc}"
            logger.warning(f"Validation error: {exc}")
        except Exception as exc:
            error = f"Prediction failed: {exc}"
            logger.error(f"Prediction error: {exc}", exc_info=True)

    return render_template(
        "predict.html", result=result, form_data=form_data, error=error
    )


@app.route("/batch", methods=["GET", "POST"])
def batch():
    """Batch CSV prediction with summary statistics and download options."""
    global _last_batch_df
    table_html = None
    summary = None
    error = None

    if request.method == "POST":
        try:
            file = request.files.get("file")
            if not file or file.filename == "":
                error = "Please upload a CSV file to continue."
            else:
                df_input = pd.read_csv(file)
                results = predict_batch(df_input)
                _last_batch_df = results

                total = len(results)
                approved = int((results["decision"] == "Approve").sum())
                rejected = total - approved
                avg_score = int(results["credit_score"].mean())
                avg_prob = round(float(results["approval_probability"].mean()) * 100, 1)
                risk_counts = results["risk_band"].value_counts().to_dict()

                summary = {
                    "total": total,
                    "approved": approved,
                    "rejected": rejected,
                    "approval_rate": round(approved / total * 100, 1) if total else 0,
                    "rejection_rate": round(rejected / total * 100, 1) if total else 0,
                    "avg_score": avg_score,
                    "avg_probability": avg_prob,
                    "risk_counts": risk_counts,
                }

                REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                results.to_csv(BATCH_OUTPUT_FILE, index=False)

                table_html = (
                    results[
                        ["credit_score", "approval_probability", "decision", "risk_band"]
                    ]
                    .head(100)
                    .to_html(
                        classes="table table-hover align-middle result-table",
                        index=False,
                        border=0,
                    )
                )

                logger.info(
                    f"Batch prediction done — {total} rows, "
                    f"{approved} approved ({summary['approval_rate']}%)"
                )

        except Exception as exc:
            error = str(exc)
            logger.error(f"Batch prediction error: {exc}", exc_info=True)

    return render_template(
        "batch.html", table_html=table_html, error=error, summary=summary
    )


@app.route("/dashboard")
def dashboard():
    """Analytics dashboard with model metrics, charts, and prediction history."""
    metrics = load_metrics()
    fi = load_feature_importance()
    metadata = load_metadata()
    history = list(prediction_history)

    total_preds = len(history)
    approved_count = sum(1 for h in history if h["decision"] == "Approve")
    avg_score = (
        round(sum(h["credit_score"] for h in history) / total_preds, 1)
        if history
        else 0
    )
    approval_rate = round(approved_count / total_preds * 100, 1) if total_preds else 0

    stats = {
        "total_predictions": total_preds,
        "approved": approved_count,
        "rejected": total_preds - approved_count,
        "avg_score": avg_score,
        "approval_rate": approval_rate,
        "rejection_rate": round(100 - approval_rate, 1),
    }

    return render_template(
        "dashboard.html",
        metrics=metrics,
        metadata=metadata,
        stats=stats,
        history=history,
        feature_importance=fi,
    )


@app.route("/developer")
def developer():
    """Developer profile page."""
    return render_template("developer.html")


@app.route("/api-docs")
def api_docs():
    """API Reference Documentation page."""
    return render_template("api.html")


@app.route("/docs")
def docs():
    """Methodology Documentation page."""
    return render_template("docs.html")


# ── File downloads ────────────────────────────────────────────────────────────

@app.route("/download-sample")
def download_sample():
    return send_file(SAMPLE_FILE, as_attachment=True, download_name="sample_applicants.csv")


@app.route("/download-batch-csv")
def download_batch_csv():
    if BATCH_OUTPUT_FILE.exists():
        return send_file(
            BATCH_OUTPUT_FILE,
            as_attachment=True,
            download_name="batch_predictions.csv",
        )
    return "No batch results available yet. Run a batch prediction first.", 404


@app.route("/download-batch-excel")
def download_batch_excel():
    global _last_batch_df
    if _last_batch_df is None:
        return "No batch results available yet. Run a batch prediction first.", 404
    buf = io.BytesIO()
    _last_batch_df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name="batch_predictions.xlsx",
        mimetype=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
    )


# ── JSON APIs ─────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        result = predict_one(data)
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/metrics")
def api_metrics():
    return jsonify(load_metrics())


@app.route("/api/feature-importance")
def api_feature_importance():
    return jsonify(load_feature_importance())


@app.route("/api/history")
def api_history():
    return jsonify(list(prediction_history))


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
