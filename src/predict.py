"""
predict.py — Singleton-based prediction engine with rich output.
Model is loaded once on first call and cached in memory for performance.
"""
from __future__ import annotations

import logging
import threading

import pandas as pd

from src.modeling import CreditScoringEngine, generate_reasons

logger = logging.getLogger(__name__)

# ── Singleton engine ──────────────────────────────────────────────────────────
_engine_instance: CreditScoringEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> CreditScoringEngine:
    """Return the cached engine, loading from disk on first call (thread-safe)."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                logger.info("Loading credit scoring model from disk …")
                engine = CreditScoringEngine()
                engine.load()
                _engine_instance = engine
                logger.info("Model loaded and cached successfully.")
    return _engine_instance


def reload_engine() -> None:
    """Force-reload the model from disk (call after retraining)."""
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
    get_engine()


# ── Public API ────────────────────────────────────────────────────────────────

def predict_one(record: dict) -> dict:
    """
    Run a single-applicant prediction.

    Returns a dict with all input fields, engineered features,
    model outputs (credit_score, approval_probability, decision, risk_band),
    explainable factors (strengths, risks), and confidence label.
    """
    engine = get_engine()
    df = pd.DataFrame([record])
    result_row = engine.predict(df).iloc[0]

    # Convert numpy scalars to native Python types for JSON serialisation
    result: dict = {}
    for k, v in result_row.items():
        result[k] = v.item() if hasattr(v, "item") else v

    strengths, risks = generate_reasons(result_row)
    result["strengths"] = strengths
    result["risks"] = risks

    # Confidence band label
    prob = float(result["approval_probability"])
    if prob >= 0.85:
        result["confidence"] = "Very High"
    elif prob >= 0.70:
        result["confidence"] = "High"
    elif prob >= 0.55:
        result["confidence"] = "Moderate"
    else:
        result["confidence"] = "Low"

    # Next-steps recommendation
    result["recommendation"] = _build_recommendation(result)

    return result


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Run batch prediction on a DataFrame of applicants."""
    engine = get_engine()
    return engine.predict(df)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_recommendation(result: dict) -> str:
    decision = result.get("decision", "")
    score = result.get("credit_score", 0)
    prob = float(result.get("approval_probability", 0))

    if decision == "Approve":
        if score >= 750:
            return "Excellent profile. Eligible for premium rates and higher loan limits."
        elif score >= 680:
            return "Strong profile. Approve with standard terms and conditions."
        else:
            return "Approve with standard terms. Monitor repayment behaviour closely."
    else:
        if prob >= 0.40:
            return "Near threshold. Consider requesting additional documents or a co-signer."
        else:
            return "High risk profile. Decline and advise applicant to reduce debt and improve credit history."
