"""
modeling.py — CreditScoringEngine: build, train, evaluate, save, load.

matplotlib is intentionally NOT imported here.
All visualisations are handled client-side via Chart.js.
Feature importances are exported as JSON for the dashboard.
"""
from __future__ import annotations

import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    METADATA_FILE,
    METRICS_FILE,
    MODEL_FILE,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.data_utils import add_engineered_features

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class CreditScoringEngine:
    """End-to-end credit scoring pipeline: preprocessing + RandomForest."""

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.threshold: float = 0.5
        self.feature_names: List[str] = []

    # ── Pipeline construction ─────────────────────────────────────────────────

    def build_pipeline(self) -> Pipeline:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_FEATURES),
                ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ]
        )

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        return Pipeline(steps=[("preprocessor", preprocessor), ("model", rf)])

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> Dict[str, float]:
        df = add_engineered_features(df)
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        self.feature_names = X.columns.tolist()

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
        )

        logger.info(
            f"Splits — train:{len(X_train)}, valid:{len(X_valid)}, test:{len(X_test)}"
        )

        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X_train, y_train)

        valid_proba = self.pipeline.predict_proba(X_valid)[:, 1]
        test_proba = self.pipeline.predict_proba(X_test)[:, 1]
        self.threshold = self._find_best_threshold(y_valid, valid_proba)
        test_pred = (test_proba >= self.threshold).astype(int)

        cm = confusion_matrix(y_test, test_pred)

        metrics: Dict = {
            "accuracy": round(float(accuracy_score(y_test, test_pred)), 4),
            "precision": round(float(precision_score(y_test, test_pred)), 4),
            "recall": round(float(recall_score(y_test, test_pred)), 4),
            "f1_score": round(float(f1_score(y_test, test_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, test_proba)), 4),
            "optimal_threshold": round(float(self.threshold), 4),
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_valid)),
            "test_rows": int(len(X_test)),
            "confusion_matrix": cm.tolist(),
        }

        logger.info(
            f"Metrics — accuracy:{metrics['accuracy']}  "
            f"f1:{metrics['f1_score']}  auc:{metrics['roc_auc']}"
        )

        self._save_artifacts(metrics)
        return metrics

    # ── Threshold optimisation ────────────────────────────────────────────────

    def _find_best_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        thresholds = np.linspace(0.35, 0.75, 81)
        best_threshold, best_score = 0.5, -1.0
        for t in thresholds:
            preds = (y_proba >= t).astype(int)
            blended = (
                0.55 * f1_score(y_true, preds)
                + 0.25 * recall_score(y_true, preds)
                + 0.20 * precision_score(y_true, preds)
            )
            if blended > best_score:
                best_threshold, best_score = t, blended
        logger.info(f"Best threshold: {best_threshold:.3f}")
        return float(best_threshold)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load() or fit() first.")

        df_feat = add_engineered_features(df)
        proba = self.pipeline.predict_proba(df_feat)[:, 1]
        decisions = (proba >= self.threshold).astype(int)
        scores = np.clip((300 + proba * 550).round().astype(int), 300, 850)

        out = df_feat.copy()
        out["approval_probability"] = np.round(proba, 4)
        out["decision"] = np.where(decisions == 1, "Approve", "Review / Decline")
        out["credit_score"] = scores
        out["risk_band"] = out["approval_probability"].apply(map_risk_band)
        return out

    # ── Feature importance ────────────────────────────────────────────────────

    def get_feature_importance(self) -> Dict[str, float]:
        """Aggregate one-hot encoded importances back to original feature names."""
        if self.pipeline is None:
            return {}
        try:
            rf = self.pipeline.named_steps["model"]
            pre = self.pipeline.named_steps["preprocessor"]
            ohe: OneHotEncoder = pre.named_transformers_["cat"].named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
            all_names = NUMERIC_FEATURES + cat_names
            importance_map: Dict[str, float] = {}
            for feat, imp in zip(all_names, rf.feature_importances_):
                origin = feat
                for col in CATEGORICAL_FEATURES:
                    if feat.startswith(col + "_"):
                        origin = col
                        break
                importance_map[origin] = importance_map.get(origin, 0.0) + float(imp)
            return dict(sorted(importance_map.items(), key=lambda x: x[1], reverse=True))
        except Exception as exc:
            logger.warning(f"Feature importance extraction failed: {exc}")
            return {}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Cannot save — pipeline not fitted.")
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "threshold": self.threshold,
            "feature_names": self.feature_names,
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "trained_at": datetime.now().isoformat(),
        }
        joblib.dump(payload, MODEL_FILE)
        logger.info(
            f"Model saved -> {MODEL_FILE}  (sklearn {sklearn.__version__})"
        )

    def load(self) -> None:
        payload = joblib.load(MODEL_FILE)
        self.pipeline = payload["pipeline"]
        self.threshold = payload["threshold"]
        self.feature_names = payload.get("feature_names", [])
        trained_ver = payload.get("sklearn_version", "unknown")
        logger.info(
            f"Model loaded. Trained with sklearn {trained_ver}, "
            f"current sklearn {sklearn.__version__}"
        )

    # ── Artifact persistence ──────────────────────────────────────────────────

    def _save_artifacts(self, metrics: Dict) -> None:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Metrics JSON
        with open(METRICS_FILE, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

        # Feature importance JSON (used by dashboard Chart.js)
        fi = self.get_feature_importance()
        fi_path = METRICS_FILE.parent / "feature_importance.json"
        with open(fi_path, "w", encoding="utf-8") as fh:
            json.dump(fi, fh, indent=2)

        # Metadata JSON
        metadata = {
            "input_columns": self.feature_names,
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "trained_at": datetime.now().isoformat(),
            "business_goal": "Predict customer creditworthiness from financial features.",
            "positive_class": "Creditworthy / Approve",
            "score_range": "300–850",
            "threshold": self.threshold,
        }
        with open(METADATA_FILE, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        logger.info("Artifacts saved: metrics, feature_importance, metadata.")


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def map_risk_band(probability: float) -> str:
    if probability >= 0.80:
        return "Low Risk"
    if probability >= 0.62:
        return "Moderate Risk"
    if probability >= 0.45:
        return "Elevated Risk"
    return "High Risk"


def generate_reasons(row: pd.Series) -> Tuple[List[str], List[str]]:
    """Generate human-readable strengths and risk flags from a result row."""
    strengths: List[str] = []
    risks: List[str] = []

    # ── Positive factors
    if row.get("annual_income", 0) >= 70000:
        strengths.append("Strong annual income supports repayment capacity.")
    if row.get("savings_balance", 0) >= 15000:
        strengths.append("Healthy savings balance improves financial resilience.")
    if row.get("credit_history_years", 0) >= 6:
        strengths.append("Long credit history indicates stable borrowing behaviour.")
    if row.get("employment_length_years", 0) >= 3:
        strengths.append("Consistent employment tenure is a positive signal.")
    if row.get("credit_utilization", 0) <= 0.35:
        strengths.append("Low credit utilisation reflects disciplined card usage.")
    if row.get("late_payments_12m", 0) == 0:
        strengths.append("No late payments in the past 12 months.")
    if row.get("delinquencies", 0) == 0:
        strengths.append("Clean delinquency record.")

    # ── Negative factors
    if row.get("credit_utilization", 0) > 0.65:
        risks.append("High credit utilisation may signal cash-flow pressure.")
    if row.get("late_payments_12m", 0) >= 2:
        risks.append("Recent late payments reduce confidence in repayment behaviour.")
    if row.get("delinquencies", 0) >= 1:
        risks.append("Past delinquencies increase credit risk rating.")
    if row.get("previous_defaults", 0) >= 1:
        risks.append("Previous defaults materially hurt creditworthiness.")
    if row.get("bankruptcies", 0) >= 1:
        risks.append("Bankruptcy history is a major negative indicator.")
    if row.get("debt_to_income_ratio", 0) > 0.40:
        risks.append("Debt-to-income ratio exceeds the recommended 40% threshold.")
    if row.get("loan_to_income_ratio", 0) > 0.35:
        risks.append("Requested loan amount is high relative to annual income.")
    if row.get("annual_income", 0) < 30000:
        risks.append("Lower income bracket limits repayment buffer.")

    if not strengths:
        strengths.append("Profile meets acceptable baseline financial criteria.")
    if not risks:
        risks.append("No major red flags detected from core financial features.")

    return strengths[:5], risks[:5]
