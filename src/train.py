"""
train.py — Training entry-point.
Run with:  python -m src.train
"""
from __future__ import annotations

import json
import logging

from src.config import METRICS_FILE
from src.data_utils import ensure_directories, generate_synthetic_credit_data, save_sample_dataset
from src.modeling import CreditScoringEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ensure_directories()

    logger.info("Generating synthetic credit dataset (8,000 samples) …")
    df = generate_synthetic_credit_data(n_samples=8000)
    save_sample_dataset(df, rows=50)
    logger.info(f"Dataset ready — {len(df):,} rows, {df.columns.tolist()}")

    logger.info("Training RandomForest credit scoring pipeline …")
    engine = CreditScoringEngine()
    metrics = engine.fit(df)
    engine.save()

    print("\n[SUCCESS] Training completed successfully!\n")
    print(json.dumps(metrics, indent=2))
    print(f"\nMetrics  -> {METRICS_FILE}")


if __name__ == "__main__":
    main()
