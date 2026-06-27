import pandas as pd

from src.data_utils import generate_synthetic_credit_data
from src.modeling import CreditScoringEngine


def test_dataset_has_target():
    df = generate_synthetic_credit_data(n_samples=200)
    assert "creditworthy" in df.columns
    assert df["creditworthy"].isin([0, 1]).all()


def test_model_predicts_valid_score():
    df = generate_synthetic_credit_data(n_samples=350)
    engine = CreditScoringEngine()
    engine.fit(df)
    sample = df.drop(columns=["creditworthy"]).head(5)
    result = engine.predict(sample)
    assert result["credit_score"].between(300, 850).all()
    assert result["approval_probability"].between(0, 1).all()
