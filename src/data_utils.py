from pathlib import Path
import numpy as np
import pandas as pd

from src.config import DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, REPORTS_DIR, SAMPLE_FILE, TARGET_COLUMN


def ensure_directories() -> None:
    for folder in [DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, REPORTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["debt_to_income_ratio"] = (out["monthly_debt"] * 12) / np.maximum(out["annual_income"], 1)
    out["loan_to_income_ratio"] = out["loan_amount"] / np.maximum(out["annual_income"], 1)
    out["payment_burden_ratio"] = (out["loan_amount"] / np.maximum(out["loan_term_months"], 1)) / np.maximum(
        out["annual_income"] / 12, 1
    )
    out["stability_score"] = (
        0.35 * np.clip(out["employment_length_years"] / 10, 0, 1)
        + 0.35 * np.clip(out["credit_history_years"] / 15, 0, 1)
        + 0.30 * np.where(out["employment_type"].eq("full_time"), 1, np.where(out["employment_type"].eq("self_employed"), 0.7, 0.4))
    )
    return out


def generate_synthetic_credit_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 66, n_samples)
    annual_income = np.clip(rng.lognormal(mean=10.7, sigma=0.45, size=n_samples), 18000, 220000)
    monthly_debt = np.clip(annual_income / 12 * rng.uniform(0.05, 0.55, n_samples), 150, 8500)
    employment_length_years = np.clip(rng.normal(5.5, 3.2, n_samples), 0, 25)
    credit_history_years = np.clip(age - rng.integers(18, 30, n_samples), 0.5, 35)
    loan_amount = np.clip(rng.normal(18000, 9000, n_samples), 1000, 60000)
    loan_term_months = rng.choice([12, 24, 36, 48, 60, 72], size=n_samples, p=[0.08, 0.12, 0.28, 0.22, 0.2, 0.1])
    open_accounts = rng.integers(1, 13, n_samples)
    credit_utilization = np.clip(rng.beta(2.1, 2.6, n_samples), 0.02, 0.98)
    late_payments_12m = rng.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.56, 0.19, 0.11, 0.08, 0.04, 0.02])
    delinquencies = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.74, 0.16, 0.07, 0.03])
    bankruptcies = rng.choice([0, 1, 2], size=n_samples, p=[0.96, 0.035, 0.005])
    savings_balance = np.clip(annual_income * rng.uniform(0.03, 0.6, n_samples), 300, 150000)
    previous_defaults = rng.choice([0, 1, 2], size=n_samples, p=[0.83, 0.14, 0.03])

    home_ownership = rng.choice(["rent", "mortgage", "own"], size=n_samples, p=[0.42, 0.38, 0.2])
    education_level = rng.choice(["high_school", "bachelor", "master", "other"], size=n_samples, p=[0.32, 0.38, 0.18, 0.12])
    employment_type = rng.choice(["full_time", "part_time", "self_employed", "contract"], size=n_samples, p=[0.58, 0.14, 0.18, 0.10])
    loan_purpose = rng.choice(["personal", "education", "business", "car", "home_improvement"], size=n_samples, p=[0.30, 0.12, 0.18, 0.20, 0.20])

    df = pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income.round(2),
            "monthly_debt": monthly_debt.round(2),
            "employment_length_years": employment_length_years.round(1),
            "credit_history_years": credit_history_years.round(1),
            "loan_amount": loan_amount.round(2),
            "loan_term_months": loan_term_months,
            "open_accounts": open_accounts,
            "credit_utilization": credit_utilization.round(3),
            "late_payments_12m": late_payments_12m,
            "delinquencies": delinquencies,
            "bankruptcies": bankruptcies,
            "savings_balance": savings_balance.round(2),
            "previous_defaults": previous_defaults,
            "home_ownership": home_ownership,
            "education_level": education_level,
            "employment_type": employment_type,
            "loan_purpose": loan_purpose,
        }
    )

    df = add_engineered_features(df)

    purpose_risk = df["loan_purpose"].map(
        {"business": -0.18, "personal": -0.1, "education": 0.06, "car": 0.02, "home_improvement": 0.04}
    )
    housing_bonus = df["home_ownership"].map({"own": 0.22, "mortgage": 0.08, "rent": -0.12})
    education_bonus = df["education_level"].map({"master": 0.12, "bachelor": 0.07, "high_school": -0.05, "other": -0.02})
    employment_bonus = df["employment_type"].map(
        {"full_time": 0.18, "self_employed": 0.04, "contract": -0.05, "part_time": -0.12}
    )

    score = (
        2.45
        + 0.000020 * df["annual_income"]
        + 0.000010 * df["savings_balance"]
        + 0.09 * df["employment_length_years"]
        + 0.07 * df["credit_history_years"]
        - 2.8 * df["debt_to_income_ratio"]
        - 1.8 * df["loan_to_income_ratio"]
        - 1.9 * df["payment_burden_ratio"]
        - 1.9 * df["credit_utilization"]
        - 0.45 * df["late_payments_12m"]
        - 0.62 * df["delinquencies"]
        - 1.1 * df["bankruptcies"]
        - 0.72 * df["previous_defaults"]
        + 1.15 * df["stability_score"]
        + purpose_risk
        + housing_bonus
        + education_bonus
        + employment_bonus
        + rng.normal(0, 0.45, len(df))
    )

    probability = sigmoid(score - 1.8)
    df[TARGET_COLUMN] = (rng.random(len(df)) < probability).astype(int)

    return df


def save_sample_dataset(df: pd.DataFrame, rows: int = 25) -> None:
    SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[TARGET_COLUMN], errors="ignore").head(rows).to_csv(SAMPLE_FILE, index=False)
