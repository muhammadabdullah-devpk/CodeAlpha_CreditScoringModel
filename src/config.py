from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
TARGET_COLUMN = "creditworthy"
MODEL_FILE = MODELS_DIR / "credit_scoring_pipeline.joblib"
METRICS_FILE = REPORTS_DIR / "metrics.json"
METADATA_FILE = REPORTS_DIR / "metadata.json"
SAMPLE_FILE = DATA_DIR / "sample_applicants.csv"
BATCH_OUTPUT_FILE = REPORTS_DIR / "batch_predictions.csv"

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "monthly_debt",
    "employment_length_years",
    "credit_history_years",
    "loan_amount",
    "loan_term_months",
    "open_accounts",
    "credit_utilization",
    "late_payments_12m",
    "delinquencies",
    "bankruptcies",
    "savings_balance",
    "previous_defaults",
    "debt_to_income_ratio",
    "loan_to_income_ratio",
    "payment_burden_ratio",
    "stability_score",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "education_level",
    "employment_type",
    "loan_purpose",
]

FORM_FEATURE_ORDER = [
    "age",
    "annual_income",
    "monthly_debt",
    "employment_length_years",
    "credit_history_years",
    "loan_amount",
    "loan_term_months",
    "open_accounts",
    "credit_utilization",
    "late_payments_12m",
    "delinquencies",
    "bankruptcies",
    "savings_balance",
    "previous_defaults",
    "home_ownership",
    "education_level",
    "employment_type",
    "loan_purpose",
]
