# src/ingestion/__init__.py

import pandas as pd
import os
import logging

# Set up a logger so we can trace what the pipeline is doing at runtime
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Expected columns and their dtypes for the main application table
# This acts as a lightweight schema contract — if the data doesn't match, we catch it early
EXPECTED_COLUMNS = {
    "SK_ID_CURR": "int64",
    "TARGET": "float64",
    "CODE_GENDER": "object",
    "FLAG_OWN_CAR": "object",
    "FLAG_OWN_REALTY": "object",
    "CNT_CHILDREN": "float64",
    "AMT_INCOME_TOTAL": "float64",
    "AMT_CREDIT": "float64",
    "AMT_ANNUITY": "float64",
    "AMT_GOODS_PRICE": "float64",
    "NAME_INCOME_TYPE": "object",
    "NAME_EDUCATION_TYPE": "object",
    "NAME_FAMILY_STATUS": "object",
    "NAME_HOUSING_TYPE": "object",
    "DAYS_BIRTH": "float64",
    "DAYS_EMPLOYED": "float64",
    "DAYS_REGISTRATION": "float64",
    "DAYS_ID_PUBLISH": "float64",
    "OCCUPATION_TYPE": "object",
    "CNT_FAM_MEMBERS": "float64",
    "REGION_RATING_CLIENT": "float64",
    "EXT_SOURCE_1": "float64",
    "EXT_SOURCE_2": "float64",
    "EXT_SOURCE_3": "float64",
}


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw CSV file from disk and return a DataFrame.
    Raises a clear error if the file doesn't exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")

    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} rows and {df.shape[1]} columns")
    return df


def validate_schema(df: pd.DataFrame, expected: dict = EXPECTED_COLUMNS) -> pd.DataFrame:
    """
    Check that all expected columns are present in the DataFrame.
    Columns not in our expected schema are dropped to prevent noise downstream.
    Missing required columns raise a ValueError immediately.
    """
    logger.info("Validating schema...")

    missing = [col for col in expected.keys() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only the columns we expect — drops irrelevant or noisy extra columns
    df = df[[col for col in expected.keys() if col in df.columns]]

    logger.info(f"Schema valid. Working with {df.shape[1]} columns.")
    return df


def report_data_quality(df: pd.DataFrame) -> None:
    """
    Print a quick summary of missing values and class balance.
    This is not a transformation — it's purely diagnostic.
    """
    logger.info("--- Data Quality Report ---")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
    missing_report = missing_report[missing_report["missing_count"] > 0].sort_values("missing_%", ascending=False)

    if missing_report.empty:
        logger.info("No missing values found.")
    else:
        logger.info(f"\n{missing_report.to_string()}")

    if "TARGET" in df.columns:
        balance = df["TARGET"].value_counts(normalize=True).round(4) * 100
        logger.info(f"\nClass balance (TARGET):\n{balance.to_string()}")


def ingest(data_dir: str = "data/raw", filename: str = "application_train.csv") -> pd.DataFrame:
    """
    Master ingestion function. This is the single entry point for loading data
    into the pipeline. It loads, validates, and reports — nothing more.
    Transformation happens in the preprocessing module.
    """
    filepath = os.path.join(data_dir, filename)
    df = load_raw_data(filepath)
    df = validate_schema(df)
    report_data_quality(df)
    return df