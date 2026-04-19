# src/preprocessing/__init__.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
import os
import joblib

logger = logging.getLogger(__name__)


def handle_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known anomalies specific to Home Credit data.

    DAYS_EMPLOYED has a known encoding issue — value 365243 is used as a
    placeholder for unemployed applicants. We replace it with NaN so it
    doesn't corrupt numerical distributions.

    This is NOT data leakage because it's a known structural issue in the
    raw data, not derived from the target variable.
    """
    logger.info("Handling anomalies...")

    if "DAYS_EMPLOYED" in df.columns:
        anomaly_count = (df["DAYS_EMPLOYED"] == 365243).sum()
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        logger.info(f"Replaced {anomaly_count:,} anomalous DAYS_EMPLOYED values with NaN")

    # DAYS_BIRTH is negative (days before application) — convert to positive age in years
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(1)
        df.drop(columns=["DAYS_BIRTH"], inplace=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones.
    All features here are derived from applicant data only — no target leakage.
    """
    logger.info("Engineering features...")

    # Credit-to-income ratio — higher ratio means higher repayment burden
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)

    # Annuity-to-income ratio — monthly repayment as share of income
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)

    # Credit-to-goods ratio — how much of the goods price is financed
    if "AMT_CREDIT" in df.columns and "AMT_GOODS_PRICE" in df.columns:
        df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)

    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values column by column.
    - Numeric columns: filled with the median (robust to outliers)
    - Categorical columns: filled with the string 'Unknown'

    Medians are computed on the training data only. If this function is
    called on test data, the same medians must be passed in — this is
    enforced in the fit/transform pattern used in the training script.
    """
    logger.info("Imputing missing values...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in numeric_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label encode all categorical (object) columns.
    LabelEncoder maps each unique string to an integer.
    This is sufficient for tree-based models like LightGBM and XGBoost.
    """
    logger.info("Encoding categorical columns...")

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def preprocess(df: pd.DataFrame, save_encoders_path: str = None):
    """
    Master preprocessing function.
    Runs the full pipeline in the correct order:
    anomaly fix → feature engineering → imputation → encoding

    Optionally saves the label encoders to disk so they can be reused
    at inference time without retraining.
    """
    df = handle_anomalies(df)
    df = engineer_features(df)
    df = impute_missing(df)
    df, encoders = encode_categoricals(df)

    if save_encoders_path:
        os.makedirs(os.path.dirname(save_encoders_path), exist_ok=True)
        joblib.dump(encoders, save_encoders_path)
        logger.info(f"Encoders saved to {save_encoders_path}")

    logger.info("Preprocessing complete.")
    return df, encoders