# src/inference/__init__.py

import pandas as pd
import numpy as np
import joblib
import logging

logger = logging.getLogger(__name__)

# These must match exactly what was used during preprocessing
EXPECTED_FEATURES = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT",
    "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "OCCUPATION_TYPE", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "AGE_YEARS", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO"
]


def load_artifacts(model_path: str, encoders_path: str):
    """Load model and encoders from disk."""
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Loaded encoders from {encoders_path}")
    return model, encoders


def prepare_input(data: dict, encoders: dict) -> pd.DataFrame:
    """
    Convert raw input dict into a model-ready DataFrame.
    Applies the same anomaly fixes, feature engineering,
    imputation and encoding used during training.
    Order of columns must match training exactly.
    """
    df = pd.DataFrame([data])

    # Anomaly fix
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(1)
        df.drop(columns=["DAYS_BIRTH"], inplace=True)

    # Feature engineering
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)

    # Impute missing numerics with 0, categoricals with Unknown
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    # Encode categoricals using saved encoders
    for col in categorical_cols:
        if col in encoders:
            le = encoders[col]
            # Handle unseen labels gracefully
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
        else:
            df[col] = 0

    # Ensure column order matches training
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df = df[EXPECTED_FEATURES]
    return df


def predict(model, encoders: dict, data: dict, model_type: str = "lgbm") -> dict:
    """
    Run a single prediction and return probability + risk label.
    Threshold of 0.5 — above this the applicant is flagged as high risk.
    """
    input_df = prepare_input(data, encoders)

    if model_type == "lgbm":
        prob = float(model.predict(input_df)[0])
    else:
        prob = float(model.predict_proba(input_df)[:, 1][0])

    label = "HIGH RISK" if prob >= 0.5 else "LOW RISK"

    return {
        "default_probability": round(prob, 4),
        "risk_label": label,
        "model_type": model_type
    }