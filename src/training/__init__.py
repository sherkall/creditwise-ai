# src/training/__init__.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import joblib
import os
import logging

logger = logging.getLogger(__name__)


def split_data(df: pd.DataFrame, target_col: str = "TARGET", test_size: float = 0.2, random_state: int = 42):
    """
    Split into train and validation sets.
    Stratified split ensures class balance is preserved in both sets —
    critical here because only 8% of samples are defaulters.
    SK_ID_CURR is an applicant ID — we drop it to prevent the model
    from memorizing IDs instead of learning patterns (data leakage).
    """
    logger.info("Splitting data...")

    if "SK_ID_CURR" in df.columns:
        df = df.drop(columns=["SK_ID_CURR"])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    logger.info(f"Train size: {len(X_train):,} | Val size: {len(X_val):,}")
    logger.info(f"Train default rate: {y_train.mean():.4f} | Val default rate: {y_val.mean():.4f}")

    return X_train, X_val, y_train, y_val


def train_logistic_regression(X_train, y_train, random_state: int = 42):
    """
    Baseline model — Logistic Regression.
    Simple, interpretable, fast to train.
    class_weight='balanced' adjusts for the 92/8 imbalance automatically
    by upweighting the minority class during training.
    max_iter=1000 ensures convergence on this dataset size.
    """
    logger.info("Training Logistic Regression baseline...")

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete.")
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, random_state: int = 42):
    """
    Main model — LightGBM gradient boosted trees.
    Better than logistic regression for tabular data with mixed feature types.
    scale_pos_weight handles class imbalance by telling the model how much
    more to penalize missing a defaulter vs a non-defaulter.
    early_stopping_rounds prevents overfitting — training stops when
    validation AUC stops improving.
    """
    logger.info("Training LightGBM model...")

    # Compute scale_pos_weight = ratio of negatives to positives
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    logger.info(f"scale_pos_weight set to {scale_pos_weight:.2f}")

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
        "verbose": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks
    )

    logger.info("LightGBM training complete.")
    return model, params


def save_model(model, path: str):
    """Save model artifact to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")