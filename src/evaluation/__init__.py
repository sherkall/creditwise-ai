# src/evaluation/__init__.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
import logging

logger = logging.getLogger(__name__)


def get_predictions(model, X_val, model_type: str = "sklearn"):
    """
    Get predicted probabilities for the positive class (default=1).
    LightGBM and sklearn models have different prediction interfaces.
    We always want probabilities, not hard class labels, so we can
    compute threshold-independent metrics like ROC-AUC.
    """
    if model_type == "lgbm":
        y_prob = model.predict(X_val)
    else:
        y_prob = model.predict_proba(X_val)[:, 1]
    return y_prob


def evaluate(model, X_val, y_val, model_name: str = "model", model_type: str = "sklearn") -> dict:
    """
    Full evaluation suite for a binary classification model.

    Metrics chosen specifically for credit risk:
    - ROC-AUC: measures ranking ability across all thresholds.
      Industry standard for credit scoring.
    - Average Precision (PR-AUC): better than ROC-AUC for imbalanced
      datasets. Focuses on the minority class (defaulters).
    - F1 Score: balance between precision and recall at 0.5 threshold.
    - Recall: how many actual defaulters we catch. Missing a defaulter
      is more costly than a false alarm in credit risk.
    - Precision: of predicted defaulters, how many actually default.
    - Confusion Matrix: raw counts of TP, FP, TN, FN.
    """
    logger.info(f"Evaluating {model_name}...")

    y_prob = get_predictions(model, X_val, model_type)
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }

    logger.info(f"\n{'='*40}")
    logger.info(f"  {model_name} Results")
    logger.info(f"{'='*40}")
    for k, v in metrics.items():
        logger.info(f"  {k:<20} {v}")
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"\nClassification Report:\n{classification_report(y_val, y_pred)}")

    return metrics