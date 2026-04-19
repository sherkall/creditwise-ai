import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import logging
import joblib

from src.ingestion import ingest
from src.preprocessing import preprocess
from src.training import split_data, train_logistic_regression, train_lightgbm, save_model
from src.evaluation import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Set the MLflow experiment name
mlflow.set_experiment("creditwise_credit_risk")

# ── Load and preprocess data ──────────────────────────────────────────────────
df = ingest()
df, encoders = preprocess(df, save_encoders_path="models/encoders.joblib")
X_train, X_val, y_train, y_val = split_data(df)

# ── Experiment 1: Logistic Regression Baseline ────────────────────────────────
with mlflow.start_run(run_name="logistic_regression_baseline"):
    logger.info("Starting MLflow run: Logistic Regression")

    params = {
        "model_type": "LogisticRegression",
        "class_weight": "balanced",
        "max_iter": 1000,
        "solver": "lbfgs",
        "test_size": 0.2,
        "random_state": 42,
    }
    mlflow.log_params(params)

    lr_model = train_logistic_regression(X_train, y_train)
    metrics = evaluate(lr_model, X_val, y_val, model_name="Logistic Regression", model_type="sklearn")

    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(lr_model, artifact_path="model")
    save_model(lr_model, "models/logistic_regression.joblib")

    logger.info("Logistic Regression run complete.")

# ── Experiment 2: LightGBM ────────────────────────────────────────────────────
with mlflow.start_run(run_name="lightgbm_main"):
    logger.info("Starting MLflow run: LightGBM")

    lgbm_model, lgbm_params = train_lightgbm(X_train, y_train, X_val, y_val)

    mlflow.log_params({**lgbm_params, "test_size": 0.2})

    metrics = evaluate(lgbm_model, X_val, y_val, model_name="LightGBM", model_type="lgbm")

    mlflow.log_metrics(metrics)
    mlflow.lightgbm.log_model(lgbm_model, artifact_path="model")
    save_model(lgbm_model, "models/lightgbm_model.joblib")

    logger.info("LightGBM run complete.")

logger.info("All experiments complete. Run: mlflow ui")
