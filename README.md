# CreditWise AI — Credit Default Risk Prediction System

A production-ready machine learning system for predicting loan default risk,
built on the Home Credit Default Risk dataset. The system includes a full
data pipeline, experiment tracking, a REST API, and Docker deployment.

---

## Project Structure
creditwise-ai/
├── data/
│   ├── raw/              # Raw downloaded dataset (not tracked in git)
│   └── processed/        # Processed outputs
├── src/
│   ├── ingestion/        # Data loading and schema validation
│   ├── preprocessing/    # Feature engineering and cleaning
│   ├── training/         # Model training scripts
│   ├── evaluation/       # Metrics and reporting
│   └── inference/        # Prediction logic
├── api/                  # FastAPI application
├── models/               # Saved model artifacts
├── mlruns/               # MLflow experiment tracking
├── reports/              # Technical report
├── notebooks/            # Exploratory analysis
├── Dockerfile
├── requirements.txt
└── README.md

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/creditwise-ai.git
cd creditwise-ai
```

### 2. Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

You will need a Kaggle account and API token set up at `~/.kaggle/kaggle.json`.
You must also accept the competition rules at:
https://www.kaggle.com/competitions/home-credit-default-risk

```bash
kaggle competitions download -c home-credit-default-risk -p data/raw/
cd data/raw && unzip home-credit-default-risk.zip && cd ../..
```

---

## Running the Training Pipeline

```bash
python3 run_experiments.py
```

This will:
- Ingest and validate the raw data
- Run preprocessing and feature engineering
- Train a Logistic Regression baseline and a LightGBM model
- Log all parameters, metrics, and artifacts to MLflow
- Save trained models to `models/`

---

## Viewing Experiment Results

```bash
mlflow ui
```

Open http://127.0.0.1:5000 in your browser to compare runs.

---

## Starting the API

```bash
uvicorn api:app --reload
```

API will be available at http://127.0.0.1:8000

Interactive docs: http://127.0.0.1:8000/docs

### Example prediction request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": "Y",
    "FLAG_OWN_REALTY": "Y",
    "CNT_CHILDREN": 0,
    "AMT_INCOME_TOTAL": 202500.0,
    "AMT_CREDIT": 406597.5,
    "AMT_ANNUITY": 24700.5,
    "AMT_GOODS_PRICE": 351000.0,
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_FAMILY_STATUS": "Single / not married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "DAYS_BIRTH": -9461,
    "DAYS_EMPLOYED": -637,
    "DAYS_REGISTRATION": -3648.0,
    "DAYS_ID_PUBLISH": -2120,
    "OCCUPATION_TYPE": "Laborers",
    "CNT_FAM_MEMBERS": 1.0,
    "REGION_RATING_CLIENT": 2,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.4
  }'
```

### Example response

```json
{
  "default_probability": 0.5683,
  "risk_label": "HIGH RISK",
  "model_type": "lgbm"
}
```

---

## Running with Docker

```bash
# Build the image
docker build -t creditwise-ai .

# Run the container
docker run -p 8001:8000 creditwise-ai
```

API will be available at http://127.0.0.1:8001

---

## Model Performance Summary

| Metric     | Logistic Regression | LightGBM |
|------------|-------------------|----------|
| ROC-AUC    | 0.6468            | 0.7589   |
| PR-AUC     | 0.1405            | 0.2482   |
| F1 Score   | 0.1961            | 0.2714   |
| Recall     | 0.6125            | 0.6812   |
| Precision  | 0.1167            | 0.1695   |

LightGBM significantly outperforms the baseline across all metrics and
is used as the production model in the inference API.

---

## Key Design Decisions

- **Stratified train/val split** preserves the 8% default rate in both sets
- **scale_pos_weight** in LightGBM handles class imbalance
- **Early stopping** prevents overfitting without manual tuning
- **Encoders saved separately** from the model for consistent inference
- **Schema validation at ingestion** catches data issues before they
  propagate through the pipeline

---

## Limitations

- Only the main `application_train.csv` table is used. Bureau, credit card,
  and installment tables are excluded — incorporating them would likely
  improve model performance significantly.
- The prediction threshold is fixed at 0.5. A lower threshold (e.g. 0.3)
  would improve recall at the cost of more false positives, which may be
  preferable in a real lending context.
- No concept drift monitoring is implemented in the current version.

---

## Tech Stack

- Python 3.11
- scikit-learn, LightGBM
- FastAPI, Uvicorn
- MLflow
- Docker
- Pandas, NumPy, Joblib