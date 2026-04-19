# api/__init__.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging
import os

from src.inference import load_artifacts, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CreditWise AI",
    description="Credit default risk prediction API powered by LightGBM",
    version="1.0.0"
)

# Load model and encoders once at startup — not on every request
MODEL_PATH = os.getenv("MODEL_PATH", "models/lightgbm_model.joblib")
ENCODERS_PATH = os.getenv("ENCODERS_PATH", "models/encoders.joblib")

model, encoders = load_artifacts(MODEL_PATH, ENCODERS_PATH)


class ApplicantInput(BaseModel):
    """
    Input schema for a loan applicant.
    All fields mirror the original Home Credit dataset columns.
    Optional fields default to None — missing values are handled
    in the inference pipeline the same way training handled them.
    """
    CODE_GENDER: str = Field(..., example="M")
    FLAG_OWN_CAR: str = Field(..., example="Y")
    FLAG_OWN_REALTY: str = Field(..., example="Y")
    CNT_CHILDREN: float = Field(..., example=0)
    AMT_INCOME_TOTAL: float = Field(..., example=202500.0)
    AMT_CREDIT: float = Field(..., example=406597.5)
    AMT_ANNUITY: float = Field(..., example=24700.5)
    AMT_GOODS_PRICE: float = Field(..., example=351000.0)
    NAME_INCOME_TYPE: str = Field(..., example="Working")
    NAME_EDUCATION_TYPE: str = Field(..., example="Secondary / secondary special")
    NAME_FAMILY_STATUS: str = Field(..., example="Single / not married")
    NAME_HOUSING_TYPE: str = Field(..., example="House / apartment")
    DAYS_BIRTH: float = Field(..., example=-9461)
    DAYS_EMPLOYED: float = Field(..., example=-637)
    DAYS_REGISTRATION: float = Field(..., example=-3648.0)
    DAYS_ID_PUBLISH: float = Field(..., example=-2120)
    OCCUPATION_TYPE: Optional[str] = Field(None, example="Laborers")
    CNT_FAM_MEMBERS: float = Field(..., example=1.0)
    REGION_RATING_CLIENT: float = Field(..., example=2)
    EXT_SOURCE_1: Optional[float] = Field(None, example=0.5)
    EXT_SOURCE_2: Optional[float] = Field(None, example=0.6)
    EXT_SOURCE_3: Optional[float] = Field(None, example=0.4)


class PredictionOutput(BaseModel):
    default_probability: float
    risk_label: str
    model_type: str


@app.get("/")
def root():
    return {"message": "CreditWise AI is running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict_default(applicant: ApplicantInput):
    """
    Accepts applicant data and returns default probability + risk label.
    Raises HTTP 500 if prediction fails for any reason.
    """
    try:
        result = predict(
            model=model,
            encoders=encoders,
            data=applicant.model_dump(),
            model_type="lgbm"
        )
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))