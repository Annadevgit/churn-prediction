"""FastAPI API serving the churn prediction model."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import add_business_features

app = FastAPI(
    title="Churn Prediction API",
    description="Customer churn prediction API",
    version="1.0.0",
)

MODEL_PATH = Path("models/churn_pipeline.pkl")
_pipeline = None


class CustomerInput(BaseModel):
    tenure_months: int = Field(..., ge=0, le=100)
    contract_type: str
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    internet_service: str
    tech_support: str
    online_security: str
    payment_method: str
    senior_citizen: int
    partner: str
    dependents: str
    paperless_billing: str

    class Config:
        json_schema_extra = {
            "example": {
                "tenure_months": 5,
                "contract_type": "Month-to-month",
                "monthly_charges": 85.5,
                "total_charges": 427.5,
                "internet_service": "Fiber optic",
                "tech_support": "No",
                "online_security": "No",
                "payment_method": "Electronic check",
                "senior_citizen": 0,
                "partner": "No",
                "dependents": "No",
                "paperless_billing": "Yes",
            }
        }


class PredictionOutput(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run `python -m src.train` first.",
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def risk_level_from_proba(p: float) -> str:
    if p >= 0.7:
        return "high"
    if p >= 0.4:
        return "medium"
    return "low"


@app.get("/")
def root():
    return {"status": "ok", "message": "Churn Prediction API - see /docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    pipeline = get_pipeline()

    df = pd.DataFrame([customer.model_dump()])
    df = add_business_features(df)

    proba = float(pipeline.predict_proba(df)[0, 1])
    prediction = int(proba >= 0.5)

    return PredictionOutput(
        churn_probability=round(proba, 4),
        churn_prediction=prediction,
        risk_level=risk_level_from_proba(proba),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)