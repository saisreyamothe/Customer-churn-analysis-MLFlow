"""
FastAPI server for churn prediction.
Real-time predictions with audit logging.
"""

import time
import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


app = FastAPI(title="Churn Prediction API", version="1.0")

# Load artifacts
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


class ChurnRequest(BaseModel):
    tenure: float
    monthly_charges: float
    total_charges: float
    # Add more features as needed


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    latency_ms: float
    timestamp: str


@app.post("/predict", response_model=ChurnResponse)
async def predict(request: ChurnRequest):
    """Make churn prediction."""
    start = time.time()
    
    # Prepare features
    features = np.array([[
        request.tenure,
        request.monthly_charges,
        request.total_charges
    ]])
    
    # Scale
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    latency = (time.time() - start) * 1000
    
    # Audit log
    _log_prediction(request, prediction, probability)
    
    return ChurnResponse(
        churn_probability=round(float(probability), 3),
        churn_prediction=bool(prediction),
        latency_ms=round(latency, 1),
        timestamp=datetime.now().isoformat(),
    )


def _log_prediction(request, prediction, probability):
    """Audit log for compliance."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prediction": int(prediction),
        "probability": float(probability),
        "input": request.dict(),
    }
    
    with open("logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
