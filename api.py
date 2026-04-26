from fastapi import FastAPI,HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
import os

class CustomerFeatures(BaseModel):
    tenure: float
    monthly_charges: float
    total_charges: float
    support_charges: float
    usage: float

app=FastAPI(title="Churn Prediction Service", version="1.0.0")
MODEL_PATH='models/churn_model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model= None

@app.get("/health")
def health_check():
    return {"status":"online","model_loaded":model is not None}
@app.post("/predict")
def predict(customer_features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")

    data_df=pd.DataFrame([customer.dict()])

    prediction = int(model.predict(data_df))
    probability = float(model.predict_proba(data_df))

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "status": "High Risk" if prediction == 1 else "Low Risk"
    }



