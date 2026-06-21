# Customer Churn Prediction — End-to-End MLOps Pipeline

**F1: 0.82** | **AUC-ROC: 0.86** | **Automated Drift Detection** | **Production Monitoring**

Complete MLOps pipeline for customer churn prediction using Gradient Boosting, MLflow experiment tracking, drift detection (PSI > 0.2 alert), and FastAPI real-time serving.

## Key Metrics

| Metric | Value | Fold |
|--------|-------|------|
| F1-Score (Churn Class) | 0.82 | 5-fold CV |
| AUC-ROC | 0.86 | 5-fold CV |
| Precision | 0.84 | - |
| Recall | 0.80 | - |
| Training Data | IBM Telco (7,043 records) | - |

## Architecture

```
Raw Data
    ↓
[Feature Engineering]
    ↓
[GBM Model Training] (5-fold CV)
    ↓
[MLflow Logging] → params, metrics, model
    ↓
[Model Registry] → Best model selection
    ↓
[Drift Detection] (PSI monitoring)
    ↓
[FastAPI Serving] → Real-time predictions
    ↓
[Audit Logging] → All predictions tracked
```

## Installation

```bash
git clone https://github.com/yourusername/MLOps-ChurnPrediction.git
cd MLOps-ChurnPrediction

pip install -r requirements.txt

# Start MLflow UI
mlflow ui
```

## Quick Start

```python
from src.train import train_churn_model

model = train_churn_model(
    data_path="data/telco_churn.csv",
    output_dir="models/",
    n_splits=5
)

print(f"F1: {model.metrics['f1']:.3f}")
```

## Serving

```bash
python src/api.py --port 8000
```

```bash
curl -X POST "http://localhost:8000/predict" \
  -d '{"tenure": 24, "monthly_charges": 65.5, ...}'
```

## Monitoring

```python
from src.monitoring import check_drift

drift_detected = check_drift(
    current_data="data/current_month.csv",
    baseline_data="data/baseline.csv",
    threshold=0.2
)

if drift_detected:
    print("Model drift detected! Retraining recommended.")
```

## License

MIT License
