# api.py
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from train_model import load_or_train_model, MODEL_DIR, DATA_DIR, SCALER_PATH, FEATURE_ORDER_PATH

# ------------------------
# FastAPI Setup
# ------------------------
app = FastAPI(title="Fraud Detection API", version="1.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.dirname(__file__)
@app.get("/")
def dashboard():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found"}

# ------------------------
# Request Model
# ------------------------
class PredictRequest(BaseModel):
    features: dict
    threshold: float = 0.5

# ------------------------
# Global artifacts
# ------------------------
model = None
scaler = None
feature_order = None
fraud_class_index = 1

# ------------------------
# Startup Event
# ------------------------
@app.on_event("startup")
def load_artifacts():
    global model, scaler, feature_order, fraud_class_index
    print("Loading model and artifacts...")

    model = load_or_train_model()
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    if os.path.exists(FEATURE_ORDER_PATH):
        with open(FEATURE_ORDER_PATH) as f:
            feature_order = [line.strip() for line in f if line.strip()]
        print(f"Feature order loaded: {feature_order}")

    if hasattr(model, "classes_") and 1 in model.classes_:
        fraud_class_index = list(model.classes_).index(1)
    print(f"Model ready! Fraud class index: {fraud_class_index}")

# ------------------------
# Health endpoint
# ------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

# ------------------------
# Predict endpoint
# ------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    if model is None or feature_order is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    features = req.features
    threshold = req.threshold

    # Prepare input in correct order
    try:
        X = np.array([features[f] for f in feature_order], dtype=float).reshape(1, -1)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {str(e)}")
    except ValueError:
        raise HTTPException(status_code=422, detail="Feature values must be numeric")

    if scaler:
        X = scaler.transform(X)

    fraud_prob = float(model.predict_proba(X)[0][fraud_class_index])
    is_fraud = fraud_prob >= threshold

    return {
        "fraud_probability": round(fraud_prob, 4),
        "is_fraud": bool(is_fraud),
        "threshold_used": threshold
    }
