import os
import pandas as pd
import joblib
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ------------------------
# Paths
# ------------------------
DATA_PATH = "data/creditcard.csv"   # path to creditcard.csv
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
SCALER_PATH = os.path.join("data", "scaler.joblib")
FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, "feature_order.txt")

os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------
# Load dataset
# ------------------------
df = pd.read_csv(DATA_PATH)
features_cols = [c for c in df.columns if c != "Class"]

X = df[features_cols].values
y = df["Class"].values

# ------------------------
# Train model
# ------------------------
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_scaled, y)

# Save artifacts
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

with open(FEATURE_ORDER_PATH, "w") as f:
    for col in features_cols:
        f.write(col + "\n")

print("Model, scaler, and feature order saved!")

# ------------------------
# Prepare test payloads
# ------------------------
X_test = df[features_cols].head(5)
y_test = df["Class"].head(5)

test_payloads = []
for i, row in X_test.iterrows():
    payload = {
        "features": {col: float(row[col]) for col in features_cols},
        "threshold": 0.5
    }
    test_payloads.append(payload)

# ------------------------
# Test API
# ------------------------
API_URL = "http://127.0.0.1:8000/predict"

print("\nTesting API predictions for first 5 rows:")
for i, payload in enumerate(test_payloads):
    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()
        print(f"Row {i}: Predicted -> {result}, Actual -> {int(y_test.iloc[i])}")
    except Exception as e:
        print(f"Row {i}: Failed to get prediction. Error: {e}")
