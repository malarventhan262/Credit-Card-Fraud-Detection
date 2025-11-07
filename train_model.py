# train_model.py
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ------------------------
# Paths
# ------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "model")
DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.joblib")
FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, "feature_order.txt")
DATASET_PATH = os.path.join(DATA_DIR, "creditcard.csv")  # CSV file path

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------
# Train or Load Model
# ------------------------
def load_or_train_model():
    # Return model if already exists
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    # Check dataset exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"{DATASET_PATH} not found. Download from Kaggle or provide CSV.")

    # Load real credit card dataset
    df = pd.read_csv(DATASET_PATH)
    features_cols = [c for c in df.columns if c != "Class"]
    X = df[features_cols].values
    y = df["Class"].values

    # Standard scaling
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURE_ORDER_PATH, "w") as f:
        for col in features_cols:
            f.write(col + "\n")

    print(f"Model trained and saved at {MODEL_PATH}")
    print(f"Scaler saved at {SCALER_PATH}")
    print(f"Feature order saved at {FEATURE_ORDER_PATH}")

    return model
