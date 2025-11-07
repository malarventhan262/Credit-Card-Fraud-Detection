import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

DATA_CSV = os.getenv("DATA_CSV", "creditcard.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if "Class" not in df.columns:
        raise ValueError("CSV must contain a 'Class' column for labels")
    return df

def preprocess(df):
    X = df.drop(columns=["Class"])
    y = df["Class"].values
    numeric_cols = X.select_dtypes(include=["float64","int64"]).columns.tolist()
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    return X, y, scaler

def balance_data(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def main():
    df = load_data(DATA_CSV)
    X, y, scaler = preprocess(df)
    print(f"Original dataset distribution: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train distribution before SMOTE: {np.bincount(y_train)}")

    X_train_res, y_train_res = balance_data(X_train, y_train)
    print(f"Train distribution after SMOTE: {np.bincount(y_train_res)}")

    np.savez_compressed(os.path.join(OUTPUT_DIR, "train.npz"), X=X_train_res, y=y_train_res)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "test.npz"), X=X_test.values, y=y_test)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))

    print(f"Data prepared and saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
