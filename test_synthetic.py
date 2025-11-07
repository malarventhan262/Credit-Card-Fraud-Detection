import pandas as pd
import requests

# Load synthetic test dataset
df = pd.read_csv("synthetic_test_data.csv")

# API endpoint
url = "http://127.0.0.1:8000/predict"

print("Testing synthetic transactions...\n")

for i, row in df.iterrows():
    payload = {"features": row.to_dict(), "threshold": 0.5}
    response = requests.post(url, json=payload)
    print(f"Transaction {i+1}: {row.to_dict()}")
    print(f"Prediction: {response.json()}\n")
