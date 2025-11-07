#!/usr/bin/env bash
URL="http://127.0.0.1:8000/predict"
# Replace fields with your model's feature names & values
json='{
  "features": {
    "V1": 0.123,
    "V2": -1.23,
    "V3": 0.45,
    "Amount": 12.34
  },
  "threshold": 0.4
}'
curl -s -H "Content-Type: application/json" -d "$json" $URL | jq .
