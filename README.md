💳 Credit Card Fraud Detection

A Linux-compatible web application for detecting credit card fraud using machine learning. Built with FastAPI for the backend and a simple web frontend for interactive testing.

⚡ Features

🚨 Predicts the probability of a credit card transaction being fraudulent.

🖥️ Interactive web frontend to input transaction features.

🔗 FastAPI backend with RESTful endpoints (/health & /predict).

📊 Preprocessing and scaling of features using scikit-learn.

⚖️ Handles imbalanced datasets using SMOTE.

🐳 Fully containerized using Docker for easy deployment.

🐧 Compatible with Linux environments.

🛠️ Requirements

Python 3.11+

Linux system

Libraries (install via requirements.txt):

fastapi

uvicorn[standard]

scikit-learn

pandas

numpy

joblib

imbalanced-learn

python-dotenv

Docker (optional) 🐳

⚙️ Setup

Clone the repository

git clone https://github.com/yourusername/credit-card-detection.git
cd credit-card-detection


Install dependencies

pip install -r requirements.txt


Prepare dataset

Place your creditcard.csv in the project root.

Or download via Kaggle API:

import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")


Generate training & test data

python data_prep.py


Train the model

python train_model.py


Run the API

uvicorn api:app --host 0.0.0.0 --port 8000


Access the frontend
Open a browser at http://localhost:8000/ 🌐

🐳 Docker Deployment

Build the Docker image

docker build -t credit-card-detection .


Run the container

docker run -p 8000:8000 credit-card-detection


Open in browser: http://localhost:8000/

🔌 API Endpoints

GET /health
Returns the status of the API and whether the model is loaded.

POST /predict
Input: JSON with transaction features and optional threshold.

{
  "features": {
    "feature0": 1.2,
    "feature1": 0.5,
    "feature2": 0.3,
    "feature3": 0.8,
    "feature4": 0.1
  },
  "threshold": 0.5
}


Output: JSON with fraud probability and classification.

{
  "fraud_probability": 0.87,
  "is_fraud": true,
  "threshold_used": 0.5
}

⚠️ Notes

The model can be trained on synthetic or real credit card data.

Input features must match the trained model’s feature_order.

You can adjust the fraud detection threshold.
