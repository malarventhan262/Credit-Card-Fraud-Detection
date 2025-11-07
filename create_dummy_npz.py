import numpy as np
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# --- Dummy training data ---
# 100 samples, 5 features
X_train = np.random.rand(100, 5)      # Features (values between 0 and 1)
y_train = np.random.randint(0, 2, 100) # Labels: 0 = non-fraud, 1 = fraud

# Save train.npz
np.savez("data/train.npz", X=X_train, y=y_train)

# --- Dummy test data ---
# 20 samples, 5 features
X_test = np.random.rand(20, 5)
y_test = np.random.randint(0, 2, 20)

# Save test.npz
np.savez("data/test.npz", X=X_test, y=y_test)

print("train.npz and test.npz created in 'data/' folder.")
