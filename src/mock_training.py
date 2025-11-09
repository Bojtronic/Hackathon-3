import numpy as np
from src.train_model import train_basic_model

# Simulated dataset for angle-based posture assessment
N_SAMPLES = 500
N_FEATURES = 2  # e.g., shoulder_angle, neck_angle

# Generate synthetic angles in degrees (0–180)
X = np.random.uniform(0, 180, size=(N_SAMPLES, N_FEATURES))

# Generate synthetic risk labels: risky if mean angle > threshold
y = (np.mean(X, axis=1) > 90).astype(int)

# Train model and save
train_basic_model(X, y, num_features=N_FEATURES)
