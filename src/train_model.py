from src.model import build_model
import numpy as np
import tensorflow as tf
import os

def train_basic_model(X, y, num_features=2, save_path="models/model.h5"):
    """
    Train a simple posture risk model using angle-based features.
    """
    model = build_model(num_features=num_features)
    X = X.reshape((X.shape[0], num_features))

    model.fit(
        X, y,
        validation_split=0.2,
        epochs=15,
        batch_size=16,
        verbose=1
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"✅ Model saved at {save_path}")

    return model
