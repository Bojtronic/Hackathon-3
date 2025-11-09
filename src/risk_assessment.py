import numpy as np
import tensorflow as tf
import os

# Load trained model once at import
MODEL_PATH = "models/model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}. Run mock_training.py first.")

model = tf.keras.models.load_model(MODEL_PATH)


def assess_posture_risk(angles: dict) -> dict:
    """
    Assess risk based on learned features.
    Returns a dictionary with score (0–100), qualitative risk, and feedback.
    """
    if not angles:
        return {
            "score": 0,
            "risk_level": "Unknown",
            "feedback": ["No posture data available."]
        }

    # Convert input angles to a feature vector
    feature_vector = np.array(list(angles.values())).reshape(1, -1)

    # Normalize roughly between 0–1 based on typical angle range (0–180°)
    feature_vector = feature_vector / 180.0

    # Predict risk probability
    risk_prob = float(model.predict(feature_vector, verbose=0)[0][0])

    # Convert probability to score and qualitative risk
    posture_score = 100 * (1 - risk_prob)
    risk_level = (
        "Low" if posture_score > 85 else
        "Medium" if posture_score > 60 else
        "High"
    )

    feedback = []
    if risk_level == "High":
        feedback.append("High-risk posture detected — likely instability or strain risk.")
    elif risk_level == "Medium":
        feedback.append("Posture slightly off ideal form, minor adjustment suggested.")
    else:
        feedback.append("Good posture detected — maintain form.")

    return {
        "score": float(np.clip(posture_score, 0, 100)),
        "risk_level": risk_level,
        "feedback": feedback
    }
