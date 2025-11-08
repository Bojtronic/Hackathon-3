# src/risk_assessment.py
import numpy as np

def assess_posture_risk(angles: dict) -> dict:
    """
    Assess risk based on posture angles.
    Returns a score (0–100) and qualitative feedback.
    """
    shoulder_angle = angles.get("shoulder_angle", 0)
    neck_angle = angles.get("neck_angle", 0)

    # Heuristic scoring (replace with ML model later)
    posture_score = 100
    feedback = []

    if shoulder_angle < 160:
        posture_score -= (160 - shoulder_angle) * 0.5
        feedback.append("Shoulders tilted forward — possible fatigue or slouching.")

    if neck_angle < 150:
        posture_score -= (150 - neck_angle) * 0.7
        feedback.append("Neck angle too sharp — risk of strain.")

    posture_score = np.clip(posture_score, 0, 100)

    risk_level = (
        "Low" if posture_score > 85 else
        "Medium" if posture_score > 60 else
        "High"
    )

    return {
        "score": float(posture_score),
        "risk_level": risk_level,
        "feedback": feedback or ["Good posture detected."]
    }
