# src/visualization.py
import cv2

def draw_feedback(frame, angles, risk_info):
    """
    Overlay posture angles and risk feedback on the frame.
    """
    color = (0, 255, 0) if risk_info["risk_level"] == "Low" else \
            (0, 255, 255) if risk_info["risk_level"] == "Medium" else \
            (0, 0, 255)

    cv2.putText(frame, f"Risk: {risk_info['risk_level']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Score: {risk_info['score']:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    y = 100
    for line in risk_info["feedback"]:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y += 25

    return frame
