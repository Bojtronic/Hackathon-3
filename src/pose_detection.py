# src/pose_detection.py
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Utility function to calculate the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


class PoseDetector:
    """
    Extracts human keypoints and calculates relevant posture angles.
    """

    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame):
        """
        Detect pose landmarks and return keypoints and angles.
        :param frame: BGR image frame
        :return: keypoints dict, angles dict
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return {}, {}

        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        # Extract main joints
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
        left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * w),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].y * h))

        # Calculate posture angles
        shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
        neck_angle = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))

        angles = {
            "shoulder_angle": shoulder_angle,
            "neck_angle": neck_angle
        }

        keypoints = {
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "left_ear": left_ear
        }

        # Optional: draw landmarks for visualization
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return keypoints, angles
