# src/video_source.py
import cv2

class VideoSource:
    """
    Unified video source class that can handle:
    - Webcam (int index)
    - RTSP stream (URL)
    - Local video file (.mp4, .avi, etc.)
    """

    def __init__(self, source=0):
        """
        :param source: int for webcam index, str for RTSP or file path
        """
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"❌ Cannot open video source: {source}")

    def get_frame(self):
        """Read a single frame from the source."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """Release the camera or video file."""
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.release()
