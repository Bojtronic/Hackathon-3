# app/app.py
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
from src.pose_detection import PoseDetector
from src.video_source import VideoSource
from src.risk_assessment import assess_posture_risk
from src.visualization import draw_feedback

app = FastAPI(title="AI Sports Coach - Injury Risk Detection")

pose_detector = PoseDetector()
video_source = None  # Initialized dynamically


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>AI Sports Coach - Global AI Hackathon</h2>
    <p>Upload a video or stream your camera to analyze athlete posture and injury risk.</p>
    <form action="/analyze_video" enctype="multipart/form-data" method="post">
      <input name="file" type="file" accept="video/*">
      <input type="submit" value="Upload and Analyze">
    </form>
    <p>Or open <a href='/video_feed'>/video_feed</a> for real-time webcam.</p>
    """


@app.post("/analyze_video")
async def analyze_video(file: UploadFile):
    contents = await file.read()
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(contents)

    video = cv2.VideoCapture(tmp_path)
    return StreamingResponse(generate_frames(video), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/video_feed")
def stream_camera():
    global video_source
    video_source = VideoSource(0)
    return StreamingResponse(generate_frames(video_source.cap), media_type="multipart/x-mixed-replace; boundary=frame")


def generate_frames(video):
    while True:
        ret, frame = video.read()
        if not ret:
            break

        keypoints, angles = pose_detector.detect(frame)
        if angles:
            risk_info = assess_posture_risk(angles)
            frame = draw_feedback(frame, angles, risk_info)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    video.release()
