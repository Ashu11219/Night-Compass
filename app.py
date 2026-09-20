# app.py
import os
import time
import uuid
import base64
import shutil

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from run_demo import process_frame, THERMAL_MODEL_PATH, RGB_MODEL_PATH

app = FastAPI(title="NightCompass API")

# Load both models once at startup, not per request
thermal_model = YOLO(THERMAL_MODEL_PATH)
rgb_model = YOLO(RGB_MODEL_PATH)

UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Frontend files (index.html, style.css, script.js) live in static/
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    return {"status": "NightCompass API running"}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    # Decode straight from memory, no temp file needed for images
    data = await file.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="That file is not a readable image. Upload a JPG or PNG.")

    annotated, detections, source, alerts = process_frame(frame, thermal_model, rgb_model)

    _, buffer = cv2.imencode('.jpg', annotated)

    return {
        "image": base64.b64encode(buffer).decode('utf-8'),
        "source": source,
        "alerts": alerts,
        "detections": [
            {"class": c, "confidence": conf, "box": box}
            for c, conf, box in detections
        ],
    }


@app.post("/detect/video")
async def upload_video(file: UploadFile = File(...)):
    # Videos need to be on disk because cv2.VideoCapture reads from a path
    ext = os.path.splitext(file.filename)[1] or '.mp4'
    video_id = f"{uuid.uuid4().hex}{ext}"
    with open(os.path.join(UPLOAD_DIR, video_id), "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"video_id": video_id}


def generate_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_time = 1 / fps

    try:
        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            annotated, _, _, _ = process_frame(frame, thermal_model, rgb_model)
            _, buffer = cv2.imencode('.jpg', annotated)

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

            # Throttle to the video's own frame rate so playback looks natural
            elapsed = time.time() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
    finally:
        cap.release()
        if os.path.exists(path):
            os.remove(path)  # uploaded video is deleted once the stream ends or is stopped


@app.get("/stream/{video_id}")
def stream_video(video_id: str):
    path = os.path.join(UPLOAD_DIR, os.path.basename(video_id))  # basename blocks ../ path tricks
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found. Upload it again.")
    return StreamingResponse(generate_frames(path),
                             media_type="multipart/x-mixed-replace; boundary=frame")