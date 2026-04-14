import base64
from pathlib import Path
from datetime import datetime, timezone

import cv2
import requests
from fastapi import FastAPI

from app.face_detector import FaceDetector

app = FastAPI(title="orchestrator")

EMOTION_HSE_URL = "http://emotion_hse:8001"
face_detector = FaceDetector()


@app.get("/health")
def health():
    return {"status": "ok", "service": "orchestrator"}


@app.get("/test-emotion")
def test_emotion():
    img_path = Path("/data/test_inputs/face.jpg")
    image_bgr = cv2.imread(str(img_path))

    if image_bgr is None:
        return {"error": f"Could not read image at {img_path}"}

    detected = face_detector.detect_largest_face(image_bgr)
    if detected is None:
        return {"error": "No face detected"}

    face_crop_bgr = detected["face_crop_bgr"]
    bbox_xyxy = detected["bbox_xyxy"]

    ok, encoded = cv2.imencode(".jpg", face_crop_bgr)
    if not ok:
        return {"error": "Failed to encode cropped face"}

    image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": "session_001",
        "frame_id": 1,
        "source_id": "manual_test",
        "face_id": 0,
        "image": {
            "encoding": "base64_jpeg",
            "data": image_b64,
        },
        "meta": {"mode": "smoke_test", "bbox_xyxy": str(bbox_xyxy)},
    }

    r = requests.post(f"{EMOTION_HSE_URL}/predict", json=payload, timeout=30)
    return {
        "bbox_xyxy": bbox_xyxy,
        "upstream_status": r.status_code,
        "upstream_response": r.json(),
    }