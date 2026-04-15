import base64
from pathlib import Path
from datetime import datetime, timezone

import cv2
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.face_detector import FaceDetector
from app.ingest.frame_store import FrameStore
from app.ingest.transport_adapters.bootstrap_http_ingest import BootstrapHttpFrameIngestAdapter
from app.routers.emotion_router import get_active_emotion_model, get_active_emotion_url
from app.state.perception_state import PerceptionState
from app.workers.emotion_worker import EmotionWorker

app = FastAPI(title="orchestrator")

face_detector = FaceDetector()

frame_store = FrameStore()
frame_ingest_adapter = BootstrapHttpFrameIngestAdapter(frame_store)
perception_state = PerceptionState()
emotion_worker = None


@app.on_event("startup")
def startup_event():
    global emotion_worker
    emotion_worker = EmotionWorker(
        frame_store=frame_store,
        perception_state=perception_state,
        interval_sec=0.05,
    )
    emotion_worker.start()


@app.on_event("shutdown")
def shutdown_event():
    global emotion_worker
    if emotion_worker is not None:
        emotion_worker.stop()


@app.get("/health")
def health():
    return {"status": "ok", "service": "orchestrator"}


@app.post("/ingest/frame")
async def ingest_frame(
    file: UploadFile = File(...),
    client_capture_timestamp: str | None = Form(default=None),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty uploaded file")

    try:
        packet = frame_ingest_adapter.ingest(
            file_bytes=content,
            client_capture_timestamp=client_capture_timestamp,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "ok",
        "frame_id": packet.frame_id,
        "client_capture_timestamp": packet.client_capture_timestamp,
        "server_ingest_timestamp": packet.server_ingest_timestamp,
        "active_model": get_active_emotion_model(),
    }


@app.get("/state/emotion")
def get_emotion_state():
    emotion_state = perception_state.get_emotion()
    if emotion_state is None:
        return {
            "status": "ok",
            "message": "No emotion state yet",
            "emotion_state": None,
        }

    return {
        "status": "ok",
        "emotion_state": emotion_state,
    }


@app.get("/metrics/live/emotion")
def get_emotion_metrics():
    metrics = perception_state.get_emotion_metrics()
    if metrics is None:
        return {
            "status": "ok",
            "message": "No live emotion metrics yet",
            "metrics": None,
        }

    return {
        "status": "ok",
        "metrics": metrics,
    }


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

    emotion_url = get_active_emotion_url()
    active_model = get_active_emotion_model()

    r = requests.post(f"{emotion_url}/predict", json=payload, timeout=30)
    return {
        "active_model": active_model,
        "bbox_xyxy": bbox_xyxy,
        "upstream_status": r.status_code,
        "upstream_response": r.json(),
    }