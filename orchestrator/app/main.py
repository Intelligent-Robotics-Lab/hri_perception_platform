import base64
from pathlib import Path
from datetime import datetime, timezone

import cv2
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.face_detector import FaceDetector
from app.ingest.audio_store import AudioStore
from app.ingest.frame_store import FrameStore
from app.ingest.transport_adapters.bootstrap_http_ingest import BootstrapHttpFrameIngestAdapter
from app.registry.perception_registry import PerceptionRegistry
from app.routers.emotion_router import get_active_emotion_model, get_active_emotion_url
from app.state.perception_state import PerceptionState
from app.workers.asr_worker import ASRWorker
from app.workers.emotion_worker import EmotionWorker

app = FastAPI(title="orchestrator")

registry = PerceptionRegistry()

face_detector = FaceDetector()

frame_store = FrameStore()
audio_store = AudioStore()

frame_ingest_adapter = BootstrapHttpFrameIngestAdapter(frame_store)

perception_state = PerceptionState()

emotion_worker = None
asr_worker = None


@app.on_event("startup")
def startup_event():
    global emotion_worker, asr_worker

    # Production policy:
    # - orchestrator is transport-agnostic
    # - live media enters only through HTTP ingest from services/media_gateway
    # - legacy direct GStreamer/WebRTC receive paths are not started here
    registry._reload_if_needed()

    emotion_worker = EmotionWorker(
        frame_store=frame_store,
        perception_state=perception_state,
        interval_sec=0.05,
    )
    emotion_worker.start()

    asr_worker = ASRWorker(
        audio_store=audio_store,
        perception_state=perception_state,
        interval_sec=0.05,
    )
    asr_worker.start()


@app.on_event("shutdown")
def shutdown_event():
    global emotion_worker, asr_worker

    if emotion_worker is not None:
        emotion_worker.stop()

    if asr_worker is not None:
        asr_worker.stop()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "orchestrator",
        "transport_policy": "http_ingest_only",
    }


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


@app.post("/ingest/audio")
async def ingest_audio(
    file: UploadFile = File(...),
    client_capture_timestamp: str | None = Form(default=None),
    source_id: str | None = Form(default="live_client"),
    sample_rate_hz: int | None = Form(default=None),
    channels: int | None = Form(default=None),
    encoding: str | None = Form(default=None),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty uploaded audio")

    packet = audio_store.update_from_bytes(
        audio_bytes=content,
        client_capture_timestamp=client_capture_timestamp,
        source_id=source_id,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        encoding=encoding,
    )

    return {
        "status": "ok",
        "chunk_id": packet.chunk_id,
        "client_capture_timestamp": packet.client_capture_timestamp,
        "server_ingest_timestamp": packet.server_ingest_timestamp,
        "source_id": packet.source_id,
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


@app.get("/state/asr")
def get_asr_state():
    asr_state = perception_state.get_asr()
    if asr_state is None:
        return {
            "status": "ok",
            "message": "No ASR state yet",
            "asr_state": None,
        }

    return {
        "status": "ok",
        "asr_state": asr_state,
    }


@app.get("/metrics/live/asr")
def get_asr_metrics():
    metrics = perception_state.get_asr_metrics()
    if metrics is None:
        return {
            "status": "ok",
            "message": "No live ASR metrics yet",
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
