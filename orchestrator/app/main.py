import base64
import requests
from datetime import datetime, timezone
from fastapi import FastAPI

app = FastAPI(title="orchestrator")

EMOTION_HSE_URL = "http://emotion_hse:8001"


@app.get("/health")
def health():
    return {"status": "ok", "service": "orchestrator"}


@app.get("/test-emotion")
def test_emotion():
    fake_image = base64.b64encode(b"fake-image-bytes").decode("utf-8")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": "session_001",
        "frame_id": 1,
        "source_id": "manual_test",
        "face_id": 0,
        "image": {
            "encoding": "base64_jpeg",
            "data": fake_image,
        },
        "meta": {"mode": "smoke_test"},
    }

    r = requests.post(f"{EMOTION_HSE_URL}/predict", json=payload, timeout=10)
    return {
        "upstream_status": r.status_code,
        "upstream_response": r.json(),
    }