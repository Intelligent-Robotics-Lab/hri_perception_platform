import base64
from pathlib import Path
from datetime import datetime, timezone

import requests
from fastapi import FastAPI

app = FastAPI(title="orchestrator")

EMOTION_HSE_URL = "http://emotion_hse:8001"


@app.get("/health")
def health():
    return {"status": "ok", "service": "orchestrator"}


@app.get("/test-emotion")
def test_emotion():
    img_path = Path("/data/test_inputs/face.jpg")
    image_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

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
        "meta": {"mode": "smoke_test"},
    }

    r = requests.post(f"{EMOTION_HSE_URL}/predict", json=payload, timeout=30)
    return {
        "upstream_status": r.status_code,
        "upstream_response": r.json(),
    }