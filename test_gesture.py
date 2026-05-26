import base64
from pathlib import Path
import requests

img_path = Path("data/debug/latest_input_frame.jpg")
image_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

payload = {
    "timestamp_utc": "2026-05-26T00:00:00+00:00",
    "session_id": "test_session",
    "frame_id": 1,
    "source_id": "manual_test",
    "image": {
        "encoding": "base64_jpeg",
        "data": image_b64,
    },
    "meta": {"mode": "manual_test"},
}

r = requests.post("http://localhost:8006/predict", json=payload, timeout=20)
print("status:", r.status_code)
print(r.json())