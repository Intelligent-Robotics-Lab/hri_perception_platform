import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import requests

from app.routers.emotion_router import get_active_emotion_model, get_active_emotion_url

from app.face_detector import FaceDetector


VIDEO_PATH = Path("/data/test_inputs/sample.mp4")
LOG_PATH = Path("/data/logs/replay_emotion.jsonl")

FRAME_STRIDE = 10  # process every 10th frame
SESSION_ID = "replay_session_001"


def encode_face_crop(face_crop_bgr):
    ok, encoded = cv2.imencode(".jpg", face_crop_bgr)
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("utf-8")

emotion_url = get_active_emotion_url()
active_model = get_active_emotion_model()

def main():
    detector = FaceDetector()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    processed = 0

    with LOG_PATH.open("w") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STRIDE != 0:
                frame_idx += 1
                continue

            detected = detector.detect_largest_face(frame)

            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "session_id": SESSION_ID,
                "frame_id": frame_idx,
                "source_id": str(VIDEO_PATH.name),
                "face_detected": detected is not None,
            }

            if detected is None:
                f.write(json.dumps(record) + "\n")
                frame_idx += 1
                processed += 1
                continue

            face_crop_bgr = detected["face_crop_bgr"]
            bbox_xyxy = detected["bbox_xyxy"]
            image_b64 = encode_face_crop(face_crop_bgr)

            if image_b64 is None:
                record["error"] = "face_crop_encode_failed"
                f.write(json.dumps(record) + "\n")
                frame_idx += 1
                processed += 1
                continue

            payload = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "session_id": SESSION_ID,
                "frame_id": frame_idx,
                "source_id": VIDEO_PATH.name,
                "face_id": 0,
                "image": {
                    "encoding": "base64_jpeg",
                    "data": image_b64,
                },
                "meta": {
                    "mode": "replay",
                    "bbox_xyxy": str(bbox_xyxy),
                },
            }

            try:
                r = requests.post(f"{emotion_url}/predict", json=payload, timeout=30)
                response = r.json()
                record["active_model"] = active_model

                record.update({
                    "bbox_xyxy": bbox_xyxy,
                    "upstream_status": r.status_code,
                    "emotion_response": response,
                })
            except Exception as e:
                record["error"] = str(e)

            f.write(json.dumps(record) + "\n")

            frame_idx += 1
            processed += 1

    cap.release()
    print(f"Done. Processed {processed} sampled frames.")
    print(f"Log written to: {LOG_PATH}")


if __name__ == "__main__":
    main()