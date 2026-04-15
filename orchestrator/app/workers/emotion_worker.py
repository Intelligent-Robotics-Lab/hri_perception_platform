import base64
import threading
import time
from typing import Optional

import cv2
import requests

from app.face_detector import FaceDetector
from app.ingest.live_ingest import FrameStore
from app.routers.emotion_router import get_active_emotion_model, get_active_emotion_url
from app.state.perception_state import PerceptionState


class EmotionWorker:
    def __init__(self, frame_store: FrameStore, perception_state: PerceptionState, interval_sec: float = 0.05):
        self.frame_store = frame_store
        self.perception_state = perception_state
        self.interval_sec = interval_sec
        self.detector = FaceDetector()
        self.running = False
        self.thread = None
        self.last_processed_frame_id: Optional[int] = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _encode_crop_to_b64(self, face_crop_bgr):
        ok, encoded = cv2.imencode(".jpg", face_crop_bgr)
        if not ok:
            return None
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    def _run(self):
        while self.running:
            packet = self.frame_store.get_latest()
            if packet is None:
                time.sleep(self.interval_sec)
                continue

            if self.last_processed_frame_id == packet.frame_id:
                time.sleep(self.interval_sec)
                continue

            self.last_processed_frame_id = packet.frame_id
            frame = packet.frame_bgr
            active_model = get_active_emotion_model()

            detected = self.detector.detect_largest_face(frame)

            if detected is None:
                self.perception_state.update_emotion({
                    "frame_id": packet.frame_id,
                    "timestamp_utc": packet.timestamp_utc,
                    "active_model": active_model,
                    "face_detected": False,
                    "bbox_xyxy": None,
                    "prediction": None,
                })
                time.sleep(self.interval_sec)
                continue

            bbox_xyxy = detected["bbox_xyxy"]
            face_crop_bgr = detected["face_crop_bgr"]
            image_b64 = self._encode_crop_to_b64(face_crop_bgr)

            if image_b64 is None:
                time.sleep(self.interval_sec)
                continue

            payload = {
                "timestamp_utc": packet.timestamp_utc,
                "session_id": "live_session_001",
                "frame_id": packet.frame_id,
                "source_id": "live_client",
                "face_id": 0,
                "image": {
                    "encoding": "base64_jpeg",
                    "data": image_b64,
                },
                "meta": {"mode": "live"},
            }

            prediction = None
            try:
                emotion_url = get_active_emotion_url()
                r = requests.post(f"{emotion_url}/predict", json=payload, timeout=5)
                if r.status_code == 200:
                    prediction = r.json()
            except Exception:
                prediction = None

            self.perception_state.update_emotion({
                "frame_id": packet.frame_id,
                "timestamp_utc": packet.timestamp_utc,
                "active_model": active_model,
                "face_detected": True,
                "bbox_xyxy": bbox_xyxy,
                "prediction": prediction,
            })

            time.sleep(self.interval_sec)