import base64
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import requests

from app.ingest.frame_store import FrameStore
from app.routers.gesture_router import get_active_gesture_model, get_active_gesture_url
from app.state.perception_state import PerceptionState


def iso_to_ts(iso_str: Optional[str]) -> Optional[float]:
    if not iso_str:
        return None
    return datetime.fromisoformat(iso_str).timestamp()


class GestureWorker:
    def __init__(self, frame_store: FrameStore, perception_state: PerceptionState, interval_sec: float = 0.05):
        self.frame_store = frame_store
        self.perception_state = perception_state
        self.interval_sec = interval_sec
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

    def _encode_frame_to_b64(self, frame_bgr):
        ok, encoded = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            return None
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    def _run(self):
        print("GestureWorker started", flush=True)

        while self.running:
            try:
                packet = self.frame_store.get_latest()
                if packet is None:
                    time.sleep(self.interval_sec)
                    continue

                if self.last_processed_frame_id == packet.frame_id:
                    time.sleep(self.interval_sec)
                    continue

                self.last_processed_frame_id = packet.frame_id

                worker_start_timestamp = datetime.now(timezone.utc).isoformat()
                worker_start_ts = iso_to_ts(worker_start_timestamp)
                server_ingest_ts = iso_to_ts(packet.server_ingest_timestamp)

                frame = packet.frame_bgr
                active_model = get_active_gesture_model()

                image_b64 = self._encode_frame_to_b64(frame)
                if image_b64 is None:
                    time.sleep(self.interval_sec)
                    continue

                payload = {
                    "timestamp_utc": packet.server_ingest_timestamp,
                    "session_id": "live_session_001",
                    "frame_id": packet.frame_id,
                    "source_id": "live_client",
                    "image": {
                        "encoding": "base64_jpeg",
                        "data": image_b64,
                    },
                    "meta": {
                        "mode": "live",
                        "input_kind": "frame",
                    },
                }

                prediction = None
                backend_inference_latency_ms = None

                try:
                    gesture_url = get_active_gesture_url()
                    r = requests.post(f"{gesture_url}/predict", json=payload, timeout=5)
                    if r.status_code == 200:
                        prediction = r.json()
                        backend_inference_latency_ms = prediction.get("latency_ms")
                    else:
                        print(f"GestureWorker backend non-200: {r.status_code} {r.text[:300]}", flush=True)
                except Exception as e:
                    print(f"GestureWorker backend request failed: {repr(e)}", flush=True)
                    prediction = None

                worker_finish_timestamp = datetime.now(timezone.utc).isoformat()
                worker_finish_ts = iso_to_ts(worker_finish_timestamp)

                face_detected = prediction.get("face_detected") if prediction else False

                result = {
                    "frame_id": packet.frame_id,
                    "client_capture_timestamp": packet.client_capture_timestamp,
                    "server_ingest_timestamp": packet.server_ingest_timestamp,
                    "worker_start_timestamp": worker_start_timestamp,
                    "worker_finish_timestamp": worker_finish_timestamp,
                    "active_model": active_model,
                    "face_detected": face_detected,
                    "bbox_xyxy": None,
                    "prediction": prediction,
                }

                metrics = {
                    "frame_id": packet.frame_id,
                    "active_model": active_model,
                    "face_detected": face_detected,
                    "worker_queue_delay_ms": round((worker_start_ts - server_ingest_ts) * 1000, 2) if server_ingest_ts else None,
                    "backend_inference_latency_ms": backend_inference_latency_ms,
                    "server_pipeline_latency_ms": round((worker_finish_ts - server_ingest_ts) * 1000, 2) if server_ingest_ts else None,
                    "end_to_end_latency_ms": None,
                    "end_to_end_latency_note": "not computed on server because client and server clocks are not guaranteed synchronized",
                }

                self.perception_state.update_gesture(result, metrics)

            except Exception as e:
                print(f"GestureWorker exception: {repr(e)}", flush=True)

            time.sleep(self.interval_sec)