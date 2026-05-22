import threading
from typing import Callable, Optional


class PerceptionState:
    def __init__(self):
        self._lock = threading.Lock()

        self._affect_result = None
        self._affect_metrics = None
        
        self._emotion_result: Optional[dict] = None
        self._emotion_metrics: Optional[dict] = None

        self._asr_result: Optional[dict] = None
        self._asr_metrics: Optional[dict] = None

        self._listeners: list[Callable[[dict], None]] = []

    def add_listener(self, listener: Callable[[dict], None]):
        with self._lock:
            self._listeners.append(listener)

    def _emit(self, event: dict):
        with self._lock:
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(event)
            except Exception:
                pass

    def update_affect(self, result: dict, metrics: dict):
        with self._lock:
            self._affect_result = result
            self._affect_metrics = metrics

    def get_affect(self):
        with self._lock:
            return self._affect_result

    def get_affect_metrics(self):
        with self._lock:
            return self._affect_metrics

    def update_emotion(self, result: dict, metrics: dict):
        with self._lock:
            self._emotion_result = result
            self._emotion_metrics = metrics

        prediction = result.get("prediction") or {}
        event = {
            "event_type": "emotion_update",
            "timestamp_utc": result.get("worker_finish_timestamp"),
            "source_id": "media_gateway_video",
            "payload": {
                "frame_id": result.get("frame_id"),
                "active_model": result.get("active_model"),
                "face_detected": result.get("face_detected"),
                "bbox_xyxy": result.get("bbox_xyxy"),
                "prediction": prediction,
                "metrics": metrics,
            },
        }
        self._emit(event)

    def get_emotion(self) -> Optional[dict]:
        with self._lock:
            return self._emotion_result

    def get_emotion_metrics(self) -> Optional[dict]:
        with self._lock:
            return self._emotion_metrics

    def update_asr(self, result: dict, metrics: dict):
        with self._lock:
            self._asr_result = result
            self._asr_metrics = metrics

        backend_result = result.get("backend_result") or {}
        event = {
            "event_type": "asr_update",
            "timestamp_utc": result.get("worker_finish_timestamp"),
            "source_id": result.get("source_id"),
            "payload": {
                "chunk_id": result.get("chunk_id"),
                "active_model": result.get("active_model"),
                "transcript": backend_result.get("transcript"),
                "backend_result": backend_result,
                "metrics": metrics,
            },
        }
        self._emit(event)

    def get_asr(self) -> Optional[dict]:
        with self._lock:
            return self._asr_result

    def get_asr_metrics(self) -> Optional[dict]:
        with self._lock:
            return self._asr_metrics