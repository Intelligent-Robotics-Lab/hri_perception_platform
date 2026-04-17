import threading
from typing import Optional


class PerceptionState:
    def __init__(self):
        self._lock = threading.Lock()

        self._emotion_result: Optional[dict] = None
        self._emotion_metrics: Optional[dict] = None

        self._asr_result: Optional[dict] = None
        self._asr_metrics: Optional[dict] = None

    def update_emotion(self, result: dict, metrics: dict):
        with self._lock:
            self._emotion_result = result
            self._emotion_metrics = metrics

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

    def get_asr(self) -> Optional[dict]:
        with self._lock:
            return self._asr_result

    def get_asr_metrics(self) -> Optional[dict]:
        with self._lock:
            return self._asr_metrics