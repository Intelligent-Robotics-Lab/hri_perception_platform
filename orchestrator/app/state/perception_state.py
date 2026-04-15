import threading
from typing import Optional


class PerceptionState:
    def __init__(self):
        self._lock = threading.Lock()
        self._emotion_result: Optional[dict] = None

    def update_emotion(self, result: dict):
        with self._lock:
            self._emotion_result = result

    def get_emotion(self) -> Optional[dict]:
        with self._lock:
            return self._emotion_result