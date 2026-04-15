import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np


@dataclass
class FramePacket:
    frame_id: int
    client_capture_timestamp: Optional[str]
    server_ingest_timestamp: str
    frame_bgr: object


class FrameStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Optional[FramePacket] = None
        self._next_frame_id = 0

    def update_from_bytes(self, file_bytes: bytes, client_capture_timestamp: Optional[str] = None) -> FramePacket:
        np_buf = np.frombuffer(file_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Failed to decode uploaded image")

        with self._lock:
            packet = FramePacket(
                frame_id=self._next_frame_id,
                client_capture_timestamp=client_capture_timestamp,
                server_ingest_timestamp=datetime.now(timezone.utc).isoformat(),
                frame_bgr=image_bgr,
            )
            self._latest = packet
            self._next_frame_id += 1
            return packet

    def get_latest(self) -> Optional[FramePacket]:
        with self._lock:
            return self._latest