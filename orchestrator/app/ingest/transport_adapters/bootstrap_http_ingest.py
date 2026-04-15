from typing import Optional

from app.ingest.frame_store import FrameStore, FramePacket


class BootstrapHttpFrameIngestAdapter:
    def __init__(self, frame_store: FrameStore):
        self.frame_store = frame_store

    def ingest(self, file_bytes: bytes, client_capture_timestamp: Optional[str] = None) -> FramePacket:
        return self.frame_store.update_from_bytes(
            file_bytes=file_bytes,
            client_capture_timestamp=client_capture_timestamp,
        )