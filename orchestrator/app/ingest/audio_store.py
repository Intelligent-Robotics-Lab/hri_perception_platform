import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class AudioPacket:
    chunk_id: int
    client_capture_timestamp: Optional[str]
    server_ingest_timestamp: str
    source_id: Optional[str]
    audio_bytes: bytes
    sample_rate_hz: Optional[int] = None
    channels: Optional[int] = None
    encoding: Optional[str] = None


class AudioStore:
    """
    Latest-audio store for low-latency live processing.

    This store intentionally keeps only the latest chunk/window for the
    current bootstrap stage. We do not build an unbounded queue here.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest: Optional[AudioPacket] = None
        self._next_chunk_id = 0

    def update_from_bytes(
        self,
        audio_bytes: bytes,
        client_capture_timestamp: Optional[str] = None,
        source_id: Optional[str] = None,
        sample_rate_hz: Optional[int] = None,
        channels: Optional[int] = None,
        encoding: Optional[str] = None,
    ) -> AudioPacket:
        with self._lock:
            packet = AudioPacket(
                chunk_id=self._next_chunk_id,
                client_capture_timestamp=client_capture_timestamp,
                server_ingest_timestamp=datetime.now(timezone.utc).isoformat(),
                source_id=source_id,
                audio_bytes=audio_bytes,
                sample_rate_hz=sample_rate_hz,
                channels=channels,
                encoding=encoding,
            )
            self._latest = packet
            self._next_chunk_id += 1
            return packet

    def get_latest(self) -> Optional[AudioPacket]:
        with self._lock:
            return self._latest