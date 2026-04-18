from typing import Optional

from app.ingest.audio_store import AudioStore
from app.ingest.frame_store import FrameStore


class WebRTCIngestAdapter:
    """
    Production transport adapter target.

    This adapter will eventually receive live audio/video from a WebRTC session
    and populate the platform's latest media stores.

    The perception platform downstream of this adapter should not need to know
    whether media arrived from bootstrap HTTP, replay, or WebRTC.
    """

    def __init__(self, frame_store: FrameStore, audio_store: AudioStore):
        self.frame_store = frame_store
        self.audio_store = audio_store

    def ingest_video_frame(
        self,
        file_bytes: bytes,
        client_capture_timestamp: Optional[str] = None,
        source_id: Optional[str] = None,
    ):
        return self.frame_store.update_from_bytes(
            file_bytes=file_bytes,
            client_capture_timestamp=client_capture_timestamp,
        )

    def ingest_audio_chunk(
        self,
        audio_bytes: bytes,
        client_capture_timestamp: Optional[str] = None,
        source_id: Optional[str] = None,
        sample_rate_hz: Optional[int] = None,
        channels: Optional[int] = None,
        encoding: Optional[str] = None,
    ):
        return self.audio_store.update_from_bytes(
            audio_bytes=audio_bytes,
            client_capture_timestamp=client_capture_timestamp,
            source_id=source_id,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            encoding=encoding,
        )