from abc import ABC, abstractmethod
from typing import Optional

from app.ingest.frame_store import FramePacket
from app.ingest.audio_store import AudioPacket


class RealtimeMediaIngestAdapter(ABC):
    """
    Abstract boundary for future production-grade live media ingest.

    A concrete implementation may later be backed by:
    - GStreamer
    - WebRTC
    - robot-side native media transport
    - another low-latency streaming mechanism

    The rest of the perception platform should not depend on the concrete transport.
    """

    @abstractmethod
    def ingest_video_frame(
        self,
        file_bytes: bytes,
        client_capture_timestamp: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> FramePacket:
        pass

    @abstractmethod
    def ingest_audio_chunk(
        self,
        audio_bytes: bytes,
        client_capture_timestamp: Optional[str] = None,
        source_id: Optional[str] = None,
        sample_rate_hz: Optional[int] = None,
        channels: Optional[int] = None,
        encoding: Optional[str] = None,
    ) -> AudioPacket:
        pass