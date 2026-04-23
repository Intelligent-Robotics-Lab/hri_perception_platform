import threading
import time
from typing import Optional

import cv2
import numpy as np

from app.ingest.frame_store import FrameStore
from app.ingest.transport_adapters.webrtc_ingest import WebRTCIngestAdapter

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


class GstVideoBridge:
    """
    Server-side GStreamer receive bridge.

    Receives video from a WebRTC session using webrtcsrc, decodes it,
    pulls frames from appsink, JPEG-encodes them, and forwards them into
    FrameStore through the transport adapter.
    """

    def __init__(
        self,
        frame_store: FrameStore,
        signaling_host: str,
        signaling_port: int = 8443,
        use_tls: bool = False,
    ):
        self.frame_store = frame_store
        self.ingest_adapter = WebRTCIngestAdapter(frame_store=frame_store, audio_store=None)

        self.signaling_host = signaling_host
        self.signaling_port = signaling_port
        self.use_tls = use_tls

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.pipeline = None
        self.appsink = None

        Gst.init(None)

    def _build_pipeline(self) -> str:
        scheme = "gstwebrtcs" if self.use_tls else "gstwebrtc"
        uri = f"{scheme}://{self.signaling_host}:{self.signaling_port}"

        return (
            f'webrtcsrc uri="{uri}" connect-to-first-producer=true '
            f'! decodebin '
            f'! videoconvert '
            f'! video/x-raw,format=BGR '
            f'! appsink name=appsink emit-signals=false sync=false max-buffers=1 drop=true'
        )

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)

    def _pull_sample_bytes(self):
        sample = self.appsink.emit("try-pull-sample", 100000000)  # 100 ms
        if sample is None:
            return None

        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)

        width = structure.get_value("width")
        height = structure.get_value("height")

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return None

        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                return None
            return encoded.tobytes()
        finally:
            buf.unmap(map_info)

    def _run(self):
        pipeline_str = self._build_pipeline()
        print(f"GstVideoBridge pipeline: {pipeline_str}", flush=True)

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("appsink")

        if self.appsink is None:
            raise RuntimeError("appsink not found in GstVideoBridge pipeline")

        self.pipeline.set_state(Gst.State.PLAYING)

        try:
            while self.running:
                frame_bytes = self._pull_sample_bytes()
                if frame_bytes is None:
                    time.sleep(0.01)
                    continue

                self.ingest_adapter.ingest_video_frame(
                    file_bytes=frame_bytes,
                    client_capture_timestamp=None,
                    source_id="gst_webrtc_bridge",
                )
        finally:
            self.pipeline.set_state(Gst.State.NULL)