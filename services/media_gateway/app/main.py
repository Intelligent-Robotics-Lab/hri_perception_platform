import os
import threading
import queue
from datetime import datetime, timezone

import cv2
import numpy as np
import requests
from fastapi import FastAPI

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version("GLib", "2.0")

from gi.repository import Gst, GLib

app = FastAPI(title="media_gateway")

Gst.init(None)

worker = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VideoForwarder:
    def __init__(
        self,
        signaling_host: str,
        signaling_port: int,
        use_tls: bool,
        orchestrator_ingest_url: str,
    ):
        self.signaling_host = signaling_host
        self.signaling_port = signaling_port
        self.use_tls = use_tls
        self.orchestrator_ingest_url = orchestrator_ingest_url

        self.pipeline = None
        self.webrtcsrc = None
        self.decodebin = None
        self.appsink = None
        self.mainloop = None
        self.loop_thread = None
        self.forward_thread = None
        self.running = False

        self.frame_queue = queue.Queue(maxsize=1)

        self._status_lock = threading.Lock()
        self._status = {
            "connected": False,
            "pipeline_state": "null",
            "last_sample_timestamp_utc": None,
            "last_forward_timestamp_utc": None,
            "last_forward_status": None,
            "forwarded_frame_count": 0,
            "last_error": None,
        }

    def _set_status(self, **kwargs):
        with self._status_lock:
            self._status.update(kwargs)

    def get_status(self):
        with self._status_lock:
            return dict(self._status)

    def _build_pipeline(self) -> str:
        scheme = "wss" if self.use_tls else "ws"
        signaller_uri = f"{scheme}://{self.signaling_host}:{self.signaling_port}"

        return (
            f'webrtcsrc name=wsrc signaller::uri="{signaller_uri}" '
            f'connect-to-first-producer=true '
            f'! decodebin name=decode '
            f'! videoconvert '
            f'! video/x-raw,format=BGR '
            f'! appsink name=appsink emit-signals=true sync=false max-buffers=1 drop=true'
        )

    def start(self):
        if self.running:
            return

        self.running = True

        self.loop_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.loop_thread.start()

        self.forward_thread = threading.Thread(target=self._run_forwarder, daemon=True)
        self.forward_thread.start()

    def stop(self):
        self.running = False

        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)

        if self.mainloop is not None:
            self.mainloop.quit()

    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        self._set_status(
            connected=True,
            last_sample_timestamp_utc=utc_now_iso(),
            last_error=None,
        )

        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)

        width = structure.get_value("width")
        height = structure.get_value("height")

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            self._set_status(last_error="failed to map buffer")
            return Gst.FlowReturn.OK

        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3))
            ok, encoded = cv2.imencode(".jpg", frame)
            if ok:
                jpeg_bytes = encoded.tobytes()

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    self.frame_queue.put_nowait(jpeg_bytes)
                except queue.Full:
                    pass
            else:
                self._set_status(last_error="jpeg encode failed")
        except Exception as e:
            self._set_status(last_error=f"sample conversion error: {repr(e)}")
        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    def _on_state_changed(self, message):
        if message.src != self.pipeline:
            return

        old_state, new_state, pending_state = message.parse_state_changed()
        self._set_status(
            pipeline_state=new_state.value_nick,
            connected=new_state == Gst.State.PLAYING,
        )

    def _on_bus_message(self, bus, message):
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[media_gateway] GStreamer ERROR: {err}; debug={debug}", flush=True)
            self._set_status(
                connected=False,
                last_error=f"{err}; debug={debug}",
            )
            if self.mainloop is not None:
                self.mainloop.quit()

        elif msg_type == Gst.MessageType.EOS:
            print("[media_gateway] GStreamer EOS", flush=True)
            self._set_status(
                connected=False,
                last_error="EOS received",
            )
            if self.mainloop is not None:
                self.mainloop.quit()

        elif msg_type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"[media_gateway] GStreamer WARNING: {err}; debug={debug}", flush=True)
            self._set_status(last_error=f"warning: {err}; debug={debug}")

        elif msg_type == Gst.MessageType.STATE_CHANGED:
            self._on_state_changed(message)

        return True

    def _run_pipeline(self):
        pipeline_str = self._build_pipeline()
        print(f"[media_gateway] pipeline: {pipeline_str}", flush=True)

        self.pipeline = Gst.parse_launch(pipeline_str)
        self.webrtcsrc = self.pipeline.get_by_name("wsrc")
        self.decodebin = self.pipeline.get_by_name("decode")
        self.appsink = self.pipeline.get_by_name("appsink")

        if self.appsink is None:
            raise RuntimeError("appsink not found in media_gateway pipeline")

        self.appsink.connect("new-sample", self._on_new_sample)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self.mainloop = GLib.MainLoop()

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        self._set_status(pipeline_state=ret.value_nick)

        try:
            self.mainloop.run()
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            self._set_status(connected=False, pipeline_state="null")

    def _forward_frame(self, jpeg_bytes: bytes):
        files = {
            "file": ("frame.jpg", jpeg_bytes, "image/jpeg")
        }
        data = {
            "client_capture_timestamp": utc_now_iso()
        }

        r = requests.post(
            self.orchestrator_ingest_url,
            files=files,
            data=data,
            timeout=2.0,
        )
        return r.status_code

    def _run_forwarder(self):
        while self.running:
            try:
                jpeg_bytes = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                status = self._forward_frame(jpeg_bytes)
                now = utc_now_iso()

                with self._status_lock:
                    self._status["last_forward_timestamp_utc"] = now
                    self._status["last_forward_status"] = status
                    if status == 200:
                        self._status["forwarded_frame_count"] += 1

                if status != 200:
                    print(f"[media_gateway] ingest/frame returned {status}", flush=True)

            except Exception as e:
                self._set_status(last_error=f"forward error: {repr(e)}")
                print(f"[media_gateway] forward error: {repr(e)}", flush=True)


@app.on_event("startup")
def startup_event():
    global worker
    worker = VideoForwarder(
        signaling_host=os.getenv("MEDIA_GATEWAY_SIGNALING_HOST", "141.210.144.85"),
        signaling_port=int(os.getenv("MEDIA_GATEWAY_SIGNALING_PORT", "8443")),
        use_tls=os.getenv("MEDIA_GATEWAY_SIGNALING_USE_TLS", "false").lower() == "true",
        orchestrator_ingest_url=os.getenv(
            "MEDIA_GATEWAY_ORCHESTRATOR_INGEST_URL",
            "http://orchestrator:8000/ingest/frame",
        ),
    )
    worker.start()


@app.on_event("shutdown")
def shutdown_event():
    global worker
    if worker is not None:
        worker.stop()


@app.get("/health")
def health():
    return {"status": "ok", "service": "media_gateway"}


@app.get("/status")
def status():
    global worker
    if worker is None:
        return {"status": "not_started"}
    return {"status": "ok", "gateway": worker.get_status()}


@app.get("/metrics")
def metrics():
    global worker
    if worker is None:
        return {"status": "not_started"}

    s = worker.get_status()
    return {
        "status": "ok",
        "metrics": {
            "connected": s["connected"],
            "pipeline_state": s["pipeline_state"],
            "forwarded_frame_count": s["forwarded_frame_count"],
            "last_sample_timestamp_utc": s["last_sample_timestamp_utc"],
            "last_forward_timestamp_utc": s["last_forward_timestamp_utc"],
            "last_forward_status": s["last_forward_status"],
            "last_error": s["last_error"],
        },
    }