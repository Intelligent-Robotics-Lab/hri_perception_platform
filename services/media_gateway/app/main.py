import io
import os
import queue
import threading
import wave
from datetime import datetime, timezone

import requests
from fastapi import FastAPI

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")

from gi.repository import Gst, GLib

app = FastAPI(title="media_gateway")

Gst.init(None)

gateway_session = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MediaGatewaySession:
    def __init__(
        self,
        signaling_host: str,
        signaling_port: int,
        use_tls: bool,
        orchestrator_video_ingest_url: str,
        orchestrator_audio_ingest_url: str,
        audio_chunk_ms: int = 1000,
        audio_sample_rate_hz: int = 16000,
        audio_channels: int = 1,
    ):
        self.signaling_host = signaling_host
        self.signaling_port = signaling_port
        self.use_tls = use_tls
        self.orchestrator_video_ingest_url = orchestrator_video_ingest_url
        self.orchestrator_audio_ingest_url = orchestrator_audio_ingest_url
        self.audio_chunk_ms = audio_chunk_ms
        self.audio_sample_rate_hz = audio_sample_rate_hz
        self.audio_channels = audio_channels
        self.audio_sample_width_bytes = 2  # S16LE

        self.pipeline = None
        self.webrtcsrc = None
        self.mainloop = None
        self.loop_thread = None
        self.running = False

        self.video_forward_thread = None
        self.audio_forward_thread = None

        self.video_queue = queue.Queue(maxsize=2)
        self.audio_queue = queue.Queue(maxsize=4)
        self.audio_buffer = bytearray()
        self.audio_target_chunk_bytes = int(
            self.audio_sample_rate_hz
            * self.audio_channels
            * self.audio_sample_width_bytes
            * self.audio_chunk_ms
            / 1000
        )

        self.video_branch_built = False
        self.audio_branch_built = False

        self.video_queue_el = None
        self.video_decodebin = None
        self.video_convert = None
        self.video_capsfilter = None
        self.video_jpegenc = None
        self.video_appsink = None

        self.audio_queue_el = None
        self.audio_decodebin = None
        self.audio_convert = None
        self.audio_resample = None
        self.audio_capsfilter = None
        self.audio_appsink = None

        self._status_lock = threading.Lock()
        self._status = {
            "session": {
                "connected": False,
                "pipeline_state": "null",
                "last_error": None,
                "last_warning": None,
            },
            "video": {
                "pad_seen": False,
                "branch_built": False,
                "last_caps": None,
                "last_sample_timestamp_utc": None,
                "last_forward_timestamp_utc": None,
                "last_forward_status": None,
                "forwarded_frame_count": 0,
            },
            "audio": {
                "pad_seen": False,
                "branch_built": False,
                "last_caps": None,
                "last_sample_timestamp_utc": None,
                "last_forward_timestamp_utc": None,
                "last_forward_status": None,
                "forwarded_audio_chunk_count": 0,
            },
        }

    def _set_session_status(self, **kwargs):
        with self._status_lock:
            self._status["session"].update(kwargs)

    def _set_video_status(self, **kwargs):
        with self._status_lock:
            self._status["video"].update(kwargs)

    def _set_audio_status(self, **kwargs):
        with self._status_lock:
            self._status["audio"].update(kwargs)

    def get_status(self):
        with self._status_lock:
            return {
                "session": dict(self._status["session"]),
                "video": dict(self._status["video"]),
                "audio": dict(self._status["audio"]),
            }

    def _signaller_uri(self) -> str:
        scheme = "wss" if self.use_tls else "ws"
        return f"{scheme}://{self.signaling_host}:{self.signaling_port}"

    def start(self):
        if self.running:
            return

        self.running = True
        self.loop_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.loop_thread.start()

        self.video_forward_thread = threading.Thread(
            target=self._run_video_forwarder, daemon=True
        )
        self.video_forward_thread.start()

        self.audio_forward_thread = threading.Thread(
            target=self._run_audio_forwarder, daemon=True
        )
        self.audio_forward_thread.start()

    def stop(self):
        self.running = False

        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)

        if self.mainloop is not None:
            self.mainloop.quit()

    def _caps_to_string(self, caps):
        if caps is None:
            return None
        try:
            return caps.to_string()
        except Exception:
            return None

    def _pad_caps_string(self, pad):
        caps = pad.get_current_caps() or pad.query_caps(None)
        return self._caps_to_string(caps)

    def _link_or_raise(self, src_pad, sink_pad, label: str):
        result = src_pad.link(sink_pad)
        if result != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"{label} link failed: {result.value_nick}")
        print(f"[media_gateway] {label} link ok", flush=True)

    def _element_or_raise(self, factory_name: str, name: str):
        el = Gst.ElementFactory.make(factory_name, name)
        if el is None:
            raise RuntimeError(f"failed to create element {factory_name}:{name}")
        return el

    # ---------------------------
    # Video path
    # ---------------------------

    def _on_video_appsink_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        self._set_video_status(last_sample_timestamp_utc=utc_now_iso())

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            self._set_session_status(last_error="failed to map jpeg buffer")
            return Gst.FlowReturn.OK

        try:
            jpeg_bytes = bytes(map_info.data)

            if self.video_queue.full():
                try:
                    self.video_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.video_queue.put_nowait(jpeg_bytes)
            except queue.Full:
                pass

        except Exception as e:
            self._set_session_status(last_error=f"jpeg sample error: {repr(e)}")
            print(f"[media_gateway] jpeg sample error: {repr(e)}", flush=True)
        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def _on_video_decodebin_pad_added(self, decodebin, pad):
        caps_str = self._pad_caps_string(pad)
        print(
            f"[media_gateway] video decodebin pad-added: {pad.get_name()} caps={caps_str}",
            flush=True,
        )

        if caps_str is None or "video/" not in caps_str.lower():
            return

        sink_pad = self.video_convert.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return

        self._link_or_raise(pad, sink_pad, "video decodebin -> videoconvert")

    def _build_video_branch(self, src_pad):
        if self.video_branch_built:
            return

        caps_str = self._pad_caps_string(src_pad)
        self._set_video_status(pad_seen=True, last_caps=caps_str)
        print(f"[media_gateway] building video branch for caps={caps_str}", flush=True)

        self.video_queue_el = self._element_or_raise("queue", "video_queue")
        self.video_decodebin = self._element_or_raise("decodebin", "video_decode")
        self.video_convert = self._element_or_raise("videoconvert", "video_convert")
        self.video_capsfilter = self._element_or_raise("capsfilter", "video_caps")
        self.video_jpegenc = self._element_or_raise("jpegenc", "video_jpegenc")
        self.video_appsink = self._element_or_raise("appsink", "video_appsink")

        self.video_queue_el.set_property("max-size-buffers", 4)
        self.video_queue_el.set_property("leaky", 2)  # downstream

        self.video_capsfilter.set_property(
            "caps", Gst.Caps.from_string("video/x-raw,format=I420")
        )

        self.video_jpegenc.set_property("quality", 95)

        self.video_appsink.set_property("emit-signals", True)
        self.video_appsink.set_property("sync", False)
        self.video_appsink.set_property("max-buffers", 1)
        self.video_appsink.set_property("drop", True)

        self.video_appsink.connect("new-sample", self._on_video_appsink_sample)
        self.video_decodebin.connect("pad-added", self._on_video_decodebin_pad_added)

        for el in [
            self.video_queue_el,
            self.video_decodebin,
            self.video_convert,
            self.video_capsfilter,
            self.video_jpegenc,
            self.video_appsink,
        ]:
            self.pipeline.add(el)

        if not self.video_queue_el.link(self.video_decodebin):
            raise RuntimeError("failed to link video_queue -> video_decodebin")

        if not self.video_convert.link(self.video_capsfilter):
            raise RuntimeError("failed to link videoconvert -> video_capsfilter")

        if not self.video_capsfilter.link(self.video_jpegenc):
            raise RuntimeError("failed to link video_capsfilter -> jpegenc")

        if not self.video_jpegenc.link(self.video_appsink):
            raise RuntimeError("failed to link jpegenc -> video_appsink")

        queue_sink_pad = self.video_queue_el.get_static_pad("sink")
        if queue_sink_pad is None:
            raise RuntimeError("video_queue sink pad not found")

        self._link_or_raise(src_pad, queue_sink_pad, "webrtc video -> video_queue")

        for el in [
            self.video_queue_el,
            self.video_decodebin,
            self.video_convert,
            self.video_capsfilter,
            self.video_jpegenc,
            self.video_appsink,
        ]:
            el.sync_state_with_parent()

        self.video_branch_built = True
        self._set_video_status(branch_built=True)

    def _forward_video_frame(self, jpeg_bytes: bytes):
        files = {"file": ("frame.jpg", jpeg_bytes, "image/jpeg")}
        data = {"client_capture_timestamp": utc_now_iso()}

        r = requests.post(
            self.orchestrator_video_ingest_url,
            files=files,
            data=data,
            timeout=2.0,
        )
        return r.status_code

    def _run_video_forwarder(self):
        while self.running:
            try:
                jpeg_bytes = self.video_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                status = self._forward_video_frame(jpeg_bytes)
                now = utc_now_iso()

                with self._status_lock:
                    self._status["video"]["last_forward_timestamp_utc"] = now
                    self._status["video"]["last_forward_status"] = status
                    if status == 200:
                        self._status["video"]["forwarded_frame_count"] += 1

                if status != 200:
                    print(
                        f"[media_gateway] video ingest/frame returned {status}",
                        flush=True,
                    )

            except Exception as e:
                self._set_session_status(last_error=f"video forward error: {repr(e)}")
                print(f"[media_gateway] video forward error: {repr(e)}", flush=True)

    # ---------------------------
    # Audio path
    # ---------------------------

    def _make_wav_bytes(self, pcm_bytes: bytes) -> bytes:
        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(self.audio_channels)
            wf.setsampwidth(self.audio_sample_width_bytes)
            wf.setframerate(self.audio_sample_rate_hz)
            wf.writeframes(pcm_bytes)
        return bio.getvalue()

    def _enqueue_audio_chunks_if_ready(self):
        while len(self.audio_buffer) >= self.audio_target_chunk_bytes:
            pcm_chunk = bytes(self.audio_buffer[: self.audio_target_chunk_bytes])
            del self.audio_buffer[: self.audio_target_chunk_bytes]

            wav_bytes = self._make_wav_bytes(pcm_chunk)

            if self.audio_queue.full():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.audio_queue.put_nowait(wav_bytes)
            except queue.Full:
                pass

    def _on_audio_appsink_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        self._set_audio_status(last_sample_timestamp_utc=utc_now_iso())

        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            self._set_session_status(last_error="failed to map audio buffer")
            return Gst.FlowReturn.OK

        try:
            self.audio_buffer.extend(map_info.data)
            self._enqueue_audio_chunks_if_ready()
        except Exception as e:
            self._set_session_status(last_error=f"audio buffering error: {repr(e)}")
        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    def _on_audio_decodebin_pad_added(self, decodebin, pad):
        caps_str = self._pad_caps_string(pad)
        print(
            f"[media_gateway] audio decodebin pad-added: {pad.get_name()} caps={caps_str}",
            flush=True,
        )

        if caps_str is None or "audio/" not in caps_str.lower():
            return

        sink_pad = self.audio_convert.get_static_pad("sink")
        if sink_pad is None or sink_pad.is_linked():
            return

        self._link_or_raise(pad, sink_pad, "audio decodebin -> audioconvert")

    def _build_audio_branch(self, src_pad):
        if self.audio_branch_built:
            return

        caps_str = self._pad_caps_string(src_pad)
        self._set_audio_status(pad_seen=True, last_caps=caps_str)
        print(f"[media_gateway] building audio branch for caps={caps_str}", flush=True)

        self.audio_queue_el = self._element_or_raise("queue", "audio_queue")
        self.audio_decodebin = self._element_or_raise("decodebin", "audio_decode")
        self.audio_convert = self._element_or_raise("audioconvert", "audio_convert")
        self.audio_resample = self._element_or_raise("audioresample", "audio_resample")
        self.audio_capsfilter = self._element_or_raise("capsfilter", "audio_caps")
        self.audio_appsink = self._element_or_raise("appsink", "audio_appsink")

        self.audio_queue_el.set_property("max-size-buffers", 8)
        self.audio_queue_el.set_property("leaky", 2)  # downstream

        self.audio_capsfilter.set_property(
            "caps",
            Gst.Caps.from_string(
                f"audio/x-raw,format=S16LE,channels={self.audio_channels},"
                f"rate={self.audio_sample_rate_hz},layout=interleaved"
            ),
        )

        self.audio_appsink.set_property("emit-signals", True)
        self.audio_appsink.set_property("sync", False)
        self.audio_appsink.set_property("max-buffers", 16)
        self.audio_appsink.set_property("drop", True)

        self.audio_appsink.connect("new-sample", self._on_audio_appsink_sample)
        self.audio_decodebin.connect("pad-added", self._on_audio_decodebin_pad_added)

        for el in [
            self.audio_queue_el,
            self.audio_decodebin,
            self.audio_convert,
            self.audio_resample,
            self.audio_capsfilter,
            self.audio_appsink,
        ]:
            self.pipeline.add(el)

        if not self.audio_queue_el.link(self.audio_decodebin):
            raise RuntimeError("failed to link audio_queue -> audio_decodebin")

        if not self.audio_convert.link(self.audio_resample):
            raise RuntimeError("failed to link audioconvert -> audioresample")

        if not self.audio_resample.link(self.audio_capsfilter):
            raise RuntimeError("failed to link audioresample -> audio_capsfilter")

        if not self.audio_capsfilter.link(self.audio_appsink):
            raise RuntimeError("failed to link audio_capsfilter -> audio_appsink")

        queue_sink_pad = self.audio_queue_el.get_static_pad("sink")
        if queue_sink_pad is None:
            raise RuntimeError("audio_queue sink pad not found")

        self._link_or_raise(src_pad, queue_sink_pad, "webrtc audio -> audio_queue")

        for el in [
            self.audio_queue_el,
            self.audio_decodebin,
            self.audio_convert,
            self.audio_resample,
            self.audio_capsfilter,
            self.audio_appsink,
        ]:
            el.sync_state_with_parent()

        self.audio_branch_built = True
        self._set_audio_status(branch_built=True)

    def _forward_audio_chunk(self, wav_bytes: bytes):
        files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
        data = {
            "client_capture_timestamp": utc_now_iso(),
            "source_id": "media_gateway_audio",
            "sample_rate_hz": str(self.audio_sample_rate_hz),
            "channels": str(self.audio_channels),
            "encoding": "wav",
        }

        r = requests.post(
            self.orchestrator_audio_ingest_url,
            files=files,
            data=data,
            timeout=3.0,
        )
        return r.status_code

    def _run_audio_forwarder(self):
        while self.running:
            try:
                wav_bytes = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                status = self._forward_audio_chunk(wav_bytes)
                now = utc_now_iso()

                with self._status_lock:
                    self._status["audio"]["last_forward_timestamp_utc"] = now
                    self._status["audio"]["last_forward_status"] = status
                    if status == 200:
                        self._status["audio"]["forwarded_audio_chunk_count"] += 1

                if status != 200:
                    print(
                        f"[media_gateway] audio ingest/audio returned {status}",
                        flush=True,
                    )

            except Exception as e:
                self._set_session_status(last_error=f"audio forward error: {repr(e)}")
                print(f"[media_gateway] audio forward error: {repr(e)}", flush=True)

    # ---------------------------
    # Shared session / bus
    # ---------------------------

    def _on_webrtc_pad_added(self, element, pad):
        caps_str = self._pad_caps_string(pad)
        print(
            f"[media_gateway] webrtc pad-added: {pad.get_name()} caps={caps_str}",
            flush=True,
        )

        if not caps_str:
            return

        caps_lower = caps_str.lower()

        try:
            if "video/" in caps_lower or "h264" in caps_lower or "vp8" in caps_lower or "vp9" in caps_lower:
                self._build_video_branch(pad)
                return

            if "audio/" in caps_lower or "opus" in caps_lower:
                self._build_audio_branch(pad)
                return

            print("[media_gateway] unsupported shared pad, ignoring", flush=True)
        except Exception as e:
            self._set_session_status(last_error=f"pad routing error: {repr(e)}")
            print(f"[media_gateway] pad routing error: {repr(e)}", flush=True)

    def _on_state_changed(self, message):
        if message.src != self.pipeline:
            return

        _, new_state, _ = message.parse_state_changed()
        self._set_session_status(
            pipeline_state=new_state.value_nick,
            connected=(new_state == Gst.State.PLAYING),
        )

    def _on_bus_message(self, bus, message):
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[media_gateway] GStreamer ERROR: {err}; debug={debug}", flush=True)
            self._set_session_status(
                connected=False,
                last_error=f"{err}; debug={debug}",
            )
            if self.mainloop is not None:
                self.mainloop.quit()

        elif msg_type == Gst.MessageType.EOS:
            print("[media_gateway] GStreamer EOS", flush=True)
            self._set_session_status(
                connected=False,
                last_error="EOS received",
            )
            if self.mainloop is not None:
                self.mainloop.quit()

        elif msg_type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(
                f"[media_gateway] GStreamer WARNING: {err}; debug={debug}",
                flush=True,
            )
            self._set_session_status(last_warning=f"{err}; debug={debug}")

        elif msg_type == Gst.MessageType.STATE_CHANGED:
            self._on_state_changed(message)

        return True

    def _run_pipeline(self):
        signaller_uri = self._signaller_uri()
        print(f'[media_gateway] shared session signaller::uri="{signaller_uri}"', flush=True)

        self.pipeline = Gst.Pipeline.new("media_gateway_pipeline")
        if self.pipeline is None:
            raise RuntimeError("failed to create media_gateway pipeline")

        self.webrtcsrc = self._element_or_raise("webrtcsrc", "shared_wsrc")
        self.webrtcsrc.set_property("connect-to-first-producer", True)
        self.webrtcsrc.set_property("message-forward", True)

        signaller = self.webrtcsrc.get_property("signaller")
        if signaller is None:
            raise RuntimeError("failed to get webrtcsrc signaller")
        signaller.set_property("uri", signaller_uri)

        self.webrtcsrc.connect("pad-added", self._on_webrtc_pad_added)

        self.pipeline.add(self.webrtcsrc)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self.mainloop = GLib.MainLoop()

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        print(f"[media_gateway] set_state returned: {ret.value_nick}", flush=True)
        self._set_session_status(pipeline_state=ret.value_nick)

        try:
            self.mainloop.run()
        finally:
            self.pipeline.set_state(Gst.State.NULL)
            self._set_session_status(connected=False, pipeline_state="null")


@app.on_event("startup")
def startup_event():
    global gateway_session

    gateway_session = MediaGatewaySession(
        signaling_host=os.getenv("MEDIA_GATEWAY_SIGNALING_HOST", "141.210.144.85"),
        signaling_port=int(os.getenv("MEDIA_GATEWAY_SIGNALING_PORT", "8443")),
        use_tls=os.getenv("MEDIA_GATEWAY_SIGNALING_USE_TLS", "false").lower() == "true",
        orchestrator_video_ingest_url=os.getenv(
            "MEDIA_GATEWAY_ORCHESTRATOR_INGEST_URL",
            "http://orchestrator:8000/ingest/frame",
        ),
        orchestrator_audio_ingest_url=os.getenv(
            "MEDIA_GATEWAY_ORCHESTRATOR_AUDIO_INGEST_URL",
            "http://orchestrator:8000/ingest/audio",
        ),
        audio_chunk_ms=int(os.getenv("MEDIA_GATEWAY_AUDIO_CHUNK_MS", "1000")),
        audio_sample_rate_hz=int(
            os.getenv("MEDIA_GATEWAY_AUDIO_SAMPLE_RATE_HZ", "16000")
        ),
        audio_channels=int(os.getenv("MEDIA_GATEWAY_AUDIO_CHANNELS", "1")),
    )
    gateway_session.start()


@app.on_event("shutdown")
def shutdown_event():
    global gateway_session
    if gateway_session is not None:
        gateway_session.stop()


@app.get("/health")
def health():
    return {"status": "ok", "service": "media_gateway"}


@app.get("/status")
def status():
    global gateway_session
    if gateway_session is None:
        return {"status": "not_started"}
    return {"status": "ok", **gateway_session.get_status()}


@app.get("/metrics")
def metrics():
    global gateway_session
    if gateway_session is None:
        return {"status": "not_started"}
    return {"status": "ok", **gateway_session.get_status()}