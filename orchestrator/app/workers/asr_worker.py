import time
from datetime import datetime, timezone
from typing import Optional

import requests

from app.ingest.audio_store import AudioStore
from app.routers.asr_router import get_active_asr_model, get_active_asr_url
from app.state.perception_state import PerceptionState


class ASRWorker:
    """
    Streaming-oriented ASR worker scaffold.

    Reads the latest audio chunk from AudioStore, calls the active ASR backend,
    and writes the latest ASR state and metrics.
    """

    def __init__(self, audio_store: AudioStore, perception_state: PerceptionState, interval_sec: float = 0.05):
        self.audio_store = audio_store
        self.perception_state = perception_state
        self.interval_sec = interval_sec
        self.running = False
        self.thread = None
        self.last_processed_chunk_id: Optional[int] = None

    def start(self):
        import threading

        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _run(self):
        print("ASRWorker started", flush=True)

        while self.running:
            try:
                packet = self.audio_store.get_latest()
                if packet is None:
                    time.sleep(self.interval_sec)
                    continue

                if self.last_processed_chunk_id == packet.chunk_id:
                    time.sleep(self.interval_sec)
                    continue

                self.last_processed_chunk_id = packet.chunk_id

                worker_start_timestamp = datetime.now(timezone.utc).isoformat()
                active_model = get_active_asr_model()

                files = {
                    "file": ("audio_chunk.bin", packet.audio_bytes, "application/octet-stream")
                }

                data = {
                    "client_capture_timestamp": packet.client_capture_timestamp,
                    "server_ingest_timestamp": packet.server_ingest_timestamp,
                    "source_id": packet.source_id,
                    "sample_rate_hz": packet.sample_rate_hz,
                    "channels": packet.channels,
                    "encoding": packet.encoding,
                }

                backend_result = None
                backend_inference_latency_ms = None

                try:
                    asr_url = get_active_asr_url()
                    r = requests.post(f"{asr_url}/transcribe", files=files, data=data, timeout=5)
                    if r.status_code == 200:
                        backend_result = r.json()
                        backend_inference_latency_ms = backend_result.get("latency_ms")
                except Exception as e:
                    print(f"ASRWorker backend request failed: {repr(e)}", flush=True)

                worker_finish_timestamp = datetime.now(timezone.utc).isoformat()

                result = {
                    "chunk_id": packet.chunk_id,
                    "client_capture_timestamp": packet.client_capture_timestamp,
                    "server_ingest_timestamp": packet.server_ingest_timestamp,
                    "worker_start_timestamp": worker_start_timestamp,
                    "worker_finish_timestamp": worker_finish_timestamp,
                    "source_id": packet.source_id,
                    "sample_rate_hz": packet.sample_rate_hz,
                    "channels": packet.channels,
                    "encoding": packet.encoding,
                    "active_model": active_model,
                    "backend_result": backend_result,
                }

                metrics = {
                    "chunk_id": packet.chunk_id,
                    "active_model": active_model,
                    "backend_inference_latency_ms": backend_inference_latency_ms,
                    "server_pipeline_latency_ms": None,
                    "end_to_end_latency_ms": None,
                    "end_to_end_latency_note": "not computed on server because client and server clocks are not guaranteed synchronized",
                }

                self.perception_state.update_asr(result, metrics)

            except Exception as e:
                print(f"ASRWorker exception: {repr(e)}", flush=True)

            time.sleep(self.interval_sec)