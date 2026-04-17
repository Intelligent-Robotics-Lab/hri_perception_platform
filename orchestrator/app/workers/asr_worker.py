import threading
import time
from datetime import datetime, timezone
from typing import Optional

from app.ingest.audio_store import AudioStore
from app.state.perception_state import PerceptionState


class ASRWorker:
    """
    Scaffold worker for future ASR integration.

    At this stage, this worker does not call a real ASR backend yet.
    It exists to establish the same perception-platform pattern used by emotion:
    - latest ingest store
    - task worker
    - latest task state
    - latency metrics
    """

    def __init__(self, audio_store: AudioStore, perception_state: PerceptionState, interval_sec: float = 0.05):
        self.audio_store = audio_store
        self.perception_state = perception_state
        self.interval_sec = interval_sec
        self.running = False
        self.thread = None
        self.last_processed_chunk_id: Optional[int] = None

    def start(self):
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
                    "transcript": None,
                    "is_partial": None,
                    "backend_name": None,
                }

                metrics = {
                    "chunk_id": packet.chunk_id,
                    "backend_inference_latency_ms": None,
                    "server_pipeline_latency_ms": None,
                    "end_to_end_latency_ms": None,
                }

                self.perception_state.update_asr(result, metrics)

            except Exception as e:
                print(f"ASRWorker exception: {repr(e)}", flush=True)

            time.sleep(self.interval_sec)