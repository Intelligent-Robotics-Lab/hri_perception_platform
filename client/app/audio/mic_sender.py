import io
import time
from datetime import datetime, timezone

import numpy as np
import requests
import sounddevice as sd
from scipy.io.wavfile import write as wav_write


class MicSender:
    def __init__(
        self,
        ingest_audio_url: str,
        sample_rate_hz: int = 16000,
        channels: int = 1,
        chunk_duration_sec: float = 1.0,
        subtype: str = "int16",
        print_metrics_every_n_chunks: int = 5,
    ):
        self.ingest_audio_url = ingest_audio_url
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.chunk_duration_sec = chunk_duration_sec
        self.subtype = subtype
        self.print_metrics_every_n_chunks = print_metrics_every_n_chunks
        self.chunk_counter = 0

    def _record_chunk(self):
        num_frames = int(self.sample_rate_hz * self.chunk_duration_sec)
        audio = sd.rec(
            frames=num_frames,
            samplerate=self.sample_rate_hz,
            channels=self.channels,
            dtype=self.subtype,
        )
        sd.wait()
        return audio

    def _encode_wav_bytes(self, audio_np: np.ndarray) -> bytes:
        buf = io.BytesIO()
        wav_write(buf, self.sample_rate_hz, audio_np)
        return buf.getvalue()

    def run(self, print_server_response: bool = False):
        while True:
            record_start = time.perf_counter()
            client_capture_timestamp = datetime.now(timezone.utc).isoformat()

            audio_np = self._record_chunk()

            encode_start = time.perf_counter()
            wav_bytes = self._encode_wav_bytes(audio_np)
            encode_end = time.perf_counter()

            files = {
                "file": ("audio_chunk.wav", wav_bytes, "audio/wav")
            }

            data = {
                "client_capture_timestamp": client_capture_timestamp,
                "source_id": "live_client",
                "sample_rate_hz": str(self.sample_rate_hz),
                "channels": str(self.channels),
                "encoding": "wav",
            }

            post_start = time.perf_counter()
            response_status = None
            response_json = None

            try:
                r = requests.post(
                    self.ingest_audio_url,
                    files=files,
                    data=data,
                    timeout=5,
                )
                response_status = r.status_code
                if r.ok:
                    response_json = r.json()
            except Exception as e:
                print(f"Audio request failed: {e}")
            post_end = time.perf_counter()

            record_end = post_end

            record_duration_ms = (encode_start - record_start) * 1000
            wav_encode_ms = (encode_end - encode_start) * 1000
            post_round_trip_ms = (post_end - post_start) * 1000
            chunk_total_ms = (record_end - record_start) * 1000

            self.chunk_counter += 1

            if print_server_response and response_json is not None:
                print(response_json)

            if self.chunk_counter % self.print_metrics_every_n_chunks == 0:
                print({
                    "chunk_counter": self.chunk_counter,
                    "response_status": response_status,
                    "server_chunk_id": response_json.get("chunk_id") if response_json else None,
                    "server_ingest_timestamp": response_json.get("server_ingest_timestamp") if response_json else None,
                    "record_duration_ms": round(record_duration_ms, 2),
                    "wav_encode_ms": round(wav_encode_ms, 2),
                    "post_round_trip_ms": round(post_round_trip_ms, 2),
                    "chunk_total_ms": round(chunk_total_ms, 2),
                })