import os
import tempfile
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from faster_whisper import WhisperModel


SERVICE_NAME = "asr_riva"
BACKEND_NAME = "faster_whisper"
MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "base.en")
DEVICE = os.getenv("ASR_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8")
LANGUAGE = os.getenv("ASR_LANGUAGE", "en")

app = FastAPI(title=SERVICE_NAME)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "backend": BACKEND_NAME,
        "mode": "real",
        "model_size": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "language": LANGUAGE,
    }


@app.get("/metadata")
def metadata():
    return {
        "service_name": SERVICE_NAME,
        "task": "speech_recognition",
        "backend_name": BACKEND_NAME,
        "backend_mode": "real",
        "supports_streaming": False,
        "supports_partial_transcripts": False,
        "input_type": "audio_chunk",
        "output_type": "transcript",
        "model_size": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "language": LANGUAGE,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    client_capture_timestamp: str | None = Form(default=None),
    server_ingest_timestamp: str | None = Form(default=None),
    source_id: str | None = Form(default="live_client"),
    sample_rate_hz: int | None = Form(default=None),
    channels: int | None = Form(default=None),
    encoding: str | None = Form(default=None),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty uploaded audio")

    now_iso = datetime.now(timezone.utc).isoformat()

    suffix = ".wav"
    if encoding:
        enc = str(encoding).lower()
        if "wav" in enc:
            suffix = ".wav"
        elif "mp3" in enc:
            suffix = ".mp3"
        elif "ogg" in enc or "opus" in enc:
            suffix = ".ogg"

    warnings = []
    transcript = None
    error = None
    latency_ms = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(content)
            tmp.flush()

            model = get_model()
            started = datetime.now(timezone.utc)

            segments, info = model.transcribe(
                tmp.name,
                language=LANGUAGE,
                vad_filter=True,
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
            )

            text_parts = [segment.text.strip() for segment in segments if segment.text and segment.text.strip()]
            transcript = " ".join(text_parts) if text_parts else None

            finished = datetime.now(timezone.utc)
            latency_ms = (finished - started).total_seconds() * 1000.0

            if transcript is None:
                warnings.append("No speech recognized in this audio chunk")

            if getattr(info, "language", None) and info.language != LANGUAGE:
                warnings.append(f"Detected language {info.language} while configured language is {LANGUAGE}")

    except Exception as e:
        error = repr(e)

    return {
        "timestamp_utc": now_iso,
        "client_capture_timestamp": client_capture_timestamp,
        "server_ingest_timestamp": server_ingest_timestamp,
        "source_id": source_id,
        "task": "speech_recognition",
        "backend_name": BACKEND_NAME,
        "backend_mode": "real",
        "sample_rate_hz": sample_rate_hz,
        "channels": channels,
        "encoding": encoding,
        "is_partial": False,
        "transcript": transcript,
        "latency_ms": latency_ms,
        "warnings": warnings,
        "error": error,
        "meta": {
            "model_size": MODEL_SIZE,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "language": LANGUAGE,
        },
    }
