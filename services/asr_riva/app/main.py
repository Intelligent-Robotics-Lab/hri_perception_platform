from datetime import datetime, timezone
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

app = FastAPI(title="asr_riva")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "asr_riva",
        "backend": "stub_riva",
        "mode": "scaffold",
    }


@app.get("/metadata")
def metadata():
    return {
        "service_name": "asr_riva",
        "task": "speech_recognition",
        "backend_name": "asr_riva",
        "backend_mode": "scaffold",
        "supports_streaming": False,
        "supports_partial_transcripts": False,
        "input_type": "audio_chunk",
        "output_type": "transcript",
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

    return {
        "timestamp_utc": now_iso,
        "client_capture_timestamp": client_capture_timestamp,
        "server_ingest_timestamp": server_ingest_timestamp,
        "source_id": source_id,
        "task": "speech_recognition",
        "backend_name": "asr_riva",
        "backend_mode": "scaffold",
        "sample_rate_hz": sample_rate_hz,
        "channels": channels,
        "encoding": encoding,
        "is_partial": False,
        "transcript": None,
        "latency_ms": 0.0,
        "warnings": ["Riva scaffold backend not yet connected to real ASR engine"],
        "error": None,
    }