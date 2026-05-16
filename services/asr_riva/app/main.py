import os
from datetime import datetime, timezone

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

SERVICE_NAME = "asr_riva"
BACKEND_NAME = "asr_riva"
UPSTREAM_TRANSCRIBE_URL = os.getenv("RIVA_UPSTREAM_TRANSCRIBE_URL")
UPSTREAM_HEALTH_URL = os.getenv("RIVA_UPSTREAM_HEALTH_URL")
LANGUAGE = os.getenv("ASR_LANGUAGE", "en")

app = FastAPI(title=SERVICE_NAME)


@app.get("/health")
def health():
    upstream_ok = None
    upstream_error = None

    if UPSTREAM_HEALTH_URL:
        try:
            r = requests.get(UPSTREAM_HEALTH_URL, timeout=2.0)
            upstream_ok = r.ok
        except Exception as e:
            upstream_ok = False
            upstream_error = repr(e)

    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "backend": BACKEND_NAME,
        "mode": "passthrough",
        "language": LANGUAGE,
        "upstream_configured": bool(UPSTREAM_TRANSCRIBE_URL),
        "upstream_ok": upstream_ok,
        "upstream_error": upstream_error,
    }


@app.get("/metadata")
def metadata():
    return {
        "service_name": SERVICE_NAME,
        "task": "speech_recognition",
        "backend_name": BACKEND_NAME,
        "backend_mode": "passthrough",
        "supports_streaming": False,
        "supports_partial_transcripts": False,
        "input_type": "audio_chunk",
        "output_type": "transcript",
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
    if not UPSTREAM_TRANSCRIBE_URL:
        raise HTTPException(
            status_code=503,
            detail="RIVA_UPSTREAM_TRANSCRIBE_URL is not configured"
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty uploaded audio")

    files = {
        "file": ("audio_chunk.bin", content, "application/octet-stream")
    }

    data = {
        "client_capture_timestamp": client_capture_timestamp,
        "server_ingest_timestamp": server_ingest_timestamp,
        "source_id": source_id,
        "sample_rate_hz": sample_rate_hz,
        "channels": channels,
        "encoding": encoding,
        "language": LANGUAGE,
    }

    try:
        r = requests.post(
            UPSTREAM_TRANSCRIBE_URL,
            files=files,
            data=data,
            timeout=15,
        )
        r.raise_for_status()
        payload = r.json()
        text_value = payload.get("text")
        transcript = text_value.strip() if isinstance(text_value, str) else None
        if transcript == "":
            transcript = None

        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "client_capture_timestamp": client_capture_timestamp,
            "server_ingest_timestamp": server_ingest_timestamp,
            "source_id": source_id,
            "task": "speech_recognition",
            "backend_name": BACKEND_NAME,
            "backend_mode": "passthrough",
            "sample_rate_hz": sample_rate_hz,
            "channels": channels,
            "encoding": encoding,
            "is_partial": False,
            "transcript": transcript,
            "latency_ms": None,
            "warnings": payload.get("warnings", []),
            "error": None,
            "meta": {
                "language": LANGUAGE,
                "upstream_configured": True,
                "raw_text": text_value,
            },
        }
    except Exception as e:
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "timestamp_utc": now_iso,
            "client_capture_timestamp": client_capture_timestamp,
            "server_ingest_timestamp": server_ingest_timestamp,
            "source_id": source_id,
            "task": "speech_recognition",
            "backend_name": BACKEND_NAME,
            "backend_mode": "passthrough",
            "sample_rate_hz": sample_rate_hz,
            "channels": channels,
            "encoding": encoding,
            "is_partial": False,
            "transcript": None,
            "latency_ms": None,
            "warnings": [],
            "error": repr(e),
            "meta": {
                "language": LANGUAGE,
                "upstream_configured": True,
            },
        }

    payload["backend_name"] = BACKEND_NAME
    payload["backend_mode"] = "passthrough"
    payload.setdefault("meta", {})
    payload["meta"]["language"] = LANGUAGE
    return payload