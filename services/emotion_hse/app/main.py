import time
from fastapi import FastAPI
from shared.contracts.schemas import EmotionPredictRequest, EmotionPredictResponse

app = FastAPI(title="emotion_hse")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "emotion_hse",
        "model_name": "stub_hse",
        "model_version": "0.1.0",
    }


@app.get("/metadata")
def metadata():
    return {
        "service_name": "emotion_hse",
        "task": "emotion_recognition",
        "model_name": "stub_hse",
        "model_version": "0.1.0",
        "input_type": "face_crop",
        "output_labels": [
            "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
        ],
        "device": "cpu",
    }


@app.post("/predict", response_model=EmotionPredictResponse)
def predict(req: EmotionPredictRequest):
    t0 = time.perf_counter()

    # Stub response for platform bring-up
    scores = {
        "angry": 0.02,
        "disgust": 0.01,
        "fear": 0.03,
        "happy": 0.72,
        "sad": 0.04,
        "surprise": 0.06,
        "neutral": 0.12,
    }

    latency_ms = (time.perf_counter() - t0) * 1000

    return EmotionPredictResponse(
        timestamp_utc=req.timestamp_utc,
        session_id=req.session_id,
        frame_id=req.frame_id,
        source_id=req.source_id,
        face_id=req.face_id,
        model_name="stub_hse",
        model_version="0.1.0",
        backend_name="emotion_hse",
        detected=True,
        dominant_label="happy",
        confidence=scores["happy"],
        scores=scores,
        latency_ms=latency_ms,
        device="cpu",
        warnings=[],
        error=None,
        meta={"note": "stub backend"},
    )