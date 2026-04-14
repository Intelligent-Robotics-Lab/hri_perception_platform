import base64
import io
import time
import urllib.request
from typing import Dict

import numpy as np
from PIL import Image
from fastapi import FastAPI
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

from shared.contracts.schemas import EmotionPredictRequest, EmotionPredictResponse

app = FastAPI(title="emotion_hse")

MODEL_NAME = "enet_b0_8_best_afew"
MODEL_VERSION = "hsemotion-onnx"
DEVICE = "cpu"

fer = HSEmotionRecognizer(model_name=MODEL_NAME)

PLATFORM_LABEL_MAP = {
    "Anger": "angry",
    "Contempt": "disgust",
    "Disgust": "disgust",
    "Fear": "fear",
    "Happiness": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
    "Surprise": "surprise",
}


def decode_base64_image(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(pil_img)


def normalize_scores(raw_scores) -> Dict[str, float]:
    raw_scores = np.array(raw_scores, dtype=np.float32).flatten()
    total = float(raw_scores.sum())
    if total > 0:
        raw_scores = raw_scores / total

    labels = [
        "Anger", "Contempt", "Disgust", "Fear",
        "Happiness", "Neutral", "Sadness", "Surprise"
    ]

    merged = {
        "angry": 0.0,
        "disgust": 0.0,
        "fear": 0.0,
        "happy": 0.0,
        "sad": 0.0,
        "surprise": 0.0,
        "neutral": 0.0,
    }

    for label, score in zip(labels, raw_scores.tolist()):
        merged[PLATFORM_LABEL_MAP[label]] += float(score)

    return merged


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "emotion_hse",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }


@app.get("/metadata")
def metadata():
    return {
        "service_name": "emotion_hse",
        "task": "emotion_recognition",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "input_type": "face_crop",
        "output_labels": [
            "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
        ],
        "device": DEVICE,
    }


@app.post("/predict", response_model=EmotionPredictResponse)
def predict(req: EmotionPredictRequest):
    t0 = time.perf_counter()

    try:
        face_img = decode_base64_image(req.image.data)

        emotion, scores = fer.predict_emotions(face_img, logits=False)
        normalized_scores = normalize_scores(scores)

        dominant_label = max(normalized_scores, key=normalized_scores.get)
        confidence = normalized_scores[dominant_label]
        latency_ms = (time.perf_counter() - t0) * 1000

        return EmotionPredictResponse(
            timestamp_utc=req.timestamp_utc,
            session_id=req.session_id,
            frame_id=req.frame_id,
            source_id=req.source_id,
            face_id=req.face_id,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            backend_name="emotion_hse",
            detected=True,
            dominant_label=dominant_label,
            confidence=confidence,
            scores=normalized_scores,
            latency_ms=latency_ms,
            device=DEVICE,
            warnings=[],
            error=None,
            meta={"note": "real hsemotion-onnx backend"},
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return EmotionPredictResponse(
            timestamp_utc=req.timestamp_utc,
            session_id=req.session_id,
            frame_id=req.frame_id,
            source_id=req.source_id,
            face_id=req.face_id,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            backend_name="emotion_hse",
            detected=False,
            dominant_label=None,
            confidence=None,
            scores={},
            latency_ms=latency_ms,
            device=DEVICE,
            warnings=["prediction_failed"],
            error=str(e),
            meta={},
        )