import base64
import io
import time
import urllib.request
from typing import Dict

import numpy as np
from PIL import Image
from fastapi import FastAPI
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

from shared.contracts.schemas import EmotionPredictRequest

app = FastAPI(title="affect_hse_va")

MODEL_NAME = "enet_b0_8_va_mtl"
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


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    return exps / np.sum(exps)


def parse_multitask_output(raw_scores) -> tuple[Dict[str, float], float, float]:
    raw_scores = np.array(raw_scores, dtype=np.float32).flatten()

    if raw_scores.shape[0] < 10:
        raise ValueError(f"Unexpected multitask output shape: {raw_scores.shape}")

    cat_logits = raw_scores[:8]
    valence = float(raw_scores[8])
    arousal = float(raw_scores[9])

    cat_probs = softmax(cat_logits)

    labels = [
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Happiness",
        "Neutral",
        "Sadness",
        "Surprise",
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

    for label, score in zip(labels, cat_probs.tolist()):
        merged[PLATFORM_LABEL_MAP[label]] += float(score)

    return merged, valence, arousal


def quadrant_label(valence: float, arousal: float) -> str:
    if valence >= 0 and arousal >= 0:
        return "pleasant-active"
    if valence >= 0 and arousal < 0:
        return "pleasant-calm"
    if valence < 0 and arousal >= 0:
        return "unpleasant-active"
    return "unpleasant-calm"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "affect_hse_va",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }


@app.get("/metadata")
def metadata():
    return {
        "service_name": "affect_hse_va",
        "task": "affect_dimensions",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "input_type": "face_crop",
        "output_dimensions": ["valence", "arousal"],
        "output_labels": [
            "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
        ],
        "device": DEVICE,
    }


@app.post("/predict")
def predict(req: EmotionPredictRequest):
    t0 = time.perf_counter()

    try:
        face_img = decode_base64_image(req.image.data)

        predicted_label, scores = fer.predict_emotions(face_img, logits=True)
        normalized_scores, valence, arousal = parse_multitask_output(scores)

        dominant_label = max(normalized_scores, key=normalized_scores.get)
        confidence = normalized_scores[dominant_label]
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "timestamp_utc": req.timestamp_utc,
            "session_id": req.session_id,
            "frame_id": req.frame_id,
            "source_id": req.source_id,
            "face_id": req.face_id,
            "task": "affect_dimensions",
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "backend_name": "affect_hse_va",
            "detected": True,
            "dominant_label": dominant_label,
            "confidence": confidence,
            "scores": normalized_scores,
            "valence": valence,
            "arousal": arousal,
            "quadrant": quadrant_label(valence, arousal),
            "latency_ms": latency_ms,
            "device": DEVICE,
            "warnings": [],
            "error": None,
            "meta": {
                "note": "parallel HSE valence-arousal backend",
                "raw_predicted_label": predicted_label,
            },
        }

    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "timestamp_utc": req.timestamp_utc,
            "session_id": req.session_id,
            "frame_id": req.frame_id,
            "source_id": req.source_id,
            "face_id": req.face_id,
            "task": "affect_dimensions",
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "backend_name": "affect_hse_va",
            "detected": False,
            "dominant_label": None,
            "confidence": None,
            "scores": {},
            "valence": None,
            "arousal": None,
            "quadrant": None,
            "latency_ms": latency_ms,
            "device": DEVICE,
            "warnings": ["prediction_failed"],
            "error": str(e),
            "meta": {},
        }