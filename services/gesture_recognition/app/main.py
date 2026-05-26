import base64
import io
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from mediapipe.python.solutions.face_mesh import FaceMesh


app = FastAPI(title="gesture_recognition")

SERVICE_NAME = "gesture_recognition"
BACKEND_NAME = "gesture_head_motion"
MODEL_NAME = "mediapipe_face_mesh_simple_temporal"
MODEL_VERSION = "v5_overlay_debug"
DEVICE = "cpu"

# Face landmark indices
UPPER_FACE_IDX = 10
NOSE_TIP_IDX = 1
CHIN_IDX = 152
LEFT_FACE_IDX = 234
RIGHT_FACE_IDX = 454

# Temporal settings
HISTORY_MAXLEN = 20
COOLDOWN_SEC = 1.0

# Simple event thresholds
NOD_DELTA_THRESH = 0.035
SHAKE_DELTA_THRESH = 0.045

MIN_VALID_FACE_WIDTH = 1e-4
MIN_VALID_FACE_HEIGHT = 1e-4

DEBUG_DIR = Path("/data/debug")
GESTURE_OVERLAY_PATH = DEBUG_DIR / "latest_gesture_overlay.jpg"


class ImagePayload(BaseModel):
    encoding: str
    data: str


class GesturePredictRequest(BaseModel):
    timestamp_utc: str
    session_id: str
    frame_id: int
    source_id: str
    image: ImagePayload
    meta: Dict = {}


@dataclass
class MotionSample:
    timestamp_monotonic: float
    x_norm: float
    y_norm: float


class HeadMotionDetector:
    def __init__(self):
        self.history: Deque[MotionSample] = deque(maxlen=HISTORY_MAXLEN)
        self.last_detection_time = 0.0

        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None

        self.nod_state: Optional[str] = None
        self.shake_state: Optional[str] = None

    def update(self, x_norm: float, y_norm: float) -> Dict:
        now = time.monotonic()
        self.history.append(MotionSample(timestamp_monotonic=now, x_norm=x_norm, y_norm=y_norm))

        dx = None if self.prev_x is None else (x_norm - self.prev_x)
        dy = None if self.prev_y is None else (y_norm - self.prev_y)

        xs = np.array([s.x_norm for s in self.history], dtype=np.float32)
        ys = np.array([s.y_norm for s in self.history], dtype=np.float32)

        x_range = float(xs.max() - xs.min()) if len(xs) > 0 else 0.0
        y_range = float(ys.max() - ys.min()) if len(ys) > 0 else 0.0

        x_centered = xs - xs.mean() if len(xs) > 0 else np.array([])
        y_centered = ys - ys.mean() if len(ys) > 0 else np.array([])

        x_zero_crossings = self._zero_crossings(x_centered) if len(x_centered) >= 3 else 0
        y_zero_crossings = self._zero_crossings(y_centered) if len(y_centered) >= 3 else 0

        x_energy = float(np.mean(np.abs(np.diff(x_centered)))) if len(x_centered) > 1 else 0.0
        y_energy = float(np.mean(np.abs(np.diff(y_centered)))) if len(y_centered) > 1 else 0.0

        nod_score = min(1.0, y_range / max(NOD_DELTA_THRESH * 2.0, 1e-6))
        shake_score = min(1.0, x_range / max(SHAKE_DELTA_THRESH * 2.0, 1e-6))

        detected_gesture = "none"
        confidence = 0.0

        in_cooldown = (now - self.last_detection_time) < COOLDOWN_SEC

        if not in_cooldown and dy is not None:
            if dy > NOD_DELTA_THRESH:
                self.nod_state = "down"
            elif dy < -NOD_DELTA_THRESH and self.nod_state == "down":
                detected_gesture = "nod"
                confidence = nod_score
                self.last_detection_time = now
                self.nod_state = "cool"
                self.shake_state = None

        if detected_gesture == "none" and not in_cooldown and dx is not None:
            if dx > SHAKE_DELTA_THRESH:
                self.shake_state = "right"
            elif dx < -SHAKE_DELTA_THRESH and self.shake_state == "right":
                detected_gesture = "shake_head"
                confidence = shake_score
                self.last_detection_time = now
                self.shake_state = "cool"
                self.nod_state = None
            elif dx < -SHAKE_DELTA_THRESH:
                self.shake_state = "left"
            elif dx > SHAKE_DELTA_THRESH and self.shake_state == "left":
                detected_gesture = "shake_head"
                confidence = shake_score
                self.last_detection_time = now
                self.shake_state = "cool"
                self.nod_state = None

        if in_cooldown:
            self.nod_state = "cool"
            self.shake_state = "cool"
        elif detected_gesture == "none":
            if self.nod_state == "cool":
                self.nod_state = None
            if self.shake_state == "cool":
                self.shake_state = None

        self.prev_x = x_norm
        self.prev_y = y_norm

        return {
            "detected_gesture": detected_gesture,
            "confidence": round(float(confidence), 4),
            "motion": {
                "pitch_score": round(float(nod_score), 4),
                "yaw_score": round(float(shake_score), 4),
                "x_range": round(float(x_range), 4),
                "y_range": round(float(y_range), 4),
                "x_zero_crossings": int(x_zero_crossings),
                "y_zero_crossings": int(y_zero_crossings),
                "x_energy": round(float(x_energy), 4),
                "y_energy": round(float(y_energy), 4),
                "dx": None if dx is None else round(float(dx), 4),
                "dy": None if dy is None else round(float(dy), 4),
                "nod_state": self.nod_state,
                "shake_state": self.shake_state,
            },
            "normalized_reference": {
                "nose_x_norm": round(float(x_norm), 4),
                "nose_y_norm": round(float(y_norm), 4),
            },
            "history_size": len(self.history),
        }

    def _zero_crossings(self, arr: np.ndarray) -> int:
        signs = np.sign(arr).copy()
        for i in range(1, len(signs)):
            if signs[i] == 0:
                signs[i] = signs[i - 1]
        return int(np.sum(signs[:-1] * signs[1:] < 0))


class FaceMeshTracker:
    def __init__(self):
        self.face_mesh = FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )

    def extract_reference(self, image_bgr: np.ndarray):
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(image_rgb)

        if not result.multi_face_landmarks:
            return None, None

        landmarks = result.multi_face_landmarks[0].landmark

        upper_face = landmarks[UPPER_FACE_IDX]
        nose = landmarks[NOSE_TIP_IDX]
        chin = landmarks[CHIN_IDX]
        left_face = landmarks[LEFT_FACE_IDX]
        right_face = landmarks[RIGHT_FACE_IDX]

        face_width = abs(right_face.x - left_face.x)
        face_height = abs(chin.y - upper_face.y)

        if face_width < MIN_VALID_FACE_WIDTH or face_height < MIN_VALID_FACE_HEIGHT:
            return None, None

        face_center_x = (left_face.x + right_face.x) / 2.0
        face_center_y = (upper_face.y + chin.y) / 2.0

        nose_x_norm = (nose.x - face_center_x) / face_width
        nose_y_norm = (nose.y - face_center_y) / face_height

        nose_x_norm = float(np.clip(nose_x_norm, -1.5, 1.5))
        nose_y_norm = float(np.clip(nose_y_norm, -1.5, 1.5))

        meta = {
            "nose_px": [int(nose.x * w), int(nose.y * h)],
            "face_width_norm": round(float(face_width), 4),
            "face_height_norm": round(float(face_height), 4),
        }

        return (nose_x_norm, nose_y_norm, meta), result.multi_face_landmarks[0]


tracker = FaceMeshTracker()
detector = HeadMotionDetector()


def decode_base64_image(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    image_rgb = np.array(pil_img)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def ensure_debug_dir():
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def draw_gesture_overlay(
    image_bgr: np.ndarray,
    face_landmarks,
    prediction_payload: dict,
):
    vis = image_bgr.copy()
    h, w = vis.shape[:2]

    if face_landmarks is not None:
        for lm in face_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(vis, (x, y), 1, (0, 255, 255), -1)

    meta = prediction_payload.get("meta", {})
    motion = prediction_payload.get("motion", {}) or {}
    detected = prediction_payload.get("detected_gesture", "none")
    confidence = prediction_payload.get("confidence", 0.0)

    nose_px = meta.get("nose_px")
    if nose_px and len(nose_px) == 2:
        cv2.circle(vis, tuple(nose_px), 4, (0, 0, 255), -1)

    lines = [
        f"Gesture: {detected}",
        f"Confidence: {confidence:.3f}",
        f"dx: {motion.get('dx', '-')}",
        f"dy: {motion.get('dy', '-')}",
        f"Pitch score: {motion.get('pitch_score', '-')}",
        f"Yaw score: {motion.get('yaw_score', '-')}",
        f"Nod state: {motion.get('nod_state', '-')}",
        f"Shake state: {motion.get('shake_state', '-')}",
        f"History: {meta.get('history_size', '-')}",
    ]

    y0 = 18
    for line in lines:
        cv2.putText(
            vis,
            str(line),
            (8, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        y0 += 18

    ensure_debug_dir()
    cv2.imwrite(str(GESTURE_OVERLAY_PATH), vis)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "backend": BACKEND_NAME,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "device": DEVICE,
    }


@app.get("/metadata")
def metadata():
    return {
        "service_name": SERVICE_NAME,
        "task": "gesture_recognition",
        "backend_name": BACKEND_NAME,
        "backend_mode": "real",
        "input_type": "frame_or_face_crop",
        "output_type": "gesture_event",
        "supported_gestures": ["nod", "shake_head"],
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "device": DEVICE,
        "debug_overlay_path": str(GESTURE_OVERLAY_PATH),
    }


@app.post("/predict")
def predict(req: GesturePredictRequest):
    started = time.perf_counter()

    try:
        frame_bgr = decode_base64_image(req.image.data)
        input_kind = str(req.meta.get("input_kind", "frame")).lower()

        ref, face_landmarks = tracker.extract_reference(frame_bgr)

        if ref is None:
            latency_ms = (time.perf_counter() - started) * 1000.0
            warning = (
                "No landmarks detected on face crop"
                if input_kind == "face_crop"
                else "No face detected"
            )
            response = {
                "timestamp_utc": req.timestamp_utc,
                "session_id": req.session_id,
                "frame_id": req.frame_id,
                "source_id": req.source_id,
                "task": "gesture_recognition",
                "backend_name": BACKEND_NAME,
                "backend_mode": "real",
                "detected_gesture": "none",
                "confidence": 0.0,
                "motion": None,
                "face_detected": False,
                "latency_ms": latency_ms,
                "warnings": [warning],
                "error": None,
                "meta": {
                    "input_kind": input_kind,
                    "frame_shape": list(frame_bgr.shape),
                },
            }
            draw_gesture_overlay(frame_bgr, None, response)
            return response

        x_norm, y_norm, ref_meta = ref
        motion_result = detector.update(x_norm=x_norm, y_norm=y_norm)
        latency_ms = (time.perf_counter() - started) * 1000.0

        response = {
            "timestamp_utc": req.timestamp_utc,
            "session_id": req.session_id,
            "frame_id": req.frame_id,
            "source_id": req.source_id,
            "task": "gesture_recognition",
            "backend_name": BACKEND_NAME,
            "backend_mode": "real",
            "detected_gesture": motion_result["detected_gesture"],
            "confidence": motion_result["confidence"],
            "motion": motion_result["motion"],
            "face_detected": True,
            "latency_ms": latency_ms,
            "warnings": [],
            "error": None,
            "meta": {
                **ref_meta,
                "input_kind": input_kind,
                "history_size": motion_result["history_size"],
                "normalized_reference": motion_result["normalized_reference"],
            },
        }
        draw_gesture_overlay(frame_bgr, face_landmarks, response)
        return response

    except Exception as e:
        latency_ms = (time.perf_counter() - started) * 1000.0
        response = {
            "timestamp_utc": req.timestamp_utc,
            "session_id": req.session_id,
            "frame_id": req.frame_id,
            "source_id": req.source_id,
            "task": "gesture_recognition",
            "backend_name": BACKEND_NAME,
            "backend_mode": "real",
            "detected_gesture": "none",
            "confidence": 0.0,
            "motion": None,
            "face_detected": False,
            "latency_ms": latency_ms,
            "warnings": ["Prediction failed"],
            "error": repr(e),
            "meta": {},
        }
        try:
            draw_gesture_overlay(np.zeros((240, 240, 3), dtype=np.uint8), None, response)
        except Exception:
            pass
        return response