import base64
import io
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
import mediapipe as mp


app = FastAPI(title="gesture_recognition")

SERVICE_NAME = "gesture_recognition"
BACKEND_NAME = "gesture_holistic_events"
MODEL_NAME = "mediapipe_holistic_event_detector"
MODEL_VERSION = "v2_old_style_port_latched_display"
DEVICE = "cpu"

DEBUG_DIR = Path("/data/debug")
GESTURE_OVERLAY_PATH = DEBUG_DIR / "latest_gesture_overlay.jpg"

ACTION_COOLDOWN_SEC = 0.9
DISPLAY_HOLD_SEC = 1.25

NOD_DELTA = 0.010
SHAKE_DELTA = 0.012
RESET_EPS = 0.003


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


class GestureEventDetector:
    def __init__(self):
        self.prev_nose_x: Optional[float] = None
        self.prev_nose_y: Optional[float] = None

        self.nod_state: Optional[str] = None
        self.shake_state: Optional[str] = None

        self.last_action = "none"
        self.last_detection_time = 0.0

        self.last_display_gesture = "none"
        self.last_display_confidence = 0.0
        self.last_display_time = 0.0

    def in_cooldown(self) -> bool:
        return (time.monotonic() - self.last_detection_time) < ACTION_COOLDOWN_SEC

    def in_display_hold(self) -> bool:
        return (time.monotonic() - self.last_display_time) < DISPLAY_HOLD_SEC

    def update(self, nose_x: Optional[float], nose_y: Optional[float]) -> Dict:
        now = time.monotonic()

        dx = None
        dy = None
        instant_gesture = "none"
        instant_confidence = 0.0

        if nose_x is not None and self.prev_nose_x is not None:
            dx = nose_x - self.prev_nose_x

        if nose_y is not None and self.prev_nose_y is not None:
            dy = nose_y - self.prev_nose_y

        if not self.in_cooldown():
            if dy is not None:
                if dy > NOD_DELTA:
                    self.nod_state = "down"
                elif dy < -NOD_DELTA and self.nod_state == "down":
                    instant_gesture = "nod"
                    instant_confidence = min(1.0, abs(dy) / max(NOD_DELTA, 1e-6))
                    self.last_detection_time = now
                    self.nod_state = "cool"
                    self.shake_state = None

            if instant_gesture == "none" and dx is not None:
                if dx > SHAKE_DELTA:
                    if self.shake_state == "left":
                        instant_gesture = "shake_head"
                        instant_confidence = min(1.0, abs(dx) / max(SHAKE_DELTA, 1e-6))
                        self.last_detection_time = now
                        self.shake_state = "cool"
                        self.nod_state = None
                    else:
                        self.shake_state = "right"
                elif dx < -SHAKE_DELTA:
                    if self.shake_state == "right":
                        instant_gesture = "shake_head"
                        instant_confidence = min(1.0, abs(dx) / max(SHAKE_DELTA, 1e-6))
                        self.last_detection_time = now
                        self.shake_state = "cool"
                        self.nod_state = None
                    else:
                        self.shake_state = "left"
        else:
            self.nod_state = "cool"
            self.shake_state = "cool"

        if dx is not None and dy is not None and abs(dx) < RESET_EPS and abs(dy) < RESET_EPS and not self.in_cooldown():
            if self.nod_state == "cool":
                self.nod_state = None
            if self.shake_state == "cool":
                self.shake_state = None

        if nose_x is not None:
            self.prev_nose_x = nose_x
        if nose_y is not None:
            self.prev_nose_y = nose_y

        if instant_gesture != "none":
            self.last_action = instant_gesture
            self.last_display_gesture = instant_gesture
            self.last_display_confidence = instant_confidence
            self.last_display_time = now

        if instant_gesture != "none":
            detected_gesture = instant_gesture
            confidence = instant_confidence
        elif self.in_display_hold():
            detected_gesture = self.last_display_gesture
            confidence = self.last_display_confidence
        else:
            detected_gesture = "none"
            confidence = 0.0

        return {
            "instant_gesture": instant_gesture,
            "detected_gesture": detected_gesture,
            "confidence": round(float(confidence), 4),
            "motion": {
                "dx": None if dx is None else round(float(dx), 4),
                "dy": None if dy is None else round(float(dy), 4),
                "nose_x": None if nose_x is None else round(float(nose_x), 4),
                "nose_y": None if nose_y is None else round(float(nose_y), 4),
                "nod_state": self.nod_state,
                "shake_state": self.shake_state,
                "cooldown_active": self.in_cooldown(),
                "display_hold_active": self.in_display_hold(),
            },
            "last_action": self.last_action,
        }


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

detector = GestureEventDetector()


def ensure_debug_dir():
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def decode_base64_image(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    image_rgb = np.array(pil_img)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def draw_text(vis: np.ndarray, text: str, x: int, y: int, scale: float = 0.48):
    cv2.putText(
        vis,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def save_overlay(frame_bgr: np.ndarray, results, response: dict):
    vis = frame_bgr.copy()

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            vis,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(vis, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(vis, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(vis, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    overlay = vis.copy()
    cv2.rectangle(overlay, (4, 4), (320, 190), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)

    motion = response.get("motion") or {}

    lines = [
        f"Gesture: {response.get('detected_gesture', 'none')}",
        f"Instant: {response.get('instant_gesture', 'none')}",
        f"Confidence: {response.get('confidence', 0.0):.3f}",
        f"dx: {motion.get('dx', '-')}",
        f"dy: {motion.get('dy', '-')}",
        f"nose_x: {motion.get('nose_x', '-')}",
        f"nose_y: {motion.get('nose_y', '-')}",
        f"Nod state: {motion.get('nod_state', '-')}",
        f"Shake state: {motion.get('shake_state', '-')}",
        f"Cooldown: {motion.get('cooldown_active', False)}",
        f"Hold: {motion.get('display_hold_active', False)}",
        f"Last action: {response.get('last_action', 'none')}",
    ]

    y = 20
    for line in lines:
        draw_text(vis, line, 10, y)
        y += 16

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
        "input_type": "full_frame",
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
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        nose_x = None
        nose_y = None

        if results.pose_landmarks:
            nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
            nose_x = float(nose.x)
            nose_y = float(nose.y)

        event = detector.update(nose_x=nose_x, nose_y=nose_y)
        latency_ms = (time.perf_counter() - started) * 1000.0

        response = {
            "timestamp_utc": req.timestamp_utc,
            "session_id": req.session_id,
            "frame_id": req.frame_id,
            "source_id": req.source_id,
            "task": "gesture_recognition",
            "backend_name": BACKEND_NAME,
            "backend_mode": "real",
            "instant_gesture": event["instant_gesture"],
            "detected_gesture": event["detected_gesture"],
            "confidence": event["confidence"],
            "motion": event["motion"],
            "last_action": event["last_action"],
            "face_detected": bool(results.face_landmarks or results.pose_landmarks),
            "latency_ms": latency_ms,
            "warnings": [] if results.pose_landmarks else ["No pose landmarks detected"],
            "error": None,
            "meta": {
                "input_kind": str(req.meta.get("input_kind", "frame")).lower(),
                "frame_shape": list(frame_bgr.shape),
            },
        }

        save_overlay(frame_bgr, results, response)
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
            "instant_gesture": "none",
            "detected_gesture": "none",
            "confidence": 0.0,
            "motion": None,
            "last_action": "none",
            "face_detected": False,
            "latency_ms": latency_ms,
            "warnings": ["Prediction failed"],
            "error": repr(e),
            "meta": {},
        }
        try:
            ensure_debug_dir()
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.imwrite(str(GESTURE_OVERLAY_PATH), blank)
        except Exception:
            pass
        return response