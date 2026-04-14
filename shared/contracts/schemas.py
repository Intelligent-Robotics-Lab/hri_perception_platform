from pydantic import BaseModel, Field
from typing import Optional, Dict, List


class ImagePayload(BaseModel):
    encoding: str = Field(description="Image encoding format, e.g. base64_jpeg")
    data: str = Field(description="Encoded image bytes")


class EmotionPredictRequest(BaseModel):
    timestamp_utc: str
    session_id: str
    frame_id: int
    source_id: str
    face_id: Optional[int] = None
    image: ImagePayload
    meta: Optional[Dict[str, str]] = None


class EmotionPredictResponse(BaseModel):
    timestamp_utc: str
    session_id: str
    frame_id: int
    source_id: str
    face_id: Optional[int] = None

    task: str = "emotion_recognition"
    model_name: str
    model_version: str
    backend_name: str

    detected: bool
    dominant_label: Optional[str] = None
    confidence: Optional[float] = None
    scores: Dict[str, float] = {}

    latency_ms: float
    device: str
    bbox_xyxy: Optional[list[int]] = None

    warnings: List[str] = []
    error: Optional[str] = None
    meta: Dict[str, str] = {}