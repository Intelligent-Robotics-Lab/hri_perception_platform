from pathlib import Path
import cv2
import numpy as np


MODEL_PATH = Path("/data/models/face_detection_yunet_2023mar.onnx")


class FaceDetector:
    def __init__(self, input_size=(320, 320), score_threshold=0.8, nms_threshold=0.3, top_k=5000):
        self.input_size = input_size
        self.detector = cv2.FaceDetectorYN.create(
            str(MODEL_PATH),
            "",
            input_size,
            score_threshold,
            nms_threshold,
            top_k,
        )

    def detect_largest_face(self, image_bgr: np.ndarray):
        h, w = image_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image_bgr)

        if faces is None or len(faces) == 0:
            return None

        # faces[:, 0:4] = x, y, w, h
        largest = max(faces, key=lambda f: f[2] * f[3])
        x, y, bw, bh = largest[:4].astype(int)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)

        if x2 <= x1 or y2 <= y1:
            return None

        face_crop = image_bgr[y1:y2, x1:x2]
        return {
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "face_crop_bgr": face_crop,
        }