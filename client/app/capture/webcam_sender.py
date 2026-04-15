import time
from datetime import datetime, timezone

import cv2
import requests


class WebcamSender:
    def __init__(self, ingest_url: str, camera_index: int = 0, target_fps: int = 8, jpeg_quality: int = 70):
        self.ingest_url = ingest_url
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {camera_index}")

    def run(self, print_server_response: bool = True):
        delay = 1.0 / self.target_fps

        try:
            while True:
                t0 = time.time()

                ret, frame = self.cap.read()
                if not ret:
                    continue

                client_capture_timestamp = datetime.now(timezone.utc).isoformat()

                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if not ok:
                    continue

                files = {
                    "file": ("frame.jpg", encoded.tobytes(), "image/jpeg")
                }

                data = {
                    "client_capture_timestamp": client_capture_timestamp
                }

                try:
                    r = requests.post(self.ingest_url, files=files, data=data, timeout=1.5)
                    if print_server_response:
                        if r.ok:
                            print(r.json())
                        else:
                            print("Server error:", r.status_code, r.text)
                except Exception as e:
                    print("Request failed:", e)

                elapsed = time.time() - t0
                sleep_time = max(0, delay - elapsed)
                time.sleep(sleep_time)

        finally:
            self.cap.release()