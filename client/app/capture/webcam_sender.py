import time
from datetime import datetime, timezone

import cv2
import requests


class WebcamSender:
    def __init__(
        self,
        ingest_url: str,
        camera_index: int = 0,
        target_fps: int = 8,
        jpeg_quality: int = 70,
        print_metrics_every_n_frames: int = 20,
    ):
        self.ingest_url = ingest_url
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality
        self.print_metrics_every_n_frames = print_metrics_every_n_frames

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {camera_index}")

        self.frame_counter = 0

    def run(self, print_server_response: bool = False):
        target_delay = 1.0 / self.target_fps

        try:
            while True:
                loop_start = time.perf_counter()

                ret, frame = self.cap.read()
                if not ret:
                    continue

                client_capture_timestamp = datetime.now(timezone.utc).isoformat()

                encode_start = time.perf_counter()
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                encode_end = time.perf_counter()

                if not ok:
                    continue

                files = {
                    "file": ("frame.jpg", encoded.tobytes(), "image/jpeg")
                }

                data = {
                    "client_capture_timestamp": client_capture_timestamp
                }

                post_start = time.perf_counter()
                response_json = None
                response_status = None

                try:
                    r = requests.post(self.ingest_url, files=files, data=data, timeout=1.5)
                    response_status = r.status_code
                    if r.ok:
                        response_json = r.json()
                except Exception as e:
                    print(f"Request failed: {e}")
                post_end = time.perf_counter()

                loop_end = time.perf_counter()

                capture_to_encode_ms = (encode_end - encode_start) * 1000
                post_round_trip_ms = (post_end - post_start) * 1000
                loop_total_ms = (loop_end - loop_start) * 1000
                achieved_fps = 1000.0 / loop_total_ms if loop_total_ms > 0 else 0.0

                self.frame_counter += 1

                if print_server_response and response_json is not None:
                    print(response_json)

                if self.frame_counter % self.print_metrics_every_n_frames == 0:
                    print({
                        "frame_counter": self.frame_counter,
                        "response_status": response_status,
                        "capture_to_encode_ms": round(capture_to_encode_ms, 2),
                        "post_round_trip_ms": round(post_round_trip_ms, 2),
                        "loop_total_ms": round(loop_total_ms, 2),
                        "achieved_fps": round(achieved_fps, 2),
                    })

                sleep_time = max(0, target_delay - (time.perf_counter() - loop_start))
                time.sleep(sleep_time)

        finally:
            self.cap.release()