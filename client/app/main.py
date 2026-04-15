from pathlib import Path
import yaml

from app.capture.webcam_sender import WebcamSender


def load_config():
    config_path = Path(__file__).parent / "config" / "client_config.yaml"
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    sender = WebcamSender(
        ingest_url=cfg["server"]["ingest_frame_url"],
        camera_index=cfg["camera"]["camera_index"],
        target_fps=cfg["camera"]["target_fps"],
        jpeg_quality=cfg["camera"]["jpeg_quality"],
    )

    sender.run(
        print_server_response=cfg["runtime"]["print_server_response"]
    )


if __name__ == "__main__":
    main()