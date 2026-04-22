from pathlib import Path
import yaml

from app.audio.mic_sender import MicSender
from app.capture.webcam_sender import WebcamSender


def load_config():
    config_path = Path(__file__).parent / "config" / "client_config.yaml"
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def run_video(cfg):
    sender = WebcamSender(
        ingest_url=cfg["server"]["ingest_frame_url"],
        camera_index=cfg["camera"]["camera_index"],
        target_fps=cfg["camera"]["target_fps"],
        jpeg_quality=cfg["camera"]["jpeg_quality"],
        print_metrics_every_n_frames=cfg["runtime"]["print_metrics_every_n_frames"],
    )
    sender.run(
        print_server_response=cfg["runtime"]["print_server_response"]
    )


def run_audio(cfg):
    sender = MicSender(
        ingest_audio_url=cfg["server"]["ingest_audio_url"],
        sample_rate_hz=cfg["audio"]["sample_rate_hz"],
        channels=cfg["audio"]["channels"],
        chunk_duration_sec=cfg["audio"]["chunk_duration_sec"],
        subtype=cfg["audio"]["subtype"],
        print_metrics_every_n_chunks=cfg["runtime"]["print_audio_metrics_every_n_chunks"],
    )
    sender.run(
        print_server_response=cfg["runtime"]["print_server_response"]
    )


def main():
    cfg = load_config()

    mode = input("Choose mode [video/audio]: ").strip().lower()

    if mode == "video":
        run_video(cfg)
    elif mode == "audio":
        run_audio(cfg)
    else:
        print("Unknown mode. Use 'video' or 'audio'.")


if __name__ == "__main__":
    main()