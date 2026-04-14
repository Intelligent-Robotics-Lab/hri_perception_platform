import json
from pathlib import Path
from collections import Counter

LOG_PATH = Path("/data/logs/replay_emotion.jsonl")


def main():
    if not LOG_PATH.exists():
        raise RuntimeError(f"Log file not found: {LOG_PATH}")

    total = 0
    face_detected = 0
    inference_success = 0
    latencies = []
    labels = Counter()

    with LOG_PATH.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total += 1
            record = json.loads(line)

            if record.get("face_detected"):
                face_detected += 1

            if record.get("upstream_status") == 200 and "emotion_response" in record:
                inference_success += 1
                resp = record["emotion_response"]

                label = resp.get("dominant_label")
                if label:
                    labels[label] += 1

                latency = resp.get("latency_ms")
                if latency is not None:
                    latencies.append(latency)

    avg_latency = sum(latencies) / len(latencies) if latencies else None

    print(f"Total sampled frames: {total}")
    print(f"Frames with detected face: {face_detected}")
    print(f"Successful emotion inferences: {inference_success}")
    print(f"Average latency (ms): {avg_latency:.2f}" if avg_latency is not None else "Average latency (ms): N/A")
    print("Emotion counts:")
    for label, count in labels.most_common():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()