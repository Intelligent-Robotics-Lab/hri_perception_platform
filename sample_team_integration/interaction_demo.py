import asyncio
from dataclasses import dataclass
from typing import Optional

from perception_client import PerceptionClient


@dataclass
class InteractionState:
    latest_transcript: Optional[str] = None
    latest_emotion: Optional[str] = None
    latest_emotion_confidence: Optional[float] = None


class SampleInteractionAgent:
    def __init__(self):
        self.state = InteractionState()

    def handle_asr(self, payload: dict):
        transcript = payload.get("transcript")
        if transcript:
            self.state.latest_transcript = transcript
            print(f"[ASR] {transcript}")
            self._react()

    def handle_emotion(self, payload: dict):
        prediction = payload.get("prediction") or {}
        label = prediction.get("dominant_label")
        confidence = prediction.get("confidence")

        if label:
            self.state.latest_emotion = label
            self.state.latest_emotion_confidence = confidence
            print(f"[EMOTION] {label} ({confidence})")

    def _react(self):
        transcript = (self.state.latest_transcript or "").lower()
        emotion = (self.state.latest_emotion or "").lower()

        if "hello" in transcript:
            if emotion in {"sad", "fear", "angry"}:
                print("[AGENT] Hi. You sound like you may need support. I’m here.")
            else:
                print("[AGENT] Hello. Good to see you.")
            return

        if "start task" in transcript:
            if emotion in {"fear", "sad"}:
                print("[AGENT] We can start gently. I’ll guide you step by step.")
            else:
                print("[AGENT] Great. Starting the task now.")
            return

        if "i am confused" in transcript or "help me" in transcript:
            print("[AGENT] I noticed that. Let me slow down and explain the next step.")
            return

        if "stop" in transcript:
            print("[AGENT] Stopping interaction.")
            return


async def main():
    client = PerceptionClient(server_host="localhost", server_port=8000)
    agent = SampleInteractionAgent()

    async for event in client.events():
        event_type = event.get("event_type")
        payload = event.get("payload", {})

        if event_type == "asr_update":
            agent.handle_asr(payload)

        elif event_type == "emotion_update":
            agent.handle_emotion(payload)


if __name__ == "__main__":
    asyncio.run(main())