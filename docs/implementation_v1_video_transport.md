# Implementation V1: Video-Only Production Transport

## Scope

This milestone proves the first production-oriented live media path:

- edge webcam capture with GStreamer
- WebRTC transport
- server-side receive with GStreamer
- handoff into the existing perception platform

This milestone does not include:
- audio
- ASR
- turn-taking policy
- dialogue logic
- avatar rendering

## Architecture

```text
edge webcam
  -> GStreamer capture pipeline
  -> webrtcsink
  -> signaling + WebRTC session
  -> webrtcsrc on server
  -> appsink
  -> Python bridge
  -> FrameStore
  -> EmotionWorker
  -> /state/emotion