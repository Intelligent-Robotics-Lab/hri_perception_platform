# Video Pipeline Plan

## Purpose

This document defines the first production-oriented GStreamer video pipeline for the edge client.

The goal is not maximum visual quality.  
The goal is low-latency, freshness-first video delivery into the live interaction stack.

This pipeline will replace the bootstrap Python webcam uploader for the production path.

---

## Role of the video pipeline

The video pipeline is responsible for:

- capturing live video from the local camera
- assigning live-source timing
- converting video into a transport-ready format
- keeping buffering shallow
- avoiding stale-frame buildup
- handing media off to the WebRTC transport layer

The video pipeline is **not** responsible for:
- perception inference
- emotion recognition
- backend selection
- server-side task logic
- dialogue logic
- avatar rendering logic

---

## Design principle

For HRI, freshness is more important than perfect frame delivery.

This means:

- stale frames should be dropped
- queues should stay small
- the server should receive recent frames, not delayed frames
- the pipeline should prioritize low latency over visual perfection

---

## First development target

### Initial operating target
- Resolution: `640x480`
- Frame rate: `15 FPS`
- Live source timestamps: enabled
- Queue depth: small
- Frame-dropping policy: enabled when needed

### Why this target
This is a practical first step because:
- it is enough for emotion and face-presence work
- it reduces bandwidth and encoder load
- it keeps development fast
- it is easier to debug than a higher-resolution pipeline

Later, if performance allows, we can move to:
- `960x540` or `1280x720`
- `20 to 30 FPS`

But not in the first production transport milestone.

---

## Conceptual media flow

```text
camera source
  -> colorspace conversion
  -> rate control
  -> small queue
  -> encoder
  -> WebRTC handoff