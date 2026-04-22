
---

## `edge_client/gst/audio_pipeline.md`

```md
# Audio Pipeline Plan

## Purpose

This document defines the first production-oriented GStreamer audio pipeline for the edge client.

The goal is low-latency continuous audio transport for live interaction.

This pipeline will replace the bootstrap WAV chunk uploader for the production path.

---

## Role of the audio pipeline

The audio pipeline is responsible for:

- capturing live microphone audio
- preserving timing
- converting and resampling audio as needed
- keeping buffering small
- preparing audio for real-time transport
- handing media off to the WebRTC transport layer

The audio pipeline is **not** responsible for:
- ASR inference
- transcript generation
- dialogue policy
- turn-taking decision logic
- server-side backend routing

---

## Design principle

For live HRI, audio is the critical modality for:

- turn-taking
- interruption handling
- speech activity
- response timing
- ASR partial transcript behavior

So the audio path must prioritize:
- continuity
- low buffering
- consistent timing
- fast delivery

---

## First development target

### Initial operating target
- Channels: `1` (mono)
- Sample rate: `16000 Hz`
- Live source timestamps: enabled
- Buffering: minimal
- Transport: continuous, not file-style upload

### Why this target
This is a strong first target because:
- it aligns well with speech-focused processing
- it reduces bandwidth and complexity
- it is enough for initial live ASR integration
- it avoids wasting time on high-fidelity settings before the path is proven

Later, if needed, we can revisit:
- `48000 Hz`
- alternate codec and transport tuning
- richer audio preprocessing

But not in the first production transport milestone.

---

## Conceptual media flow

```text
microphone source
  -> audio convert / resample
  -> small queue
  -> encoder
  -> WebRTC handoff