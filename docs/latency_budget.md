# Live Interaction Latency Budget

## Purpose

This document defines the latency budget targets for the perception platform as it evolves from bootstrap validation toward production-grade live interaction support.

This repository owns the perception-side portion of the latency budget.

---

## Principles

- Freshness is more important than processing every input in order
- Latest-state semantics are preferred over deep queues
- Perception latency must be measured by stage
- Backend model latency is only one component of interaction latency
- Audio timing will become critical for turn-taking and interruption handling

---

## Current measured bootstrap path

### Current trusted measurements
- Client JPEG encode: low single-digit milliseconds
- Bootstrap HTTP POST round-trip: low tens of milliseconds with jitter
- Emotion backend inference: around 18 ms
- Server pipeline latency: tens of milliseconds

### Current limitation
Cross-machine wall-clock timestamps are not yet a trustworthy final end-to-end KPI.

---

## Target budget categories

### Edge capture
- Video frame capture timestamp
- Audio chunk capture timestamp

### Transport
- Edge send to server ingest arrival

### Server pipeline
- Server ingest to worker update

### Backend inference
- Worker request to backend result

### Downstream availability
- State update to downstream consumer access

---

## Initial engineering targets

### Emotion perception
- Backend inference latency: under 30 ms steady-state
- Server pipeline latency: under 20 ms beyond model time
- Live visual perception cadence: 15 FPS minimum, 20 to 30 FPS preferred for richer interaction

### ASR / audio perception
- Partial transcript availability: target under 300 ms from chunk capture in early production path
- Better targets to be refined after backend selection

### Product-facing interaction
These are not fully owned by this repo, but this repo should support them:
- Turn end to agent speech onset: target under 700 ms
- Interruption reaction: target under 250 ms
- TTS time to first audio: target under 250 ms

---

## Current development guidance

### Allowed for bootstrap validation
- JPEG over HTTP frame ingest
- Basic audio placeholder ingest
- Replay and benchmarking

### Required for production path
- Replace bootstrap live transport with lower-latency media ingest
- Preserve worker/state/backend abstractions
- Avoid rewriting task workers during transport migration

---

## Team summary

The perception platform should be designed so that transport can improve without changing:
- backend services
- workers
- state schema
- task routing

That is the key to evolving from MVP validation to product-grade live interaction support.