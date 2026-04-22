
---

## `edge_client/signaling/README.md`

```md
# Signaling and Session Plan

## Purpose

This document defines the role of signaling in the future production live transport path.

Signaling exists only to establish and manage real-time media sessions.

It is not part of the perception logic.

---

## Core rule

Signaling is for:
- session setup
- offer / answer exchange
- ICE candidate exchange
- connection lifecycle

Signaling is **not** for:
- transporting media payloads
- running perception logic
- routing model outputs
- replacing the ingest adapters

Media should flow through the real-time media transport path, not through the signaling channel.

---

## High-level architecture

```text
edge GStreamer media pipeline
  -> WebRTC session
  -> server-side WebRTC ingest adapter
  -> media stores
  -> workers
  -> latest state