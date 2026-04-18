# Live Transport Upgrade Plan

## Current state
The platform currently supports a bootstrap live ingest path using per-frame HTTP upload.

This path exists for:
- validation
- regression testing
- rapid debugging

It is not the intended production transport.

## Production target
The production live path will use:
- GStreamer on the edge/client
- WebRTC as the real-time session transport
- transport adapters in the perception platform to populate latest media stores

## Architectural rule
Transport upgrades must not require rewriting:
- workers
- state
- backend routing
- backend services

Only the ingest adapter layer should change.

## Immediate next engineering steps
1. Keep bootstrap HTTP path for fallback
2. Add WebRTC ingest adapter scaffold
3. Add real ASR backend
4. Define edge-client responsibilities
5. Implement production session signaling and media handoff