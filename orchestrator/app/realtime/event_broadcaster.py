import asyncio
import json
import threading
from typing import Any

from fastapi import WebSocket


class EventBroadcaster:
    def __init__(self):
        self._clients: set[WebSocket] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            self._clients.discard(websocket)

    async def _broadcast_async(self, event: dict[str, Any]):
        dead = []
        message = json.dumps(event)

        with self._lock:
            clients = list(self._clients)

        for ws in clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)

        if dead:
            with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    def publish(self, event: dict[str, Any]):
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast_async(event),
            self._loop,
        )