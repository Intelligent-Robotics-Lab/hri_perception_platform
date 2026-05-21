import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import websockets


class PerceptionClient:
    def __init__(self, server_host: str = "localhost", server_port: int = 8000):
        self.server_host = server_host
        self.server_port = server_port
        self.uri = f"ws://{server_host}:{server_port}/ws/events"

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        async with websockets.connect(self.uri, ping_interval=20, ping_timeout=20) as ws:
            print(f"[PerceptionClient] connected to {self.uri}")

            async def keepalive():
                while True:
                    await asyncio.sleep(10)
                    try:
                        await ws.send("ping")
                    except Exception:
                        break

            keepalive_task = asyncio.create_task(keepalive())

            try:
                async for message in ws:
                    yield json.loads(message)
            finally:
                keepalive_task.cancel()