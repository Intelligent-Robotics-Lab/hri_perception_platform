from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


def build_ws_router(broadcaster):
    @router.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket):
        await broadcaster.connect(websocket)
        try:
            while True:
                # Keep the socket open. Clients usually only receive.
                await websocket.receive_text()
        except WebSocketDisconnect:
            broadcaster.disconnect(websocket)
        except Exception:
            broadcaster.disconnect(websocket)

    return router