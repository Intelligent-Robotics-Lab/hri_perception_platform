from pathlib import Path

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse


router = APIRouter()

DEBUG_DIR = Path("/data/debug")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MEDIA_GATEWAY_STATUS_URL = "http://media_gateway:8010/status"


def _file_response(path: Path, media_type: str):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path.name}")
    return FileResponse(path, media_type=media_type)


@router.get("/debug/live", response_class=HTMLResponse)
def debug_live_page():
    html_path = STATIC_DIR / "debug_dashboard.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="debug_dashboard.html not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@router.get("/debug/image/input")
def debug_input_image():
    return _file_response(DEBUG_DIR / "latest_input_frame.jpg", "image/jpeg")


@router.get("/debug/image/annotated")
def debug_annotated_image():
    return _file_response(DEBUG_DIR / "latest_annotated_frame.jpg", "image/jpeg")


@router.get("/debug/image/face")
def debug_face_image():
    return _file_response(DEBUG_DIR / "latest_face_crop.jpg", "image/jpeg")


@router.get("/debug/gateway-status")
def debug_gateway_status():
    try:
        response = requests.get(MEDIA_GATEWAY_STATUS_URL, timeout=2.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "status": "error",
            "session": {"connected": False},
            "video": {},
            "audio": {},
            "error": repr(e),
        }
