from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from . import config
from .compliance import measure_compliance
from .crop import crop_passport, encode_jpeg
from .gate import _decode_image, warmup, validate_image
from .openrouter import OpenRouterError, edit_selfie
from .whitening import force_white_background

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gosphoto-gate")

ALLOWED_CT = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "application/octet-stream",
}


def _guess_mime(filename: str | None, content_type: str | None) -> str:
    ct = (content_type or "").lower()
    if ct in ("image/png", "image/webp", "image/jpeg", "image/jpg"):
        return "image/jpeg" if ct == "image/jpg" else ct
    name = (filename or "").lower()
    if name.endswith(".png"):
        return "image/png"
    if name.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("Loading Face Landmarker from %s", config.MODEL_PATH)
    warmup()
    log.info(
        "Gate ready; OpenRouter model=%s key=%s",
        config.OPENROUTER_IMAGE_MODEL,
        "set" if config.OPENROUTER_API_KEY else "MISSING",
    )
    yield


app = FastAPI(title="Gosphoto photo gate", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "gosphoto-gate",
        "openrouter": bool(config.OPENROUTER_API_KEY),
        "edit_model": config.OPENROUTER_IMAGE_MODEL,
    }


@app.post("/api/validate")
async def validate(file: UploadFile = File(...)):
    ct = (file.content_type or "").lower()
    if ct and ct not in ALLOWED_CT and not ct.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are allowed")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {config.MAX_UPLOAD_BYTES} bytes)",
        )

    result = validate_image(data)
    body = {
        "ok": result.ok,
        "reason": result.reason,
        "message": result.message,
        "face_count": result.face_count,
        "metrics": result.metrics,
    }
    return JSONResponse(content=body, status_code=200)


@app.get("/api/validate")
def validate_info():
    return {
        "method": "POST multipart field 'file'",
        "max_bytes": config.MAX_UPLOAD_BYTES,
        "checks": ["face_count", "yaw", "pitch", "roll", "blur"],
    }


@app.post("/api/process")
async def process(file: UploadFile = File(...), format: str = "json"):
    """Gate → OpenRouter edit (white bg) → local passport crop."""
    ct = (file.content_type or "").lower()
    if ct and ct not in ALLOWED_CT and not ct.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are allowed")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {config.MAX_UPLOAD_BYTES} bytes)",
        )

    gate = validate_image(data)
    if not gate.ok:
        return JSONResponse(
            {
                "ok": False,
                "stage": "gate",
                "reason": gate.reason,
                "message": gate.message,
                "metrics": gate.metrics,
            }
        )

    if not config.OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY not configured on server",
        )

    mime = _guess_mime(file.filename, file.content_type)
    try:
        edited = edit_selfie(data, mime=mime)
    except OpenRouterError as e:
        raise HTTPException(
            status_code=502,
            detail={"message": str(e), "provider_status": e.status, "body": e.body},
        ) from e

    bgr = _decode_image(edited)
    if bgr is None:
        raise HTTPException(status_code=502, detail="Edited image decode failed")

    bgr = force_white_background(bgr)

    try:
        cropped, crop_metrics = crop_passport(bgr)
    except ValueError as e:
        log.warning("Crop fallback after edit: %s", e)
        h, w = bgr.shape[:2]
        target_ratio = config.PASSPORT_WIDTH / config.PASSPORT_HEIGHT
        cur = w / h
        if cur > target_ratio:
            new_w = int(h * target_ratio)
            x0 = (w - new_w) // 2
            patch = bgr[:, x0 : x0 + new_w]
        else:
            new_h = int(w / target_ratio)
            y0 = (h - new_h) // 2
            patch = bgr[y0 : y0 + new_h, :]
        cropped = cv2.resize(
            patch,
            (config.PASSPORT_WIDTH, config.PASSPORT_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        crop_metrics = {"fallback": True, "error": str(e)}

    # Whitening again after crop (edges may reintroduce gray)
    cropped = force_white_background(cropped, tol=55)
    compliance = measure_compliance(cropped)

    jpeg = encode_jpeg(cropped)
    if format == "jpeg":
        return Response(content=jpeg, media_type="image/jpeg")

    return JSONResponse(
        {
            "ok": True,
            "stage": "done",
            "message": "Фото обработано",
            "mime": "image/jpeg",
            "width": config.PASSPORT_WIDTH,
            "height": config.PASSPORT_HEIGHT,
            "image_base64": base64.b64encode(jpeg).decode("ascii"),
            "model": config.OPENROUTER_IMAGE_MODEL,
            "gate": gate.metrics,
            "crop": crop_metrics,
            "compliance": compliance,
        }
    )


@app.get("/api/process")
def process_info():
    return {
        "method": "POST multipart field 'file'",
        "pipeline": [
            "gate",
            "openrouter_edit",
            "force_white_bg",
            "local_passport_crop",
            "compliance",
        ],
        "model": config.OPENROUTER_IMAGE_MODEL,
        "openrouter_configured": bool(config.OPENROUTER_API_KEY),
        "output": "json (image_base64) or ?format=jpeg",
    }
