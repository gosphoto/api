from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from . import config
from .bg import warmup_cutout
from .crop import encode_jpeg, run_crop_stage
from .edit import edit_selfie_local, run_edit_stage
from .gate import _decode_image, warmup, validate_image
from .openrouter import OpenRouterError
from .pairs import save_pair
from .rejecteds import save_rejected

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
    try:
        warmup_cutout()
        log.info("Cutout ready backend=%s", config.EDIT_CUTOUT)
    except Exception as e:
        log.warning("Cutout warmup failed (will retry on request): %s", e)
    log.info(
        "Gate ready; edit_backend=%s cutout=%s openrouter=%s",
        config.EDIT_BACKEND,
        config.EDIT_CUTOUT,
        "set" if config.OPENROUTER_API_KEY else "MISSING",
    )
    yield


app = FastAPI(title="Gosphoto photo gate", version="0.5.0", lifespan=lifespan)
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
        "version": "0.5.0",
        "pipeline": ["gate", "local_white_bg", "crop"],
        "edit_backend": "local",
        "edit_cutout": config.EDIT_CUTOUT,
        "openrouter": bool(config.OPENROUTER_API_KEY),
        "edit_model": config.OPENROUTER_IMAGE_MODEL,
        "note": "/api/process never uses generative OpenRouter; optional /api/edit may",
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
    if not result.ok:
        save_rejected(
            data,
            reason=result.reason,
            message=result.message,
            metrics=result.metrics,
            filename=file.filename,
        )
    return JSONResponse(
        {
            "ok": result.ok,
            "reason": result.reason,
            "message": result.message,
            "face_count": result.face_count,
            "metrics": result.metrics,
        }
    )


@app.get("/api/validate")
def validate_info():
    return {
        "method": "POST multipart field 'file'",
        "max_bytes": config.MAX_UPLOAD_BYTES,
        "checks": ["face_count", "yaw", "pitch", "roll", "blur"],
    }


async def _read_upload(file: UploadFile) -> bytes:
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
    return data


@app.post("/api/edit")
async def edit_only(file: UploadFile = File(...), format: str = "json"):
    """Step 1: white background + normalize. No passport crop."""
    data = await _read_upload(file)
    gate = validate_image(data)
    if not gate.ok:
        save_rejected(
            data,
            reason=gate.reason,
            message=gate.message,
            metrics=gate.metrics,
            filename=file.filename,
        )
        return JSONResponse(
            {
                "ok": False,
                "stage": "gate",
                "reason": gate.reason,
                "message": gate.message,
                "metrics": gate.metrics,
            }
        )

    mime = _guess_mime(file.filename, file.content_type)
    try:
        edited, edit_meta = run_edit_stage(data, mime=mime)
    except OpenRouterError as e:
        raise HTTPException(
            status_code=502,
            detail={"message": str(e), "provider_status": e.status, "body": e.body},
        ) from e
    except Exception as e:
        log.exception("Edit stage failed")
        raise HTTPException(status_code=502, detail=f"Edit failed: {e}") from e

    jpeg = encode_jpeg(edited)
    if format == "jpeg":
        return Response(content=jpeg, media_type="image/jpeg")

    return JSONResponse(
        {
            "ok": True,
            "stage": "edit",
            "message": "Фон и свет подготовлены — можно кропать",
            "mime": "image/jpeg",
            "width": int(edited.shape[1]),
            "height": int(edited.shape[0]),
            "image_base64": base64.b64encode(jpeg).decode("ascii"),
            "edit": edit_meta,
            "gate": gate.metrics,
        }
    )


@app.post("/api/crop")
async def crop_only(file: UploadFile = File(...), format: str = "json"):
    """Step 2: passport 35×45 crop from an already-edited (white bg) photo."""
    data = await _read_upload(file)
    bgr = _decode_image(data)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    try:
        cropped, crop_metrics, compliance = run_crop_stage(bgr)
    except Exception as e:
        log.exception("Crop stage failed")
        raise HTTPException(status_code=502, detail=f"Crop failed: {e}") from e

    jpeg = encode_jpeg(cropped)
    if format == "jpeg":
        return Response(content=jpeg, media_type="image/jpeg")

    return JSONResponse(
        {
            "ok": True,
            "stage": "crop",
            "message": "Кадр 35×45 готов",
            "mime": "image/jpeg",
            "width": config.PASSPORT_WIDTH,
            "height": config.PASSPORT_HEIGHT,
            "image_base64": base64.b64encode(jpeg).decode("ascii"),
            "crop": crop_metrics,
            "compliance": compliance,
        }
    )


@app.post("/api/process")
async def process(file: UploadFile = File(...), format: str = "json"):
    """RF passport: gate → local white bg (pixels kept) → align/fit/crop 35×45.

    Never calls generative OpenRouter — identity-safe cutout only.
    """
    data = await _read_upload(file)

    gate = validate_image(data)
    if not gate.ok:
        save_rejected(
            data,
            reason=gate.reason,
            message=gate.message,
            metrics=gate.metrics,
            filename=file.filename,
        )
        return JSONResponse(
            {
                "ok": False,
                "stage": "gate",
                "reason": gate.reason,
                "message": gate.message,
                "metrics": gate.metrics,
            }
        )

    bgr = _decode_image(data)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    try:
        edited, edit_meta = edit_selfie_local(bgr)
        edit_meta = {**edit_meta, "stage": "edit", "model": edit_meta.get("cutout", "mediapipe")}
    except Exception as e:
        log.exception("Local white-bg edit failed")
        raise HTTPException(status_code=502, detail=f"Edit failed: {e}") from e

    try:
        cropped, crop_metrics, compliance = run_crop_stage(edited)
    except Exception as e:
        log.exception("Crop stage failed")
        raise HTTPException(status_code=502, detail=f"Crop failed: {e}") from e

    jpeg = encode_jpeg(cropped)
    save_pair(
        data,
        jpeg,
        filename=file.filename,
        meta={
            "gate": gate.metrics,
            "edit": edit_meta,
            "crop": crop_metrics,
            "compliance": compliance,
            "pipeline": ["gate", "local_white_bg", "crop"],
        },
    )
    if format == "jpeg":
        return Response(content=jpeg, media_type="image/jpeg")

    return JSONResponse(
        {
            "ok": True,
            "stage": "done",
            "message": "Фото на белом фоне, 35×45",
            "mime": "image/jpeg",
            "width": config.PASSPORT_WIDTH,
            "height": config.PASSPORT_HEIGHT,
            "image_base64": base64.b64encode(jpeg).decode("ascii"),
            "pipeline": ["gate", "local_white_bg", "crop"],
            "model": edit_meta.get("model"),
            "edit_backend": "local",
            "edit_cutout": config.EDIT_CUTOUT,
            "gate": gate.metrics,
            "edit": edit_meta,
            "crop": crop_metrics,
            "compliance": compliance,
        }
    )


@app.get("/api/process")
@app.get("/api/edit")
@app.get("/api/crop")
def process_info():
    return {
        "pipeline": [
            "1. gate — face/pose/blur",
            "2. local_white_bg — MediaPipe/rembg cutout → #FFFFFF (no generative rewrite)",
            "3. crop — roll-correct + fit face/margins + 35×45 @300dpi",
        ],
        "endpoints": {
            "/api/process": "gate + local white bg + crop (never OpenRouter)",
            "/api/edit": "white-bg only; may use OpenRouter if EDIT_BACKEND=openrouter",
            "/api/crop": "crop only",
        },
        "edit_backend": config.EDIT_BACKEND,
        "edit_cutout": config.EDIT_CUTOUT,
        "openrouter_model": config.OPENROUTER_IMAGE_MODEL,
        "openrouter_configured": bool(config.OPENROUTER_API_KEY),
        "output": "json (image_base64) or ?format=jpeg",
    }
