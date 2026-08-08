"""HTTP API used by https://gosphoto.ru

Live surface:
  GET  /health
  POST /api/process/nano-banana
  GET  /api/result/{id}
  GET  /api/result/{id}/digital.jpg
  GET  /api/result/{id}/print.jpg
"""

from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from . import config
from .crop import encode_jpeg, run_crop_stage
from .edit import run_edit_nano_banana
from .gate import warmup, validate_image
from .openrouter import OpenRouterError
from .pairs import save_pair
from .print_sheet import encode_print_jpeg, make_print_sheet_bgr
from .rejecteds import save_rejected
from .results import is_valid_result_id, load_file, load_meta, save_result

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gosphoto-gate")

ALLOWED_CT = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "application/octet-stream",
}


def _print_payload(passport_bgr) -> tuple[bytes, dict]:
    sheet_bgr, sheet_meta = make_print_sheet_bgr(passport_bgr)
    sheet_jpeg = encode_print_jpeg(sheet_bgr)
    payload = {
        "mime": "image/jpeg",
        "width": sheet_meta["width"],
        "height": sheet_meta["height"],
        "dpi": sheet_meta["dpi"],
        "size_cm": sheet_meta["size_cm"],
        "copies": sheet_meta["copies"],
        "layout": sheet_meta["layout"],
        "bytes": len(sheet_jpeg),
        "image_base64": base64.b64encode(sheet_jpeg).decode("ascii"),
        "meta": sheet_meta,
    }
    return sheet_jpeg, payload


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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("Loading Face Landmarker from %s", config.MODEL_PATH)
    warmup()
    log.info(
        "Gate ready; nano_banana=%s openrouter=%s",
        config.NANO_BANANA_MODEL,
        "set" if config.OPENROUTER_API_KEY else "MISSING",
    )
    yield


app = FastAPI(title="Gosphoto photo gate", version="0.7.0", lifespan=lifespan)
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
        "version": "0.7.0",
        "pipeline": ["gate", "nano_banana", "crop", "print_10x15"],
        "nano_banana_model": config.NANO_BANANA_MODEL,
        "openrouter": bool(config.OPENROUTER_API_KEY),
        "note": "gosphoto.ru → POST /api/process/nano-banana + GET /api/result/{id}",
    }


@app.post("/api/process/nano-banana")
async def process_nano_banana(file: UploadFile = File(...), format: str = "json"):
    """gate → Nano Banana Pro (Gosuslugi one-pass) → 35×45 crop → print sheet."""
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
        edited, edit_meta = run_edit_nano_banana(data, mime=mime)
    except OpenRouterError as e:
        raise HTTPException(
            status_code=502,
            detail={"message": str(e), "provider_status": e.status, "body": e.body},
        ) from e
    except Exception as e:
        log.exception("Nano Banana edit failed")
        raise HTTPException(status_code=502, detail=f"Edit failed: {e}") from e

    try:
        cropped, crop_metrics, compliance = run_crop_stage(edited)
    except Exception as e:
        log.exception("Crop stage failed")
        raise HTTPException(status_code=502, detail=f"Crop failed: {e}") from e

    jpeg = encode_jpeg(cropped)
    print_jpeg, print_sheet = _print_payload(cropped)
    compliance = {
        **compliance,
        "jpeg_bytes": len(jpeg),
        "jpeg_max_bytes": config.JPEG_MAX_BYTES,
        "jpeg_size_ok": len(jpeg) <= config.JPEG_MAX_BYTES,
    }
    print_meta = {
        k: print_sheet[k]
        for k in ("width", "height", "dpi", "copies", "bytes", "size_cm", "mime", "layout")
        if k in print_sheet
    }
    pair_meta = {
        "gate": gate.metrics,
        "edit": edit_meta,
        "crop": crop_metrics,
        "compliance": compliance,
        "print_sheet": print_meta,
        "pipeline": [
            "gate",
            "nano_banana",
            "crop",
            "print_10x15",
        ],
        "width": config.PASSPORT_WIDTH,
        "height": config.PASSPORT_HEIGHT,
        "dpi": config.PASSPORT_DPI,
        "mime": "image/jpeg",
    }
    save_pair(
        data,
        jpeg,
        filename=file.filename,
        meta=pair_meta,
    )
    result_id = save_result(
        jpeg,
        print_jpeg,
        meta=pair_meta,
    )
    if format == "jpeg":
        return Response(content=jpeg, media_type="image/jpeg")
    if format == "print":
        return Response(content=print_jpeg, media_type="image/jpeg")

    payload = {
        "ok": True,
        "stage": "done",
        "message": "Фото 35×45 (Nano Banana Pro) + лист 10×15",
        "mime": "image/jpeg",
        "width": config.PASSPORT_WIDTH,
        "height": config.PASSPORT_HEIGHT,
        "dpi": config.PASSPORT_DPI,
        "image_base64": base64.b64encode(jpeg).decode("ascii"),
        "print_sheet": print_sheet,
        "pipeline": pair_meta["pipeline"],
        "model": edit_meta.get("model"),
        "edit_backend": "nano_banana",
        "gate": gate.metrics,
        "edit": edit_meta,
        "crop": crop_metrics,
        "compliance": compliance,
    }
    if result_id:
        payload["result_id"] = result_id
        payload["result_path"] = f"/result/{result_id}"
    return JSONResponse(payload)


@app.get("/api/result/{result_id}")
def get_result(result_id: str):
    if not is_valid_result_id(result_id):
        raise HTTPException(status_code=404, detail="Result not found")
    meta = load_meta(result_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Result not found")
    print_meta = meta.get("print_sheet") or {}
    return {
        "ok": True,
        "result_id": result_id,
        "result_path": f"/result/{result_id}",
        "mime": meta.get("mime") or "image/jpeg",
        "width": meta.get("width") or config.PASSPORT_WIDTH,
        "height": meta.get("height") or config.PASSPORT_HEIGHT,
        "dpi": meta.get("dpi") or config.PASSPORT_DPI,
        "compliance": meta.get("compliance") or {},
        "print_sheet": print_meta,
        "digital_url": f"/api/result/{result_id}/digital.jpg",
        "print_url": f"/api/result/{result_id}/print.jpg",
        "gate": meta.get("gate"),
        "crop": meta.get("crop"),
        "saved_at": meta.get("saved_at"),
    }


@app.get("/api/result/{result_id}/digital.jpg")
def get_result_digital(result_id: str):
    data = load_file(result_id, "digital.jpg")
    if not data:
        raise HTTPException(status_code=404, detail="Result not found")
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/result/{result_id}/print.jpg")
def get_result_print(result_id: str):
    data = load_file(result_id, "print.jpg")
    if not data:
        raise HTTPException(status_code=404, detail="Result not found")
    return Response(content=data, media_type="image/jpeg")
