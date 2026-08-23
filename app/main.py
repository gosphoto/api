from __future__ import annotations

import asyncio
import base64
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from concurrent.futures import ThreadPoolExecutor
from . import config
from . import compliance as compliance_mod
from . import doc_metrics as doc_metrics_mod
from . import feedback as feedback_mod
from . import payments as payments_mod
from . import progress_events as progress_events_mod
from . import result_email as result_email_mod
from . import torso as torso_mod
from .bg import warmup_cutout
from .crop import encode_jpeg, run_crop_stage
from .edit import run_edit_stage, run_resume_suit_edit
from .gate import _decode_image, prepare_upload, warmup, validate_image
from .openrouter import OpenRouterError
from .readiness import assess_readiness
from .pairs import save_pair
from .print_sheet import encode_print_jpeg, make_print_sheet_bgr
from .process_cache import lookup as process_cache_lookup
from .process_cache import put as process_cache_put
from .rejecteds import save_rejected
from .results import (
    is_paid,
    is_paid_resume,
    is_valid_result_id,
    load_file,
    load_meta,
    save_result,
)
from .tochka import TochkaError, get_tochka_client

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gosphoto-gate")

ALLOWED_CT = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "application/octet-stream",
}


def _print_payload(passport_bgr, *, include_base64: bool = True) -> tuple[bytes, dict]:
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
        "meta": sheet_meta,
    }
    if include_base64:
        payload["image_base64"] = base64.b64encode(sheet_jpeg).decode("ascii")
    return sheet_jpeg, payload


def _require_paid(result_id: str) -> None:
    if not is_valid_result_id(result_id) or not load_meta(result_id):
        raise HTTPException(status_code=404, detail="Result not found")
    if not is_paid(result_id):
        raise HTTPException(
            status_code=403,
            detail="Payment required. Pay via POST /api/result/{id}/pay",
        )


def _require_paid_resume(result_id: str) -> None:
    if not is_valid_result_id(result_id) or not load_meta(result_id):
        raise HTTPException(status_code=404, detail="Result not found")
    if not is_paid_resume(result_id):
        raise HTTPException(
            status_code=403,
            detail="Payment required. Pay via POST /api/result/{id}/pay-resume",
        )


def _result_public_payload(result_id: str, meta: dict) -> dict:
    paid = bool(meta.get("paid"))
    paid_resume = bool(meta.get("paid_resume"))
    resume_offer = bool(meta.get("resume_offer"))
    print_meta = meta.get("print_sheet") or {}
    body = {
        "ok": True,
        "result_id": result_id,
        "result_path": f"/result/{result_id}",
        "mime": meta.get("mime") or "image/jpeg",
        "width": meta.get("width") or config.PASSPORT_WIDTH,
        "height": meta.get("height") or config.PASSPORT_HEIGHT,
        "dpi": meta.get("dpi") or config.PASSPORT_DPI,
        "doc_type": meta.get("doc_type") or config.DEFAULT_DOC_TYPE,
        "doc_label": meta.get("doc_label")
        or config.DOC_PRESETS[config.DEFAULT_DOC_TYPE]["label"],
        "compliance": compliance_mod.apply_pass(meta.get("compliance") or {}),
        "print_sheet": print_meta,
        "preview_digital_url": f"/api/result/{result_id}/preview_digital.jpg",
        "preview_print_url": f"/api/result/{result_id}/preview_print.jpg",
        "paid": paid,
        "price_kopecks": config.PRICE_KOPECKS,
        "price_rub": payments_mod.price_rub(),
        "resume_offer": resume_offer,
        "paid_resume": paid_resume,
        "price_resume_kopecks": config.RESUME_PRICE_KOPECKS,
        "price_resume_rub": payments_mod.resume_price_rub(),
        "gate": meta.get("gate"),
        "crop": meta.get("crop"),
        "saved_at": meta.get("saved_at"),
        "torso_ok": bool(meta.get("torso_ok")),
    }
    if paid:
        body["digital_url"] = f"/api/result/{result_id}/digital.jpg"
        body["print_url"] = f"/api/result/{result_id}/print.jpg"
    else:
        body["digital_url"] = None
        body["print_url"] = None
    if resume_offer:
        body["preview_resume_url"] = f"/api/result/{result_id}/preview_resume.jpg"
        body["resume_url"] = (
            f"/api/result/{result_id}/resume.jpg" if paid_resume else None
        )
    else:
        body["preview_resume_url"] = None
        body["resume_url"] = None
    return body


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


async def _payment_sync_loop(stop: asyncio.Event) -> None:
    interval = config.PAYMENT_SYNC_INTERVAL_SECONDS
    if interval <= 0:
        await stop.wait()
        return
    while not stop.is_set():
        try:
            await asyncio.to_thread(payments_mod.sync_all_pending)
        except Exception as e:
            log.warning("Payment sync failed: %s", e)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("Loading Face Landmarker from %s", config.MODEL_PATH)
    warmup()
    try:
        torso_mod.warmup()
        log.info(
            "Pose torso ready model=%s upsell=%s",
            config.POSE_MODEL_PATH,
            config.RESUME_UPSELL_ENABLED,
        )
    except Exception as e:
        log.warning("Pose warmup failed (resume upsell may skip): %s", e)
    try:
        warmup_cutout()
        log.info("Cutout ready backend=%s", config.EDIT_CUTOUT)
    except Exception as e:
        log.warning("Cutout warmup failed (will retry on request): %s", e)
    log.info(
        "Gate ready; edit_backend=%s riverflow=%s cutout=%s openrouter=%s "
        "payments=%s free_unlock=%s resume_upsell=%s resume_price=%s",
        config.EDIT_BACKEND,
        config.RIVERFLOW_MODEL,
        config.EDIT_CUTOUT,
        "set" if config.OPENROUTER_API_KEY else "MISSING",
        "tochka" if config.TOCHKA_ACCESS_TOKEN else "stub",
        config.FREE_DOWNLOAD_UNLOCK,
        config.RESUME_UPSELL_ENABLED,
        payments_mod.resume_price_rub(),
    )
    if config.TOCHKA_ACCESS_TOKEN:
        try:
            code = await asyncio.to_thread(get_tochka_client().resolve_customer_code)
            log.info("Tochka customerCode ready: %s", code)
        except Exception as e:
            log.warning("Tochka customerCode resolve failed at startup: %s", e)
    stop = asyncio.Event()
    sync_task = asyncio.create_task(_payment_sync_loop(stop))
    try:
        yield
    finally:
        stop.set()
        sync_task.cancel()
        try:
            await sync_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Gosphoto photo gate", version="0.12.0", lifespan=lifespan)
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
        "version": "0.12.0",
        "pipeline": ["gate", "readiness|riverflow", "crop", "print_10x15"],
        "edit_backend": config.EDIT_BACKEND,
        "skip_edit_if_ready": config.SKIP_EDIT_IF_READY,
        "riverflow_model": config.RIVERFLOW_MODEL,
        "riverflow_bg_mode": config.RIVERFLOW_BG_MODE,
        "edit_cutout": config.EDIT_CUTOUT,
        "openrouter": bool(config.OPENROUTER_API_KEY),
        "smtp": bool(config.SMTP_PASSWORD),
        "payments": "tochka" if config.TOCHKA_ACCESS_TOKEN else "stub",
        "price_rub": payments_mod.price_rub(),
        "price_resume_rub": payments_mod.resume_price_rub(),
        "resume_upsell": config.RESUME_UPSELL_ENABLED,
        "free_download_unlock": config.FREE_DOWNLOAD_UNLOCK,
        "note": (
            "/api/process/stream or /api/process → "
            "gate → readiness|riverflow (+ optional resume suit) → crop → result_id"
        ),
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

    _oriented, result = prepare_upload(data)
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
    data, gate = prepare_upload(data)
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
    if format in ("jpeg", "print"):
        raise HTTPException(
            status_code=403,
            detail="Прямое скачивание отключено. Используйте /api/process → страница результата.",
        )

    return JSONResponse(
        {
            "ok": True,
            "stage": "edit",
            "message": "Фон и свет подготовлены — можно кропать",
            "mime": "image/jpeg",
            "width": int(edited.shape[1]),
            "height": int(edited.shape[0]),
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
    print_jpeg, print_sheet = _print_payload(cropped, include_base64=False)
    compliance = {
        **compliance,
        "jpeg_bytes": len(jpeg),
        "jpeg_max_bytes": config.JPEG_MAX_BYTES,
        "jpeg_size_ok": len(jpeg) <= config.JPEG_MAX_BYTES,
    }
    if format in ("jpeg", "print"):
        raise HTTPException(
            status_code=403,
            detail="Прямое скачивание отключено. Используйте /api/process → страница результата.",
        )

    print_meta = {
        k: print_sheet[k]
        for k in ("width", "height", "dpi", "copies", "bytes", "size_cm", "mime", "layout")
        if k in print_sheet
    }
    return JSONResponse(
        {
            "ok": True,
            "stage": "crop",
            "message": "Кадр 35×45 + лист 10×15 (4 фото)",
            "mime": "image/jpeg",
            "width": config.PASSPORT_WIDTH,
            "height": config.PASSPORT_HEIGHT,
            "dpi": config.PASSPORT_DPI,
            "print_sheet": print_meta,
            "crop": crop_metrics,
            "compliance": compliance,
        }
    )


def _run_passport_stages(
    data: bytes,
    *,
    mime: str,
    preset: dict | None = None,
) -> dict:
    """Gate-passed input → readiness/edit → crop → JPEGs. Raises or returns error dict."""
    preset = preset or config.resolve_doc_preset(None)
    out_w = int(preset["width"])
    out_h = int(preset["height"])
    dpi = int(preset["dpi"])
    jpeg_max = int(preset["jpeg_max_bytes"])
    readiness_meta: dict | None = None
    skipped_edit = False
    decoded = _decode_image(data)
    if decoded is not None:
        readiness = assess_readiness(decoded)
        readiness_meta = readiness.as_dict()
        log.info(
            "Readiness ready=%s reason=%s scores=%s (edit always runs)",
            readiness.ready,
            readiness.reason,
            readiness.scores,
        )

    try:
        edited, edit_meta = run_edit_stage(data, mime=mime)
    except OpenRouterError as e:
        log.exception("Edit stage OpenRouter error")
        return {
            "ok": False,
            "stage": "edit",
            "message": str(e),
            "provider_status": e.status,
            "body": e.body,
        }
    except Exception as e:
        log.exception("Edit stage failed")
        return {"ok": False, "stage": "edit", "message": f"Edit failed: {e}"}

    try:
        cropped, crop_metrics, compliance = run_crop_stage(
            edited, width=out_w, height=out_h, dpi=dpi
        )
    except Exception as e:
        log.exception("Crop stage failed")
        return {"ok": False, "stage": "crop", "message": f"Crop failed: {e}"}

    jpeg = encode_jpeg(cropped, max_bytes=jpeg_max, dpi=dpi)
    print_jpeg, print_sheet = _print_payload(cropped, include_base64=False)
    compliance = {
        **compliance,
        "jpeg_bytes": len(jpeg),
        "jpeg_max_bytes": jpeg_max,
        "jpeg_size_ok": len(jpeg) <= jpeg_max,
    }
    print_meta = {
        k: print_sheet[k]
        for k in ("width", "height", "dpi", "copies", "bytes", "size_cm", "mime", "layout")
        if k in print_sheet
    }
    cutout = edit_meta.get("cutout") or "riverflow"
    if cutout == "skip_edit":
        edit_stage_name = "skip_edit"
    elif cutout == "riverflow":
        edit_stage_name = "riverflow"
    else:
        edit_stage_name = "local_person"
    return {
        "ok": True,
        "jpeg": jpeg,
        "print_jpeg": print_jpeg,
        "print_meta": print_meta,
        "edit_meta": edit_meta,
        "crop_metrics": crop_metrics,
        "compliance": compliance,
        "readiness_meta": readiness_meta,
        "skipped_edit": skipped_edit,
        "edit_stage_name": edit_stage_name,
        "preset": preset,
    }


def _build_process_done_payload(
    result_id: str | None,
    meta: dict,
    *,
    from_cache: bool = False,
) -> dict:
    edit_meta = meta.get("edit") or {}
    print_meta = meta.get("print_sheet") or {}
    compliance = meta.get("compliance") or {}
    readiness_meta = meta.get("readiness") or {}
    crop_metrics = meta.get("crop") or {}
    pipeline = meta.get("pipeline") or []
    resume_offer = bool(meta.get("resume_offer"))
    edit_stage = pipeline[1] if len(pipeline) > 1 else ""
    if edit_stage == "skip_edit":
        done_msg = "Фото 35×45 (crop-only, без Riverflow) + лист 10×15 (4 фото)"
    else:
        done_msg = "Фото 35×45 (Riverflow) + лист 10×15 (4 фото)"
    if resume_offer:
        done_msg += " + превью для резюме"
    if from_cache:
        done_msg += " (из кеша)"
    payload = {
        "ok": True,
        "stage": "done",
        "message": done_msg,
        "mime": meta.get("mime") or "image/jpeg",
        "doc_type": meta.get("doc_type") or config.DEFAULT_DOC_TYPE,
        "doc_label": meta.get("doc_label")
        or config.DOC_PRESETS[config.DEFAULT_DOC_TYPE]["label"],
        "width": meta.get("width") or config.PASSPORT_WIDTH,
        "height": meta.get("height") or config.PASSPORT_HEIGHT,
        "dpi": meta.get("dpi") or config.PASSPORT_DPI,
        "print_sheet": print_meta,
        "pipeline": pipeline,
        "model": edit_meta.get("model"),
        "edit_backend": edit_meta.get("cutout") or config.EDIT_BACKEND,
        "edit_cutout": edit_meta.get("cutout") or config.EDIT_CUTOUT,
        "gate": meta.get("gate") or {},
        "torso": meta.get("torso") or {},
        "resume_offer": resume_offer,
        "price_resume_rub": payments_mod.resume_price_rub(),
        "readiness": readiness_meta,
        "edit": edit_meta,
        "crop": crop_metrics,
        "compliance": compliance,
        "price_rub": payments_mod.price_rub(),
        "from_cache": from_cache,
    }
    if result_id:
        payload["result_id"] = result_id
        payload["result_path"] = f"/result/{result_id}"
    return payload


def _run_process_pipeline(
    data: bytes,
    *,
    filename: str | None,
    mime: str,
    doc_type: str | None = None,
) -> dict:
    """Sync gate → optional parallel passport+resume → save. Returns ok/error dict."""
    preset = config.resolve_doc_preset(doc_type)
    data, gate = prepare_upload(data)
    if not gate.ok:
        save_rejected(
            data,
            reason=gate.reason,
            message=gate.message,
            metrics=gate.metrics,
            filename=filename,
        )
        return {
            "ok": False,
            "stage": "gate",
            "reason": gate.reason,
            "message": gate.message,
            "metrics": gate.metrics,
        }

    cached_id = process_cache_lookup(data, preset["doc_type"])
    if cached_id:
        cached_meta = load_meta(cached_id)
        if cached_meta:
            return _build_process_done_payload(
                cached_id,
                cached_meta,
                from_cache=True,
            )

    torso = torso_mod.assess_torso(data)
    log.info(
        "Torso check ok=%s reason=%s upsell=%s doc_type=%s",
        torso.ok,
        torso.reason,
        config.RESUME_UPSELL_ENABLED,
        preset["doc_type"],
    )
    run_suit = bool(config.RESUME_UPSELL_ENABLED and torso.ok)

    resume_jpeg: bytes | None = None
    resume_meta: dict | None = None
    resume_error: str | None = None

    if run_suit:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_pass = pool.submit(
                _run_passport_stages, data, mime=mime, preset=preset
            )
            fut_suit = pool.submit(run_resume_suit_edit, data, mime)
            passport = fut_pass.result()
            try:
                resume_jpeg, resume_meta = fut_suit.result()
            except Exception as e:
                log.warning("Resume suit edit failed (passport continues): %s", e)
                resume_error = str(e)[:300]
                resume_jpeg = None
                resume_meta = None
    else:
        passport = _run_passport_stages(data, mime=mime, preset=preset)

    if not passport.get("ok"):
        return passport

    jpeg = passport["jpeg"]
    print_jpeg = passport["print_jpeg"]
    print_meta = passport["print_meta"]
    edit_meta = passport["edit_meta"]
    crop_metrics = passport["crop_metrics"]
    compliance = passport["compliance"]
    readiness_meta = passport["readiness_meta"]
    skipped_edit = passport["skipped_edit"]
    edit_stage_name = passport["edit_stage_name"]

    pipeline = ["gate", edit_stage_name, "crop", "print_10x15"]
    if resume_jpeg:
        pipeline.append("resume_suit")

    pair_meta = {
        "doc_type": preset["doc_type"],
        "doc_label": preset["label"],
        "gate": gate.metrics,
        "torso": {"ok": torso.ok, "reason": torso.reason, **torso.metrics},
        "torso_ok": torso.ok,
        "readiness": readiness_meta,
        "edit": edit_meta,
        "crop": crop_metrics,
        "compliance": compliance,
        "print_sheet": print_meta,
        "resume": resume_meta,
        "resume_error": resume_error,
        "pipeline": pipeline,
        "width": preset["width"],
        "height": preset["height"],
        "dpi": preset["dpi"],
        "mime": "image/jpeg",
    }
    save_pair(
        data,
        jpeg,
        filename=filename,
        meta=pair_meta,
    )
    result_id = save_result(
        jpeg,
        print_jpeg,
        meta=pair_meta,
        resume_jpeg=resume_jpeg,
    )
    if result_id:
        process_cache_put(data, preset["doc_type"], result_id)
    doc_metrics_mod.record_doc_type(
        doc_type=preset["doc_type"],
        result_id=result_id,
        dpi=preset["dpi"],
        source="process",
    )
    if skipped_edit:
        done_msg = "Фото 35×45 (crop-only, без Riverflow) + лист 10×15 (4 фото)"
    else:
        done_msg = "Фото 35×45 (Riverflow) + лист 10×15 (4 фото)"
    if resume_jpeg:
        done_msg += " + превью для резюме"
    payload = _build_process_done_payload(result_id, pair_meta)
    payload["message"] = done_msg
    return payload


@app.post("/api/process")
async def process(
    file: UploadFile = File(...),
    format: str = "json",
    doc_type: str | None = Form(None),
):
    """RF passport / zagran: gate → edit → 35×45 crop at doc preset DPI."""
    data = await _read_upload(file)
    mime = _guess_mime(file.filename, file.content_type)
    payload = await asyncio.to_thread(
        _run_process_pipeline,
        data,
        filename=file.filename,
        mime=mime,
        doc_type=doc_type,
    )
    if not payload.get("ok"):
        if payload.get("stage") == "gate":
            return JSONResponse(payload)
        status = 502
        detail = payload.get("message") or "Process failed"
        if payload.get("provider_status") is not None:
            raise HTTPException(
                status_code=status,
                detail={
                    "message": detail,
                    "provider_status": payload.get("provider_status"),
                    "body": payload.get("body"),
                },
            )
        raise HTTPException(status_code=status, detail=detail)

    if format in ("jpeg", "print"):
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Прямое скачивание отключено. Откройте страницу результата и оплатите.",
                "result_id": payload.get("result_id"),
                "result_path": payload.get("result_path"),
            },
        )
    return JSONResponse(payload)


@app.post("/api/process/stream")
async def process_stream(
    file: UploadFile = File(...),
    doc_type: str | None = Form(None),
):
    """Same pipeline as /api/process, but SSE progress every ~4s until done/error."""
    data = await _read_upload(file)
    mime = _guess_mime(file.filename, file.content_type)
    filename = file.filename

    async def work():
        return await asyncio.to_thread(
            _run_process_pipeline,
            data,
            filename=filename,
            mime=mime,
            doc_type=doc_type,
        )

    async def events():
        async for frame in progress_events_mod.iter_process_sse(
            work=work,
            interval_sec=progress_events_mod.PROGRESS_INTERVAL_SEC,
        ):
            yield frame

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/result/{result_id}")
def get_result(result_id: str):
    if not is_valid_result_id(result_id):
        raise HTTPException(status_code=404, detail="Result not found")
    meta = load_meta(result_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Result not found")
    return _result_public_payload(result_id, meta)


@app.post("/api/result/{result_id}/pay")
def pay_result(result_id: str):
    if not is_valid_result_id(result_id):
        raise HTTPException(status_code=404, detail="Result not found")
    try:
        return payments_mod.create_checkout(result_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Result not found") from None
    except TochkaError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except ValueError:
        raise HTTPException(status_code=404, detail="Result not found") from None


@app.post("/api/result/{result_id}/pay-resume")
def pay_resume_result(result_id: str):
    if not is_valid_result_id(result_id):
        raise HTTPException(status_code=404, detail="Result not found")
    try:
        return payments_mod.create_checkout_resume(result_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Result not found") from None
    except TochkaError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except ValueError as e:
        detail = str(e)
        if "resume offer" in detail:
            raise HTTPException(status_code=404, detail="Resume offer not available") from e
        raise HTTPException(status_code=404, detail="Result not found") from e


@app.get("/api/result/{result_id}/payment-status")
def payment_status(result_id: str):
    if not is_valid_result_id(result_id):
        raise HTTPException(status_code=404, detail="Result not found")
    meta = load_meta(result_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Result not found")
    payments_mod.sync_pending_for_result(result_id)
    meta = load_meta(result_id) or meta
    return _result_public_payload(result_id, meta)


@app.post("/api/payments/tochka/webhook")
async def tochka_webhook(request: Request):
    raw = (await request.body()).decode("utf-8", errors="replace")
    signature = request.headers.get("x-signature") or request.headers.get(
        "X-Signature"
    )
    content_type = request.headers.get("content-type", "")
    log.info(
        "Tochka webhook received bytes=%s content_type=%s has_signature=%s body_prefix=%r",
        len(raw),
        content_type,
        bool(signature),
        raw[:96],
    )
    # Always ACK 200 so Tochka does not retry forever on our processing bugs.
    try:
        result = payments_mod.handle_webhook(raw, signature)
    except Exception as e:
        log.exception("Tochka webhook handler error: %s", e)
        result = {"ok": True, "error": "internal"}
    log.info("Tochka webhook result=%s", result)
    return result


@app.get("/pay/success")
def pay_success(result_id: str = ""):
    if is_valid_result_id(result_id):
        return RedirectResponse(url=f"/result/{result_id}?paid=1", status_code=302)
    return RedirectResponse(url="/", status_code=302)


@app.get("/pay/fail")
def pay_fail(result_id: str = ""):
    if is_valid_result_id(result_id):
        return RedirectResponse(url=f"/result/{result_id}?paid=0", status_code=302)
    return RedirectResponse(url="/", status_code=302)


@app.get("/api/result/{result_id}/preview_digital.jpg")
def get_result_preview_digital(result_id: str):
    data = load_file(result_id, "preview_digital.jpg")
    if not data:
        raise HTTPException(status_code=404, detail="Result not found")
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/result/{result_id}/preview_print.jpg")
def get_result_preview_print(result_id: str):
    data = load_file(result_id, "preview_print.jpg")
    if not data:
        raise HTTPException(status_code=404, detail="Result not found")
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/result/{result_id}/preview_resume.jpg")
def get_result_preview_resume(result_id: str):
    data = load_file(result_id, "preview_resume.jpg")
    if not data:
        raise HTTPException(status_code=404, detail="Result not found")
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/result/{result_id}/digital.jpg")
def get_result_digital(result_id: str):
    _require_paid(result_id)
    data = load_file(result_id, "digital.jpg")
    if not data:
        raise HTTPException(status_code=404, detail="Result not found")
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/result/{result_id}/print.jpg")
def get_result_print(result_id: str):
    _require_paid(result_id)
    data = load_file(result_id, "print.jpg")
    if not data:
        raise HTTPException(status_code=404, detail="Result not found")
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/result/{result_id}/resume.jpg")
def get_result_resume(result_id: str):
    _require_paid_resume(result_id)
    data = load_file(result_id, "resume.jpg")
    if not data:
        raise HTTPException(status_code=404, detail="Result not found")
    return Response(content=data, media_type="image/jpeg")


class ResultEmailBody(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)


def _client_ip(request: Request) -> str:
    ip = request.client.host if request.client else "unknown"
    xff = request.headers.get("x-forwarded-for")
    if xff:
        ip = xff.split(",")[0].strip() or ip
    return ip


@app.post("/api/result/{result_id}/email")
async def email_result(result_id: str, body: ResultEmailBody, request: Request):
    """Send digital + print JPEGs to the given email (paid results only)."""
    _require_paid(result_id)
    digital = load_file(result_id, "digital.jpg")
    print_jpeg = load_file(result_id, "print.jpg")
    if not digital or not print_jpeg:
        raise HTTPException(status_code=404, detail="Result not found")
    try:
        result_email_mod.check_rate_limit(_client_ip(request))
        email_n = result_email_mod.validate_email(body.email)
        msg = result_email_mod.build_result_email(
            email=email_n,
            result_id=result_id,
            digital_jpeg=digital,
            print_jpeg=print_jpeg,
        )
        await asyncio.to_thread(result_email_mod.send_result_email, msg)
    except result_email_mod.FeedbackRateLimitError as e:
        raise HTTPException(status_code=429, detail=e.detail) from e
    except result_email_mod.FeedbackValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    except result_email_mod.FeedbackConfigError as e:
        raise HTTPException(status_code=503, detail=e.detail) from e
    except result_email_mod.FeedbackSmtpError as e:
        raise HTTPException(status_code=502, detail=e.detail) from e
    return {"ok": True, "email": email_n, "result_id": result_id}


@app.post("/api/feedback")
async def post_feedback(
    request: Request,
    email: str = Form(...),
    message: str = Form(...),
    photo: UploadFile | None = File(None),
):
    ip = _client_ip(request)
    ua = request.headers.get("user-agent", "")
    try:
        feedback_mod.check_rate_limit(ip)
        email_n = feedback_mod.validate_email(email)
        message_n = feedback_mod.validate_message(message)
        raw = await photo.read() if photo is not None else None
        photo_n = feedback_mod.validate_photo(
            photo.filename if photo else None,
            photo.content_type if photo else None,
            raw,
        )
        msg = feedback_mod.build_feedback_email(
            email=email_n,
            message=message_n,
            client_ip=ip,
            user_agent=ua,
            photo=photo_n,
        )
        await asyncio.to_thread(feedback_mod.send_feedback_email, msg)
    except feedback_mod.FeedbackRateLimitError as e:
        raise HTTPException(status_code=429, detail=e.detail) from e
    except feedback_mod.FeedbackValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    except feedback_mod.FeedbackConfigError as e:
        raise HTTPException(status_code=503, detail=e.detail) from e
    except feedback_mod.FeedbackSmtpError as e:
        raise HTTPException(status_code=502, detail=e.detail) from e
    return {"ok": True}


@app.get("/api/feedback")
def feedback_info():
    return {
        "endpoint": "/api/feedback",
        "method": "POST",
        "fields": {
            "email": "required, reply-to",
            "message": "required, 10–4000 chars",
            "photo": "optional, JPEG/PNG/WebP ≤5MB",
        },
        "to": config.FEEDBACK_TO,
    }


@app.get("/api/process")
@app.get("/api/edit")
@app.get("/api/crop")
def process_info():
    return {
        "pipeline": [
            "1. gate — face/pose/blur",
            "2. openrouter_bg — white bg + shoulders (face forbidden)",
            "3. face_protect — MediaPipe no-retouch face zone from original",
            "4. crop — roll-correct + 35×45 @600dpi (FMS §34.3)",
            "5. print_sheet — 10×15 cm @300dpi with 4 copies",
        ],
        "endpoints": {
            "/api/process": "digital 35×45 + print 10×15 (JSON)",
            "/api/process/stream": "same pipeline + SSE progress every 4s",
            "/api/edit": "white-bg edit only",
            "/api/crop": "crop + print sheet",
            "/api/feedback": "contact form → SMTP email",
            "/api/result/{id}/email": "POST {email} → send paid JPEGs",
            "/api/result/{id}/pay-resume": "Tochka checkout for resume suit (300 ₽)",
        },
        "passport": {
            "size_mm": [35, 45],
            "dpi": config.PASSPORT_DPI,
            "pixels": [config.PASSPORT_WIDTH, config.PASSPORT_HEIGHT],
            "face_ratio": config.PASSPORT_FACE_RATIO,
            "head_height_mm": [
                config.HEAD_HEIGHT_MM_MIN,
                config.HEAD_HEIGHT_MM_MAX,
            ],
            "jpeg_max_bytes": config.JPEG_MAX_BYTES,
            "source": "https://rg.ru/documents/2011/08/22/pasport-dok.html",
        },
        "print_sheet": {
            "size_cm": [10, 15],
            "dpi": 300,
            "copies": 4,
            "layout": "2x2",
        },
        "edit_backend": config.EDIT_BACKEND,
        "edit_cutout": config.EDIT_CUTOUT,
        "openrouter_model": config.OPENROUTER_IMAGE_MODEL,
        "openrouter_configured": bool(config.OPENROUTER_API_KEY),
        "output": "json with result_id; download after POST /api/result/{id}/pay",
        "payments": {
            "provider": "tochka",
            "price_rub": payments_mod.price_rub(),
            "pay": "POST /api/result/{id}/pay",
            "webhook": "POST /api/payments/tochka/webhook",
            "email": "POST /api/result/{id}/email after paid",
        },
    }
