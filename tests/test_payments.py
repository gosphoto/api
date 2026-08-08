"""Paywall + Tochka stub payments (no mediapipe import)."""

from __future__ import annotations

import io

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.testclient import TestClient
from PIL import Image

from app import config
from app import payments as payments_mod
from app import results
from app.tochka import StubTochkaClient, TochkaError, reset_tochka_client


def _tiny_jpeg(color=(200, 180, 160), size=(80, 100)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _setup_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "PAYMENTS_DIR", tmp_path / "payments")
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    monkeypatch.setattr(config, "PAYMENTS_ENABLED", True)
    monkeypatch.setattr(config, "PRICE_KOPECKS", 10000)
    monkeypatch.setattr(config, "FREE_DOWNLOAD_UNLOCK", False)
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", "https://gosphoto.ru")
    monkeypatch.setattr(config, "TOCHKA_ACCESS_TOKEN", "")
    reset_tochka_client()
    stub = StubTochkaClient()
    monkeypatch.setattr(payments_mod, "get_tochka_client", lambda: stub)
    return stub


def _result_public_payload(result_id: str, meta: dict) -> dict:
    paid = bool(meta.get("paid"))
    body = {
        "ok": True,
        "result_id": result_id,
        "paid": paid,
        "price_kopecks": config.PRICE_KOPECKS,
        "price_rub": payments_mod.price_rub(),
        "preview_digital_url": f"/api/result/{result_id}/preview_digital.jpg",
        "preview_print_url": f"/api/result/{result_id}/preview_print.jpg",
        "digital_url": f"/api/result/{result_id}/digital.jpg" if paid else None,
        "print_url": f"/api/result/{result_id}/print.jpg" if paid else None,
        "compliance": meta.get("compliance") or {},
    }
    return body


def _mini_pay_app():
    app = FastAPI()

    @app.get("/api/result/{result_id}")
    def get_result(result_id: str):
        if not results.is_valid_result_id(result_id):
            raise HTTPException(status_code=404, detail="Result not found")
        meta = results.load_meta(result_id)
        if not meta:
            raise HTTPException(status_code=404, detail="Result not found")
        return _result_public_payload(result_id, meta)

    @app.post("/api/result/{result_id}/pay")
    def pay_result(result_id: str):
        if not results.is_valid_result_id(result_id):
            raise HTTPException(status_code=404, detail="Result not found")
        try:
            return payments_mod.create_checkout(result_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Result not found") from None
        except TochkaError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

    @app.get("/api/result/{result_id}/payment-status")
    def payment_status(result_id: str):
        if not results.is_valid_result_id(result_id):
            raise HTTPException(status_code=404, detail="Result not found")
        meta = results.load_meta(result_id)
        if not meta:
            raise HTTPException(status_code=404, detail="Result not found")
        payments_mod.sync_pending_for_result(result_id)
        meta = results.load_meta(result_id) or meta
        return _result_public_payload(result_id, meta)

    @app.post("/api/payments/tochka/webhook")
    async def tochka_webhook(request: Request):
        raw = (await request.body()).decode("utf-8", errors="replace")
        return payments_mod.handle_webhook(raw, request.headers.get("x-signature"))

    @app.get("/api/result/{result_id}/preview_digital.jpg")
    def get_preview(result_id: str):
        data = results.load_file(result_id, "preview_digital.jpg")
        if not data:
            raise HTTPException(status_code=404, detail="Result not found")
        return Response(content=data, media_type="image/jpeg")

    @app.get("/api/result/{result_id}/digital.jpg")
    def get_digital(result_id: str):
        if not results.is_valid_result_id(result_id) or not results.load_meta(result_id):
            raise HTTPException(status_code=404, detail="Result not found")
        if not results.is_paid(result_id):
            raise HTTPException(status_code=403, detail="Payment required")
        data = results.load_file(result_id, "digital.jpg")
        if not data:
            raise HTTPException(status_code=404, detail="Result not found")
        return Response(content=data, media_type="image/jpeg")

    @app.get("/api/result/{result_id}/print.jpg")
    def get_print(result_id: str):
        if not results.is_valid_result_id(result_id) or not results.load_meta(result_id):
            raise HTTPException(status_code=404, detail="Result not found")
        if not results.is_paid(result_id):
            raise HTTPException(status_code=403, detail="Payment required")
        data = results.load_file(result_id, "print.jpg")
        if not data:
            raise HTTPException(status_code=404, detail="Result not found")
        return Response(content=data, media_type="image/jpeg")

    return app


def test_save_result_has_preview_and_unpaid(tmp_path, monkeypatch):
    _setup_dirs(tmp_path, monkeypatch)
    rid = results.save_result(
        _tiny_jpeg(), _tiny_jpeg(color=(100, 100, 100), size=(120, 80))
    )
    assert rid
    meta = results.load_meta(rid)
    assert meta["paid"] is False
    assert results.load_file(rid, "preview_digital.jpg")
    assert results.load_file(rid, "preview_print.jpg")


def test_jpeg_forbidden_until_paid(tmp_path, monkeypatch):
    _setup_dirs(tmp_path, monkeypatch)
    rid = results.save_result(_tiny_jpeg(), _tiny_jpeg())
    client = TestClient(_mini_pay_app())

    r = client.get(f"/api/result/{rid}")
    assert r.status_code == 200
    body = r.json()
    assert body["paid"] is False
    assert body["price_rub"] == 100
    assert body["digital_url"] is None
    assert body["preview_digital_url"].endswith("preview_digital.jpg")

    assert client.get(f"/api/result/{rid}/digital.jpg").status_code == 403
    assert client.get(f"/api/result/{rid}/print.jpg").status_code == 403
    assert client.get(f"/api/result/{rid}/preview_digital.jpg").status_code == 200

    pay = client.post(f"/api/result/{rid}/pay")
    assert pay.status_code == 200
    pay_body = pay.json()
    assert pay_body["payment_required"] is True
    assert pay_body["payment_url"]
    payment_id = pay_body["payment_id"]

    webhook = client.post(
        "/api/payments/tochka/webhook",
        content=(
            f'{{"payment_id":"tochka_{payment_id[:8]}",'
            f'"payment_link_id":"{payment_id}","status":"paid"}}'
        ),
        headers={"content-type": "application/json"},
    )
    assert webhook.status_code == 200
    assert webhook.json().get("paid") is True

    meta = results.load_meta(rid)
    assert meta["paid"] is True
    assert meta["payment_id"] == payment_id

    unlocked = client.get(f"/api/result/{rid}")
    assert unlocked.json()["paid"] is True
    assert unlocked.json()["digital_url"]

    dig = client.get(f"/api/result/{rid}/digital.jpg")
    assert dig.status_code == 200
    assert dig.headers["content-type"].startswith("image/jpeg")


def test_free_download_unlock(tmp_path, monkeypatch):
    _setup_dirs(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "FREE_DOWNLOAD_UNLOCK", True)
    rid = results.save_result(_tiny_jpeg(), _tiny_jpeg())
    client = TestClient(_mini_pay_app())
    pay = client.post(f"/api/result/{rid}/pay")
    assert pay.status_code == 200
    assert pay.json()["paid"] is True
    assert pay.json()["payment_required"] is False
    assert client.get(f"/api/result/{rid}/digital.jpg").status_code == 200


def test_payment_binds_to_result_id(tmp_path, monkeypatch):
    _setup_dirs(tmp_path, monkeypatch)
    rid_a = results.save_result(_tiny_jpeg(), _tiny_jpeg())
    rid_b = results.save_result(
        _tiny_jpeg(color=(10, 20, 30)), _tiny_jpeg(color=(40, 50, 60))
    )
    out = payments_mod.create_checkout(rid_a)
    assert out["result_id"] == rid_a
    payment_id = out["payment_id"]
    record = payments_mod.load_payment(payment_id)
    assert record["result_id"] == rid_a
    payments_mod.handle_webhook(
        f'{{"payment_link_id":"{payment_id}","payment_id":"op1","status":"APPROVED"}}'
    )
    assert results.is_paid(rid_a) is True
    assert results.is_paid(rid_b) is False
