import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.testclient import TestClient

from app import config
from app.results import (
    is_valid_result_id,
    load_file,
    load_meta,
    new_result_id,
    save_result,
)


def test_new_result_id_is_32_hex():
    rid = new_result_id()
    assert is_valid_result_id(rid)
    assert len(rid) == 32


def test_save_and_load_result(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    from PIL import Image
    import io

    def jpeg_bytes(color):
        buf = io.BytesIO()
        Image.new("RGB", (40, 50), color).save(buf, format="JPEG")
        return buf.getvalue()

    dig = jpeg_bytes((10, 20, 30))
    prn = jpeg_bytes((40, 50, 60))
    rid = save_result(
        dig,
        prn,
        meta={
            "width": 827,
            "height": 1063,
            "dpi": 600,
            "compliance": {"pass": True},
            "print_sheet": {"copies": 4},
        },
    )
    assert rid is not None
    meta = load_meta(rid)
    assert meta is not None
    assert meta["result_id"] == rid
    assert meta["compliance"]["pass"] is True
    assert meta["paid"] is False
    assert load_file(rid, "digital.jpg") == dig
    assert load_file(rid, "print.jpg") == prn
    assert load_file(rid, "preview_digital.jpg")
    assert load_file(rid, "evil.txt") is None
    assert load_meta("../x") is None


def test_save_result_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", False)
    assert save_result(b"a", b"b") is None
    assert list(tmp_path.iterdir()) == []


def test_save_result_meta_json_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (20, 20), (1, 2, 3)).save(buf, format="JPEG")
    jpeg = buf.getvalue()
    rid = save_result(jpeg, jpeg, meta={"dpi": 600})
    raw = json.loads((tmp_path / rid / "meta.json").read_text(encoding="utf-8"))
    assert "image_base64" not in raw
    assert raw["dpi"] == 600
    assert raw["paid"] is False


def _mini_result_app():
    """HTTP surface for result routes without loading mediapipe gate stack."""
    from app.results import is_paid

    app = FastAPI()

    @app.get("/api/result/{result_id}")
    def get_result(result_id: str):
        if not is_valid_result_id(result_id):
            raise HTTPException(status_code=404, detail="Result not found")
        meta = load_meta(result_id)
        if not meta:
            raise HTTPException(status_code=404, detail="Result not found")
        paid = bool(meta.get("paid"))
        return {
            "ok": True,
            "result_id": result_id,
            "paid": paid,
            "preview_digital_url": f"/api/result/{result_id}/preview_digital.jpg",
            "digital_url": (
                f"/api/result/{result_id}/digital.jpg" if paid else None
            ),
            "print_url": f"/api/result/{result_id}/print.jpg" if paid else None,
            "compliance": meta.get("compliance") or {},
            "print_sheet": meta.get("print_sheet") or {},
            "width": meta.get("width"),
            "height": meta.get("height"),
            "dpi": meta.get("dpi"),
        }

    @app.get("/api/result/{result_id}/preview_digital.jpg")
    def get_preview(result_id: str):
        data = load_file(result_id, "preview_digital.jpg")
        if not data:
            raise HTTPException(status_code=404, detail="Result not found")
        return Response(content=data, media_type="image/jpeg")

    @app.get("/api/result/{result_id}/digital.jpg")
    def get_digital(result_id: str):
        if not is_paid(result_id):
            raise HTTPException(status_code=403, detail="Payment required")
        data = load_file(result_id, "digital.jpg")
        if not data:
            raise HTTPException(status_code=404, detail="Result not found")
        return Response(content=data, media_type="image/jpeg")

    @app.get("/api/result/{result_id}/print.jpg")
    def get_print(result_id: str):
        if not is_paid(result_id):
            raise HTTPException(status_code=403, detail="Payment required")
        data = load_file(result_id, "print.jpg")
        if not data:
            raise HTTPException(status_code=404, detail="Result not found")
        return Response(content=data, media_type="image/jpeg")

    return app


def test_get_result_http(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    # Valid JPEG so preview generation works
    from PIL import Image
    import io

    def jpeg_bytes():
        buf = io.BytesIO()
        Image.new("RGB", (40, 50), (180, 160, 140)).save(buf, format="JPEG")
        return buf.getvalue()

    dig = jpeg_bytes()
    prn = jpeg_bytes()
    rid = save_result(
        dig,
        prn,
        meta={
            "width": 827,
            "height": 1063,
            "dpi": 600,
            "compliance": {"pass": True, "jpeg_bytes": 10},
            "print_sheet": {"copies": 4, "width": 1181},
        },
    )
    client = TestClient(_mini_result_app())
    r = client.get(f"/api/result/{rid}")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["result_id"] == rid
    assert body["paid"] is False
    assert body["digital_url"] is None
    assert body["preview_digital_url"] == f"/api/result/{rid}/preview_digital.jpg"

    assert client.get(f"/api/result/{rid}/digital.jpg").status_code == 403
    prev = client.get(body["preview_digital_url"])
    assert prev.status_code == 200

    assert client.get("/api/result/not-a-valid-id").status_code == 404
    assert client.get(f"/api/result/{'a' * 32}").status_code == 404
