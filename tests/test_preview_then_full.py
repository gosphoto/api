"""Gemini preview before pay; full Pro frame after passport payment."""

from __future__ import annotations

import io
import time

from PIL import Image

from app import config
from app import full_render
from app import payments as payments_mod
from app import results


def _jpeg(size=(1000, 800), color=(180, 160, 140)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    monkeypatch.setattr(config, "PAIRS_DIR", tmp_path / "pairs")
    monkeypatch.setattr(config, "PAYMENTS_DIR", tmp_path / "payments")
    monkeypatch.setattr(config, "PAYMENTS_ENABLED", True)


def test_preview_edit_uses_gemini(monkeypatch):
    import pytest

    pytest.importorskip("mediapipe")
    from app import edit as edit_mod

    seen = {}

    def fake(image_bytes, mime="image/jpeg", *, model=None, prompt=None, reasoning=None):
        seen["model"] = model
        seen["reasoning"] = reasoning
        seen["bytes"] = image_bytes
        return _jpeg(size=(64, 80))

    monkeypatch.setattr(edit_mod, "edit_selfie_riverflow", fake)
    monkeypatch.setattr(edit_mod, "force_white_background", lambda bgr, tol=48: bgr)
    _bgr, meta = edit_mod.run_preview_edit(_jpeg(size=(1000, 800)))
    sent = Image.open(io.BytesIO(seen["bytes"]))
    assert max(sent.size) == 1000
    assert seen["model"] == config.RIVERFLOW_MODEL
    assert seen["model"] == "google/gemini-2.5-flash-image"
    assert seen["reasoning"] is None
    assert meta["preview"] is True
    assert "riverflow" not in seen["model"]


def test_preview_payload_is_plain_gemini():
    from app.openrouter import build_generic_edit_images_payload

    payload = build_generic_edit_images_payload(
        b"fake",
        "image/jpeg",
        model="google/gemini-2.5-flash-image",
    )
    assert payload["model"] == "google/gemini-2.5-flash-image"
    assert "reasoning" not in payload
    assert "image_config" not in payload


def test_save_preview_does_not_write_full_jpeg(tmp_path, monkeypatch):
    _dirs(tmp_path, monkeypatch)
    rid = results.save_preview_result(
        _jpeg(size=(80, 100)),
        _jpeg(size=(120, 80)),
        meta={"source_pair": "20261002T120000Z_selfie", "doc_type": "passport_rf"},
    )
    assert rid
    folder = results.result_dir(rid)
    assert (folder / "preview_digital.jpg").is_file()
    assert (folder / "preview_print.jpg").is_file()
    assert not (folder / "digital.jpg").exists()
    assert not (folder / "print.jpg").exists()
    meta = results.load_meta(rid)
    assert meta["source_pair"] == "20261002T120000Z_selfie"
    assert meta.get("paid") is False
    cloned = results.clone_result(rid)
    assert results.load_meta(cloned)["source_pair"] == meta["source_pair"]


def test_full_render_starts_only_after_passport_payment(tmp_path, monkeypatch):
    _dirs(tmp_path, monkeypatch)
    pair = tmp_path / "pairs" / "20261002T120000Z_selfie"
    pair.mkdir(parents=True)
    (pair / "in.jpg").write_bytes(_jpeg(size=(40, 50)))
    rid = results.save_preview_result(
        _jpeg(size=(40, 50)),
        _jpeg(size=(60, 40)),
        meta={"source_pair": "20261002T120000Z_selfie", "doc_type": "passport_rf"},
    )
    calls = []

    def fake(data, *, mime, preset, edit_model):
        calls.append(edit_model)
        return {
            "ok": True,
            "jpeg": _jpeg(size=(30, 40)),
            "print_jpeg": _jpeg(size=(50, 40)),
            "crop_metrics": {"width": 30},
            "compliance": {"pass": True},
            "print_meta": {"copies": 4},
            "edit_meta": {"model": edit_model},
        }

    monkeypatch.setattr(full_render, "_passport_stages", fake)
    full_render.ensure_started(rid)
    time.sleep(0.15)
    assert calls == []
    assert not (results.result_dir(rid) / "digital.jpg").exists()

    assert results.set_paid(
        rid, payment_id="pay-1", tochka_operation_id="op-1"
    )
    full_render.ensure_started(rid)
    deadline = time.time() + 3
    folder = results.result_dir(rid)
    while time.time() < deadline and not (
        calls and (folder / "digital.jpg").is_file() and (folder / "print.jpg").is_file()
    ):
        time.sleep(0.05)
    assert calls == [config.RIVERFLOW_PRO_MODEL]
    assert results.load_meta(rid)["full"]["status"] == "ready"
    full_render.ensure_started(rid)
    time.sleep(0.15)
    assert calls == [config.RIVERFLOW_PRO_MODEL]


def test_resume_and_repeat_mark_paid_do_not_start_full(tmp_path, monkeypatch):
    _dirs(tmp_path, monkeypatch)
    rid = results.save_preview_result(
        _jpeg(size=(40, 50)),
        _jpeg(size=(60, 40)),
        meta={"resume_offer": True, "doc_type": "passport_rf"},
    )
    started = []
    monkeypatch.setattr(
        full_render, "ensure_started", lambda *args, **kwargs: started.append(args[0])
    )
    payments_mod._apply_unlock(
        {
            "product": "resume",
            "result_id": rid,
            "payment_id": "resume-pay",
        },
        tochka_operation_id="op-resume",
        paid_at="20261002T120000Z",
    )
    assert started == []

    payments_mod._apply_unlock(
        {
            "product": "passport",
            "result_id": rid,
            "payment_id": "passport-pay",
        },
        tochka_operation_id="op-passport",
        paid_at="20261002T120100Z",
    )
    assert started == [rid]

    record = {
        "payment_id": "already",
        "result_id": rid,
        "product": "passport",
        "status": "paid",
        "amount_kopecks": 30000,
    }
    payments_mod._write_payment(record)
    payments_mod.mark_paid("already", tochka_operation_id="op-again", source="test")
    assert started == [rid]


def test_email_waits_until_two_full_files(tmp_path, monkeypatch):
    _dirs(tmp_path, monkeypatch)
    rid = results.save_preview_result(
        _jpeg(size=(40, 50)),
        _jpeg(size=(60, 40)),
        meta={"doc_type": "passport_rf"},
    )
    results.set_paid(rid, payment_id="pay-2", tochka_operation_id="op-2")
    monkeypatch.setattr(
        full_render,
        "_passport_stages",
        lambda *args, **kwargs: {"ok": False, "stage": "edit", "message": "down"},
    )
    status = full_render.wait_until_ready(rid, timeout_sec=2)
    assert status == "error"
    folder = results.result_dir(rid)
    assert not (folder / "digital.jpg").exists()
    assert not (folder / "print.jpg").exists()

    ready_id = results.save_result(
        _jpeg(size=(30, 40)),
        _jpeg(size=(50, 40)),
        meta={"doc_type": "passport_rf"},
    )
    results.set_paid(ready_id, payment_id="pay-3", tochka_operation_id="op-3")

    def boom(*_args, **_kwargs):
        raise AssertionError("full pipeline must not rerun a ready result")

    monkeypatch.setattr(full_render, "_passport_stages", boom)
    assert full_render.wait_until_ready(ready_id, timeout_sec=1) == "ready"
