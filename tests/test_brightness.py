# tests/test_brightness.py
import io
import pytest
from PIL import Image

from app.brightness import (
    DEFAULT_BRIGHTNESS,
    apply_adjust,
    apply_brightness,
    apply_contrast,
    apply_saturation,
    is_contrast_dirty,
    is_dirty,
    is_saturation_dirty,
    parse_brightness,
    parse_contrast,
    parse_saturation,
)


def test_parse_brightness_default_and_valid():
    assert parse_brightness(1.0) == pytest.approx(1.0)
    assert parse_brightness(1.25) == pytest.approx(1.25)
    assert parse_brightness(0.7) == pytest.approx(0.7)
    assert parse_brightness(1.35) == pytest.approx(1.35)


def test_parse_brightness_rejects_out_of_range():
    with pytest.raises(ValueError):
        parse_brightness(2.5)
    with pytest.raises(ValueError):
        parse_brightness(0.5)
    with pytest.raises(ValueError):
        parse_brightness("bright")


def test_is_dirty():
    assert is_dirty(1.0) is False
    assert is_dirty(1.009) is False
    assert is_dirty(1.25) is True


def test_apply_brightness_brightens_mid_gray():
    img = Image.new("RGB", (32, 32), (128, 128, 128))
    out = apply_brightness(img, 1.25)
    assert out.getpixel((0, 0))[0] > 128
    same = apply_brightness(img, 1.0)
    assert same.getpixel((0, 0)) == (128, 128, 128)


def test_parse_contrast_valid_and_rejects():
    assert parse_contrast(1.0) == pytest.approx(1.0)
    assert parse_contrast(1.2) == pytest.approx(1.2)
    with pytest.raises(ValueError):
        parse_contrast(2.0)
    with pytest.raises(ValueError):
        parse_contrast("x")


def test_is_contrast_dirty():
    assert is_contrast_dirty(1.0) is False
    assert is_contrast_dirty(1.2) is True


def test_apply_contrast_increases_spread():
    img = Image.new("RGB", (2, 1))
    img.putpixel((0, 0), (80, 80, 80))
    img.putpixel((1, 0), (180, 180, 180))
    out = apply_contrast(img, 1.3)
    assert out.getpixel((1, 0))[0] - out.getpixel((0, 0))[0] > 100


def test_apply_adjust_order_brightness_then_contrast():
    img = Image.new("RGB", (32, 32), (128, 128, 128))
    out = apply_adjust(img, 1.2, 1.2)
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) != (128, 128, 128)


def test_parse_saturation_valid_and_rejects():
    assert parse_saturation(1.0) == pytest.approx(1.0)
    assert parse_saturation(1.2) == pytest.approx(1.2)
    with pytest.raises(ValueError):
        parse_saturation(2.0)
    with pytest.raises(ValueError):
        parse_saturation("x")


def test_is_saturation_dirty():
    assert is_saturation_dirty(1.0) is False
    assert is_saturation_dirty(1.2) is True


def test_apply_saturation_boosts_chroma():
    img = Image.new("RGB", (1, 1), (200, 80, 80))
    boosted = apply_saturation(img, 1.3)
    muted = apply_saturation(img, 0.7)
    # Higher saturation → larger channel spread from gray mean.
    def spread(px):
        return max(px) - min(px)

    assert spread(boosted.getpixel((0, 0))) > spread(img.getpixel((0, 0)))
    assert spread(muted.getpixel((0, 0))) < spread(img.getpixel((0, 0)))


def test_save_preview_can_store_digital_base(tmp_path, monkeypatch):
    from app import config, results

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    monkeypatch.setattr(config, "PAIRS_DIR", tmp_path / "pairs")

    def jpeg(size=(70, 90), color=(100, 110, 120)):
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    base = jpeg()
    rid = results.save_preview_result(
        jpeg(size=(48, 60)),
        jpeg(size=(80, 50)),
        meta={"doc_type": "passport_rf"},
        digital_base_jpeg=base,
    )
    assert rid
    folder = results.result_dir(rid)
    assert (folder / "digital_base.jpg").read_bytes() == base
    meta = results.load_meta(rid)
    assert meta.get("brightness") == pytest.approx(1.0)
    assert "digital_base.jpg" not in results.ALLOWED_FILES

    # existing brightness in meta is preserved
    rid2 = results.save_preview_result(
        jpeg(size=(48, 60)),
        jpeg(size=(80, 50)),
        meta={"brightness": 1.2},
    )
    assert results.load_meta(rid2)["brightness"] == pytest.approx(1.2)
    assert not (results.result_dir(rid2) / "digital_base.jpg").exists()


def _rebake_setup(tmp_path, monkeypatch, *, with_base=True):
    from app import config, results

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    monkeypatch.setattr(config, "PAIRS_DIR", tmp_path / "pairs")

    def jpeg(color, size=(70, 90)):
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    rid = results.save_preview_result(
        jpeg((80, 80, 80), size=(48, 60)),
        jpeg((80, 80, 80), size=(100, 60)),
        meta={"doc_type": "passport_rf"},
        digital_base_jpeg=jpeg((80, 80, 80)) if with_base else None,
    )
    assert rid
    return results, rid, jpeg


def _mid_red(data: bytes) -> int:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img.getpixel((img.size[0] // 2, img.size[1] // 2))[0]


def test_rebake_brightness_updates_previews_and_meta(tmp_path, monkeypatch):
    results, rid, _ = _rebake_setup(tmp_path, monkeypatch)
    folder = results.result_dir(rid)
    before = _mid_red((folder / "preview_digital.jpg").read_bytes())
    meta = results.rebake_brightness(rid, 1.25)
    assert meta["brightness"] == pytest.approx(1.25)
    assert results.load_meta(rid)["brightness"] == pytest.approx(1.25)
    after = _mid_red((folder / "preview_digital.jpg").read_bytes())
    assert after > before
    assert (folder / "preview_print.jpg").is_file()
    # unpaid preview-only: no full-size files created
    assert not (folder / "digital.jpg").exists()
    assert not (folder / "print.jpg").exists()


def test_rebake_adjust_writes_brightness_and_contrast_meta(tmp_path, monkeypatch):
    results, rid, _ = _rebake_setup(tmp_path, monkeypatch)
    meta = results.rebake_adjust(rid, 1.12345, 1.23456, 1.34567)
    assert meta["brightness"] == pytest.approx(1.1235)
    assert meta["contrast"] == pytest.approx(1.2346)
    assert meta["saturation"] == pytest.approx(1.3457)
    stored = results.load_meta(rid)
    assert stored["brightness"] == pytest.approx(1.1235)
    assert stored["contrast"] == pytest.approx(1.2346)
    assert stored["saturation"] == pytest.approx(1.3457)


def test_rebake_brightness_preserves_stored_contrast(tmp_path, monkeypatch):
    results, rid, jpeg = _rebake_setup(tmp_path, monkeypatch)
    folder = results.result_dir(rid)
    base = Image.new("RGB", (100, 100), (80, 80, 80))
    for x in range(50, 100):
        for y in range(100):
            base.putpixel((x, y), (160, 160, 160))
    buf = io.BytesIO()
    base.save(buf, format="JPEG", quality=100, subsampling=0)
    (folder / results.DIGITAL_BASE_FILE).write_bytes(buf.getvalue())
    (folder / "digital.jpg").write_bytes(jpeg((100, 100, 100)))
    (folder / "print.jpg").write_bytes(jpeg((100, 100, 100)))
    meta = results.load_meta(rid)
    meta["contrast"] = 1.2
    meta["saturation"] = 1.15
    results._write_meta(rid, meta)

    updated = results.rebake_brightness(rid, 1.0)

    rendered = Image.open(folder / "digital.jpg").convert("RGB")
    spread = rendered.getpixel((75, 50))[0] - rendered.getpixel((25, 50))[0]
    assert spread > 90
    assert updated["brightness"] == pytest.approx(1.0)
    assert updated["contrast"] == pytest.approx(1.2)
    assert updated["saturation"] == pytest.approx(1.15)


def test_rebake_brightness_rewrites_full_files_when_present(tmp_path, monkeypatch):
    results, rid, jpeg = _rebake_setup(tmp_path, monkeypatch)
    folder = results.result_dir(rid)
    (folder / "digital.jpg").write_bytes(jpeg((80, 80, 80)))
    (folder / "print.jpg").write_bytes(jpeg((80, 80, 80), size=(60, 90)))
    results.rebake_brightness(rid, 1.3)
    assert _mid_red((folder / "digital.jpg").read_bytes()) > 90
    sheet = Image.open(io.BytesIO((folder / "print.jpg").read_bytes())).convert("RGB")
    assert sheet.size == (1181, 1772)
    px = sheet.getpixel((sheet.size[0] // 4, sheet.size[1] // 4))[0]
    assert px > 90 and px < 250
    meta = results.load_meta(rid)
    assert meta["digital_bytes"] == (folder / "digital.jpg").stat().st_size
    assert meta["print_bytes"] == (folder / "print.jpg").stat().st_size


def test_rebake_brightness_missing_base_raises(tmp_path, monkeypatch):
    results, rid, _ = _rebake_setup(tmp_path, monkeypatch, with_base=False)
    with pytest.raises(FileNotFoundError):
        results.rebake_brightness(rid, 1.2)
    with pytest.raises(FileNotFoundError):
        results.rebake_brightness("a" * 32, 1.2)


# ---------------------------------------------------------------------------
# API (real app.main:app)
# ---------------------------------------------------------------------------


def _import_app_main():
    """Import real app.main; stub mediapipe only if it is not installed (CI-light venvs)."""
    import importlib
    import sys
    from unittest.mock import MagicMock

    try:
        import mediapipe  # noqa: F401
    except ModuleNotFoundError:
        for name in ("mediapipe", "mediapipe.tasks", "mediapipe.tasks.python"):
            sys.modules.setdefault(name, MagicMock())
    return importlib.import_module("app.main")


def _jpeg(color=(120, 110, 100), size=(64, 80)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def preview_result(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import config, results

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    main = _import_app_main()
    rid = results.save_preview_result(
        _jpeg(),
        _jpeg(),
        meta={"width": 64, "height": 80, "dpi": 600},
        digital_base_jpeg=_jpeg(size=(827, 1063)),
    )
    assert rid
    return TestClient(main.app), rid


def test_get_result_includes_default_brightness(preview_result):
    client, rid = preview_result
    r = client.get(f"/api/result/{rid}")
    assert r.status_code == 200
    assert r.json()["brightness"] == pytest.approx(1.0)


def test_post_brightness_updates_and_rejects_oob(preview_result):
    from app import results

    client, rid = preview_result
    bad = client.post(f"/api/result/{rid}/brightness", json={"brightness": 2.5})
    assert bad.status_code == 422
    ok = client.post(f"/api/result/{rid}/brightness", json={"brightness": 1.25})
    assert ok.status_code == 200
    assert ok.json()["brightness"] == pytest.approx(1.25)
    assert results.load_meta(rid)["brightness"] == pytest.approx(1.25)
    # unpaid: digital still locked
    assert ok.json().get("digital_url") in (None,)
    assert client.get(f"/api/result/{rid}").json()["brightness"] == pytest.approx(1.25)
    reset = client.post(f"/api/result/{rid}/brightness", json={"brightness": 1.0})
    assert reset.json()["brightness"] == pytest.approx(1.0)


def test_post_brightness_unknown_result_404(preview_result):
    client, _ = preview_result
    r = client.post(f"/api/result/{'a' * 32}/brightness", json={"brightness": 1.1})
    assert r.status_code == 404
    r = client.post("/api/result/not-a-valid-id/brightness", json={"brightness": 1.1})
    assert r.status_code == 404


def test_post_brightness_missing_base_409(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import config, results

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    main = _import_app_main()
    rid = results.save_preview_result(_jpeg(), _jpeg(), meta={})  # no digital_base
    r = TestClient(main.app).post(
        f"/api/result/{rid}/brightness", json={"brightness": 1.1}
    )
    assert r.status_code == 409
    assert "digital_base" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Paid promote keeps brightness
# ---------------------------------------------------------------------------


def _mean_luma(data: bytes) -> float:
    import numpy as np

    return float(np.asarray(Image.open(io.BytesIO(data)).convert("L")).mean())


def _promote_setup(tmp_path, monkeypatch, *, brightness=None):
    import numpy as np

    from app import config, results

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    monkeypatch.setattr(config, "PAIRS_DIR", tmp_path / "pairs")
    monkeypatch.setattr(config, "PAIRS_ENABLED", True)
    main = _import_app_main()  # noqa: F841 - promote imports app.main

    pair = tmp_path / "pairs" / "20261002T140000Z_selfie"
    pair.mkdir(parents=True)
    (pair / "in.jpg").write_bytes(_jpeg(size=(40, 50)))
    (pair / "edit_gemini.jpg").write_bytes(_jpeg((100, 100, 100), size=(200, 260)))

    from app import crop

    def fake_crop(bgr, *, width, height, dpi):
        cropped = np.full((height, width, 3), 100, dtype=np.uint8)
        return cropped, {"fake": True}, {}

    monkeypatch.setattr(crop, "run_crop_stage", fake_crop)

    meta = {"source_pair": pair.name, "doc_type": "passport_rf"}
    if brightness is not None:
        meta["brightness"] = brightness
    rid = results.save_preview_result(_jpeg(), _jpeg(), meta=meta)
    assert rid
    results.set_paid(rid, payment_id="pay-b", tochka_operation_id="op-b")
    return results, rid


def test_brightness_survives_paid_promote(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import full_render

    results, rid = _promote_setup(tmp_path, monkeypatch)
    main = _import_app_main()
    client = TestClient(main.app)
    meta = results.load_meta(rid)
    meta["brightness"] = 1.25  # unpaid slider value (no digital_base yet)
    results._write_meta(rid, meta)

    assert full_render.promote_paid_from_edit_gemini(rid, results.load_meta(rid))

    folder = results.result_dir(rid)
    base = (folder / "digital_base.jpg").read_bytes()
    digital = (folder / "digital.jpg").read_bytes()
    assert _mean_luma(base) == pytest.approx(100, abs=3)
    assert _mean_luma(digital) > _mean_luma(base) * 1.15
    assert _mean_luma((folder / "preview_digital.jpg").read_bytes()) > 110
    assert results.load_meta(rid)["brightness"] == pytest.approx(1.25)
    assert client.get(f"/api/result/{rid}").json()["brightness"] == pytest.approx(1.25)

    # reset after pay works on full files
    r = client.post(f"/api/result/{rid}/brightness", json={"brightness": 1.0})
    assert r.status_code == 200
    reset = (folder / "digital.jpg").read_bytes()
    assert _mean_luma(reset) == pytest.approx(100, abs=3)


def test_paid_promote_default_brightness_is_neutral(tmp_path, monkeypatch):
    from app import full_render

    results, rid = _promote_setup(tmp_path, monkeypatch)
    assert full_render.promote_paid_from_edit_gemini(rid, results.load_meta(rid))
    folder = results.result_dir(rid)
    assert (folder / "digital_base.jpg").is_file()
    assert _mean_luma((folder / "digital.jpg").read_bytes()) == pytest.approx(100, abs=3)


# ---------------------------------------------------------------------------
# Final-review fixes (I-1 / I-2 / I-3)
# ---------------------------------------------------------------------------


def test_rebake_zagran_uses_preset_dpi(tmp_path, monkeypatch):
    from app import config, results

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    monkeypatch.setattr(config, "PAIRS_DIR", tmp_path / "pairs")
    rid = results.save_preview_result(
        _jpeg(size=(48, 60)),
        _jpeg(size=(80, 50)),
        meta={"doc_type": "zagran"},
        digital_base_jpeg=_jpeg((80, 80, 80), size=(497, 638)),
    )
    assert rid
    folder = results.result_dir(rid)
    (folder / "digital.jpg").write_bytes(_jpeg((80, 80, 80), size=(497, 638)))
    (folder / "print.jpg").write_bytes(_jpeg((80, 80, 80), size=(60, 90)))
    results.rebake_brightness(rid, 1.2)
    with Image.open(folder / "digital.jpg") as img:
        dpi = tuple(round(v) for v in img.info["dpi"])
    assert dpi == (360, 360) == (config.ZAGRAN_DPI, config.ZAGRAN_DPI)


def test_rebake_rereads_meta_before_write(tmp_path, monkeypatch):
    """A concurrent set_paid during the slow render must not be clobbered."""
    results, rid, _ = _rebake_setup(tmp_path, monkeypatch)
    assert not results.load_meta(rid).get("paid")
    orig = results.make_preview_jpeg
    fired = []

    def racing_preview(data):
        if not fired:
            fired.append(1)
            results.set_paid(rid, payment_id="pay-race", tochka_operation_id="op-race")
        return orig(data)

    monkeypatch.setattr(results, "make_preview_jpeg", racing_preview)
    results.rebake_brightness(rid, 1.2)
    meta = results.load_meta(rid)
    assert fired
    assert meta["paid"] is True
    assert meta.get("payment_id") == "pay-race"
    assert meta["brightness"] == pytest.approx(1.2)


def test_post_brightness_409_while_full_render_pending(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import config, results

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    main = _import_app_main()
    rid = results.save_preview_result(
        _jpeg(), _jpeg(), meta={}, digital_base_jpeg=_jpeg(size=(827, 1063))
    )
    assert rid
    results.set_paid(rid, payment_id="pay-p", tochka_operation_id="op-p")
    client = TestClient(main.app)
    # paid, no digital.jpg yet -> full render pending
    r = client.post(f"/api/result/{rid}/brightness", json={"brightness": 1.2})
    assert r.status_code == 409
    assert "render" in r.json()["detail"].lower()
    assert results.load_meta(rid)["brightness"] == pytest.approx(1.0)


def test_brightness_editable_flag(tmp_path, monkeypatch, preview_result):
    from app import config, results

    client, rid = preview_result
    assert client.get(f"/api/result/{rid}").json()["brightness_editable"] is True

    rid2 = results.save_preview_result(_jpeg(), _jpeg(), meta={})  # legacy: no base
    assert rid2
    assert client.get(f"/api/result/{rid2}").json()["brightness_editable"] is False


# ---------------------------------------------------------------------------
# Contrast: GET payload, POST /adjust, /brightness alias, paid promote
# ---------------------------------------------------------------------------


def test_get_result_includes_default_contrast(preview_result):
    client, rid = preview_result
    r = client.get(f"/api/result/{rid}")
    assert r.status_code == 200
    body = r.json()
    assert body["brightness"] == pytest.approx(1.0)
    assert body["contrast"] == pytest.approx(1.0)
    assert body["saturation"] == pytest.approx(1.0)


def test_post_adjust_sets_both(preview_result):
    client, rid = preview_result
    r = client.post(
        f"/api/result/{rid}/adjust",
        json={"brightness": 1.15, "contrast": 1.2, "saturation": 1.1},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["brightness"] == pytest.approx(1.15)
    assert body["contrast"] == pytest.approx(1.2)
    assert body["saturation"] == pytest.approx(1.1)
    g = client.get(f"/api/result/{rid}").json()
    assert g["contrast"] == pytest.approx(1.2)
    assert g["saturation"] == pytest.approx(1.1)


def test_post_adjust_out_of_range_422(preview_result):
    client, rid = preview_result
    r = client.post(
        f"/api/result/{rid}/adjust",
        json={"brightness": 1.0, "contrast": 2.5},
    )
    assert r.status_code == 422
    r2 = client.post(
        f"/api/result/{rid}/adjust",
        json={"brightness": 1.0, "contrast": 1.0, "saturation": 2.5},
    )
    assert r2.status_code == 422


def test_post_brightness_alias_keeps_contrast(preview_result):
    client, rid = preview_result
    client.post(
        f"/api/result/{rid}/adjust",
        json={"brightness": 1.0, "contrast": 1.25, "saturation": 1.18},
    )
    r = client.post(f"/api/result/{rid}/brightness", json={"brightness": 1.1})
    assert r.status_code == 200
    assert r.json()["contrast"] == pytest.approx(1.25)
    assert r.json()["saturation"] == pytest.approx(1.18)
    assert r.json()["brightness"] == pytest.approx(1.1)


def test_post_adjust_unknown_404_and_missing_base_409(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import config, results

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    main = _import_app_main()
    client = TestClient(main.app)
    body = {"brightness": 1.0, "contrast": 1.1}
    assert client.post(f"/api/result/{'a' * 32}/adjust", json=body).status_code == 404
    rid = results.save_preview_result(_jpeg(), _jpeg(), meta={})  # no digital_base
    r = client.post(f"/api/result/{rid}/adjust", json=body)
    assert r.status_code == 409
    assert "digital_base" in r.json()["detail"]


def test_contrast_survives_paid_promote(tmp_path, monkeypatch):
    import numpy as np

    results, rid = _promote_setup(tmp_path, monkeypatch)  # stubs mediapipe if absent
    from app import crop, full_render

    def two_tone_crop(bgr, *, width, height, dpi):
        cropped = np.full((height, width, 3), 80, dtype=np.uint8)
        cropped[:, width // 2 :] = 160
        return cropped, {"fake": True}, {}

    monkeypatch.setattr(crop, "run_crop_stage", two_tone_crop)
    meta = results.load_meta(rid)
    meta["contrast"] = 1.3  # unpaid slider value (no digital_base yet)
    meta["saturation"] = 1.2
    results._write_meta(rid, meta)

    assert full_render.promote_paid_from_edit_gemini(rid, results.load_meta(rid))

    folder = results.result_dir(rid)

    def spread(data: bytes) -> int:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        return img.getpixel((w * 3 // 4, h // 2))[0] - img.getpixel((w // 4, h // 2))[0]

    base_spread = spread((folder / "digital_base.jpg").read_bytes())
    assert base_spread == pytest.approx(80, abs=6)
    assert spread((folder / "digital.jpg").read_bytes()) > base_spread + 15
    assert results.load_meta(rid)["contrast"] == pytest.approx(1.3)
    assert results.load_meta(rid)["saturation"] == pytest.approx(1.2)
