# tests/test_brightness.py
import io
import pytest
from PIL import Image

from app.brightness import (
    DEFAULT_BRIGHTNESS,
    apply_brightness,
    is_dirty,
    parse_brightness,
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
