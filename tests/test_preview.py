import io

from PIL import Image

from app.preview import WATERMARK_TEXT, make_preview_jpeg


def _solid_jpeg(color=(255, 255, 255), size=(373, 480)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_preview_is_smaller_or_equal_side():
    src = _solid_jpeg(size=(1200, 1600))
    out = make_preview_jpeg(src)
    img = Image.open(io.BytesIO(out))
    assert max(img.size) <= 480


def test_preview_has_watermark_on_white():
    src = _solid_jpeg((255, 255, 255))
    out = make_preview_jpeg(src)
    pixels = list(Image.open(io.BytesIO(out)).convert("RGB").getdata())
    darkened = sum(1 for r, g, b in pixels if (r + g + b) / 3 < 250)
    assert darkened > 200, f"expected watermark pixels, got {darkened}"


def test_preview_preserves_subject_colors_away_from_strip():
    """Face-ish mid band should stay close to source; strip/watermark may tint edges."""
    src = _solid_jpeg((180, 140, 120), size=(373, 480))
    out = make_preview_jpeg(src)
    img = Image.open(io.BytesIO(out)).convert("RGB")
    # Sample a few mid-frame points; watermark is translucent so allow drift.
    samples = [
        img.getpixel((img.size[0] // 2, img.size[1] // 3)),
        img.getpixel((img.size[0] // 3, img.size[1] // 2)),
        img.getpixel((2 * img.size[0] // 3, img.size[1] // 2)),
    ]
    ok = False
    for r, g, b in samples:
        delta = abs(r - 180) + abs(g - 140) + abs(b - 120)
        if delta < 80:
            ok = True
            break
    assert ok, f"subject colors lost under watermark: {samples}"


def test_watermark_constant_exported():
    assert WATERMARK_TEXT == "ГОСФОТО"
