import io

from PIL import Image

from app.preview import make_preview_jpeg


def _solid_jpeg(color=(255, 255, 255), size=(373, 480)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_preview_is_smaller_or_equal_side():
    src = _solid_jpeg(size=(1200, 1600))
    out = make_preview_jpeg(src)
    img = Image.open(io.BytesIO(out))
    assert max(img.size) <= 480


def test_preview_has_no_watermark_on_white():
    src = _solid_jpeg((255, 255, 255))
    out = make_preview_jpeg(src)
    pixels = list(Image.open(io.BytesIO(out)).convert("RGB").getdata())
    darkened = sum(1 for r, g, b in pixels if (r + g + b) / 3 < 250)
    assert darkened == 0, f"unexpected non-white pixels: {darkened}"


def test_preview_preserves_subject_colors():
    src = _solid_jpeg((180, 140, 120), size=(373, 480))
    out = make_preview_jpeg(src)
    img = Image.open(io.BytesIO(out)).convert("RGB")
    r, g, b = img.getpixel((img.size[0] // 2, img.size[1] // 2))
    delta = abs(r - 180) + abs(g - 140) + abs(b - 120)
    assert delta < 15, f"color drift: {(r, g, b)}"
