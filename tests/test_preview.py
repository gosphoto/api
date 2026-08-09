import io

from PIL import Image, ImageDraw, ImageFont

from app.preview import WATERMARK_TEXT, make_preview_jpeg


def _solid_jpeg(color=(255, 255, 255), size=(373, 480)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _portrait_on_white_jpeg(size=(373, 480)) -> bytes:
    """Dark subject blob on white — mimics passport cutout."""
    im = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(im)
    # Head + shoulders roughly centered
    cx, cy = size[0] // 2, size[1] // 2
    d.ellipse((cx - 90, cy - 130, cx + 90, cy + 40), fill=(180, 140, 120))
    d.rectangle((cx - 110, cy + 20, cx + 110, size[1] - 20), fill=(30, 50, 90))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_preview_is_smaller_or_equal_side():
    src = _solid_jpeg(size=(1200, 1600))
    out = make_preview_jpeg(src)
    img = Image.open(io.BytesIO(out))
    assert max(img.size) <= 480


def test_preview_watermark_on_white_background():
    src = _solid_jpeg((255, 255, 255))
    out = make_preview_jpeg(src)
    pixels = list(Image.open(io.BytesIO(out)).convert("RGB").getdata())
    darkened = sum(1 for r, g, b in pixels if (r + g + b) / 3 < 245)
    assert darkened > 800, f"too few darkened pixels: {darkened}"


def test_preview_watermark_stays_behind_subject():
    src = _portrait_on_white_jpeg()
    out = make_preview_jpeg(src)
    img = Image.open(io.BytesIO(out)).convert("RGB")
    # Sample cheek area (subject fill color ~180,140,120) — must stay close to source.
    src_img = Image.open(io.BytesIO(src)).convert("RGB")
    cx, cy = img.size[0] // 2, img.size[1] // 2 - 40
    sr, sg, sb = src_img.getpixel((cx, cy))
    pr, pg, pb = img.getpixel((cx, cy))
    delta = abs(sr - pr) + abs(sg - pg) + abs(sb - pb)
    assert delta < 25, f"watermark leaked onto subject: src={(sr,sg,sb)} prev={(pr,pg,pb)}"
    # White margins must show watermark (scan top strip outside subject).
    top = [img.getpixel((x, 8)) for x in range(0, img.size[0], 4)]
    darkened = sum(1 for r, g, b in top if (r + g + b) / 3 < 250)
    assert darkened >= 3, f"no watermark on background strip: darkened={darkened}"


def test_preview_uses_truetype_when_available():
    """Regression: slim Docker without fonts made Cyrillic render as junk."""
    try:
        ImageFont.truetype("/app/fonts/DejaVuSans.ttf", 24)
        available = True
    except OSError:
        try:
            ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 24
            )
            available = True
        except OSError:
            available = False
    if not available:
        return
    from app import preview as preview_mod

    font = preview_mod._font(40)
    im = Image.new("RGB", (10, 10))
    bbox = ImageDraw.Draw(im).textbbox((0, 0), WATERMARK_TEXT, font=font)
    width = bbox[2] - bbox[0]
    assert width > 80, f"Cyrillic watermark width too small ({width}); font likely broken"
