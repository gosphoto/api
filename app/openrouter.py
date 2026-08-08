"""OpenRouter Image API — white/transparent background edit, no precise crop."""

from __future__ import annotations

import base64
import logging
import re
from typing import Any

import httpx

from . import config

log = logging.getLogger("gosphoto-gate")

EDIT_PROMPT = (
    "Russian passport / Gosuslugi photo. PRIMARY RULE: stay as close as possible "
    "to the ORIGINAL input — treat it as a copy-paste of the person onto a white studio. "
    "ALLOWED only: (1) replace background with transparent alpha or pure #FFFFFF; "
    "(2) minor cleanup of clothing/shoulder outline against white. "
    "MUST KEEP FROM ORIGINAL (do not invent or restyle): face identity, eyes, brows, "
    "nose, mouth, ears, skin tone and texture (pores, freckles, redness, wrinkles), "
    "neck color matching the original neck, hair shape/color/texture/strands "
    "(no cartoon, plastic, or smoothed hair), expression, makeup, age, lighting on skin. "
    "Prefer near pixel-identical head and hair; do not redraw, beautify, morph, "
    "relight, recolor, or generate a new person. "
    "BACKGROUND RULE: pure #FFFFFF only OUTSIDE the outer hair silhouette and "
    "where the room wall clearly shows through sparse outer strands. "
    "FORBIDDEN inside the hair mass: white holes, swiss-cheese gaps, salt-and-pepper "
    "white speckles, or cutting light/blonde strands into #FFFFFF — those are hair "
    "(including highlights), not background. If unsure whether a bright pixel is "
    "hair or wall, keep the hair. Also clean room crumbs behind ears / ear edges "
    "without erasing cartilage. "
    "If unsure about face/identity pixels, leave them unchanged. "
    "Only background (and tiny edge cleanup on clothes) may change."
)

# Nano Banana Pro — RF / Gosuslugi passport requirements (pass 1).
GOSUSLUGI_NANO_PROMPT = (
    "Сделай фото на документы для Госуслуг / загранпаспорта РФ строго по требованиям. "
    "Один кадр: человек на чистом белом фоне #FFFFFF, без теней на фоне, без виньетки. "
    "Формат портрета: лицо анфас, взгляд в камеру, нейтральное выражение, плечи видны. "
    "КРИТИЧНО — сохранить идентичность 1:1 с исходным селфи: "
    "то же лицо, возраст, черты, цвет и текстура кожи (включая шею и лоб — без «маски»), "
    "те же волосы (натуральные пряди, без мультяшной заливки), одежда, украшения. "
    "Не ретушируй, не омолаживай, не меняй макияж, свет на коже и геометрию головы. "
    "Разрешено только: заменить фон на чисто белый и слегка подчистить контур одежды/плеч. "
    "ФОН: чисто белый #FFFFFF только СНАРУЖИ силуэта причёски и там, где сквозь "
    "редкие крайние пряди реально видна стена комнаты. "
    "ВНУТРИ массы волос запрещены белые дыры, «сыр», соль-перец и вырезание "
    "светлых/блондовых прядей или бликов в #FFFFFF — это волосы, не фон. "
    "Если сомневаешься (волос или стена) — оставь волос. "
    "За ушами и по краям раковин убери куски стены/ореол, но не хрящ уха. "
    "Без водяных знаков, текста, рамок, фильтров. Высокое качество, естественный вид."
)

class OpenRouterError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


def _data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _aspect_ratio_for_model(model: str) -> str:
    """gpt-image-* accepts 1:1|3:2|2:3|auto — not 3:4 (causes HTTP 400)."""
    m = (model or "").lower()
    if "gpt-image" in m:
        return "2:3"
    return "3:4"


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://gosphoto.ru",
        "X-Title": "Gosphoto",
    }


def build_edit_payload(
    image_bytes: bytes,
    mime: str = "image/jpeg",
    *,
    model: str | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    transparent = bool(config.OPENROUTER_TRANSPARENT_BG)
    model = model or config.OPENROUTER_IMAGE_MODEL
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt or EDIT_PROMPT,
        "aspect_ratio": _aspect_ratio_for_model(model),
        "output_format": "png" if transparent else "jpeg",
        "input_references": [
            {
                "type": "image_url",
                "image_url": {"url": _data_url(image_bytes, mime)},
            }
        ],
    }
    if transparent:
        payload["background"] = "transparent"
    return payload


def _decode_b64_image(b64: str) -> bytes:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def _extract_image_bytes_from_images_api(data: dict[str, Any]) -> bytes:
    items = data.get("data") or []
    if not items:
        raise OpenRouterError("OpenRouter returned no image", body=data)
    item = items[0]
    b64 = item.get("b64_json")
    if b64:
        return _decode_b64_image(b64)
    img_url = item.get("url")
    if img_url:
        with httpx.Client(timeout=60) as client:
            r = client.get(img_url)
            r.raise_for_status()
            return r.content
    raise OpenRouterError("OpenRouter image missing b64_json/url", body=item)


def _extract_image_bytes_from_chat(data: dict[str, Any]) -> bytes:
    """Parse chat/completions multimodal image response."""
    choices = data.get("choices") or []
    if not choices:
        raise OpenRouterError("OpenRouter chat returned no choices", body=data)
    msg = choices[0].get("message") or {}

    for img in msg.get("images") or []:
        url = (
            (img.get("image_url") or {}).get("url")
            if isinstance(img, dict)
            else None
        )
        if not url and isinstance(img, dict):
            url = img.get("url") or img.get("b64_json")
        if isinstance(url, str) and url:
            if url.startswith("data:") or re.fullmatch(r"[A-Za-z0-9+/=\s]+", url[:80]):
                try:
                    return _decode_b64_image(url)
                except Exception:
                    pass
            if url.startswith("http"):
                with httpx.Client(timeout=60) as client:
                    r = client.get(url)
                    r.raise_for_status()
                    return r.content

    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("image_url", "output_image", "image"):
                url = (part.get("image_url") or part.get("image") or {}).get("url")
                if not url:
                    url = part.get("url") or part.get("b64_json")
                if isinstance(url, str) and url:
                    if url.startswith("http"):
                        with httpx.Client(timeout=60) as client:
                            r = client.get(url)
                            r.raise_for_status()
                            return r.content
                    return _decode_b64_image(url)
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                m = re.search(
                    r"data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)", part["text"]
                )
                if m:
                    return base64.b64decode(m.group(1))

    if isinstance(content, str) and "base64," in content:
        m = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)", content)
        if m:
            return base64.b64decode(m.group(1))

    raise OpenRouterError("OpenRouter chat response missing image", body=data)


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not config.OPENROUTER_API_KEY:
        raise OpenRouterError("OPENROUTER_API_KEY is not configured")
    url = f"{config.OPENROUTER_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    with httpx.Client(timeout=config.OPENROUTER_TIMEOUT_SEC) as client:
        resp = client.post(url, json=payload, headers=_headers())
    if resp.status_code >= 400:
        try:
            err_body = resp.json()
        except Exception:
            err_body = resp.text[:500]
        log.error("OpenRouter error %s %s: %s", path, resp.status_code, err_body)
        raise OpenRouterError(
            f"OpenRouter request failed ({resp.status_code})",
            status=resp.status_code,
            body=err_body,
        )
    return resp.json()


def edit_selfie(
    image_bytes: bytes,
    mime: str = "image/jpeg",
    *,
    model: str | None = None,
    prompt: str | None = None,
) -> bytes:
    payload = build_edit_payload(image_bytes, mime, model=model, prompt=prompt)
    data = _post_json("images", payload)
    return _extract_image_bytes_from_images_api(data)


def _nano_banana_chat(
    prompt: str,
    image_parts: list[tuple[bytes, str]],
) -> bytes:
    """Chat/completions image edit; fallback to /images using the last image."""
    model = config.NANO_BANANA_MODEL
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img_bytes, mime in image_parts:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _data_url(img_bytes, mime)},
            }
        )
    chat_payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["image", "text"],
        "image_config": {"aspect_ratio": "3:4"},
    }
    try:
        data = _post_json("chat/completions", chat_payload)
        return _extract_image_bytes_from_chat(data)
    except OpenRouterError as e:
        log.warning("Nano Banana chat path failed (%s), trying /images", e)

    last_bytes, last_mime = image_parts[-1]
    return edit_selfie(last_bytes, last_mime, model=model, prompt=prompt)


def edit_selfie_nano_banana(
    image_bytes: bytes,
    mime: str = "image/jpeg",
) -> bytes:
    """Nano Banana Pro for Gosuslugi white-bg (one-pass)."""
    return _nano_banana_chat(GOSUSLUGI_NANO_PROMPT, [(image_bytes, mime)])
