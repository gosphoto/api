"""OpenRouter Image API — Riverflow white-bg edit (+ legacy /images helper)."""

from __future__ import annotations

import base64
import logging
import re
from typing import Any

import httpx

from . import config

log = logging.getLogger("gosphoto-gate")

EDIT_PROMPT = (
    "STRICT edit scope for a Russian passport / Gosuslugi photo. "
    "ALLOWED only: (1) replace background with transparent alpha or pure #FFFFFF; "
    "(2) clean the shoulder/clothing outline BELOW the neck. "
    "FORBIDDEN: any change to the head — face, eyes, brows, nose, mouth, ears, "
    "skin, pores, freckles, wrinkles, stubble, expression, makeup, hair style/color/volume. "
    "The entire head must look pixel-identical to the input photo; do not redraw, "
    "beautify, morph, smooth, relight, or re-age the face. "
    "Do not invent a new face. Keep identity 100%. "
    "Only shoulders and background may differ. Remove colour spill / halo on clothes edges."
)

GOSUSLUGI_EDIT_PROMPT = (
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

GOSUSLUGI_SCORING_PROMPT = (
    "Prefer a pure #FFFFFF studio background with no room props or shadows on the "
    "backdrop; keep face identity pixel-faithful to the input selfie; never punch "
    "white holes into hair; clean ear edges without erasing cartilage."
)

GOSUSLUGI_SCORING_RUBRIC: list[dict[str, Any]] = [
    {
        "key": "white_bg",
        "label": "Pure white background",
        "description": "Backdrop is solid #FFFFFF with no shadows, vignette, or room objects.",
        "weight": 0.3,
    },
    {
        "key": "identity",
        "label": "Identity fidelity",
        "description": "Same face, age, skin texture, hair, clothing as the input; no beautify.",
        "weight": 0.3,
    },
    {
        "key": "hair_integrity",
        "label": "Hair without white holes",
        "description": "No swiss-cheese / salt-pepper white gaps inside the hair mass.",
        "weight": 0.25,
    },
    {
        "key": "ear_edges",
        "label": "Clean ear edges",
        "description": "No wall crumbs behind ears; cartilage preserved.",
        "weight": 0.15,
    },
]


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


def build_edit_payload(image_bytes: bytes, mime: str = "image/jpeg") -> dict[str, Any]:
    transparent = bool(config.OPENROUTER_TRANSPARENT_BG)
    model = config.OPENROUTER_IMAGE_MODEL
    payload: dict[str, Any] = {
        "model": model,
        "prompt": EDIT_PROMPT,
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


def _post_json(
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    if not config.OPENROUTER_API_KEY:
        raise OpenRouterError("OPENROUTER_API_KEY is not configured")
    url = f"{config.OPENROUTER_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    with httpx.Client(timeout=timeout or config.OPENROUTER_TIMEOUT_SEC) as client:
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


def edit_selfie(image_bytes: bytes, mime: str = "image/jpeg") -> bytes:
    """Legacy OpenRouter /images edit (non-Riverflow models)."""
    data = _post_json(
        "images",
        build_edit_payload(image_bytes, mime),
        timeout=config.OPENROUTER_TIMEOUT_SEC,
    )
    return _extract_image_bytes_from_images_api(data)


def _riverflow_image_config() -> dict[str, Any]:
    bg_mode = config.RIVERFLOW_BG_MODE or "solid"
    if bg_mode not in ("solid", "transparent", "original"):
        bg_mode = "solid"
    cfg: dict[str, Any] = {
        "aspect_ratio": "3:4",
        "image_size": config.RIVERFLOW_IMAGE_SIZE or "1K",
        "background_mode": bg_mode,
        "scoring_prompt": GOSUSLUGI_SCORING_PROMPT,
        "scoring_rubric": GOSUSLUGI_SCORING_RUBRIC,
    }
    if bg_mode == "solid":
        hex_color = config.RIVERFLOW_BG_HEX or "#FFFFFF"
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", hex_color):
            hex_color = "#FFFFFF"
        cfg["background_hex_color"] = hex_color
    return cfg


def build_riverflow_images_payload(
    image_bytes: bytes,
    mime: str = "image/jpeg",
    *,
    prompt: str | None = None,
) -> dict[str, Any]:
    bg_mode = config.RIVERFLOW_BG_MODE or "solid"
    if bg_mode not in ("solid", "transparent", "original"):
        bg_mode = "solid"
    out_fmt = "png" if bg_mode == "transparent" else "jpeg"
    payload: dict[str, Any] = {
        "model": config.RIVERFLOW_MODEL,
        "prompt": prompt or GOSUSLUGI_EDIT_PROMPT,
        "aspect_ratio": "3:4",
        "resolution": config.RIVERFLOW_IMAGE_SIZE or "1K",
        "output_format": out_fmt,
        "input_references": [
            {
                "type": "image_url",
                "image_url": {"url": _data_url(image_bytes, mime)},
            }
        ],
        "image_config": _riverflow_image_config(),
    }
    effort = (config.RIVERFLOW_REASONING or "medium").strip().lower()
    if effort and effort != "none":
        payload["reasoning"] = {"effort": effort}
    return payload


def edit_selfie_riverflow(
    image_bytes: bytes,
    mime: str = "image/jpeg",
) -> bytes:
    """Riverflow v2.5 (fast/pro) — solid/transparent bg via OpenRouter /images."""
    payload = build_riverflow_images_payload(
        image_bytes, mime, prompt=GOSUSLUGI_EDIT_PROMPT
    )
    data = _post_json(
        "images",
        payload,
        timeout=config.RIVERFLOW_TIMEOUT_SEC,
    )
    return _extract_image_bytes_from_images_api(data)
