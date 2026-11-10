"""Hosted background removal via Replicate (851-labs/background-remover)."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from . import config

log = logging.getLogger("gosphoto-gate")

# Cloudflare blocks Python-urllib UA; keep a browser-like client string.
_USER_AGENT = "gosphoto-gate/1.0 (+https://gosphoto.ru)"


class ReplicateBgError(RuntimeError):
    """Replicate cutout failed; callers may fail-open."""

    def __init__(self, message: str, *, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


def _headers(*, content_type: str | None = None) -> dict[str, str]:
    token = config.REPLICATE_API_TOKEN
    if not token:
        raise ReplicateBgError("REPLICATE_API_TOKEN is not configured")
    h = {
        "Authorization": f"Bearer {token}",
        "User-Agent": _USER_AGENT,
    }
    if content_type:
        h["Content-Type"] = content_type
    return h


def _model_slug() -> str:
    slug = (config.REPLICATE_BG_MODEL or "851-labs/background-remover").strip()
    if "/" not in slug:
        raise ReplicateBgError(f"invalid REPLICATE_BG_MODEL: {slug!r}")
    return slug


def resolve_version(client: httpx.Client) -> str:
    pinned = (config.REPLICATE_BG_VERSION or "").strip()
    if pinned:
        return pinned
    slug = _model_slug()
    url = f"{config.REPLICATE_API_BASE}/models/{slug}"
    resp = client.get(url, headers=_headers())
    if resp.status_code >= 400:
        raise ReplicateBgError(
            f"Replicate model lookup failed ({resp.status_code})",
            status=resp.status_code,
            body=_safe_body(resp),
        )
    data = resp.json()
    version = ((data.get("latest_version") or {}).get("id") or "").strip()
    if not version:
        raise ReplicateBgError("Replicate model has no latest_version.id")
    return version


def upload_image(client: httpx.Client, image_bytes: bytes, *, filename: str = "in.jpg") -> str:
    """Upload bytes via Files API; return the authenticated get URL."""
    url = f"{config.REPLICATE_API_BASE}/files"
    files = {"content": (filename, image_bytes, "image/jpeg")}
    resp = client.post(url, headers=_headers(), files=files)
    if resp.status_code >= 400:
        raise ReplicateBgError(
            f"Replicate file upload failed ({resp.status_code})",
            status=resp.status_code,
            body=_safe_body(resp),
        )
    data = resp.json()
    get_url = ((data.get("urls") or {}).get("get") or "").strip()
    if not get_url:
        raise ReplicateBgError("Replicate file upload missing urls.get")
    return get_url


def create_prediction(
    client: httpx.Client,
    *,
    version: str,
    image_url: str,
) -> dict[str, Any]:
    url = f"{config.REPLICATE_API_BASE}/predictions"
    payload = {
        "version": version,
        "input": {
            "image": image_url,
            "format": "png",
            "background_type": "rgba",
            "threshold": 0,
            "reverse": False,
        },
    }
    resp = client.post(url, headers=_headers(content_type="application/json"), json=payload)
    if resp.status_code >= 400:
        raise ReplicateBgError(
            f"Replicate prediction create failed ({resp.status_code})",
            status=resp.status_code,
            body=_safe_body(resp),
        )
    return resp.json()


def wait_prediction(client: httpx.Client, prediction_id: str) -> dict[str, Any]:
    url = f"{config.REPLICATE_API_BASE}/predictions/{prediction_id}"
    deadline = time.monotonic() + float(config.REPLICATE_TIMEOUT_SEC)
    interval = max(0.2, float(config.REPLICATE_POLL_INTERVAL_SEC))
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        resp = client.get(url, headers=_headers())
        if resp.status_code >= 400:
            raise ReplicateBgError(
                f"Replicate prediction poll failed ({resp.status_code})",
                status=resp.status_code,
                body=_safe_body(resp),
            )
        last = resp.json()
        status = (last.get("status") or "").strip()
        if status in ("succeeded", "failed", "canceled"):
            return last
        time.sleep(interval)
    raise ReplicateBgError(
        f"Replicate prediction timed out after {config.REPLICATE_TIMEOUT_SEC}s",
        body=last or None,
    )


def _output_url(pred: dict[str, Any]) -> str:
    out = pred.get("output")
    if isinstance(out, str) and out.startswith("http"):
        return out
    if isinstance(out, list):
        for item in out:
            if isinstance(item, str) and item.startswith("http"):
                return item
    if isinstance(out, dict):
        for item in out.values():
            if isinstance(item, str) and item.startswith("http"):
                return item
    raise ReplicateBgError("Replicate prediction missing output URL", body=pred)


def download_bytes(client: httpx.Client, url: str) -> bytes:
    resp = client.get(url, headers={"User-Agent": _USER_AGENT}, follow_redirects=True)
    if resp.status_code >= 400:
        raise ReplicateBgError(
            f"Replicate output download failed ({resp.status_code})",
            status=resp.status_code,
            body=resp.text[:500],
        )
    data = resp.content
    if not data:
        raise ReplicateBgError("Replicate output download empty")
    return data


def _safe_body(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return resp.text[:500]


def remove_background(image_bytes: bytes, *, mime: str = "image/jpeg") -> tuple[bytes, dict[str, Any]]:
    """Return PNG RGBA cutout bytes and timing/meta.

    Raises ReplicateBgError on failure.
    """
    if not image_bytes:
        raise ReplicateBgError("empty image_bytes")
    timeout = httpx.Timeout(float(config.REPLICATE_TIMEOUT_SEC))
    meta: dict[str, Any] = {
        "model": _model_slug(),
        "applied": False,
    }
    with httpx.Client(timeout=timeout) as client:
        version = resolve_version(client)
        meta["version"] = version
        filename = "in.jpg" if "jpeg" in (mime or "").lower() or "jpg" in (mime or "").lower() else "in.png"
        image_url = upload_image(client, image_bytes, filename=filename)
        pred = create_prediction(client, version=version, image_url=image_url)
        pred_id = (pred.get("id") or "").strip()
        if not pred_id:
            raise ReplicateBgError("Replicate prediction missing id", body=pred)
        meta["prediction_id"] = pred_id
        if (pred.get("status") or "") != "succeeded":
            pred = wait_prediction(client, pred_id)
        status = (pred.get("status") or "").strip()
        if status != "succeeded":
            raise ReplicateBgError(
                f"Replicate prediction {status or 'unknown'}: {pred.get('error')}",
                body=pred,
            )
        metrics = pred.get("metrics") or {}
        if isinstance(metrics, dict):
            meta["predict_time"] = metrics.get("predict_time")
            meta["total_time"] = metrics.get("total_time")
        out_url = _output_url(pred)
        meta["output_url"] = out_url
        png = download_bytes(client, out_url)
    meta["applied"] = True
    meta["ok"] = True
    log.info(
        "Replicate cutout ok model=%s pred=%s predict_time=%s bytes=%s",
        meta.get("model"),
        meta.get("prediction_id"),
        meta.get("predict_time"),
        len(png),
    )
    return png, meta
