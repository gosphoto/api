"""Persist successful process pairs: input upload + output JPEG."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import config

log = logging.getLogger("gosphoto-gate")

_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")

EDIT_GEMINI = "edit_gemini.jpg"
EDIT_RIVERFLOW = "edit_riverflow.jpg"


def _safe_stem(name: str | None) -> str:
    raw = (name or "upload").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = Path(raw).stem or "upload"
    cleaned = _SAFE.sub("_", stem).strip("._")[:40]
    return cleaned or "upload"


def _in_ext(data: bytes, filename: str | None) -> str:
    name = (filename or "").lower()
    if name.endswith(".png"):
        return ".png"
    if name.endswith(".webp"):
        return ".webp"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def pre_crop_edit_name(model: str | None) -> str:
    """Filename for the post-model / pre-crop frame."""
    m = (model or "").lower()
    if "riverflow" in m:
        return EDIT_RIVERFLOW
    return EDIT_GEMINI


def encode_pre_crop_jpeg(bgr: np.ndarray, *, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("pre_crop_jpeg_encode_failed")
    return buf.tobytes()


def resolve_pair_dir(rel: str | None) -> Path | None:
    """Resolve a pairs-relative folder from meta.source_pair."""
    if not isinstance(rel, str) or not rel or rel.startswith(("/", "\\")) or ".." in rel:
        return None
    root = Path(config.PAIRS_DIR).resolve()
    folder = (root / rel).resolve()
    if folder != root and root not in folder.parents:
        return None
    if not folder.is_dir():
        return None
    return folder


def save_pre_crop_edit(
    folder: Path | str | None,
    jpeg: bytes,
    *,
    model: str | None = None,
    name: str | None = None,
) -> Path | None:
    """
    Write edit_gemini.jpg or edit_riverflow.jpg into an existing pair folder.

    Returns the image path, or None if disabled / empty / IO error.
    """
    if not config.PAIRS_ENABLED or not jpeg or folder is None:
        return None
    try:
        path = Path(folder)
        if not path.is_dir():
            return None
        fname = name or pre_crop_edit_name(model)
        if fname not in (EDIT_GEMINI, EDIT_RIVERFLOW):
            fname = pre_crop_edit_name(model)
        out = path / fname
        out.write_bytes(jpeg)

        meta_path = path / "meta.json"
        payload: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        edits = payload.get("pre_crop_edits")
        if not isinstance(edits, dict):
            edits = {}
        edits[fname] = {
            "bytes": len(jpeg),
            "model": model,
            "saved_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        }
        payload["pre_crop_edits"] = edits
        payload[fname.replace(".jpg", "")] = fname
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(meta_path)
        log.info("Saved pre-crop edit model=%s path=%s", model, out)
        return out
    except Exception as e:
        log.warning("Failed to save pre-crop edit: %s", e)
        return None


def save_pair(
    input_bytes: bytes,
    output_jpeg: bytes,
    *,
    filename: str | None = None,
    meta: dict[str, Any] | None = None,
    pre_crop_edits: dict[str, bytes] | None = None,
) -> Path | None:
    """
    Create pairs/<timestamp>_<stem>/ with in.* and out.jpg (+ meta.json).

    Optional pre_crop_edits maps edit_gemini.jpg / edit_riverflow.jpg → JPEG bytes
    (model output before passport crop).

    Returns the pair folder path, or None if disabled / empty / IO error.
    """
    if not config.PAIRS_ENABLED or not input_bytes or not output_jpeg:
        return None
    try:
        root = Path(config.PAIRS_DIR)
        root.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder = root / f"{ts}_{_safe_stem(filename)}"
        if folder.exists():
            folder = root / f"{ts}_{_safe_stem(filename)}_{len(input_bytes)}"
        folder.mkdir(parents=True, exist_ok=False)

        in_path = folder / f"in{_in_ext(input_bytes, filename)}"
        out_path = folder / "out.jpg"
        in_path.write_bytes(input_bytes)
        out_path.write_bytes(output_jpeg)

        edit_meta: dict[str, Any] = {}
        for fname, jpeg in (pre_crop_edits or {}).items():
            if fname not in (EDIT_GEMINI, EDIT_RIVERFLOW) or not jpeg:
                continue
            (folder / fname).write_bytes(jpeg)
            edit_meta[fname] = {
                "bytes": len(jpeg),
                "saved_at": ts,
            }

        payload = {
            "saved_at": ts,
            "original_filename": filename,
            "in": in_path.name,
            "out": out_path.name,
            "in_bytes": len(input_bytes),
            "out_bytes": len(output_jpeg),
            **(meta or {}),
        }
        if edit_meta:
            payload["pre_crop_edits"] = edit_meta
            for fname in edit_meta:
                payload[fname.replace(".jpg", "")] = fname
        (folder / "meta.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Saved process pair dir=%s pre_crop=%s", folder, list(edit_meta))
        return folder
    except Exception as e:
        log.warning("Failed to save process pair: %s", e)
        return None
