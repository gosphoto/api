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
EDIT_REPLICATE = "edit_replicate.jpg"
OUT_GEMINI = "out_gemini.jpg"
OUT_RIVERFLOW = "out_riverflow.jpg"

# Side files archived next to in.*/out.jpg
# edit_replicate = Replicate bg-remover on white, before Gemini/Riverflow.
ARCHIVE_FILES = frozenset(
    {EDIT_GEMINI, EDIT_RIVERFLOW, EDIT_REPLICATE, OUT_GEMINI, OUT_RIVERFLOW}
)


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


def out_crop_name(model: str | None) -> str:
    """Filename for the post-crop frame from a model."""
    m = (model or "").lower()
    if "riverflow" in m:
        return OUT_RIVERFLOW
    return OUT_GEMINI


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


def save_pair_file(
    folder: Path | str | None,
    jpeg: bytes,
    *,
    name: str,
    model: str | None = None,
) -> Path | None:
    """
    Write an allowed archive JPEG (edit_* / out_*) into an existing pair folder.

    Returns the image path, or None if disabled / empty / IO error.
    """
    fname = name if name in ARCHIVE_FILES else ""
    if not fname:
        log.warning("Skip pair file save reason=bad_name name=%s", name)
        return None
    if not config.PAIRS_ENABLED:
        log.info("Skip pair file save file=%s reason=pairs_disabled", fname)
        return None
    if not jpeg:
        log.info("Skip pair file save file=%s reason=empty_jpeg", fname)
        return None
    if folder is None:
        log.info("Skip pair file save file=%s reason=no_pair_dir", fname)
        return None
    try:
        path = Path(folder)
        if not path.is_dir():
            log.warning(
                "Skip pair file save file=%s reason=not_a_dir path=%s", fname, path
            )
            return None
        out = path / fname
        out.write_bytes(jpeg)

        meta_path = path / "meta.json"
        payload: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        archive = payload.get("archive_files")
        if not isinstance(archive, dict):
            archive = {}
        # Keep legacy pre_crop_edits in sync for edit_* files.
        edits = payload.get("pre_crop_edits")
        if not isinstance(edits, dict):
            edits = {}
        entry = {
            "bytes": len(jpeg),
            "model": model,
            "saved_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        }
        archive[fname] = entry
        payload["archive_files"] = archive
        if fname in (EDIT_GEMINI, EDIT_RIVERFLOW):
            edits[fname] = entry
            payload["pre_crop_edits"] = edits
        payload[fname.replace(".jpg", "")] = fname
        tmp = meta_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(meta_path)
        log.info(
            "Saved pair file=%s model=%s bytes=%s path=%s",
            fname,
            model,
            len(jpeg),
            out,
        )
        return out
    except Exception as e:
        log.warning("Failed to save pair file=%s model=%s: %s", fname, model, e)
        return None


def save_pre_crop_edit(
    folder: Path | str | None,
    jpeg: bytes,
    *,
    model: str | None = None,
    name: str | None = None,
) -> Path | None:
    """Write edit_gemini.jpg / edit_riverflow.jpg (or out_*) into a pair folder."""
    fname = name or pre_crop_edit_name(model)
    if fname not in ARCHIVE_FILES:
        fname = pre_crop_edit_name(model)
    return save_pair_file(folder, jpeg, name=fname, model=model)


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

    Optional pre_crop_edits maps archive filenames (edit_* / out_*) → JPEG bytes.

    Returns the pair folder path, or None if disabled / empty / IO error.
    """
    if not config.PAIRS_ENABLED:
        log.info("Skip process pair save reason=pairs_disabled")
        return None
    if not input_bytes or not output_jpeg:
        log.info(
            "Skip process pair save reason=empty_bytes in=%s out=%s",
            len(input_bytes or b""),
            len(output_jpeg or b""),
        )
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

        archive_meta: dict[str, Any] = {}
        edit_meta: dict[str, Any] = {}
        for fname, jpeg in (pre_crop_edits or {}).items():
            if fname not in ARCHIVE_FILES or not jpeg:
                continue
            (folder / fname).write_bytes(jpeg)
            entry = {"bytes": len(jpeg), "saved_at": ts}
            archive_meta[fname] = entry
            if fname in (EDIT_GEMINI, EDIT_RIVERFLOW):
                edit_meta[fname] = entry
            log.info(
                "Wrote archive into pair file=%s bytes=%s dir=%s",
                fname,
                len(jpeg),
                folder.name,
            )

        payload = {
            "saved_at": ts,
            "original_filename": filename,
            "in": in_path.name,
            "out": out_path.name,
            "in_bytes": len(input_bytes),
            "out_bytes": len(output_jpeg),
            **(meta or {}),
        }
        if archive_meta:
            payload["archive_files"] = archive_meta
            for fname in archive_meta:
                payload[fname.replace(".jpg", "")] = fname
        if edit_meta:
            payload["pre_crop_edits"] = edit_meta
        (folder / "meta.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(
            "Saved process pair dir=%s in_bytes=%s out_bytes=%s archive=%s",
            folder,
            len(input_bytes),
            len(output_jpeg),
            {k: v.get("bytes") for k, v in archive_meta.items()},
        )
        return folder
    except Exception as e:
        log.warning("Failed to save process pair: %s", e)
        return None
