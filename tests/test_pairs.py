import json
from pathlib import Path

import numpy as np

from app import config
from app.pairs import (
    EDIT_GEMINI,
    EDIT_RIVERFLOW,
    encode_pre_crop_jpeg,
    pre_crop_edit_name,
    save_pair,
    save_pre_crop_edit,
)


def test_save_pair_writes_in_out_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PAIRS_DIR", tmp_path)
    monkeypatch.setattr(config, "PAIRS_ENABLED", True)
    folder = save_pair(
        b"\xff\xd8\xffin",
        b"\xff\xd8\xffout",
        filename="selfie.jpg",
        meta={"model": "test"},
    )
    assert folder is not None
    assert (folder / "in.jpg").read_bytes() == b"\xff\xd8\xffin"
    assert (folder / "out.jpg").read_bytes() == b"\xff\xd8\xffout"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "test"
    assert meta["in"] == "in.jpg"
    assert meta["out"] == "out.jpg"


def test_save_pair_writes_pre_crop_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PAIRS_DIR", tmp_path)
    monkeypatch.setattr(config, "PAIRS_ENABLED", True)
    folder = save_pair(
        b"\xff\xd8\xffin",
        b"\xff\xd8\xffout",
        filename="selfie.jpg",
        pre_crop_edits={
            EDIT_GEMINI: b"\xff\xd8\xffgem",
            EDIT_RIVERFLOW: b"\xff\xd8\xffrf",
        },
    )
    assert folder is not None
    assert (folder / EDIT_GEMINI).read_bytes() == b"\xff\xd8\xffgem"
    assert (folder / EDIT_RIVERFLOW).read_bytes() == b"\xff\xd8\xffrf"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert EDIT_GEMINI in meta["pre_crop_edits"]
    assert EDIT_RIVERFLOW in meta["pre_crop_edits"]


def test_save_pre_crop_edit_updates_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PAIRS_DIR", tmp_path)
    monkeypatch.setattr(config, "PAIRS_ENABLED", True)
    folder = save_pair(b"\xff\xd8\xffin", b"\xff\xd8\xffout", filename="x.jpg")
    path = save_pre_crop_edit(
        folder,
        b"\xff\xd8\xffrf",
        model="sourceful/riverflow-v2.5-pro",
    )
    assert path is not None
    assert path.name == EDIT_RIVERFLOW
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["pre_crop_edits"][EDIT_RIVERFLOW]["model"] == (
        "sourceful/riverflow-v2.5-pro"
    )


def test_pre_crop_edit_name_and_encode():
    assert pre_crop_edit_name("google/gemini-2.5-flash-image") == EDIT_GEMINI
    assert pre_crop_edit_name("sourceful/riverflow-v2.5-pro") == EDIT_RIVERFLOW
    jpeg = encode_pre_crop_jpeg(np.zeros((8, 8, 3), dtype=np.uint8))
    assert jpeg[:3] == b"\xff\xd8\xff"


def test_save_pair_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PAIRS_DIR", tmp_path)
    monkeypatch.setattr(config, "PAIRS_ENABLED", False)
    assert save_pair(b"a", b"b") is None
    assert list(tmp_path.iterdir()) == []
