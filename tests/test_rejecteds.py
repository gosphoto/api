import json
from pathlib import Path

from app import config
from app.rejecteds import save_rejected


def test_save_rejected_writes_image_and_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REJECTEDS_DIR", tmp_path)
    monkeypatch.setattr(config, "REJECTEDS_ENABLED", True)
    data = b"\xff\xd8\xfffakejpeg"
    path = save_rejected(
        data,
        reason="no_face",
        message="no face",
        metrics={"face_count": 0},
        filename="bad selfie!.jpg",
    )
    assert path is not None
    assert path.is_file()
    assert path.read_bytes() == data
    assert "no_face" in path.name
    meta_path = Path(str(path) + ".json")
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["reason"] == "no_face"
    assert meta["bytes"] == len(data)


def test_save_rejected_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REJECTEDS_DIR", tmp_path)
    monkeypatch.setattr(config, "REJECTEDS_ENABLED", False)
    assert save_rejected(b"abc", reason="blur") is None
    assert list(tmp_path.iterdir()) == []


def test_save_rejected_writes_bisenet_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REJECTEDS_DIR", tmp_path)
    monkeypatch.setattr(config, "REJECTEDS_ENABLED", True)
    path = save_rejected(
        b"\xff\xd8\xfffakejpeg",
        reason="bangs_on_brows",
        filename="fringe.jpg",
        bisenet_overlay_png=b"overlay-bytes",
        bisenet_mask_png=b"mask-bytes",
    )
    assert path is not None
    side = tmp_path / f"{path.stem}_bisenet"
    assert (side / "overlay.png").read_bytes() == b"overlay-bytes"
    assert (side / "mask.png").read_bytes() == b"mask-bytes"
    meta = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    assert meta["bisenet_dir"] == side.name
