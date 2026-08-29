import hashlib
import io
import json
from datetime import datetime, timezone

from PIL import Image

from app import config
from app import process_cache as cache_mod
from app.process_cache import content_hash, lookup, put
from app.results import clone_result, load_file, load_meta, save_result


def _jpeg(color=(10, 20, 30), *, quality=92, size=(40, 50)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _enable_cache(tmp_path, monkeypatch, *, version="1"):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    monkeypatch.setattr(config, "PROCESS_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "PROCESS_CACHE_ENABLED", True)
    monkeypatch.setattr(config, "PROCESS_CACHE_VERSION", version)


def _freeze_utc_day(monkeypatch, ymd: str) -> None:
    year, month, day = int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8])

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(year, month, day, 12, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(cache_mod, "datetime", FrozenDateTime)


def test_content_hash_stable():
    data = b"same-bytes"
    assert content_hash(data) == content_hash(data)
    assert content_hash(b"other") != content_hash(data)


def test_clone_result_copies_files_and_resets_payment(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    dig = _jpeg()
    prn = _jpeg((40, 50, 60))
    source_id = save_result(
        dig,
        prn,
        meta={"doc_type": "passport_rf", "paid": True, "payment_id": "x"},
    )
    assert source_id is not None
    cloned_id = clone_result(source_id)
    assert cloned_id is not None
    assert cloned_id != source_id
    meta = load_meta(cloned_id)
    assert meta is not None
    assert meta["paid"] is False
    assert meta["cloned_from"] == source_id
    assert "payment_id" not in meta
    assert load_file(cloned_id, "digital.jpg") == dig
    assert load_file(cloned_id, "print.jpg") == prn


def test_process_cache_put_and_lookup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    monkeypatch.setattr(config, "PROCESS_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "PROCESS_CACHE_ENABLED", True)
    monkeypatch.setattr(config, "PROCESS_CACHE_VERSION", "1")

    data = b"oriented-upload-bytes"
    dig = _jpeg()
    prn = _jpeg((1, 2, 3))
    source_id = save_result(
        dig,
        prn,
        meta={
            "doc_type": "passport_rf",
            "doc_label": "Паспорт РФ",
            "pipeline": ["gate", "riverflow", "crop", "print_10x15"],
            "edit": {"model": "test-model"},
            "compliance": {"pass": True},
            "print_sheet": {"copies": 4},
        },
    )
    assert source_id is not None

    put(data, "passport_rf", source_id)
    cache_files = list((tmp_path / "cache").rglob("*.json"))
    assert len(cache_files) == 1
    entry = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert entry["result_id"] == source_id

    hit_id = lookup(data, "passport_rf")
    assert hit_id is not None
    assert hit_id != source_id
    assert load_file(hit_id, "digital.jpg") == dig

    assert lookup(data, "zagran") is None
    assert lookup(b"other-bytes", "passport_rf") is None


def test_process_cache_put_is_first_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    monkeypatch.setattr(config, "PROCESS_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "PROCESS_CACHE_ENABLED", True)
    monkeypatch.setattr(config, "PROCESS_CACHE_VERSION", "1")

    data = b"same-photo"
    first_id = save_result(_jpeg(), _jpeg((1, 1, 1)))
    second_id = save_result(_jpeg((2, 2, 2)), _jpeg((3, 3, 3)))
    assert first_id and second_id

    put(data, "passport_rf", first_id)
    put(data, "passport_rf", second_id)

    cache_path = next((tmp_path / "cache").rglob("*.json"))
    entry = json.loads(cache_path.read_text(encoding="utf-8"))
    assert entry["result_id"] == first_id


def test_process_cache_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    monkeypatch.setattr(config, "PROCESS_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "PROCESS_CACHE_ENABLED", False)

    rid = save_result(_jpeg(), _jpeg((5, 5, 5)))
    put(b"x", "passport_rf", rid)
    assert not (tmp_path / "cache").exists()
    assert lookup(b"x", "passport_rf") is None


def test_content_hash_is_full_sha256_of_exact_bytes():
    data = b"oriented-upload-bytes"
    digest = content_hash(data)
    assert digest == hashlib.sha256(data).hexdigest()
    assert len(digest) == 64
    assert digest != hashlib.sha256(data + b"\x00").hexdigest()


def test_cache_filename_uses_full_digest_not_truncated(tmp_path, monkeypatch):
    _enable_cache(tmp_path, monkeypatch)
    data = b"oriented-upload-bytes"
    source_id = save_result(_jpeg(), _jpeg((1, 2, 3)))
    assert source_id is not None
    put(data, "passport_rf", source_id)

    digest = content_hash(data)
    cache_files = list((tmp_path / "cache").rglob("*.json"))
    assert len(cache_files) == 1
    assert cache_files[0].name == f"{digest}_passport_rf_v1.json"


def test_same_bytes_same_doc_is_cache_hit(tmp_path, monkeypatch):
    _enable_cache(tmp_path, monkeypatch)
    data = _jpeg((12, 34, 56))
    dig = _jpeg((9, 8, 7))
    source_id = save_result(dig, _jpeg((1, 1, 1)))
    put(data, "passport_rf", source_id)

    hit_id = lookup(data, "passport_rf")
    assert hit_id is not None
    assert hit_id != source_id
    assert load_file(hit_id, "digital.jpg") == dig
    meta = load_meta(hit_id)
    assert meta is not None
    assert meta["cloned_from"] == source_id
    assert meta["paid"] is False


def _jpeg_with_block(block_color, *, bg=(240, 240, 240)):
    """Solid-color 40×50 JPEGs can quantize to the same bytes; a large block does not."""
    img = Image.new("RGB", (120, 160), bg)
    for x in range(20, 90):
        for y in range(20, 90):
            img.putpixel((x, y), block_color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def test_visually_similar_jpegs_do_not_collide(tmp_path, monkeypatch):
    _enable_cache(tmp_path, monkeypatch)
    photo_a = _jpeg_with_block((10, 20, 30))
    photo_b = _jpeg_with_block((10, 20, 80))
    assert photo_a != photo_b
    assert content_hash(photo_a) != content_hash(photo_b)

    source_id = save_result(_jpeg(), _jpeg((1, 1, 1)))
    put(photo_a, "passport_rf", source_id)

    assert lookup(photo_a, "passport_rf") is not None
    assert lookup(photo_b, "passport_rf") is None


def test_recompressed_jpeg_is_cache_miss(tmp_path, monkeypatch):
    _enable_cache(tmp_path, monkeypatch)
    original = _jpeg((40, 80, 120), quality=95)
    recompressed = _jpeg((40, 80, 120), quality=50)
    assert original != recompressed
    assert content_hash(original) != content_hash(recompressed)

    source_id = save_result(_jpeg(), _jpeg((1, 1, 1)))
    put(original, "passport_rf", source_id)

    assert lookup(original, "passport_rf") is not None
    assert lookup(recompressed, "passport_rf") is None


def test_same_bytes_different_doc_type_is_miss(tmp_path, monkeypatch):
    _enable_cache(tmp_path, monkeypatch)
    data = _jpeg((7, 8, 9))
    source_id = save_result(_jpeg(), _jpeg((1, 1, 1)))
    put(data, "passport_rf", source_id)

    assert lookup(data, "passport_rf") is not None
    assert lookup(data, "zagran") is None


def test_cache_version_bump_is_miss(tmp_path, monkeypatch):
    _enable_cache(tmp_path, monkeypatch, version="1")
    data = _jpeg((11, 22, 33))
    source_id = save_result(_jpeg(), _jpeg((1, 1, 1)))
    put(data, "passport_rf", source_id)
    assert lookup(data, "passport_rf") is not None

    monkeypatch.setattr(config, "PROCESS_CACHE_VERSION", "2")
    assert lookup(data, "passport_rf") is None


def test_next_utc_day_is_cache_miss(tmp_path, monkeypatch):
    _enable_cache(tmp_path, monkeypatch)
    _freeze_utc_day(monkeypatch, "20260829")
    data = _jpeg((1, 2, 3))
    source_id = save_result(_jpeg(), _jpeg((4, 5, 6)))
    put(data, "passport_rf", source_id)
    assert lookup(data, "passport_rf") is not None

    _freeze_utc_day(monkeypatch, "20260830")
    assert lookup(data, "passport_rf") is None


def test_two_uploads_of_same_file_clone_first_result(tmp_path, monkeypatch):
    _enable_cache(tmp_path, monkeypatch)
    shared_file = _jpeg((50, 60, 70))
    first_digital = _jpeg((200, 10, 10))
    source_id = save_result(first_digital, _jpeg((1, 1, 1)))
    put(shared_file, "passport_rf", source_id)

    user_b = lookup(shared_file, "passport_rf")
    user_c = lookup(shared_file, "passport_rf")
    assert user_b and user_c
    assert len({source_id, user_b, user_c}) == 3
    assert load_file(user_b, "digital.jpg") == first_digital
    assert load_file(user_c, "digital.jpg") == first_digital


def test_distinct_payloads_never_share_a_cache_key(tmp_path, monkeypatch):
    _enable_cache(tmp_path, monkeypatch)
    payloads = [bytes([i, j, 255 - i]) * 16 for i in range(8) for j in range(8)]
    hashes = [content_hash(p) for p in payloads]
    assert len(hashes) == len(set(hashes))

    for i, data in enumerate(payloads):
        rid = save_result(_jpeg((i, 0, 0)), _jpeg((0, i, 0)))
        put(data, "passport_rf", rid)

    cache_files = list((tmp_path / "cache").rglob("*.json"))
    assert len(cache_files) == len(payloads)
    names = {p.name for p in cache_files}
    assert len(names) == len(payloads)
