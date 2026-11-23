import os
import zipfile
from pathlib import Path

from PIL import Image

from src.dataset import (
    MAX_IMAGE_PIXELS,
    _extract,
    _safe_member,
    collect_images,
    collect_media,
    compute_scaled_hw,
    is_image_file,
    is_video_file,
    validate_image,
)


def _rgb(path: Path, size=(32, 32), color=(10, 20, 30)):
    Image.new("RGB", size, color).save(path)


def test_zip_slip_is_rejected(tmp_path):
    zpath = tmp_path / "evil.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("ok/hello.txt", "safe")
        z.writestr("../escape.txt", "nope")
        z.writestr("/abs/escape2.txt", "nope")
        info = zipfile.ZipInfo("nested/../../outside.txt")
        z.writestr(info, "nope")
    dest = tmp_path / "out"
    slipped = _extract(str(zpath), str(dest))
    assert slipped >= 1
    root = dest.resolve()
    for p in dest.rglob("*"):
        assert str(p.resolve()).startswith(str(root))
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "outside.txt").exists()
    assert (dest / "ok" / "hello.txt").is_file()


def test_safe_member_blocks_dotdot(tmp_path):
    base = str(tmp_path / "root")
    os.makedirs(base)
    assert _safe_member(base, "../x") is None
    assert _safe_member(base, "/etc/passwd") is None
    assert _safe_member(base, "ok/img.jpg") is not None


def test_pixel_limit_rejects_huge_image(tmp_path, monkeypatch):
    import src.dataset as ds
    monkeypatch.setattr(ds, "MAX_IMAGE_PIXELS", 1000)
    path = tmp_path / "big.jpg"
    Image.new("RGB", (40, 40), (1, 2, 3)).save(path)
    ok, err, _ = ds.validate_image(str(path))
    assert ok is False
    assert err and ("exceeds" in err or "decompression" in err.lower() or "bomb" in err.lower())


def test_pixel_limit_accepts_small_image(tmp_path):
    path = tmp_path / "ok.png"
    _rgb(path, (64, 48))
    ok, err, size = validate_image(str(path))
    assert ok is True and err is None
    assert size == (64, 48)


def test_byte_limit_rejects_large_file(tmp_path, monkeypatch):
    import src.dataset as ds
    monkeypatch.setattr(ds, "MAX_IMAGE_BYTES", 200)
    path = tmp_path / "fat.jpg"
    Image.new("RGB", (200, 200), (9, 9, 9)).save(path, quality=95)
    ok, err, _ = ds.validate_image(str(path))
    assert ok is False
    assert "byte limit" in (err or "")


def test_collect_media_keeps_images_and_videos_ignores_junk(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "__MACOSX").mkdir()
    _rgb(tmp_path / "sub" / "cat.png")
    (tmp_path / "sub" / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 80)
    _rgb(tmp_path / ".hidden.jpg")
    _rgb(tmp_path / "__MACOSX" / "._x.jpg")
    (tmp_path / "notes.txt").write_text("not media")
    found = collect_media(str(tmp_path))
    rels = [os.path.relpath(p, tmp_path) for p in found]
    assert os.path.join("sub", "cat.png") in rels
    assert os.path.join("sub", "clip.mp4") in rels
    assert all(not p.startswith(".") and "__MACOSX" not in p for p in rels)
    assert collect_images(str(tmp_path)) == [p for p in found if p.endswith(".png")]


def test_official_image_extensions_match_seedvr2():
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        assert is_image_file(f"x{ext}")
    assert not is_image_file("x.gif")
    assert is_video_file("x.mp4")
    assert not is_video_file("x.png")


def test_scale_helper_maps_pil_size_to_res_h_w():
    # PIL size is (width, height); official flags are res_h, res_w.
    assert compute_scaled_hw((512, 768), 2.0) == (1536, 1024)
    assert compute_scaled_hw(None, 2.0) is None
    assert compute_scaled_hw((100, 100), 0) is None
