"""Fetch, unpack and validate images and videos (zip-slip and decompression-bomb safe)."""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image, ImageFile, UnidentifiedImageError

from .secret_utils import redact

log = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 2 * 2**30
MAX_UNPACKED_BYTES = 2 * 2**30
MAX_IMAGE_BYTES = 40 * 2**20  # 40 MiB per still
MAX_VIDEO_BYTES = 2 * 2**30
MAX_IMAGE_PIXELS = 40_000_000  # 40 megapixels

# Official is_image_file set in inference_seedvr2_3b.py (no .gif / .tif).
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg")
MEDIA_EXTS = IMAGE_EXTS + VIDEO_EXTS

CONTENT_TYPE_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
    "application/zip": ".zip",
    "application/x-tar": ".tar",
    "application/gzip": ".tar.gz",
}


class DatasetError(ValueError):
    pass


@dataclass
class DatasetReport:
    source: str
    images_found: int = 0
    videos_found: int = 0
    media_used: int = 0
    read_errors: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    example_files: list[str] = field(default_factory=list)
    media_dir: str = ""
    media_paths: list[str] = field(default_factory=list)
    media_types: dict[str, str] = field(default_factory=dict)
    first_size: tuple[int, int] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "images_found": self.images_found,
            "videos_found": self.videos_found,
            "media_used": self.media_used,
            "read_errors": self.read_errors[:20],
            "skipped": self.skipped[:20],
            "example_files": self.example_files[:8],
            "media_dir": self.media_dir,
            "first_size": list(self.first_size) if self.first_size else None,
        }


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTS


def is_video_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in VIDEO_EXTS


def media_kind(filename: str) -> str | None:
    if is_image_file(filename):
        return "image"
    if is_video_file(filename):
        return "video"
    return None


def _download(url: str, dest: str, limit: int = MAX_DOWNLOAD_BYTES) -> tuple[str, str]:
    log.info("downloading %s", redact(url).split("?")[0])
    with requests.get(url, stream=True, timeout=(15, 300), allow_redirects=True) as r:
        if r.status_code != 200:
            raise DatasetError(f"download returned HTTP {r.status_code}")
        size = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                size += len(chunk)
                if size > limit:
                    raise DatasetError(f"download is larger than {limit} bytes")
                f.write(chunk)
        return dest, r.headers.get("Content-Type", "")


def _safe_member(base: str, name: str) -> str | None:
    # Zip-slip: refuse members that would escape dest (including names with .. or abs paths).
    cleaned = name.replace("\\", "/")
    if cleaned.startswith("/") or cleaned.startswith("\\") or (len(cleaned) > 1 and cleaned[1] == ":"):
        return None
    target = os.path.realpath(os.path.join(base, cleaned))
    root = os.path.realpath(base)
    if not target.startswith(root + os.sep) and target != root:
        return None
    return target


def _extract(archive: str, dest: str) -> int:
    """Extract a zip/tar into dest. Returns the number of members skipped as zip-slip."""
    os.makedirs(dest, exist_ok=True)
    total = 0
    slipped = 0
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                if m.is_dir():
                    continue
                total += max(m.file_size, 0)
                if total > MAX_UNPACKED_BYTES:
                    raise DatasetError("dataset archive unpacks to more than 2 GiB")
                target = _safe_member(dest, m.filename)
                if target is None:
                    slipped += 1
                    log.warning("zip-slip member skipped: %s", m.filename)
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                written = 0
                with z.open(m) as src, open(target, "wb") as out:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > MAX_UNPACKED_BYTES:
                            raise DatasetError("dataset archive unpacks to more than 2 GiB")
                        out.write(chunk)
        return slipped
    try:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if not m.isfile():
                    continue
                total += max(m.size, 0)
                if total > MAX_UNPACKED_BYTES:
                    raise DatasetError("dataset archive unpacks to more than 2 GiB")
                target = _safe_member(dest, m.name)
                if target is None:
                    slipped += 1
                    log.warning("tar-slip member skipped: %s", m.name)
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                src = t.extractfile(m)
                if src is None:
                    continue
                written = 0
                with src, open(target, "wb") as out:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > MAX_UNPACKED_BYTES:
                            raise DatasetError("dataset archive unpacks to more than 2 GiB")
                        out.write(chunk)
        return slipped
    except tarfile.TarError as exc:
        raise DatasetError(f"archive is not a zip or tar: {exc}") from exc


def _is_junk(name: str) -> bool:
    base = os.path.basename(name)
    rel = name.replace("\\", "/")
    return base.startswith(".") or base.startswith("__MACOSX") or "/__MACOSX/" in rel or base == "Thumbs.db"


def collect_media(root: str) -> list[str]:
    out: list[str] = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__MACOSX"))]
        for f in files:
            if _is_junk(f) or not f.lower().endswith(MEDIA_EXTS):
                continue
            out.append(os.path.join(dirpath, f))
    out.sort()
    return out


def collect_images(root: str) -> list[str]:
    return [p for p in collect_media(root) if is_image_file(p)]


def validate_image(path: str) -> tuple[bool, str | None, tuple[int, int] | None]:
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return False, f"{os.path.basename(path)}: {exc}", None
    if size > MAX_IMAGE_BYTES:
        return False, f"{os.path.basename(path)}: {size} bytes exceeds the {MAX_IMAGE_BYTES} byte limit", None
    if size < 32:
        return False, f"{os.path.basename(path)}: file is too small to be an image", None
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    previous_trunc = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
            pixels = int(w) * int(h)
            if pixels > MAX_IMAGE_PIXELS:
                return False, f"{os.path.basename(path)}: {w}x{h} exceeds the {MAX_IMAGE_PIXELS} pixel limit", None
            im.load()
        return True, None, (int(w), int(h))
    except Image.DecompressionBombError as exc:
        return False, f"{os.path.basename(path)}: rejected as a decompression bomb ({exc})", None
    except UnidentifiedImageError:
        return False, f"{os.path.basename(path)}: not a recognizable image", None
    except Exception as exc:
        return False, f"{os.path.basename(path)}: {exc}", None
    finally:
        Image.MAX_IMAGE_PIXELS = previous
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_trunc


def validate_video(path: str) -> tuple[bool, str | None, tuple[int, int] | None]:
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return False, f"{os.path.basename(path)}: {exc}", None
    if size > MAX_VIDEO_BYTES:
        return False, f"{os.path.basename(path)}: {size} bytes exceeds the {MAX_VIDEO_BYTES} byte limit", None
    if size < 64:
        return False, f"{os.path.basename(path)}: file is too small to be a video", None
    head = b""
    with open(path, "rb") as f:
        head = f.read(12)
    # Cheap container sniff; decode happens later in the official script.
    if not (head[4:8] == b"ftyp" or head.startswith(b"\x1aE\xdf\xa3") or head.startswith(b"RIFF") or head.startswith(b"\x00\x00\x01") or b"ftyp" in head):
        # Still accept unknown containers with a known extension; do not hard-fail.
        log.info("video %s has an unrecognized header; keeping it on extension", os.path.basename(path))
    return True, None, None


def _guess_ext(url: str, content_type: str, data_head: bytes, allowed: tuple[str, ...]) -> str:
    parsed = urlparse(url)
    suffix = os.path.splitext(parsed.path)[1].lower()
    if suffix in allowed:
        return suffix
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype in CONTENT_TYPE_EXT and CONTENT_TYPE_EXT[ctype] in allowed:
        return CONTENT_TYPE_EXT[ctype]
    if data_head.startswith(b"\x89PNG"):
        return ".png"
    if data_head.startswith(b"\xff\xd8"):
        return ".jpg"
    if data_head.startswith(b"RIFF") and b"WEBP" in data_head[:16]:
        return ".webp"
    if data_head[4:8] == b"ftyp":
        return ".mp4"
    return allowed[0]


def _add_single(url: str, dest_dir: str, allowed: tuple[str, ...], limit: int, label: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    tmp = os.path.join(dest_dir, "download.bin")
    try:
        _, ctype = _download(url, tmp, limit=limit)
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"failed to download {label}: {redact(str(exc)).splitlines()[0][:300]}") from exc
    with open(tmp, "rb") as f:
        head = f.read(16)
    ext = _guess_ext(url, ctype, head, allowed)
    dest = os.path.join(dest_dir, f"input{ext}")
    os.replace(tmp, dest)
    return dest


def _unique_dest(media_dir: str, rel: str, used: set[str]) -> str:
    # Official generation_loop writes os.path.basename only, so flatten safely.
    flat = rel.replace("\\", "/").replace("/", "__")
    base, ext = os.path.splitext(flat)
    name = flat
    n = 1
    while name.lower() in used:
        name = f"{base}_{n}{ext}"
        n += 1
    used.add(name.lower())
    return os.path.join(media_dir, name)


def prepare_dataset(cfg, work: str) -> DatasetReport:
    media_dir = os.path.join(work, "media")
    os.makedirs(media_dir, exist_ok=True)
    sources: list[str] = []
    used_names: set[str] = set()

    if getattr(cfg, "dataset_url", ""):
        sources.append(cfg.dataset_url)
        raw_dir = os.path.join(work, "archive")
        archive = os.path.join(work, "dataset.bin")
        try:
            _download(cfg.dataset_url, archive)
        except DatasetError:
            raise
        except Exception as exc:
            raise DatasetError(f"failed to download dataset_url: {redact(str(exc)).splitlines()[0][:300]}") from exc
        try:
            slipped = _extract(archive, raw_dir)
        finally:
            try:
                os.remove(archive)
            except OSError:
                pass
        copied = 0
        for path in collect_media(raw_dir):
            rel = os.path.relpath(path, raw_dir)
            dest = _unique_dest(media_dir, rel, used_names)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.realpath(path) != os.path.realpath(dest):
                shutil.copy2(path, dest)
            copied += 1
        if slipped:
            log.warning("skipped %s zip-slip archive member(s)", slipped)

    if getattr(cfg, "image_url", ""):
        sources.append(cfg.image_url)
        path = _add_single(cfg.image_url, os.path.join(work, "_single_image"), IMAGE_EXTS, MAX_IMAGE_BYTES, "image_url")
        dest = _unique_dest(media_dir, os.path.basename(path), used_names)
        shutil.copy2(path, dest)

    if getattr(cfg, "video_url", ""):
        sources.append(cfg.video_url)
        path = _add_single(cfg.video_url, os.path.join(work, "_single_video"), VIDEO_EXTS, MAX_VIDEO_BYTES, "video_url")
        dest = _unique_dest(media_dir, os.path.basename(path), used_names)
        shutil.copy2(path, dest)

    found = collect_media(media_dir)
    report = DatasetReport(
        source=" + ".join(sources),
        media_dir=media_dir,
        images_found=sum(1 for p in found if is_image_file(p)),
        videos_found=sum(1 for p in found if is_video_file(p)),
    )
    if not found:
        raise DatasetError("no images or videos found (jpg/png/webp/bmp/tiff or mp4/mov/webm/mkv/...)")

    used: list[str] = []
    for path in found:
        kind = media_kind(path)
        if kind == "image":
            ok, err, size = validate_image(path)
        else:
            ok, err, size = validate_video(path)
        if not ok:
            report.read_errors.append(err or os.path.basename(path))
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        used.append(path)
        report.media_types[path] = kind or "unknown"
        if report.first_size is None and size:
            report.first_size = size
        rel = os.path.relpath(path, media_dir)
        if len(report.example_files) < 8:
            report.example_files.append(rel)

    report.media_paths = used
    report.media_used = len(used)
    if not used:
        raise DatasetError(
            "no usable media remained after validation "
            f"({len(report.read_errors)} rejected for size, pixels, or format)"
        )
    log.info("dataset ready: %s image(s), %s video(s)", report.images_found, report.videos_found)
    return report


def compute_scaled_hw(first_size: tuple[int, int] | None, scale: float) -> tuple[int, int] | None:
    """Map an optional scale helper onto official res_h / res_w (source h, w * scale)."""
    if not first_size or scale is None or scale <= 0:
        return None
    w, h = first_size  # PIL size is (width, height)
    # Official flags are res_h, res_w (height, width).
    rh = max(16, int(round(h * scale)))
    rw = max(16, int(round(w * scale)))
    return rh, rw
