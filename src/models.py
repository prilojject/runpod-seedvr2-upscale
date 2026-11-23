"""Download and verify SeedVR2 weights into the volume (or ephemeral) cache.

Weights are never baked into the image. The official apex wheel is installed
in the Docker image; it is not stored in the weight cache.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from .paths import (
    EXPECTED_3B_FILES,
    EXPECTED_7B_FILES,
    MODEL_REPOS,
    SEEDVR_DIR,
    model_cache_dir,
)
from .secret_utils import redact

log = logging.getLogger(__name__)

Progress = Callable[[dict], None]

# Do not pull the prebuilt apex wheels or the 7B-sharp checkpoint into the cache.
ALLOW_PATTERNS_3B = ("*.pth", "*.pt", "*.json", "*.md", "*.txt")
ALLOW_PATTERNS_7B = ("seedvr2_ema_7b.pth", "ema_vae.pth", "*.json", "*.md", "*.txt")
IGNORE_PATTERNS = ("*.whl", "*sharp*", "apex-*.whl")


class ModelError(RuntimeError):
    pass


def local_model_dir(key: str) -> str:
    return os.path.join(model_cache_dir(), key)


def _expected_files(model: str) -> dict[str, int]:
    return dict(EXPECTED_3B_FILES if model == "3b" else EXPECTED_7B_FILES)


def shards_ok(dest: str, model: str, require_embeddings: bool = True) -> bool:
    expected = _expected_files(model)
    for name, size in expected.items():
        if name.endswith(".pt") and not require_embeddings:
            continue
        path = os.path.join(dest, name)
        if not os.path.isfile(path):
            return False
        if os.path.getsize(path) != size:
            return False
    if require_embeddings:
        for name in ("pos_emb.pt", "neg_emb.pt"):
            path = os.path.join(dest, name)
            if not os.path.isfile(path):
                # 7B repo has no embeddings; they may live in the 3B cache.
                if model == "7b":
                    alt = os.path.join(local_model_dir("3b"), name)
                    if os.path.isfile(alt) and os.path.getsize(alt) == EXPECTED_3B_FILES[name]:
                        continue
                return False
            if model == "3b" and os.path.getsize(path) != EXPECTED_3B_FILES[name]:
                return False
    return True


def check_access(cfg, token: str | None) -> dict[str, Any]:
    """HEAD-level reachability check for a dry run (no 13+ GiB download)."""
    from huggingface_hub import HfApi

    info_meta = MODEL_REPOS[cfg.model]
    repo_id = info_meta["repo_id"]
    revision = cfg.model_revision or info_meta["revision"]
    info: dict[str, Any] = {
        "model": cfg.model,
        "repo_id": repo_id,
        "revision": revision,
        "ok": False,
        "gated": None,
    }
    try:
        api = HfApi(token=token)
        model = api.model_info(repo_id, revision=revision)
        siblings = {s.rfilename: getattr(s, "size", None) for s in (model.siblings or [])}
        expected = _expected_files(cfg.model)
        missing = [n for n in expected if n not in siblings]
        info["files"] = len(siblings)
        info["gated"] = bool(getattr(model, "gated", False))
        if cfg.model == "3b":
            for name in ("pos_emb.pt", "neg_emb.pt"):
                if name not in siblings:
                    missing.append(name)
        if missing:
            info["error"] = f"missing files on the Hub: {missing}"
            return info
        # Refuse to treat sharp as the 7B default.
        if cfg.model == "7b" and "seedvr2_ema_7b_sharp.pth" in siblings:
            info["note"] = "7B-sharp is present on the Hub but will not be downloaded"
        info["ok"] = True
        info["size_gb"] = round(sum(expected.values()) / 2**30, 2)
    except Exception as exc:
        info["error"] = redact(str(exc)).splitlines()[0][:400]
    return info


def _snapshot(repo_id: str, revision: str, dest: str, token: str | None,
              allow_patterns: tuple[str, ...]) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=dest,
        token=token,
        allow_patterns=list(allow_patterns),
        ignore_patterns=list(IGNORE_PATTERNS),
    )


def ensure_models(cfg, token: str | None, progress: Progress | None = None) -> str:
    """Return the cache directory that holds the resolved checkpoint + VAE + embeddings."""
    dest = local_model_dir(cfg.model)
    os.makedirs(dest, exist_ok=True)
    if shards_ok(dest, cfg.model, require_embeddings=(cfg.model == "3b")) and (
        cfg.model == "3b" or shards_ok(local_model_dir("3b"), "3b", require_embeddings=True)
        or all(os.path.isfile(os.path.join(dest, n)) for n in ("pos_emb.pt", "neg_emb.pt"))
    ):
        log.info("using cached weights at %s", dest)
        _materialize_runtime_links(cfg, dest)
        return dest

    info = MODEL_REPOS[cfg.model]
    if progress:
        progress({"stage": "downloading_weights", "repo_id": info["repo_id"], "dest": dest})
    log.info("downloading %s @ %s into cache (weights only, no apex wheels)", info["repo_id"], cfg.model_revision)
    try:
        patterns = ALLOW_PATTERNS_3B if cfg.model == "3b" else ALLOW_PATTERNS_7B
        _snapshot(info["repo_id"], cfg.model_revision or info["revision"], dest, token, patterns)
    except Exception as exc:
        raise ModelError(
            f"failed to download {info['repo_id']}@{cfg.model_revision}: "
            f"{redact(str(exc)).splitlines()[0][:400]}"
        ) from exc

    # 7B repo has no pos_emb.pt / neg_emb.pt — pull them from the 3B snapshot.
    if cfg.model == "7b":
        _ensure_embeddings(token, progress)

    if not shards_ok(dest, cfg.model, require_embeddings=(cfg.model == "3b")):
        sizes = {
            name: (os.path.getsize(os.path.join(dest, name)) if os.path.isfile(os.path.join(dest, name)) else None)
            for name in _expected_files(cfg.model)
        }
        raise ModelError(f"weight download incomplete or size mismatch: {sizes}")

    # Never keep a sharp checkpoint if a broad pattern slipped.
    sharp = os.path.join(dest, "seedvr2_ema_7b_sharp.pth")
    if os.path.isfile(sharp):
        log.warning("removing unexpected 7B-sharp checkpoint from the cache")
        try:
            os.remove(sharp)
        except OSError:
            pass

    _materialize_runtime_links(cfg, dest)
    return dest


def _ensure_embeddings(token: str | None, progress: Progress | None) -> str:
    dest3 = local_model_dir("3b")
    os.makedirs(dest3, exist_ok=True)
    if all(
        os.path.isfile(os.path.join(dest3, n)) and os.path.getsize(os.path.join(dest3, n)) == EXPECTED_3B_FILES[n]
        for n in ("pos_emb.pt", "neg_emb.pt")
    ):
        return dest3
    info = MODEL_REPOS["3b"]
    if progress:
        progress({"stage": "downloading_embeddings", "repo_id": info["repo_id"]})
    _snapshot(info["repo_id"], info["revision"], dest3, token, ("pos_emb.pt", "neg_emb.pt", "*.json", "*.md"))
    return dest3


def _link(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.isfile(dst):
        try:
            os.remove(dst)
        except OSError:
            pass
    os.symlink(os.path.abspath(src), dst)


def _materialize_runtime_links(cfg, dest: str) -> None:
    """Place checkpoints and embeddings where the official script expects them.

    Official 3B script: ``./ckpts/seedvr2_ema_3b.pth``, ``./ckpts/ema_vae.pth``,
    and ``pos_emb.pt`` / ``neg_emb.pt`` in the process cwd (SEEDVR_DIR).
    """
    info = MODEL_REPOS[cfg.model]
    ckpts = os.path.join(SEEDVR_DIR, "ckpts")
    os.makedirs(ckpts, exist_ok=True)
    _link(os.path.join(dest, info["checkpoint"]), os.path.join(ckpts, info["checkpoint"]))
    vae_src = os.path.join(dest, "ema_vae.pth")
    if not os.path.isfile(vae_src):
        vae_src = os.path.join(local_model_dir("3b"), "ema_vae.pth")
    _link(vae_src, os.path.join(ckpts, "ema_vae.pth"))

    for name in ("pos_emb.pt", "neg_emb.pt"):
        src = os.path.join(dest, name)
        if not os.path.isfile(src):
            src = os.path.join(local_model_dir("3b"), name)
        if not os.path.isfile(src):
            raise ModelError(f"missing required embedding {name}")
        _link(src, os.path.join(SEEDVR_DIR, name))
