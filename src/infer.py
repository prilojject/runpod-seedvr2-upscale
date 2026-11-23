"""Headless wrapper around official projects/inference_seedvr2_{3b,7b}.py.

Patches the image-only fps_lists bug in the official write loop, forces
use_colorfix=False, and never imports or runs SeedVR v1 scripts.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from types import ModuleType
from typing import Any, Callable

from .paths import (
    COLOR_FIX_RELPATH,
    FORBIDDEN_V1_SCRIPTS,
    MODEL_REPOS,
    SEEDVR_DIR,
    SEEDVR_SHA,
)
from .secret_utils import redact

log = logging.getLogger(__name__)

Progress = Callable[[dict], None]

IMAGE_FPS_PLACEHOLDER = 24.0

# Official generation_loop (SHA e4de8c24441a67e1b7df56abea10645059bb1185)
# only appends to fps_lists in the *video* branch. An image-only batch leaves
# fps_lists empty, so `zip(..., fps_lists)` writes nothing. Insert a dummy FPS
# after the official image-branch ValueError so images are written too.
_FPS_NEEDLE = (
    "                if sp_size > 1:\n"
    "                    raise ValueError(\"Sp size should be set to 1 for image inputs!\")"
)
_FPS_PATCH = (
    "                if sp_size > 1:\n"
    "                    raise ValueError(\"Sp size should be set to 1 for image inputs!\")\n"
    "                fps_lists.append(out_fps if out_fps is not None else 24.0)"
)

_RUNNER = None
_RUNNER_KEY: tuple | None = None
_OFFICIAL_MOD: ModuleType | None = None
_NORM_NOTE: str | None = None


class InferError(RuntimeError):
    pass


def is_image_file(filename: str) -> bool:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    return os.path.splitext(filename.lower())[1] in image_exts


def pad_fps_lists(
    n_items: int,
    fps_lists: list,
    *,
    out_fps: float | None = None,
    default: float = IMAGE_FPS_PLACEHOLDER,
) -> list:
    """Ensure fps_lists is long enough for zip(..., fps_lists) to emit every item.

    Official generation_loop only appends FPS for video files. An image-only
    batch leaves fps_lists empty, so the write zip produces no files.
    """
    if n_items < 0:
        raise ValueError("n_items must be >= 0")
    fps = list(fps_lists)
    fill = default if out_fps is None else out_fps
    if len(fps) < n_items:
        fps.extend([fill] * (n_items - len(fps)))
    return fps


def official_write_zip_keeps_images(paths: list[str], fps_lists: list) -> list[str]:
    """Reproduce the official zip write loop (bug included) for unit tests."""
    kept = []
    for path, save_fps in zip(paths, fps_lists):
        kept.append(path)
        _ = save_fps
    return kept


def patch_generation_loop_source(src: str) -> str:
    """Return official generation_loop source with the image-only fps_lists fix.

    Also forces the color-fix branch off: color_fix.py is S-Lab NC and is not shipped.
    """
    if _FPS_NEEDLE not in src:
        raise InferError(
            "could not patch the official image-only fps_lists bug: "
            "expected official generation_loop source was not found (upstream changed?)"
        )
    if "fps_lists.append(out_fps if out_fps is not None else 24.0)" not in src:
        src = src.replace(_FPS_NEEDLE, _FPS_PATCH, 1)
    # Never run color fix, even if someone later drops color_fix.py into the image.
    src = src.replace("if use_colorfix:", "if False:  # color_fix is S-Lab NC; never enable")
    # Official configure_runner leaves DiT (fp32 ~13.6 GiB) and VAE on CUDA
    # together. generation_loop swaps them for 24GB cards, but without
    # empty_cache the allocator can keep ~21 GiB allocated and OOM on the
    # next 7–8 GiB workspace (what the 4090 r2 smoke hit).
    src = src.replace(
        'runner.dit.to("cpu")\n',
        'runner.dit.to("cpu")\n        gc.collect()\n        torch.cuda.empty_cache()\n',
    )
    src = src.replace(
        'runner.vae.to("cpu")\n',
        'runner.vae.to("cpu")\n        gc.collect()\n        torch.cuda.empty_cache()\n',
    )
    return src


def assert_not_v1_script(path: str) -> None:
    norm = path.replace("\\", "/").lower()
    base = os.path.basename(norm)
    for forbidden in FORBIDDEN_V1_SCRIPTS:
        if norm.endswith(forbidden) or base == os.path.basename(forbidden):
            raise InferError(
                f"refusing to import SeedVR v1 script {path}; "
                "this worker only wraps projects/inference_seedvr2_3b.py "
                "(and optionally inference_seedvr2_7b.py)"
            )


def official_script_path(model: str) -> str:
    info = MODEL_REPOS[model]
    path = os.path.join(SEEDVR_DIR, info["script"])
    assert_not_v1_script(path)
    if not os.path.isfile(path):
        raise InferError(f"official SeedVR2 script not found at {path}")
    # Defense in depth: the path must contain seedvr2, not seedvr_3b without the 2.
    if "seedvr2" not in os.path.basename(path).lower():
        raise InferError(f"refusing to run unexpected script {path}")
    return path


def color_fix_present() -> bool:
    return os.path.isfile(os.path.join(SEEDVR_DIR, COLOR_FIX_RELPATH))


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"gpu": None}
    try:
        import torch
        if not torch.cuda.is_available():
            return info
        idx = 0
        props = torch.cuda.get_device_properties(idx)
        reserved = torch.cuda.memory_reserved(idx) if torch.cuda.is_initialized() else 0
        allocated = torch.cuda.memory_allocated(idx) if torch.cuda.is_initialized() else 0
        info.update({
            "gpu": torch.cuda.get_device_name(idx),
            "vram_gb": round(props.total_memory / 2**30, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "cuda": torch.version.cuda,
            "memory_allocated_gb": round(allocated / 2**30, 2),
            "memory_reserved_gb": round(reserved / 2**30, 2),
        })
    except Exception as exc:
        info["error"] = redact(str(exc)).splitlines()[0][:200]
    return info


def _prepare_distributed_env() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")


def _load_official_module(model: str) -> ModuleType:
    """Import the official SeedVR2 inference script from the pinned SHA clone."""
    global _OFFICIAL_MOD
    script = official_script_path(model)
    stamp = os.path.join(os.path.dirname(SEEDVR_DIR.rstrip("/")), "seedvr.commit")
    if not os.path.isfile(stamp):
        stamp = "/opt/seedvr.commit"
    if os.path.isfile(stamp):
        got = open(stamp).read().strip()
        if got != SEEDVR_SHA:
            log.warning("SeedVR SHA mismatch: %s=%s expected %s", stamp, got, SEEDVR_SHA)

    if color_fix_present():
        # Must never ship; if it appears at runtime, refuse to enable it.
        log.warning("color_fix.py is present under %s — it will not be used", COLOR_FIX_RELPATH)

    if SEEDVR_DIR not in sys.path:
        sys.path.insert(0, SEEDVR_DIR)

    # Import under a stable name so configure_runner / generation_loop stay patchable.
    mod_name = f"seedvr2_official_{model}"
    if mod_name in sys.modules and _OFFICIAL_MOD is not None and getattr(_OFFICIAL_MOD, "_seedvr_model", None) == model:
        return _OFFICIAL_MOD

    spec = importlib.util.spec_from_file_location(mod_name, script)
    if spec is None or spec.loader is None:
        raise InferError(f"could not import {script}")
    # Official scripts call print(os.getcwd()) and load color_fix at import time.
    cwd = os.getcwd()
    try:
        os.chdir(SEEDVR_DIR)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)

    mod.use_colorfix = False
    if hasattr(mod, "wavelet_reconstruction"):
        # Belt and braces: even a leftover import must not be called.
        mod.wavelet_reconstruction = None  # type: ignore[assignment]

    src = open(script, encoding="utf-8").read()
    # Only patch the generation_loop function, not the whole module (imports already ran).
    try:
        import inspect
        loop_src = inspect.getsource(mod.generation_loop)
    except (OSError, TypeError) as exc:
        raise InferError(f"could not read official generation_loop source: {exc}") from exc
    patched = patch_generation_loop_source(loop_src)
    ns: dict[str, Any] = {}
    exec(compile(patched, "<patched_generation_loop>", "exec"), mod.__dict__, ns)
    if "generation_loop" not in ns:
        raise InferError("patched generation_loop was not defined")
    mod.generation_loop = ns["generation_loop"]
    mod._seedvr_model = model  # type: ignore[attr-defined]
    _OFFICIAL_MOD = mod
    log.info("loaded official %s with fps_lists image-only patch; use_colorfix=False", script)
    return mod


def _idempotent_init_torch(mod: ModuleType) -> None:
    """Official configure_runner always calls init_torch (nccl). Make it re-entrant."""
    try:
        import torch.distributed as dist
        from common.distributed import basic as dist_basic
    except Exception:
        return

    orig = dist_basic.init_torch

    def _init_once(*args, **kwargs):
        if dist.is_available() and dist.is_initialized():
            return
        return orig(*args, **kwargs)

    dist_basic.init_torch = _init_once  # type: ignore[assignment]
    if hasattr(mod, "init_torch"):
        mod.init_torch = _init_once  # type: ignore[assignment]



def apex_fused_norm_supported(device_index: int = 0) -> bool:
    """Official SeedVR2-3B apex wheel ships only sm_90 (H100) kernels.

    A 4090 (sm_89) will load the 3B weights and then fail at the first
    FusedLayerNorm / FusedRMSNorm with "no kernel image is available".
    """
    try:
        import glob
        import importlib.util
        import torch
        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability(device_index)
        token = f"sm_{cap[0]}{cap[1]}".encode()
        roots: list[str] = []
        spec = importlib.util.find_spec("apex")
        if spec and spec.origin:
            roots.append(os.path.dirname(spec.origin))
            roots.append(os.path.dirname(os.path.dirname(spec.origin)))
        hits = []
        for root in roots:
            hits.extend(glob.glob(os.path.join(root, "fused_layer_norm_cuda*.so")))
            hits.extend(glob.glob(os.path.join(root, "**", "fused_layer_norm_cuda*.so"), recursive=True))
        if not hits:
            return False
        data = open(hits[0], "rb").read()
        return token in data
    except Exception:
        return False


def patch_fused_norms_if_needed() -> str | None:
    """If apex fused kernels cannot run on this GPU, use PyTorch LayerNorm / RMSNorm.

    Official get_norm_layer("fusedln"|"fusedrms") always imports apex. The
    official wheel is H100-only, so the 4090 smoke path must remap.
    """
    if apex_fused_norm_supported():
        return None
    try:
        import models.dit.nadit as dit_nadit
        import models.dit.normalization as dit_norm
        import models.dit_v2.nadit as dit_v2_nadit
        import models.dit_v2.normalization as dit_v2_norm
    except Exception as exc:
        log.warning("could not import official norm helpers to patch apex: %s", exc)
        return "apex fused norms likely unsupported; official import patch failed"
    # nadit.py does `from .normalization import get_norm_layer`, so patch both
    # the definition module and the already-bound name.
    pairs = (
        (dit_norm, (dit_norm, dit_nadit)),
        (dit_v2_norm, (dit_v2_norm, dit_v2_nadit)),
    )
    for src, targets in pairs:
        orig = src.get_norm_layer

        def _wrap(orig_fn=orig):
            def wrapped(norm_type):
                if norm_type == "fusedln":
                    norm_type = "layer"
                elif norm_type == "fusedrms":
                    norm_type = "rms"
                return orig_fn(norm_type)
            return wrapped

        patched = _wrap()
        for mod in targets:
            mod.get_norm_layer = patched  # type: ignore[assignment]
    msg = (
        "official apex wheel has no kernels for this GPU; "
        "using PyTorch LayerNorm/RMSNorm instead of fusedln/fusedrms"
    )
    log.warning(msg)
    return msg


def _get_runner(mod: ModuleType, model: str, sp_size: int):
    global _RUNNER, _RUNNER_KEY
    key = (model, sp_size)
    if _RUNNER is not None and _RUNNER_KEY == key:
        return _RUNNER
    _prepare_distributed_env()
    _idempotent_init_torch(mod)
    global _NORM_NOTE
    _NORM_NOTE = patch_fused_norms_if_needed()
    if _NORM_NOTE:
        log.warning("%s", _NORM_NOTE)
    cwd = os.getcwd()
    try:
        os.chdir(SEEDVR_DIR)
        runner = mod.configure_runner(sp_size)
    finally:
        os.chdir(cwd)
    # Free DiT VRAM before generation_loop's VAE encode (24GB / 4090).
    try:
        import gc
        import torch
        if hasattr(runner, "dit"):
            runner.dit.requires_grad_(False).eval()
            props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
            # Official keeps fp32 weights + bf16 autocast. On <=24GB that
            # barely fits next to the VAE; use bf16 weights on those cards.
            if props is not None and props.total_memory <= 30 * (1024 ** 3):
                runner.dit = runner.dit.to(dtype=torch.bfloat16)
                log.info("cast DiT weights to bfloat16 for %.1f GiB card", props.total_memory / 1024 ** 3)
            runner.dit.to("cpu")
        if hasattr(runner, "vae"):
            runner.vae.requires_grad_(False).eval()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        log.warning("post-configure offload failed: %s", exc)
    _RUNNER = runner
    _RUNNER_KEY = key
    return runner


def run_upscale(
    cfg,
    media_dir: str,
    output_dir: str,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run the official SeedVR2 loop (in-process) on a clean media directory."""
    if cfg.sp_size > 1 and any(is_image_file(p) for p in os.listdir(media_dir)):
        raise InferError("sp_size must be 1 when the batch contains images")
    if progress:
        progress({"stage": "loading_official_script", "model": cfg.model})
    mod = _load_official_module(cfg.model)
    # Re-assert after import: official module sets use_colorfix from file presence.
    mod.use_colorfix = False

    if progress:
        progress({"stage": "configuring_runner", "model": cfg.model})
    runner = _get_runner(mod, cfg.model, cfg.sp_size)

    out_fps = cfg.out_fps if cfg.out_fps else None
    os.makedirs(output_dir, exist_ok=True)
    if progress:
        progress({
            "stage": "running_seedvr2",
            "res_h": cfg.res_h,
            "res_w": cfg.res_w,
            "sample_steps": cfg.sample_steps,
            "seed": cfg.seed,
        })
    cwd = os.getcwd()
    try:
        os.chdir(SEEDVR_DIR)
        # Official argparse defaults: res_h=720, res_w=1280, sample_steps=1 (kwarg).
        mod.generation_loop(
            runner,
            video_path=media_dir,
            output_dir=output_dir,
            batch_size=1,
            cfg_scale=1.0,
            cfg_rescale=0.0,
            sample_steps=int(cfg.sample_steps),
            seed=int(cfg.seed),
            res_h=int(cfg.res_h),
            res_w=int(cfg.res_w),
            sp_size=int(cfg.sp_size),
            out_fps=out_fps,
        )
    except Exception as exc:
        raise InferError(redact(str(exc)).splitlines()[0][:500]) from exc
    finally:
        os.chdir(cwd)

    written = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    if not written:
        raise InferError(
            "official loop produced no output files "
            "(image-only fps_lists bug should have been patched; check logs)"
        )
    out = {
        "output_dir": output_dir,
        "files": [os.path.basename(p) for p in written],
        "count": len(written),
        "use_colorfix": False,
        "script": MODEL_REPOS[cfg.model]["script"],
        "sample_steps": int(cfg.sample_steps),
        "dtype": "bfloat16",
    }
    if _NORM_NOTE:
        out["apex_fallback"] = _NORM_NOTE
    return out
