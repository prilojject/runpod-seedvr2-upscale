"""Runpod Serverless entry point: headless SeedVR2 batch upscale.

Wraps official projects/inference_seedvr2_3b.py (or optionally _7b.py).
No Gradio, no ComfyUI, no SeedVR v1 scripts.

Local dry run (no GPU, no model weights download beyond a Hub reachability check):
    python handler.py --test_input '{"input": {"dry_run": true, "image_url": "https://.../tiny_pd.png"}}'
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import __version__  # noqa: E402
from src.config import ConfigError, resolve_config  # noqa: E402
from src.dataset import DatasetError, compute_scaled_hw, prepare_dataset  # noqa: E402
from src.paths import configure_caches, has_volume, output_root  # noqa: E402
from src.secret_utils import RedactFilter, get_hf_token, redact  # noqa: E402

configure_caches()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for h in logging.getLogger().handlers:
    h.addFilter(RedactFilter())
log = logging.getLogger("seedvr2-worker")


def _progress_fn(job: dict):
    def send(payload: dict) -> None:
        log.info("progress: %s", payload)
        if job.get("id") and not str(job["id"]).startswith("local") and os.environ.get("RUNPOD_POD_ID"):
            try:
                import runpod
                runpod.serverless.progress_update(job, payload)
            except Exception as exc:  # best effort
                log.debug("progress_update failed: %s", exc)
    return send


def handler(job: dict) -> dict[str, Any]:
    result = _handle(job)
    # Keep the ~13.6 GiB 3B weights in worker memory across jobs. Do not copy
    # refresh_worker=true from the trainer listings.
    result["refresh_worker"] = False
    return result


def _handle(job: dict) -> dict[str, Any]:
    started = time.time()
    try:
        cfg = resolve_config(job.get("input") or {})
    except ConfigError as exc:
        return {"error": f"invalid input: {exc}"}

    run_name = cfg.run_name or str(job.get("id") or f"run-{int(started)}")
    cfg.values["run_name"] = run_name
    token = get_hf_token()
    progress = _progress_fn(job)
    persistent = has_volume()
    run_dir = os.path.join(output_root(), run_name)
    work = os.path.join("/tmp/seedvr2/work", run_name)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    result: dict[str, Any] = {
        "worker_version": __version__,
        "run_name": run_name,
        "status": "error",
        "config": cfg.public_dict(),
        "warnings": list(cfg.warnings),
    }
    if not persistent:
        size = "~13.57 GiB" if cfg.model == "3b" else "~31.62 GiB (without sharp)"
        result["warnings"].append(
            f"no network volume: SeedVR2-{cfg.model.upper()} weights ({size}) are re-downloaded "
            "on every cold start (attach one to cache them under /runpod-volume/seedvr2/models)"
        )

    try:
        progress({"stage": "preparing_dataset"})
        report = prepare_dataset(cfg, work)
    except DatasetError as exc:
        return {"error": f"dataset: {exc}", **{k: result[k] for k in ("run_name", "config")}}
    except Exception as exc:
        log.error("dataset failed: %s", redact(traceback.format_exc()))
        return {"error": f"dataset failed: {redact(str(exc)).splitlines()[0][:400]}"}

    # Optional scale helper: compute res_* from the first still, unless the job set them.
    if cfg.scale and cfg.scale > 0 and not cfg.res_from_input:
        scaled = compute_scaled_hw(report.first_size, cfg.scale)
        if scaled:
            cfg.values["res_h"], cfg.values["res_w"] = scaled
            result["config"] = cfg.public_dict()
            result["warnings"].append(
                f"scale={cfg.scale} set res_h={cfg.res_h}, res_w={cfg.res_w} "
                f"from first input size {report.first_size}"
            )
    elif cfg.scale and cfg.scale > 0 and cfg.res_from_input:
        result["warnings"].append("scale ignored because res_h/res_w were set in the job input")

    result["dataset"] = report.as_dict()
    result["warnings"] += report.warnings

    if cfg.dry_run:
        from src.models import check_access
        access = check_access(cfg, token)
        result.update({
            "dry_run": True,
            "status": "ok" if access.get("ok") else "error",
            "model_access": access,
            "media_to_upscale": report.media_used,
            "elapsed_seconds": round(time.time() - started, 2),
        })
        if result["status"] == "error":
            result["error"] = "weights not reachable: " + str(access.get("error"))
        shutil.rmtree(work, ignore_errors=True)
        return result

    from src.infer import InferError, gpu_info, run_upscale
    from src.models import ModelError, ensure_models
    from src.outputs import collect_results, deliver, write_results_json, zip_results

    gpu = gpu_info()
    result["gpu"] = gpu
    if not gpu.get("gpu"):
        return {"error": "no CUDA GPU visible to the worker", **{k: result[k] for k in ("run_name", "config")}}

    try:
        ensure_models(cfg, token, progress)
    except ModelError as exc:
        return {"error": str(exc), **{k: result[k] for k in ("run_name", "config")}}

    out_media = os.path.join(work, "output")
    os.makedirs(out_media, exist_ok=True)
    try:
        cap = run_upscale(cfg, report.media_dir, out_media, progress)
    except InferError as exc:
        log.error("upscale failed: %s", redact(traceback.format_exc()))
        return {
            "error": f"upscale failed: {exc}",
            "run_name": run_name,
            "gpu": gpu_info() or gpu,
        }
    except Exception as exc:
        log.error("upscale crashed: %s", redact(traceback.format_exc()))
        return {
            "error": f"upscale crashed: {redact(str(exc)).splitlines()[0][:500]}",
            "run_name": run_name,
            "gpu": gpu_info() or gpu,
        }

    result["upscale"] = cap
    result["gpu"] = gpu_info() or gpu
    if cap.get("apex_fallback"):
        result["warnings"].append(cap["apex_fallback"])
    if cap.get("attn_fallback"):
        result["warnings"].append(cap["attn_fallback"])

    # Copy official outputs into the persistent run dir (same basenames).
    dest_media = os.path.join(run_dir, "output")
    if os.path.isdir(dest_media):
        shutil.rmtree(dest_media)
    shutil.copytree(out_media, dest_media)

    input_names = [os.path.basename(p) for p in report.media_paths]
    rows = collect_results(dest_media, input_names)
    missing = [r["file"] for r in rows if r.get("status") != "ok"]
    if missing:
        result["warnings"].append(f"{len(missing)} output file(s) missing or empty: {missing[:8]}")

    json_path = os.path.join(run_dir, "results.json")
    write_results_json(json_path, rows, cfg, extra={"run_name": run_name})
    zip_path = os.path.join(run_dir, "results.zip")
    zip_results(dest_media, [r["file"] for r in rows if r.get("status") == "ok"], zip_path, json_path)

    progress({"stage": "uploading"})
    artifacts, warns = deliver(cfg, run_name, run_dir, zip_path, json_path, token, persistent)
    result["warnings"] += warns
    result["artifacts"] = artifacts
    result["results"] = rows
    failed = len(missing)
    result.update({
        "status": "completed" if failed == 0 else "error",
        "elapsed_seconds": round(time.time() - started, 1),
    })
    if failed and "error" not in result:
        result["error"] = f"upscale failed for {failed} file(s)"
    shutil.rmtree(work, ignore_errors=True)
    return result


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
