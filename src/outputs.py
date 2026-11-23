"""Deliver upscale artifacts: results.zip + results.json."""

from __future__ import annotations

import json
import logging
import os
import zipfile
from typing import Any

import requests
from PIL import Image

from .dataset import is_image_file, is_video_file
from .secret_utils import redact

log = logging.getLogger(__name__)

BUCKET_ENV = ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")
UPSTREAM_SHA = "e4de8c24441a67e1b7df56abea10645059bb1185"


def bucket_configured() -> bool:
    return all(os.environ.get(k) for k in BUCKET_ENV)


def _image_size(path: str) -> tuple[int, int] | None:
    try:
        with Image.open(path) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        return None


def collect_results(output_dir: str, input_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in input_names:
        path = os.path.join(output_dir, name)
        # Official writer keeps the original basename.
        if not os.path.isfile(path):
            # mediapy may rewrite an extension in rare cases; try stem match.
            stem, ext = os.path.splitext(name)
            matches = [
                f for f in os.listdir(output_dir)
                if os.path.splitext(f)[0] == stem and os.path.isfile(os.path.join(output_dir, f))
            ]
            if not matches:
                rows.append({"file": name, "status": "missing"})
                continue
            path = os.path.join(output_dir, matches[0])
            name = matches[0]
        kind = "image" if is_image_file(name) else ("video" if is_video_file(name) else "file")
        row: dict[str, Any] = {
            "file": name,
            "type": kind,
            "size_bytes": os.path.getsize(path),
            "status": "ok" if os.path.getsize(path) > 0 else "empty",
        }
        if kind == "image":
            wh = _image_size(path)
            if wh:
                row["width"], row["height"] = wh
        rows.append(row)
    return rows


def write_results_json(path: str, rows: list[dict[str, Any]], cfg, extra: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        "model": cfg.model,
        "repo_id": cfg.public_dict().get("repo_id"),
        "revision": cfg.model_revision,
        "dtype": "bfloat16",
        "sample_steps": cfg.sample_steps,
        "res_h": cfg.res_h,
        "res_w": cfg.res_w,
        "seed": cfg.seed,
        "sp_size": cfg.sp_size,
        "use_colorfix": False,
        "results": rows,
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def zip_results(output_dir: str, files: list[str], zip_path: str, results_json: str) -> str:
    os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in files:
            path = os.path.join(output_dir, name)
            if os.path.isfile(path):
                z.write(path, name)
        if results_json and os.path.isfile(results_json):
            z.write(results_json, "results.json")
    return zip_path


def push_to_hf(cfg, token: str, files: list[str], card: str) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=cfg.hf_repo_id, repo_type="model", private=cfg.hf_private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode())]
    for path in files:
        ops.append(CommitOperationAdd(path_in_repo=os.path.basename(path), path_or_fileobj=path))
    api.create_commit(
        repo_id=cfg.hf_repo_id, repo_type="model", operations=ops,
        commit_message=f"SeedVR2 {cfg.model} upscale ({cfg.run_name or 'run'})",
    )
    return f"https://huggingface.co/{cfg.hf_repo_id}"


def upload_bucket(path: str, prefix: str) -> str:
    from runpod.serverless.utils.rp_upload import upload_file_to_bucket
    return upload_file_to_bucket(
        file_name=os.path.basename(path), file_location=path,
        bucket_name=os.environ.get("BUCKET_NAME") or None, prefix=prefix,
    )


def put_presigned(url: str, path: str) -> int:
    with open(path, "rb") as f:
        r = requests.put(url, data=f, headers={"Content-Type": "application/octet-stream"}, timeout=(15, 900))
    if r.status_code >= 300:
        raise RuntimeError(f"presigned PUT returned HTTP {r.status_code}: {r.text[:200]}")
    return r.status_code


def _card(cfg) -> str:
    return "\n".join([
        "---",
        "tags:",
        "- seedvr2",
        "- upscale",
        "- video-to-video",
        "- runpod",
        "---",
        "",
        "# SeedVR2 upscale",
        "",
        "Artifacts produced with "
        "[ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR) "
        f"(`{UPSTREAM_SHA}`) and "
        f"[{cfg.public_dict().get('repo_id')}](https://huggingface.co/{cfg.public_dict().get('repo_id')}) "
        f"(revision `{cfg.model_revision}`) on Runpod Serverless using "
        "[prilojject/runpod-seedvr2-upscale](https://github.com/prilojject/runpod-seedvr2-upscale).",
        "",
        f"- model: SeedVR2-{cfg.model.upper()}  dtype: bfloat16  steps: {cfg.sample_steps}",
        f"- res_h={cfg.res_h}  res_w={cfg.res_w}  seed={cfg.seed}  sp_size={cfg.sp_size}",
        "- color fix: not used (S-Lab License 1.0, non-commercial; file not shipped)",
        "",
        "SeedVR and SeedVR2 are licensed under the Apache 2.0 license, © 2025 ByteDance Seed.",
        "",
    ])


def deliver(cfg, run_name: str, run_dir: str, zip_path: str, json_path: str,
            token: str | None, persistent: bool) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {"persistent_storage": persistent}
    warnings: list[str] = []
    files: list[str] = []
    if zip_path and os.path.isfile(zip_path):
        out["zip_file"] = os.path.basename(zip_path)
        out["zip_path"] = zip_path
        out["zip_size_mb"] = round(os.path.getsize(zip_path) / 2**20, 2)
        files.append(zip_path)
        if persistent:
            out["zip_volume_path"] = zip_path
    if json_path and os.path.isfile(json_path):
        out["results_json"] = os.path.basename(json_path)
        out["results_json_path"] = json_path
        files.append(json_path)
        if persistent:
            out["results_json_volume_path"] = json_path
    if persistent:
        out["volume_dir"] = run_dir

    delivered = persistent and bool(files)
    card = _card(cfg)
    with open(os.path.join(run_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)

    if cfg.hf_repo_id and token and files:
        try:
            out["hf_repo_url"] = push_to_hf(cfg, token, files, card)
            delivered = True
        except Exception as exc:
            out["hf_error"] = redact(str(exc)).splitlines()[0][:400]
            log.error("HF upload failed: %s", out["hf_error"])
    if cfg.upload_url and zip_path and os.path.isfile(zip_path):
        try:
            out["upload_url_status"] = put_presigned(cfg.upload_url, zip_path)
            delivered = True
        except Exception as exc:
            out["upload_url_error"] = redact(str(exc))[:400].replace(cfg.upload_url, "<presigned url>")
    if cfg.upload_to_bucket and bucket_configured() and files:
        try:
            prefix = f"seedvr2/{run_name}"
            for path in files:
                key = "bucket_zip_url" if path.endswith(".zip") else "bucket_json_url"
                out[key] = upload_bucket(path, prefix)
            delivered = True
        except Exception as exc:
            out["bucket_error"] = redact(str(exc))[:400]
    if not delivered:
        warnings.append(
            "the upscaled files were not delivered anywhere persistent: attach a network volume, "
            "or set hf_repo_id (+HF_TOKEN), upload_url, or BUCKET_* env vars. They are lost when the worker stops."
        )
    return out, warnings
