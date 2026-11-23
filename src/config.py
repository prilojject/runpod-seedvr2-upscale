"""Input schema for the headless SeedVR2 upscale worker.

Every parameter comes from (highest priority first):

1. the job ``input`` object,
2. an environment variable (what the Runpod Hub deploy form sets),
3. the preset defaults,
4. the built-in default.

Dependency-free so it can be unit-tested and used for a dry run without a GPU.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .paths import (
    DEFAULT_MODEL,
    DEFAULT_RES_H,
    DEFAULT_RES_W,
    DEFAULT_SAMPLE_STEPS,
    DEFAULT_SEED,
    DEFAULT_SP_SIZE,
    MODEL_REPOS,
)

HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
URL_RE = re.compile(r"^https?://\S+$")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")

TRUE_STRINGS = {"1", "true", "yes", "y", "on", "enabled"}
FALSE_STRINGS = {"0", "false", "no", "n", "off", "disabled", ""}

PRESETS: dict[str, dict[str, Any]] = {
    "default": {
        "model": "3b",
        "res_h": DEFAULT_RES_H,
        "res_w": DEFAULT_RES_W,
        "seed": DEFAULT_SEED,
        "sp_size": DEFAULT_SP_SIZE,
        "sample_steps": DEFAULT_SAMPLE_STEPS,
    },
    "photo": {
        "model": "3b",
        "res_h": DEFAULT_RES_H,
        "res_w": DEFAULT_RES_W,
        "seed": DEFAULT_SEED,
        "sp_size": 1,
        "sample_steps": DEFAULT_SAMPLE_STEPS,
    },
    "quick": {
        "model": "3b",
        "res_h": 512,
        "res_w": 768,
        "seed": DEFAULT_SEED,
        "sp_size": 1,
        "sample_steps": DEFAULT_SAMPLE_STEPS,
    },
    "video": {
        "model": "3b",
        "res_h": DEFAULT_RES_H,
        "res_w": DEFAULT_RES_W,
        "seed": DEFAULT_SEED,
        "sp_size": 1,
        "sample_steps": DEFAULT_SAMPLE_STEPS,
    },
}


@dataclass(frozen=True)
class Param:
    name: str
    type: type
    default: Any
    env: str | None = None
    min: float | None = None
    max: float | None = None
    choices: tuple | None = None
    help: str = ""


PARAMS: list[Param] = [
    Param("image_url", str, "", "IMAGE_URL",
          help="https URL of a single image (jpg/png/webp/bmp/tiff)."),
    Param("video_url", str, "", "VIDEO_URL",
          help="https URL of a single video (mp4/mov/webm/mkv/...)."),
    Param("dataset_url", str, "", "DATASET_URL",
          help="https URL of a .zip / .tar(.gz) of images and/or videos."),
    Param("model", str, DEFAULT_MODEL, "MODEL",
          help="SeedVR2 checkpoint: 3b (default) or 7b (48 GB+ only)."),
    Param("model_revision", str, "", "MODEL_REVISION",
          help="Pinned Hugging Face revision. Default depends on the model. Do not use a floating main."),
    Param("res_h", int, DEFAULT_RES_H, "RES_H", min=16, max=4096,
          help="Official argparse height used as the geometric-mean area with res_w (NaResize). Default 720."),
    Param("res_w", int, DEFAULT_RES_W, "RES_W", min=16, max=4096,
          help="Official argparse width used as the geometric-mean area with res_h (NaResize). Default 1280."),
    Param("scale", float, 0.0, "SCALE", min=0.0, max=8.0,
          help="Optional helper: if >0, compute res_h/res_w from the first input's size times scale, unless res_* were set in the job input."),
    Param("seed", int, DEFAULT_SEED, "SEED", min=0,
          help="Official default is 666."),
    Param("out_fps", float, 0.0, "OUT_FPS", min=0.0, max=120.0,
          help="Output FPS for videos. 0 keeps the source FPS. Unused for images."),
    Param("sp_size", int, DEFAULT_SP_SIZE, "SP_SIZE", min=1, max=8,
          help="Sequence-parallel size. Official image path requires 1. Default 1."),
    Param("sample_steps", int, DEFAULT_SAMPLE_STEPS, "SAMPLE_STEPS", min=1, max=8,
          help="SeedVR2 is one-step. Leave this at 1; do not raise the training yaml's 50."),
    Param("hf_repo_id", str, "", "HF_OUTPUT_REPO",
          help="Upload the results zip + results.json here (needs HF_TOKEN with write)."),
    Param("hf_private", bool, True, "HF_PRIVATE"),
    Param("upload_url", str, "", None, help="Presigned HTTP PUT URL for the results zip."),
    Param("upload_to_bucket", bool, True, "UPLOAD_TO_BUCKET",
          help="If BUCKET_* env vars are set, upload the artifacts and return URLs."),
    Param("run_name", str, "", "RUN_NAME",
          help="Folder name for outputs. Default: the Runpod job id."),
    Param("dry_run", bool, False, "DRY_RUN",
          help="Download + validate inputs, check that weights are reachable. No GPU work."),
    Param("preset", str, "default", "PRESET", choices=tuple(PRESETS),
          help="Named defaults: default/photo (720x1280 area), quick (512x768 smoke), video."),
    Param("infer_timeout_seconds", int, 0, "INFER_TIMEOUT_SECONDS", min=0, max=86400,
          help="Advisory watchdog in seconds (0 = off). Keep below the endpoint execution timeout."),
    Param("use_colorfix", bool, False, "USE_COLORFIX",
          help="Must stay false. color_fix.py is S-Lab License 1.0 (non-commercial) and is not shipped."),
    Param("prompt", str, "", "PROMPT",
          help="Ignored. Official SeedVR2 always loads baked pos_emb.pt / neg_emb.pt; there is no text encoder in the script."),
]

PARAM_BY_NAME = {p.name: p for p in PARAMS}

ALIASES = {
    "dataset": "dataset_url",
    "zip_url": "dataset_url",
    "image": "image_url",
    "url": "image_url",
    "video": "video_url",
    "output_repo": "hf_repo_id",
    "presigned_url": "upload_url",
    "model_id": "model",
    "checkpoint": "model",
    "steps": "sample_steps",
    "fps": "out_fps",
    "revision": "model_revision",
}

# Tokens must never arrive in the job input.
BANNED_INPUT_KEYS = {
    "hf_token", "token", "authorization", "api_key", "secret", "password",
    "aws_secret_access_key", "bucket_secret_access_key",
}

V1_MODEL_TOKENS = ("seedvr1", "seedvr-v1", "seedvr_v1", "v1-3b", "v1-7b")


class ConfigError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class ResolvedConfig:
    values: dict[str, Any]
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __getattr__(self, item: str) -> Any:
        try:
            return self.values[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    def public_dict(self) -> dict[str, Any]:
        out = dict(self.values)
        if out.get("upload_url"):
            out["upload_url"] = "<presigned url>"
        # Surface the resolved model metadata used at runtime (no secrets).
        info = MODEL_REPOS[out["model"]]
        out["repo_id"] = info["repo_id"]
        if not out.get("model_revision"):
            out["model_revision"] = info["revision"]
        out["revision"] = out["model_revision"]
        out["dtype"] = "bfloat16"
        out["script"] = info["script"]
        out["checkpoint"] = info["checkpoint"]
        if not out.get("out_fps"):
            out["out_fps"] = None
        if not out.get("scale"):
            out["scale"] = None
        out.pop("prompt", None)  # never a real prompt; embeddings are baked
        out.pop("use_colorfix", None)
        return out


def _coerce(param: Param, raw: Any) -> Any:
    t = param.type
    if t is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)) and raw in (0, 1):
            return bool(raw)
        s = str(raw).strip().lower()
        if s in TRUE_STRINGS:
            return True
        if s in FALSE_STRINGS:
            return False
        raise ValueError(f"{param.name}: expected boolean, got {raw!r}")
    if t is int:
        if isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected integer, got boolean")
        if isinstance(raw, float):
            if not raw.is_integer():
                raise ValueError(f"{param.name}: expected integer, got {raw!r}")
            return int(raw)
        s = str(raw).strip()
        try:
            return int(float(s)) if re.fullmatch(r"-?\d+(\.0+)?", s) else int(s)
        except ValueError as exc:
            raise ValueError(f"{param.name}: expected integer, got {raw!r}") from exc
    if t is float:
        if isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected number, got boolean")
        if isinstance(raw, (int, float)):
            return float(raw)
        try:
            return float(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"{param.name}: expected number, got {raw!r}") from exc
    if t is str:
        if not isinstance(raw, (str, int, float)) or isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected string")
        return str(raw).strip()
    raise TypeError(t)  # pragma: no cover


def normalize_model(raw: str) -> str:
    s = raw.strip().lower().replace("_", "-")
    s = s.replace("bytedance-seed/", "")
    aliases = {
        "3b": "3b",
        "7b": "7b",
        "seedvr2": "3b",
        "seedvr2-3b": "3b",
        "seedvr2-7b": "7b",
        "seedvr2-3b-instruct": "3b",
        "default": "3b",
    }
    if s in aliases:
        return aliases[s]
    if "sharp" in s:
        raise ValueError("model: 7B-sharp is not a default option and is not downloaded with 7B")
    if s in V1_MODEL_TOKENS or s in {"seedvr", "seedvr-3b", "seedvr-7b"} and "seedvr2" not in s:
        raise ValueError("model: SeedVR v1 is not supported; use 3b or 7b (SeedVR2 only)")
    if "seedvr2-7b" in s or s.endswith("/seedvr2-7b"):
        return "7b"
    if "seedvr2-3b" in s or s.endswith("/seedvr2-3b"):
        return "3b"
    raise ValueError(f"model: expected 3b or 7b, got {raw!r}")


def resolve_config(job_input: Mapping[str, Any] | None,
                   env: Mapping[str, str] | None = None) -> ResolvedConfig:
    env = dict(os.environ if env is None else env)
    raw_in = dict(job_input or {})
    errors: list[str] = []
    warnings: list[str] = []
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}

    normalized_input: dict[str, Any] = {}
    for key, val in raw_in.items():
        name = ALIASES.get(key, key)
        if name in normalized_input and name != key:
            warnings.append(f"alias {key} ignored; {name} already set")
            continue
        normalized_input[name] = val

    preset_name = "default"
    for candidate, _source in (
        (normalized_input.get("preset"), "input"),
        (env.get("PRESET"), "env"),
    ):
        if candidate is not None and str(candidate).strip() != "":
            name = str(candidate).strip().lower()
            if name not in PRESETS:
                errors.append(f"preset: unknown value {candidate!r} (use default, photo, quick or video)")
            else:
                preset_name = name
            break

    preset = dict(PRESETS[preset_name])
    # Track whether the job input explicitly set res_h / res_w (for the scale helper).
    res_from_input = any(
        k in normalized_input and normalized_input[k] not in (None, "")
        for k in ("res_h", "res_w")
    )

    for param in PARAMS:
        raw = None
        source = "default"
        if param.name in normalized_input and normalized_input[param.name] is not None and normalized_input[param.name] != "":
            raw = normalized_input[param.name]
            source = "input"
        elif param.env and env.get(param.env) not in (None, ""):
            raw = env[param.env]
            source = "env"
        elif param.name in preset:
            raw = preset[param.name]
            source = f"preset:{preset_name}"
        else:
            raw = param.default
            source = "default"

        if raw is None:
            values[param.name] = param.default
            sources[param.name] = source
            continue
        try:
            val = _coerce(param, raw)
            if param.name == "model":
                val = normalize_model(str(val))
            if param.name == "model_revision":
                val = str(val).strip()
                if val.lower() in {"main", "master", "latest"}:
                    raise ValueError("model_revision: do not use a floating branch name; pin a commit SHA")
            if param.choices and val not in param.choices:
                raise ValueError(f"{param.name}: expected one of {param.choices}, got {val!r}")
            if param.min is not None and isinstance(val, (int, float)) and val < param.min:
                raise ValueError(f"{param.name}: {val} is below the minimum {param.min}")
            if param.max is not None and isinstance(val, (int, float)) and val > param.max:
                raise ValueError(f"{param.name}: {val} is above the maximum {param.max}")
            values[param.name] = val
            sources[param.name] = source
        except ValueError as exc:
            errors.append(str(exc))

    if any(k.lower() in BANNED_INPUT_KEYS for k in raw_in):
        errors.append("secrets must not be sent in the job input; use Runpod secrets / env")

    if not values.get("image_url") and not values.get("video_url") and not values.get("dataset_url"):
        errors.append("image_url, video_url or dataset_url is required")
    for url_key in ("image_url", "video_url", "dataset_url", "upload_url"):
        if values.get(url_key) and not URL_RE.match(values[url_key]):
            errors.append(f"{url_key} must be an http(s) URL")
    if values.get("hf_repo_id") and not HF_REPO_RE.match(values["hf_repo_id"]):
        errors.append("hf_repo_id must look like 'user/repo'")
    if values.get("run_name") and not RUN_NAME_RE.match(values["run_name"]):
        errors.append("run_name may only contain letters, digits, '.', '_' and '-'")

    if values.get("use_colorfix"):
        errors.append(
            "use_colorfix must stay false: color_fix.py is licensed under S-Lab License 1.0 "
            "(non-commercial) and is not included in this worker"
        )

    if values.get("sample_steps") and values["sample_steps"] != DEFAULT_SAMPLE_STEPS:
        warnings.append(
            f"sample_steps={values['sample_steps']} is not the official SeedVR2 one-step default; "
            "leaving your value, but the published path is sample_steps=1"
        )

    if values.get("prompt"):
        warnings.append(
            "prompt is ignored: official SeedVR2 always loads baked pos_emb.pt / neg_emb.pt "
            "and the inference script has no text encoder"
        )

    if values.get("model") == "7b":
        warnings.append(
            "SeedVR2-7B is an option for 48 GB+ GPUs only; it is not the default or the smoke path. "
            "The sharp 7B checkpoint is not downloaded with this option."
        )

    if values.get("sp_size", 1) > 1:
        warnings.append(
            "sp_size>1 is official multi-GPU sequence parallelism (e.g. 4x H100 for 1080p/2K). "
            "Images require sp_size=1; long 720p video is documented on H100, not a 4090."
        )

    # Fill the default revision for the chosen model when the user did not pin one.
    if values.get("model") in MODEL_REPOS and not values.get("model_revision"):
        values["model_revision"] = MODEL_REPOS[values["model"]]["revision"]
        sources["model_revision"] = "default"

    values["res_from_input"] = res_from_input
    values["dtype"] = "bfloat16"
    values["preset"] = preset_name
    sources.setdefault("preset", "default" if preset_name == "default" else sources.get("preset", "input"))

    if errors:
        raise ConfigError(errors)

    return ResolvedConfig(values=values, sources=sources, warnings=warnings)
