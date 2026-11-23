"""Where models, datasets and outputs live."""

from __future__ import annotations

import os

NETWORK_VOLUME = "/runpod-volume"
SEEDVR_DIR = os.environ.get("SEEDVR_DIR", "/opt/seedvr")
SEEDVR_SHA = "e4de8c24441a67e1b7df56abea10645059bb1185"

DEFAULT_MODEL = "3b"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_SAMPLE_STEPS = 1
DEFAULT_SEED = 666
DEFAULT_RES_H = 720
DEFAULT_RES_W = 1280
DEFAULT_SP_SIZE = 1

# Official SeedVR2-3B / 7B Hugging Face repos. Weights are public (gated: false).
MODEL_REPOS = {
    "3b": {
        "repo_id": "ByteDance-Seed/SeedVR2-3B",
        "revision": "37255ff8cccfb01071b87f635a5948ca8d53117c",
        "script": "projects/inference_seedvr2_3b.py",
        "checkpoint": "seedvr2_ema_3b.pth",
        "config_dir": "configs_3b",
    },
    "7b": {
        "repo_id": "ByteDance-Seed/SeedVR2-7B",
        "revision": "eb0c4281d41ba3767d4f14370f0e37e9e9180c16",
        "script": "projects/inference_seedvr2_7b.py",
        "checkpoint": "seedvr2_ema_7b.pth",
        "config_dir": "configs_7b",
    },
}

# Sizes on the pinned 3B revision (bytes). Used to verify a complete download.
# pos_emb.pt / neg_emb.pt sizes are from the Hugging Face tree of that revision
# (the two files are 595_100 and 656_540 bytes respectively).
EXPECTED_3B_FILES = {
    "seedvr2_ema_3b.pth": 13_566_090_228,
    "ema_vae.pth": 1_002_691_902,
    "pos_emb.pt": 595_100,
    "neg_emb.pt": 656_540,
}

# 7B sharp is the same size as the regular 7B checkpoint; never download it
# together with the default 7B option.
EXPECTED_7B_FILES = {
    "seedvr2_ema_7b.pth": 32_958_774_606,
    "ema_vae.pth": 1_002_691_902,
}

# SeedVR v1 scripts must never be imported or executed.
FORBIDDEN_V1_SCRIPTS = (
    "projects/inference_seedvr_3b.py",
    "projects/inference_seedvr_7b.py",
    "inference_seedvr_3b.py",
    "inference_seedvr_7b.py",
)

COLOR_FIX_RELPATH = "projects/video_diffusion_sr/color_fix.py"


def has_volume() -> bool:
    return os.path.isdir(NETWORK_VOLUME) and os.access(NETWORK_VOLUME, os.W_OK)


def model_cache_dir() -> str:
    if os.environ.get("MODEL_CACHE_DIR"):
        return os.environ["MODEL_CACHE_DIR"]
    if has_volume():
        return os.path.join(NETWORK_VOLUME, "seedvr2", "models")
    return "/tmp/seedvr2/models"


def output_root() -> str:
    if os.environ.get("OUTPUT_ROOT"):
        return os.environ["OUTPUT_ROOT"]
    if has_volume():
        return os.path.join(NETWORK_VOLUME, "seedvr2", "runs")
    return "/tmp/seedvr2/runs"


def configure_caches() -> None:
    if has_volume():
        os.environ.setdefault("HF_HOME", os.path.join(NETWORK_VOLUME, "seedvr2", "hf-home"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(NETWORK_VOLUME, "seedvr2", "hf-hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(NETWORK_VOLUME, "seedvr2", "hf-hub"))
    os.makedirs(model_cache_dir(), exist_ok=True)
    os.makedirs(output_root(), exist_ok=True)
