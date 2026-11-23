import pytest

from src.config import ConfigError, resolve_config
from src.paths import (
    DEFAULT_RES_H,
    DEFAULT_RES_W,
    DEFAULT_SAMPLE_STEPS,
    DEFAULT_SEED,
    MODEL_REPOS,
)


def test_default_is_3b_bf16_one_step():
    cfg = resolve_config({"image_url": "https://example.com/a.png"})
    assert cfg.model == "3b"
    assert cfg.res_h == DEFAULT_RES_H
    assert cfg.res_w == DEFAULT_RES_W
    assert cfg.seed == DEFAULT_SEED
    assert cfg.sample_steps == DEFAULT_SAMPLE_STEPS
    assert cfg.sp_size == 1
    pub = cfg.public_dict()
    assert pub["dtype"] == "bfloat16"
    assert pub["repo_id"] == "ByteDance-Seed/SeedVR2-3B"
    assert pub["revision"] == MODEL_REPOS["3b"]["revision"]
    assert pub["script"].endswith("inference_seedvr2_3b.py")
    assert "inference_seedvr_3b.py" not in pub["script"]


def test_quick_preset_uses_smoke_resolution():
    cfg = resolve_config({"image_url": "https://example.com/a.png", "preset": "quick"})
    assert cfg.res_h == 512
    assert cfg.res_w == 768
    assert cfg.model == "3b"
    assert cfg.sample_steps == 1


def test_input_overrides_env_and_preset(monkeypatch):
    monkeypatch.setenv("MODEL", "7b")
    monkeypatch.setenv("RES_H", "512")
    cfg = resolve_config({
        "image_url": "https://example.com/a.png",
        "model": "3b",
        "res_h": 720,
    })
    assert cfg.model == "3b"
    assert cfg.res_h == 720
    assert cfg.sources["model"] == "input"
    assert cfg.sources["res_h"] == "input"


def test_env_overrides_preset(monkeypatch):
    monkeypatch.setenv("IMAGE_URL", "https://example.com/from-env.png")
    monkeypatch.setenv("SEED", "42")
    cfg = resolve_config({}, env=None)
    assert cfg.image_url.endswith("from-env.png")
    assert cfg.seed == 42
    assert cfg.sources["image_url"] == "env"


def test_requires_media_url():
    with pytest.raises(ConfigError, match="image_url, video_url or dataset_url"):
        resolve_config({"model": "3b"})


def test_accepts_video_or_dataset_alone():
    v = resolve_config({"video_url": "https://example.com/a.mp4"})
    assert v.video_url.endswith("a.mp4")
    d = resolve_config({"dataset_url": "https://example.com/a.zip"})
    assert d.dataset_url.endswith("a.zip")


def test_rejects_bad_urls():
    with pytest.raises(ConfigError, match="dataset_url must be an http"):
        resolve_config({"dataset_url": "ftp://x/y.zip"})
    with pytest.raises(ConfigError, match="image_url must be an http"):
        resolve_config({"image_url": "not-a-url"})
    with pytest.raises(ConfigError, match="video_url must be an http"):
        resolve_config({"video_url": "file:///tmp/a.mp4"})


def test_rejects_secrets_in_input():
    with pytest.raises(ConfigError, match="secrets must not"):
        resolve_config({"image_url": "https://example.com/a.png", "hf_token": "hf_xxx"})
    with pytest.raises(ConfigError, match="secrets must not"):
        resolve_config({"image_url": "https://example.com/a.png", "api_key": "x"})


def test_redacts_upload_url_in_public_dict():
    cfg = resolve_config({
        "image_url": "https://example.com/a.png",
        "upload_url": "https://bucket.example/presigned?sig=secret",
    })
    assert cfg.public_dict()["upload_url"] == "<presigned url>"


def test_rejects_color_fix():
    with pytest.raises(ConfigError, match="use_colorfix"):
        resolve_config({"image_url": "https://example.com/a.png", "use_colorfix": True})


def test_rejects_seedvr_v1_model():
    with pytest.raises(ConfigError, match="SeedVR v1"):
        resolve_config({"image_url": "https://example.com/a.png", "model": "seedvr-3b"})
    with pytest.raises(ConfigError, match="SeedVR v1"):
        resolve_config({"image_url": "https://example.com/a.png", "model": "seedvr1"})


def test_rejects_7b_sharp():
    with pytest.raises(ConfigError, match="sharp"):
        resolve_config({"image_url": "https://example.com/a.png", "model": "7b-sharp"})


def test_rejects_floating_revision():
    with pytest.raises(ConfigError, match="floating"):
        resolve_config({"image_url": "https://example.com/a.png", "model_revision": "main"})


def test_prompt_is_ignored_with_warning():
    cfg = resolve_config({
        "image_url": "https://example.com/a.png",
        "prompt": "make it cinematic",
    })
    assert any("ignored" in w and "pos_emb" in w for w in cfg.warnings)
    assert "prompt" not in cfg.public_dict()


def test_7b_warns_about_vram():
    cfg = resolve_config({"image_url": "https://example.com/a.png", "model": "7b"})
    assert cfg.model == "7b"
    assert cfg.model_revision == MODEL_REPOS["7b"]["revision"]
    assert cfg.public_dict()["script"].endswith("inference_seedvr2_7b.py")
    assert any("48 GB" in w for w in cfg.warnings)
    assert any("sharp" in w.lower() for w in cfg.warnings)


def test_model_aliases():
    for raw in ("3b", "SeedVR2-3B", "ByteDance-Seed/SeedVR2-3B"):
        assert resolve_config({"image_url": "https://example.com/a.png", "model": raw}).model == "3b"
