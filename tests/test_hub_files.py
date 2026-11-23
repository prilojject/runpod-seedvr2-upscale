"""Static checks of .runpod/hub.json and tests.json against the spec."""
import json
import os

from src.config import PARAMS
from src.paths import MODEL_REPOS, SEEDVR_SHA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA_ENV = {"HF_TOKEN", "PRESET", "INFER_TIMEOUT_SECONDS", "USE_COLORFIX", "SCALE", "OUT_FPS", "SP_SIZE", "SAMPLE_STEPS", "MODEL"}


def load(name):
    with open(os.path.join(ROOT, ".runpod", name)) as fh:
        return json.load(fh)


def test_required_files():
    for f in (
        "handler.py", "Dockerfile", "README.md", "LICENSE", "LICENSE.upstream",
        "NOTICE",
        ".runpod/hub.json", ".runpod/tests.json", ".runpod/icon.png",
        "examples/images/tiny_pd.png", "examples/SOURCES.md",
        "llms.txt", "AGENTS.md", "CITATION.cff", "openapi.yaml", "context7.json",
        "scripts/smoke_runpod.py",
    ):
        assert os.path.exists(os.path.join(ROOT, f)), f


def test_hub_json():
    hub = load("hub.json")
    for k in ("title", "description", "type", "category", "config", "iconUrl"):
        assert hub.get(k), k
    assert hub["type"] == "serverless"
    assert hub["category"] in {"video", "image"}
    cfg = hub["config"]
    assert cfg["runsOn"] == "GPU" and cfg["containerDiskInGb"] == 80 and cfg["gpuCount"] == 1
    assert cfg["gpuIds"].startswith("ADA_24,AMPERE_24,ADA_48_PRO,AMPERE_48")
    assert "1g.24gb" in cfg["gpuIds"] and "2g.48gb" in cfg["gpuIds"]
    # Image is torch 2.4.0+cu121 + sm_90-only apex / flash-attn 2.5.9: no Blackwell, CUDA ≤12.7.
    cuda_cap = ["12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1"]
    assert cfg["allowedCudaVersions"] == cuda_cap
    blackwell_ids = [
        "NVIDIA GeForce RTX 5080",
        "NVIDIA GeForce RTX 5090",
        "NVIDIA B200",
        "NVIDIA B300 SXM6 AC",
        "NVIDIA RTX PRO 4000 Blackwell",
        "NVIDIA RTX PRO 4500 Blackwell",
        "NVIDIA RTX PRO 4500 Blackwell Server Edition",
        "NVIDIA RTX PRO 5000 Blackwell",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb",
        "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    ]
    ids = cfg["gpuIds"].split(",")
    assert ids[:4] == ["ADA_24", "AMPERE_24", "ADA_48_PRO", "AMPERE_48"]
    for gpu in blackwell_ids:
        assert f"-{gpu}" in ids, gpu
    assert "NVIDIA RTX 5000 Ada Generation" not in cfg["gpuIds"]
    known = {p.env for p in PARAMS if p.env} | EXTRA_ENV
    keys = [e["key"] for e in cfg["env"]]
    assert len(keys) == len(set(keys))
    required = {
        "MODEL", "MODEL_REVISION", "IMAGE_URL", "VIDEO_URL", "DATASET_URL",
        "RES_H", "RES_W", "SEED", "HF_OUTPUT_REPO", "HF_TOKEN",
    }
    assert required <= set(keys)
    assert set(keys) <= known, set(keys) - known
    for preset in cfg["presets"]:
        assert set(preset["defaults"]) <= set(keys), preset["name"]
    names = [p["name"] for p in cfg["presets"]]
    assert any("Photo" in n or "photo" in n.lower() for n in names)
    assert any("Quick test" in n or "quick" in n.lower() for n in names)
    for e in cfg["env"]:
        inp = e["input"]
        assert inp["type"] in {"string", "number", "boolean", "huggingface"}
        if inp["type"] == "number" and "min" in inp:
            assert inp["min"] <= inp["default"] <= inp["max"]
        if inp["type"] == "boolean":
            assert {inp["trueValue"], inp["falseValue"]} == {"true", "false"}
        for o in inp.get("options", []):
            assert o["label"] and o["value"] is not None
        if e["key"] == "HF_TOKEN":
            assert "RUNPOD_SECRET" in inp["description"] or "secret" in inp["description"].lower()


def test_tests_json():
    t = load("tests.json")
    assert t["config"]["gpuTypeId"] == "NVIDIA GeForce RTX 4090"
    assert t["config"]["gpuCount"] == 1
    assert t["config"]["allowedCudaVersions"] == [
        "12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1"
    ]
    assert "12.8" not in t["config"]["allowedCudaVersions"]
    assert len(t["tests"]) == 1
    case = t["tests"][0]
    assert case["name"] == "seedvr2_3b_image_smoke"
    assert case["input"]["model"] == "3b"
    assert case["input"]["res_h"] == 512
    assert case["input"]["res_w"] == 768
    assert case["input"]["seed"] == 666
    assert "tiny_pd.png" in case["input"]["image_url"]
    assert "prilojject/runpod-seedvr2-upscale/main" in case["input"]["image_url"]
    assert case.get("timeout", 0) >= 1_200_000
    # One cheap image test only — no 7B, no long video.
    blob = json.dumps(t)
    assert "7b" not in blob.lower()
    assert "video_url" not in blob


def test_dockerfile_pins_sha_and_stack():
    text = open(os.path.join(ROOT, "Dockerfile")).read()
    assert f"SEEDVR_SHA={SEEDVR_SHA}" in text
    assert "git checkout ${SEEDVR_SHA}" in text
    assert "git clone" in text and "origin/main" not in text
    assert "--branch main" not in text
    assert "pip install gradio" not in text.lower()
    assert "pip install comfy" not in text.lower()
    assert "comfyui" not in text.lower() or "never" in text.lower() or "not" in text.lower()
    assert "inference_seedvr_3b.py" not in text  # v1 must not be invoked
    assert "inference_seedvr_7b.py" not in text
    assert "color_fix.py" in text  # only to *assert it is absent* or forbid it
    assert "use_colorfix=True" not in text
    assert "torch==2.4.0" in text
    assert "cu121" in text
    assert "2.3.0" not in text or "not" in text.lower()  # may mention the mismatch
    assert "torch==2.7.0" not in text and "torch==2.7.1" not in text
    assert "download.pytorch.org/whl/cu128" not in text
    assert "--index-url https://download.pytorch.org/whl/cu121" in text
    assert "python=3.10" in text or "3, 10" in text or "python3.10" in text.lower() or "(3, 10)" in text
    assert "flash_attn" in text.lower() or "flash-attn" in text.lower() or "flash_attn-" in text
    assert "apex-0.1-cp310-cp310-linux_x86_64.whl" in text
    assert "37255ff8cccfb01071b87f635a5948ca8d53117c" in text  # apex wheel pin, not baked weights
    assert "snapshot_download" not in text
    assert "huggingface-cli download ByteDance-Seed/SeedVR2-3B" not in text
    assert "seedvr2_ema_3b.pth" not in text or "not bake" in text.lower() or "not baked" in text.lower()
    assert 'CMD ["python", "-u", "/app/handler.py"]' in text
    # environment.yml must not be installed wholesale.
    assert "environment.yml" not in text or "not" in text.lower()


def test_handler_refresh_worker_false():
    text = open(os.path.join(ROOT, "handler.py"), encoding="utf-8").read()
    assert 'refresh_worker"] = False' in text or "refresh_worker'] = False" in text
    assert "inference_seedvr_3b.py" not in text
    assert "import gradio" not in text.lower() and "from gradio" not in text.lower()
    assert "import comfy" not in text.lower()


def test_readme_and_notice_cover_licenses_and_limits():
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    notice = open(os.path.join(ROOT, "NOTICE"), encoding="utf-8").read()
    for text in (readme, notice):
        assert "Apache" in text
        assert "ByteDance" in text or "Byte Dance" in text or "seed" in text.lower()
        assert "MIT" in text
    assert SEEDVR_SHA in readme
    assert MODEL_REPOS["3b"]["repo_id"] in readme
    assert "bfloat16" in readme.lower() or "bf16" in readme.lower()
    assert "H100" in readme
    assert "color fix" in readme.lower() or "color_fix" in readme.lower()
    assert "S-Lab" in readme
    assert "one-step" in readme.lower() or "sample_steps=1" in readme or "one step" in readme.lower()
    assert "Gradio" in readme or "gradio" in readme
    assert "ComfyUI" in readme
    assert "SeedVR v1" in readme or "SeedVR2" in readme
    # Do not promise 720p 100-frame on a 4090.
    assert "100" in readme and "720" in readme
    # Prompts are baked.
    assert "pos_emb" in readme or "baked" in readme.lower() or "text encoder" in readme.lower()


def test_requirements_do_not_pin_torch_230():
    req = open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8").read()
    # Mentions of torch==2.3.0 are allowed only as a warning not to install it.
    for line in req.splitlines():
        s = line.strip()
        if s.startswith("#") or not s:
            continue
        assert "torch==2.3.0" not in s
    assert "runpod" in req
