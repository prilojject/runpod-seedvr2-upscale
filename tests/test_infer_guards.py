import os
from pathlib import Path

import pytest

from src.infer import InferError, assert_not_v1_script, official_script_path
from src.paths import COLOR_FIX_RELPATH, FORBIDDEN_V1_SCRIPTS


def test_v1_scripts_are_rejected():
    for name in (
        "projects/inference_seedvr_3b.py",
        "projects/inference_seedvr_7b.py",
        "/opt/seedvr/projects/inference_seedvr_3b.py",
        "inference_seedvr_7b.py",
    ):
        with pytest.raises(InferError, match="SeedVR v1"):
            assert_not_v1_script(name)
    assert_not_v1_script("projects/inference_seedvr2_3b.py")
    assert_not_v1_script("projects/inference_seedvr2_7b.py")


def test_forbidden_list_covers_v1_only():
    for name in FORBIDDEN_V1_SCRIPTS:
        assert "seedvr2" not in os.path.basename(name).lower()
        assert "seedvr_" in os.path.basename(name) or "inference_seedvr_" in name


def test_official_script_path_requires_existing_seedvr2_file(tmp_path, monkeypatch):
    import src.infer as infer
    fake = tmp_path / "opt" / "seedvr"
    (fake / "projects").mkdir(parents=True)
    monkeypatch.setattr(infer, "SEEDVR_DIR", str(fake))
    with pytest.raises(InferError, match="not found"):
        official_script_path("3b")
    script = fake / "projects" / "inference_seedvr2_3b.py"
    script.write_text("# official placeholder\n")
    assert official_script_path("3b") == str(script)


def test_color_fix_is_not_shipped_in_this_repo():
    root = Path(__file__).resolve().parents[1]
    hits = []
    for p in root.rglob("color_fix.py"):
        if any(part in {".venv", "__pycache__", ".git"} for part in p.parts):
            continue
        hits.append(str(p))
    assert hits == []
    assert not (root / COLOR_FIX_RELPATH).exists()
