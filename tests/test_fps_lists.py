"""Unit tests for the official image-only fps_lists bug and the wrapper fix."""

from src.infer import (
    official_write_zip_keeps_images,
    pad_fps_lists,
    patch_generation_loop_source,
)

# Minimal excerpt of the official generation_loop write / read logic
# (Apache-2.0, ByteDance-Seed/SeedVR @ e4de8c24441a67e1b7df56abea10645059bb1185).
OFFICIAL_SNIPPET = """
        fps_lists = []
        for video in videos:
            if is_image_file(video):
                video = read_image(
                    os.path.join(video_path, video)
                ).unsqueeze(0) / 255.0
                if sp_size > 1:
                    raise ValueError("Sp size should be set to 1 for image inputs!")
            else:
                video, _, info = read_video(
                    os.path.join(video_path, video), output_format="TCHW"
                    )
                video = video / 255.0
                fps_lists.append(info["video_fps"] if out_fps is None else out_fps)
        for path, input, sample, ori_length, save_fps in zip(
            videos, input_videos, samples, ori_lengths, fps_lists
        ):
            mediapy.write_image(filename, sample.squeeze(0))
"""


def test_official_image_only_zip_drops_all_outputs():
    paths = ["a.png", "b.jpg"]
    fps_lists = []  # official loop never appends for images
    kept = official_write_zip_keeps_images(paths, fps_lists)
    assert kept == [], "the official zip(..., fps_lists) must drop image-only batches"


def test_official_video_batch_is_fine():
    paths = ["a.mp4", "b.mp4"]
    fps_lists = [24.0, 30.0]
    kept = official_write_zip_keeps_images(paths, fps_lists)
    assert kept == paths


def test_pad_fps_lists_repairs_image_only_batch():
    paths = ["a.png", "b.jpg"]
    fixed = pad_fps_lists(len(paths), [], out_fps=None)
    assert len(fixed) == 2
    assert fixed == [24.0, 24.0]
    assert official_write_zip_keeps_images(paths, fixed) == paths


def test_pad_fps_lists_respects_out_fps_and_leaves_video_fps():
    assert pad_fps_lists(2, [30.0], out_fps=12.5) == [30.0, 12.5]
    assert pad_fps_lists(2, [24.0, 25.0]) == [24.0, 25.0]


def test_patch_inserts_fps_append_and_disables_color_fix():
    patched = patch_generation_loop_source(
        OFFICIAL_SNIPPET + "\n        if use_colorfix:\n            wavelet_reconstruction(sample, input)\n"
    )
    assert "fps_lists.append(out_fps if out_fps is not None else 24.0)" in patched
    # The image branch now appends, so an image-only batch is no longer empty.
    assert patched.index("fps_lists.append") < patched.index("else:")
    assert "if False:  # color_fix is S-Lab NC; never enable" in patched
    assert "if use_colorfix:" not in patched


def test_patch_against_official_checked_in_source():
    # Use the real official file when we have a cached copy (tests stay offline
    # if it is absent; the snippet test above always runs).
    import os
    candidates = [
        "/tmp/inference_seedvr2_3b.py",
        os.path.join(os.path.dirname(__file__), "fixtures", "inference_seedvr2_3b.py"),
    ]
    src = None
    for path in candidates:
        if os.path.isfile(path):
            src = open(path, encoding="utf-8").read()
            break
    if src is None:
        return
    start = src.index("def generation_loop")
    patched = patch_generation_loop_source(src[start:])
    assert "fps_lists.append(out_fps if out_fps is not None else 24.0)" in patched
    img = patched.index("if is_image_file(video):")
    append = patched.index("fps_lists.append(out_fps if out_fps is not None else 24.0)")
    assert img < append
    assert "if False:  # color_fix is S-Lab NC; never enable" in patched
