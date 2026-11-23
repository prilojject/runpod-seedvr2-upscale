# SeedVR2 Upscale for Runpod

[![Runpod](https://img.shields.io/badge/Runpod-Hub-7c3aed)](https://console.runpod.io/hub/listing/prilojject/runpod-seedvr2-upscale)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A **Runpod Serverless** worker that **batch-upscales and restores videos and photos** with official [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) (one-step diffusion, ICLR 2026).

**Send** one `image_url`, one `video_url`, and/or a zip (`dataset_url`). **Get** files of the same type and basename in an output folder, plus a `results.json` manifest.

It is a headless wrapper around official `projects/inference_seedvr2_3b.py` (and optionally `_7b.py`). Gradio, ComfyUI, and the official web Space are never started. This is **not** SeedVR v1 (the multi-step `projects/inference_seedvr_3b.py` / `_7b.py` scripts are never called) and **not** a community ComfyUI port.

Worker code in this repository is MIT. SeedVR / SeedVR2 code and the 3B / 7B weights are **Apache-2.0** (© 2025 ByteDance Seed). See [License](#license).

## What you send / what you get

**Send** `image_url` and/or `video_url` and/or `dataset_url` (a `.zip` or `.tar.gz` of jpg/png/webp/bmp/tiff and mp4/mov/webm/mkv/…). Images and videos may be mixed in one batch.

```
input/
  frame.png
  clip.mp4
```

**Get** the same basenames, plus a manifest:

```
frame.png
clip.mp4
results.json
```

Delivered as a zip to a Hugging Face repo, an S3-compatible bucket, a presigned PUT URL, or a Runpod network volume (`/runpod-volume/seedvr2/runs/<run_name>/`). If none of those are configured, the job still returns metadata in the response, and a warning that the files will disappear when the worker stops.

## Defaults

| Setting | Default | Notes |
| --- | --- | --- |
| Model | **SeedVR2-3B** | `ByteDance-Seed/SeedVR2-3B` @ `37255ff8cccfb01071b87f635a5948ca8d53117c` |
| Dtype | **bfloat16** autocast (not fp16) | Official `torch.autocast("cuda", torch.bfloat16)`. Checkpoints on disk are large `.pth` files (~12.63 GiB + 0.93 GiB VAE + embeddings ≈ **13.57 GiB**) |
| Steps | **1** (one-step SeedVR2) | Do not raise the training yaml's 50 steps |
| Output size | `res_h=720`, `res_w=1280` | Official argparse. The script resizes to the geometric mean `sqrt(res_h*res_w)` (`NaResize`) then `DivisibleCrop((16,16))` — **not** exactly 720×1280 |
| CFG | `cfg_scale=1.0` | Official `generation_loop` default |
| `sp_size` | **1** | Required for images. Multi-GPU (`sp_size=4`) for 1080p/2K is not the v1 path |
| Seed | **666** | Official default |
| GPU | **RTX 4090 24 GB** for smoke / a small photo or a short clip at a moderate size | Official README: **1× H100 80 GB** can handle videos with **100×720×1280**; 4× H100-80G for 1080p and 2K. That 100-frame 720p regime is **not** a 4090 guarantee. **Blackwell is not supported** (see below). |
| `refresh_worker` | **false** | The model stays in worker memory |

Public 3B weights (~13.57 GiB, no token, Apache-2.0) are downloaded at job start and cached under `/runpod-volume/seedvr2/models/` when a network volume is attached. They are **not** baked into the image. Without a volume, every cold start re-downloads them — attach a volume for anything other than a one-off test.

Upstream SeedVR code is pinned to SHA `e4de8c24441a67e1b7df56abea10645059bb1185` (the official repo has no tags). That SHA is the **upstream** pin only. Releases of *this* repo use `targetCommitish=main`.

The image stack is **Python 3.10 + torch 2.4.0+cu121 + the official apex `cp310` wheel + flash-attn 2.5.9**, as in the SeedVR README. Upstream `requirements.txt` pins `torch==2.3.0`, which conflicts with that wheel — this worker follows the README / wheel, not 2.3.0, and does not upgrade to torch 2.7 or a Blackwell-capable CUDA build. The host driver must be **CUDA ≤ 12.7**. **Blackwell is not supported.** 4090 / CUDA 12.1–12.7 supply may be limited.

Optional checkpoint (same Apache-2.0 license, **not** the smoke path): `ByteDance-Seed/SeedVR2-7B` @ `eb0c4281d41ba3767d4f14370f0e37e9e9180c16` (~31.62 GiB without sharp). Raw 7B weights are already ~30.7 GiB; even with bf16 autocast this is a **48 GB+** class, not a 24 GB smoke. The `seedvr2_ema_7b_sharp.pth` checkpoint is **not** downloaded with the 7B option (it would roughly double the download). 7B has no `pos_emb.pt` / `neg_emb.pt`; the worker takes those from the 3B snapshot.

## Color fix is not included

`color_fix.py` from [pkuliyi2015/sd-webui-stablesr](https://github.com/pkuliyi2015/sd-webui-stablesr) is licensed under **S-Lab License 1.0 (non-commercial only)** and is **not** shipped. Official SeedVR already treats color fix as optional. Setting `use_colorfix: true` is rejected.

## Prompts are baked

The official script **ignores** text prompts and always loads `pos_emb.pt` / `neg_emb.pt`. There is no text encoder on this path. A custom `prompt` in the job input is accepted only to warn that it is ignored. Do not expect arbitrary prompting in v1 of this listing.

## Example job

Photo upscale (3B, official 720×1280 area):

```json
{
  "input": {
    "image_url": "https://example.com/photo.png",
    "model": "3b",
    "res_h": 720,
    "res_w": 1280,
    "seed": 666
  }
}
```

Quick test (one small public-domain image, moderate output area — this is the Hub smoke):

```json
{
  "input": {
    "model": "3b",
    "image_url": "https://raw.githubusercontent.com/prilojject/runpod-seedvr2-upscale/main/examples/images/tiny_pd.png",
    "res_h": 512,
    "res_w": 768,
    "seed": 666
  }
}
```

`scale` is an optional helper: if you set `scale: 2` and do not also set `res_h`/`res_w`, the worker computes `res_*` from the first input's size times 2. A short video job uses `video_url` the same way; long 720p video is H100 work, not the smoke test.

Input priority is job input > environment (Hub form) > defaults. Never put tokens in the job input; use Runpod secrets / env (`HF_TOKEN`, `BUCKET_*`).

## Smoke test

A paid Serverless smoke already ran on 23 Nov 2026 (Asia/Bangkok) on an **RTX 4090**. This v1.0.0 publish does not run another GPU test.

- Model: **SeedVR2-3B**, `sample_steps=1`, bfloat16, seed 666
- Input: the official NASA / ESA / Hubble public-domain `~small.jpg` of NASA ID `GSFC_20171208_Archive_e000842` (**640×599**), not the bundled 512×768 crop in `examples/images/tiny_pd.png`
- Output: 640×592 (crop to a multiple of 16)
- Job execution: **~78 s** (`executionTime` 77575 ms); spend **≈ $0.40**
- The official apex wheel is **sm_90 only**, so this 4090 run used the PyTorch `LayerNorm` / `RMSNorm` fallback
- **DiT/VAE offload** between steps is enabled on 24 GB cards
- No network volume was attached, so the **~13.57 GiB** weights were downloaded on that cold start
- **SeedVR2-7B** and long **720p** video were **not** smoked

The Hub automated test (`seedvr2_3b_image_smoke` in `.runpod/tests.json`) still uses the bundled `tiny_pd.png` at `res_h=512` / `res_w=768`.

## FAQ

**The official apex wheel and a 4090.** The prebuilt `apex-0.1-cp310-cp310-linux_x86_64.whl` in the SeedVR2-3B repo contains fused LayerNorm / RMSNorm kernels for **sm_90 (H100) only**. On an RTX 4090 (sm_89) the 3B weights load, then the first fused norm fails with `CUDA error: no kernel image is available for execution on the device`. This worker detects that and falls back to PyTorch `LayerNorm` / `RMSNorm` so the official 3B script can still run on a 4090. On H100 the official fused kernels are used unchanged.

**Is Blackwell supported?** No. This image is torch 2.4.0+cu121 with that sm_90-only apex wheel and flash-attn 2.5.9; it has no Blackwell kernels. A Hub test that falls back to a Blackwell GPU fails with `no kernel image is available`. The listing caps `allowedCudaVersions` at **12.7** and excludes Blackwell GPU IDs so that cannot happen. The host driver must be CUDA **≤ 12.7**. Because of that cap, 4090 / CUDA 12.1–12.7 supply may be limited. This listing does not upgrade torch to fix Blackwell.

**Why does the worker offload DiT and VAE on a 4090?** Official `configure_runner` places DiT and VAE on CUDA together. On 24 GB that combination OOMs. This worker offloads DiT to CPU, empties the CUDA cache after each swap, and casts DiT weights to bf16 on those cards. That offload was part of the paid 4090 smoke above.

**Will a 100-frame 720p video fit on a 4090?** Not as a guarantee of the official script. The official README documents that regime on 1× H100 80 GB. This listing's paid smoke was a single 640×599 photo on a 4090. Long 720p video was not smoked. Community ComfyUI tiling / FP8 / GGUF / BlockSwap is not mixed into the official path.

**Why is there no color fix?** The usual `color_fix.py` is S-Lab License 1.0, non-commercial. It is not included.

**Can I pass a text prompt?** No. Embeddings are baked in `pos_emb.pt` / `neg_emb.pt`.

**Is this SeedVR v1?** No. Only `inference_seedvr2_*.py` is called.

**Are the weights baked into the image?** No. Attach a network volume to cache them under `/runpod-volume/seedvr2/models/`.

## Changelog

- **v1.0.2** (23 Nov 2026, Asia/Bangkok): Re-run the Hub build/test. v1.0.1 could not start because no free GPU was available (RTX 4090; CUDA capped at 12.1–12.7). The image, CUDA caps, and gpuIds are unchanged.

## License

- Worker code: [MIT](LICENSE)
- Upstream SeedVR / SeedVR2 code: [Apache-2.0](LICENSE.upstream), © 2025 seed / ByteDance Seed
- Weights `ByteDance-Seed/SeedVR2-3B` and `SeedVR2-7B`: Apache-2.0 on the model card (`gated: false`). The weight repos do not ship a LICENSE file; see [NOTICE](NOTICE).

Example image: public-domain NASA / ESA / Hubble photograph documented in [examples/SOURCES.md](examples/SOURCES.md).
