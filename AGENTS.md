# Agent notes — runpod-seedvr2-upscale

## What this worker is

A Runpod Serverless **headless SeedVR2 batch upscaler** for videos and photos. Input: `image_url` and/or `video_url` and/or a zip of mixed media. Output: files of the same type and basename plus `results.json`.

Wrap official `projects/inference_seedvr2_3b.py` (optionally `_7b.py`) in-process. Do **not** start Gradio, ComfyUI, or the official web Space. Do **not** call SeedVR v1 scripts (`projects/inference_seedvr_3b.py` / `_7b.py`). Do **not** fold this into the LoRA trainers or captioners. Do **not** bake weights into the image. Do **not** add `color_fix.py` from sd-webui-stablesr (S-Lab License 1.0, non-commercial).

## Pinned upstream

- Code: `ByteDance-Seed/SeedVR` @ SHA `e4de8c24441a67e1b7df56abea10645059bb1185` (written to `/opt/seedvr.commit`). There are no upstream tags. Do **not** clone a floating `main` as the runtime.
- Weights: `ByteDance-Seed/SeedVR2-3B` @ `37255ff8cccfb01071b87f635a5948ca8d53117c` (~13.57 GiB, public, Apache-2.0, no token). Cache under `/runpod-volume/seedvr2/models/`. Optional 7B @ `eb0c4281d41ba3767d4f14370f0e37e9e9180c16`. Do not download `seedvr2_ema_7b_sharp.pth` with the 7B option. 7B has no embeddings — take `pos_emb.pt` / `neg_emb.pt` from the 3B snapshot.
- The SHA is the upstream pin only. GitHub releases of *this* repo must use `targetCommitish=main` (a SHA target breaks Runpod Hub's `git clone -b`).

## Stack (strict)

- Python **3.10**, `torch==2.4.0+cu121`, official `apex-0.1-cp310-cp310-linux_x86_64.whl` (python=3.10, torch=2.4.0, cuda=12.1), `flash_attn==2.5.9.post1` from a prebuilt wheel.
- Upstream `requirements.txt` pins `torch==2.3.0` — that conflicts with the README / apex wheel. Follow the README / wheel, not 2.3.0. Do not upgrade to torch 2.7 / cu128. Do not install `environment.yml` wholesale.

## Defaults to keep

- SeedVR2-3B, **bfloat16** autocast (not fp16), `sample_steps=1`, `sp_size=1`, seed 666, `res_h=720` `res_w=1280` (geometric-mean resize + crop 16)
- `refresh_worker=false`, `containerDiskInGb=80`
- The wrapper **must** pad `fps_lists` for image-only batches (official `zip(..., fps_lists)` otherwise writes nothing)
- Force `use_colorfix=False`
- Official apex wheel is **sm_90 only** (H100). On other GPUs (including 4090 sm_89) the wrapper remaps `fusedln`/`fusedrms` to PyTorch LayerNorm/RMSNorm before `configure_runner`. Patch both `normalization.py` and the bound name in `nadit.py`.
- Official `configure_runner` places DiT and VAE on CUDA together. On 24GB cards, offload DiT to CPU, `empty_cache` after each swap, and cast DiT weights to bf16. The image-only `fps_lists.append` must be a sibling of `if sp_size > 1`, not dead code under the `raise`.
- Custom text prompts are ignored (baked `pos_emb.pt` / `neg_emb.pt`)
- 7B is an option only for 48 GB+; do not smoke it and do not make it the default
- Official long 720p video (100×720×1280) is documented on H100 80 GB — do not promise it on a 4090

## License (strict)

- Worker code: MIT
- Upstream code and 3B/7B weights: **Apache-2.0**. README and NOTICE must say so and attribute ByteDance Seed. Ship `LICENSE.upstream`. The weight repos have no LICENSE file — do not invent a different one.
- Never vendor `color_fix.py` (S-Lab NC).

## Tests

- Unit tests (`tests/`) must run without a GPU and without model weights.
- The Hub smoke test is one case only: `seedvr2_3b_image_smoke` — 3B, one small public-domain image (`examples/images/tiny_pd.png`, documented in `examples/SOURCES.md`), `res_h=512` `res_w=768`, seed 666, RTX 4090, timeout 20 min. Success: a non-empty output image, larger than the source or with the declared side after the crop-16. Stop if spend approaches $3–4. Do not smoke 7B, sharp, long video, or `sp_size>1` on the paid Hub test.
- After a paid GPU test: delete the endpoint, pod/template and registry auth. Confirm with GraphQL that leftover smoke endpoints are gone. Leave protected pods `92anyumh6q7jrf` and `oejzh2afh8hcym` (and volumes `13faqbphuo`, `fv2c638zex`) untouched.

## Secrets

Never print `RUNPOD_API_KEY`, `HF_TOKEN`, `gh` tokens, or `BUCKET_*` values. Never accept tokens in the job input.
