# Runpod Hub always builds from a Dockerfile. This thin file lets the Hub
# reuse the already-built image instead of compiling the worker again.
# The real build recipe lives in the repository-root Dockerfile.
#
# Pin the digest (not a floating :latest). Releases of this repo must still
# use targetCommitish=main — this digest is the image pin, not the git target.
#
# Base: :v1.0.0 / :v1.0.1 / :latest / :smoke-20261123-r3 (Docker-v2 manifest
# sha256:a9e1c48dae820ae642a4136301b7ff186834ae5d54931374b04d2f2bc0a28f54,
# config sha256:0272500851ff2110629eeb46928de4a973cffe8831fa72608786adf21a8fef0c).
FROM ghcr.io/prilojject/runpod-seedvr2-upscale:v1.0.0@sha256:a9e1c48dae820ae642a4136301b7ff186834ae5d54931374b04d2f2bc0a28f54

# v1.0.3 — Blackwell support.
# The base image ships torch 2.4.0+cu121, the official sm_90-only apex wheel,
# and flash-attn 2.5.9 built for torch 2.4. Those have no Blackwell kernels
# (and the compiled extensions are ABI-tied to torch 2.4). Swap in
# torch 2.7.1+cu128 (archs sm_86, sm_90, sm_100, sm_120 — cu128 wheels do not
# list sm_89; Ada uses sm_86 code). Do not require sm_89 (LLaMA-Factory v1.0.1 bug).
# COPY the current worker so the SDPA attention fallback is in the Hub image.
# Uninstall the torch-2.4 flash-attn and apex wheels first. A matching
# flash-attn 2.8.3.post2 torch2.7 wheel is installed when the ABI matches;
# if that wheel is missing or has no kernel for the current GPU, src/infer.py
# uses a PyTorch SDPA varlen fallback. Apex fused norms stay on the existing
# PyTorch LayerNorm/RMSNorm fallback.
COPY handler.py /app/handler.py
COPY src /app/src
RUN python -m pip uninstall -y flash-attn flash_attn apex || true
RUN python -m pip install --no-cache-dir \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && ABI=$(python -c "import torch; print('TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')") \
    && python -m pip install --no-cache-dir \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2+cu12torch2.7cxx11abi${ABI}-cp310-cp310-linux_x86_64.whl" \
        || echo "flash-attn 2.8.3.post2 wheel not installed; worker will use SDPA" \
    && python -c "import torch, torchvision, torchaudio, runpod; \
flags = torch._C._cuda_getArchFlags(); \
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| archs', flags); \
assert torch.__version__.startswith('2.7.1') and torch.version.cuda == '12.8', (torch.__version__, torch.version.cuda); \
assert all(a in flags for a in ('sm_86', 'sm_90', 'sm_100', 'sm_120')), flags" \
    && python -m pip freeze > /app/requirements.lock
