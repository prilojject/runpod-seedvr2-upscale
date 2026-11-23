# Headless SeedVR2 batch upscaler for Runpod Serverless.
# No model weights are baked in: SeedVR2-3B (~13.57 GiB) is downloaded at job
# start and cached on the network volume (/runpod-volume/seedvr2/models) when
# one is attached.
#
# Official SeedVR code has no tags; pin the SHA below. This SHA is only the
# Docker / upstream pin; our GitHub release for the Hub must use
# targetCommitish=main.
#
# Stack follows the official README + prebuilt apex wheel, NOT upstream
# requirements.txt (which pins torch==2.3.0). We install Python 3.10,
# torch 2.4.0+cu121, and apex-0.1-cp310-cp310-linux_x86_64.whl (python=3.10,
# torch=2.4.0, cuda=12.1). Do not "upgrade" to torch 2.7 / cu128.
# environment.yml is bloated (tensorflow etc.) and is not installed.
ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG SEEDVR_SHA=e4de8c24441a67e1b7df56abea10645059bb1185
ARG APEX_REVISION=37255ff8cccfb01071b87f635a5948ca8d53117c
ARG FLASH_ATTN_WHL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    SEEDVR_DIR=/opt/seedvr \
    SEEDVR_SHA=${SEEDVR_SHA} \
    MODEL=3b \
    MODEL_REVISION=37255ff8cccfb01071b87f635a5948ca8d53117c

# python3.10 is the Ubuntu 22.04 default (required by the official apex wheel).
# ffmpeg: video containers. git: clone the pinned SHA. libgl: opencv.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git ffmpeg ca-certificates wget \
        python3 python3-venv python3-pip python3-dev \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && python -c "import sys; assert sys.version_info[:2] == (3, 10), sys.version"

# Clone the official repo and check out the pinned SHA. Do not clone a floating main.
# Gradio / ComfyUI are not started. color_fix.py must not be present (S-Lab NC).
# Drop .git after the pin.
RUN git clone https://github.com/ByteDance-Seed/SeedVR.git ${SEEDVR_DIR} \
    && cd ${SEEDVR_DIR} \
    && git checkout ${SEEDVR_SHA} \
    && git rev-parse HEAD > /opt/seedvr.commit \
    && test "$(cat /opt/seedvr.commit)" = "${SEEDVR_SHA}" \
    && test ! -f ${SEEDVR_DIR}/projects/video_diffusion_sr/color_fix.py \
    && rm -f ${SEEDVR_DIR}/projects/video_diffusion_sr/color_fix.py \
    && rm -rf .git \
    && mkdir -p ${SEEDVR_DIR}/ckpts \
    && chmod -R a+rwX ${SEEDVR_DIR}

# Use a venv so we do not fight the Debian pip install (no RECORD file).
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Stage 1: official torch 2.4.0+cu121 (README / apex wheel). Not torch 2.3.0.
WORKDIR ${SEEDVR_DIR}
RUN pip install --upgrade pip \
    && pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && python -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+)==' > /tmp/constraints.txt \
    && cat /tmp/constraints.txt

# Stage 2: official Python deps from requirements.txt EXCEPT torch==2.3.0 and
# torchvision==0.18.0 (those conflict with the apex wheel). Keep transformers
# 4.38.2 as written by upstream; do not upgrade torch afterwards.
# Plus av/tqdm/numpy/pillow needed by the official scripts at runtime.
RUN python -m pip install \
        einops==0.7.0 \
        omegaconf==2.3.0 \
        opencv-python==4.9.0.80 \
        diffusers==0.29.1 \
        rotary-embedding-torch==0.5.3 \
        transformers==4.38.2 \
        mediapy==1.2.0 \
        tqdm numpy pillow av packaging \
        -c /tmp/constraints.txt \
    && python -m pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+)==' > /tmp/constraints.txt \
    && python -c "import torch; assert torch.__version__.startswith('2.4.0'), torch.__version__; assert 'cu121' in torch.__version__ or torch.version.cuda.startswith('12.1'), (torch.__version__, torch.version.cuda)"

# Prebuilt flash-attn 2.5.9.post1 wheel for torch 2.4 + cp310. Prefer a wheel
# over a long source build; do not upgrade torch to make a newer wheel fit.
RUN python -m pip install --no-build-isolation "${FLASH_ATTN_WHL}" -c /tmp/constraints.txt

# Official prebuilt apex wheel from the SeedVR2-3B repo at the pinned weights
# revision (python=3.10, torch=2.4.0, cuda=12.1). Install into the image, not
# into the weight cache. This is a small wheel, not the 13 GiB checkpoint.
RUN wget -q -O /tmp/apex-0.1-cp310-cp310-linux_x86_64.whl \
        "https://huggingface.co/ByteDance-Seed/SeedVR2-3B/resolve/${APEX_REVISION}/apex-0.1-cp310-cp310-linux_x86_64.whl" \
    && python -m pip install /tmp/apex-0.1-cp310-cp310-linux_x86_64.whl -c /tmp/constraints.txt \
    && rm -f /tmp/apex-0.1-cp310-cp310-linux_x86_64.whl

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt -c /tmp/constraints.txt \
    && python -m pip freeze > /app/requirements.lock \
    && python -c "import sys,torch; import runpod; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| python', sys.version.split()[0], '| runpod', runpod.__version__, '| seedvr sha', open('/opt/seedvr.commit').read().strip())"

COPY handler.py /app/handler.py
COPY src /app/src

# No secrets are baked in: HF_TOKEN / BUCKET_* come from Runpod secrets at runtime.
# The official Gradio / ComfyUI / web Space is never started. Weights are not baked.
# color_fix.py is not added (S-Lab License 1.0, non-commercial).
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
