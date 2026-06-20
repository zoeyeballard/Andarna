# Reproducible CPU simulation environment for robotics-test-infra.
#
# Why a container: GitHub Actions runners have no MuJoCo, no GPU, and no display.
# This image bakes the exact sim stack (LeRobot 0.5.1 / MuJoCo 3.8 / gym-aloha) plus
# the OSMesa software-GL libraries needed for headless offscreen rendering, so a sim
# evaluation runs identically on a laptop and in CI.
#
# The policy checkpoint is NOT baked in (it's 198 MB). At runtime it is either:
#   - pulled from the Hugging Face Hub (set RTI_CHECKPOINT_PATH=<org/repo>), or
#   - bind-mounted (e.g. -v "$PWD/outputs:/outputs" and RTI_CHECKPOINT_PATH=/outputs/...).
#
# Build:  docker build -t robotics-test-infra robotics-test-infra/
# Test :  docker run --rm robotics-test-infra pytest tests/unit -q
# Eval :  docker run --rm -e RTI_CHECKPOINT_PATH=<hub-id> robotics-test-infra \
#             python scripts/run_evaluation.py --num_episodes 3

FROM python:3.12-slim

# --- system libs for headless MuJoCo (OSMesa software GL) + video encoding ---
# libgl1-mesa-glx was removed in Debian trixie (python:3.12-slim's base); libgl1 is the
# modern GL loader package. Headless OSMesa rendering needs libosmesa6 + the GL loader.
# build-essential + python3-dev + linux-libc-dev: LeRobot pulls evdev (a teleop dep),
# which is sdist-only and compiles a C extension against <linux/input.h> — the slim
# image ships no compiler or kernel headers, so the wheel build fails without these.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libosmesa6 \
        libglfw3 \
        libgl1-mesa-dri \
        ffmpeg \
        git \
        build-essential \
        python3-dev \
        linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Headless software rendering — no GPU, no display required.
ENV MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    RTI_RENDER_BACKEND=osmesa \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install CPU-only torch + torchvision FIRST so the heavy sim deps resolve against them
# instead of pulling multi-GB CUDA wheels (the Project-1 lesson). torchvision MUST come
# from the same CPU index as torch — otherwise pip pulls the PyPI (CUDA-ABI) torchvision
# and its compiled ops fail to bind ("operator torchvision::nms does not exist") the
# moment lerobot imports torchvision.transforms. 2.10.0 ↔ 0.25.0 is the matching pair.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.10.0" "torchvision==0.25.0"

# Dependency install is its own layer (cached unless pyproject changes).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[sim,dev]"

# Project source (changes most often → last layer).
COPY tests ./tests
COPY scripts ./scripts
COPY baselines ./baselines

# Sanity check at build time: the sim stack imports and MuJoCo can render headless.
RUN python -c "import lerobot, gym_aloha, mujoco; import gymnasium as gym; \
    e=gym.make('gym_aloha/AlohaTransferCube-v0'); e.reset(seed=0); \
    f=e.unwrapped.render(); print('build-check render', None if f is None else f.shape)"

CMD ["pytest", "tests/unit", "-q"]
