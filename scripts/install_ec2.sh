#!/usr/bin/env bash
#
# install_ec2.sh — Set up the OpenVLA profiling environment on an AWS EC2 A10G instance.
#
# What it does:
#   1. Verifies the NVIDIA driver + CUDA toolkit (nvidia-smi / nvcc) and Nsight tools.
#   2. Creates an isolated Python 3.10 venv (avoids clobbering system / other venvs).
#   3. Installs PyTorch (CUDA build), transformers, flash-attn, bitsandbytes.
#   4. Clones and `pip install -e` OpenVLA and LIBERO.
#   5. Runs a GPU smoke test to confirm CUDA is actually usable from PyTorch.
#
# Usage:
#   bash scripts/install_ec2.sh                 # full install
#   bash scripts/install_ec2.sh --verify-only   # just run the checks + GPU smoke test
#   bash scripts/install_ec2.sh --skip-flash    # skip flash-attn (slow to build; optional)
#
# Safe to re-run: each step checks before doing work. Run from the repo root.

set -euo pipefail

# ----------------------------------------------------------------------------- config
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-$HOME/openvla-profiling/.venv}"
SRC_DIR="${SRC_DIR:-$HOME/src}"               # where OpenVLA / LIBERO get cloned
TORCH_CUDA_CHANNEL="${TORCH_CUDA_CHANNEL:-cu121}"   # PyTorch CUDA 12.1 wheels (A10G / Ampere)
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.40.1}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.5.5}"

VERIFY_ONLY=0
SKIP_FLASH=0
for arg in "$@"; do
  case "$arg" in
    --verify-only) VERIFY_ONLY=1 ;;
    --skip-flash)  SKIP_FLASH=1 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# ----------------------------------------------------------------------------- logging
log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------- driver / CUDA
check_gpu_driver() {
  log "Checking NVIDIA driver (nvidia-smi)..."
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not working. This A10G AMI often needs a manual driver install."
    cat <<'EOF'

  The NVIDIA driver is not loaded. Install it before continuing, e.g.:

      sudo apt-get update
      sudo apt-get install -y nvidia-driver-535   # or the version your CUDA needs
      sudo reboot

  After reboot, confirm with `nvidia-smi` (should list the A10G, 24 GB) and re-run this script.
EOF
    die "NVIDIA driver not available."
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  ok "NVIDIA driver present."
}

check_cuda_toolkit() {
  log "Checking CUDA toolkit (nvcc)..."
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | grep -i release || true
    ok "nvcc present."
  else
    warn "nvcc not found. PyTorch's bundled CUDA runtime is enough for inference/profiling,"
    warn "but building flash-attn from source needs the full toolkit. Install with:"
    warn "    sudo apt-get install -y cuda-toolkit-12-1"
  fi
}

check_nsight() {
  log "Checking Nsight tools (nsys / ncu)..."
  command -v nsys >/dev/null 2>&1 && nsys --version | head -1 || warn "nsys not found (install CUDA toolkit / Nsight Systems)."
  command -v ncu  >/dev/null 2>&1 && ncu  --version | head -1 || warn "ncu not found (install Nsight Compute)."
}

# --------------------------------------------------------------------------- python env
setup_venv() {
  log "Setting up Python venv at $VENV_DIR ..."
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "$PYTHON_BIN not found. Install Python 3.10 (sudo apt-get install -y python3.10 python3.10-venv)."
  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    ok "Created venv."
  else
    ok "venv already exists — reusing."
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip setuptools wheel >/dev/null
  ok "pip/setuptools/wheel upgraded. Active python: $(which python)"
}

install_pytorch() {
  log "Installing PyTorch (CUDA $TORCH_CUDA_CHANNEL build)..."
  if python -c 'import torch' 2>/dev/null; then
    ok "torch already installed: $(python -c 'import torch; print(torch.__version__)')"
  else
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL}"
    ok "PyTorch installed."
  fi
}

install_python_deps() {
  log "Installing transformers / bitsandbytes / supporting deps..."
  pip install \
    "transformers==${TRANSFORMERS_VERSION}" \
    "tokenizers" "accelerate" "bitsandbytes" \
    "timm" "einops" "sentencepiece" "protobuf" \
    "numpy<2" "matplotlib" "pandas" "pytest"
  ok "Core Python deps installed."

  if [[ "$SKIP_FLASH" -eq 1 ]]; then
    warn "Skipping flash-attn (--skip-flash)."
  else
    log "Installing flash-attn ${FLASH_ATTN_VERSION} (this can take several minutes to build)..."
    pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation \
      || warn "flash-attn install failed — re-run with the full CUDA toolkit present, or use --skip-flash."
  fi
}

clone_and_install() {  # $1 = repo url, $2 = dir name
  local url="$1" name="$2" dest="$SRC_DIR/$2"
  if [[ -d "$dest/.git" ]]; then
    ok "$name already cloned at $dest."
  else
    log "Cloning $name ..."
    git clone "$url" "$dest"
  fi
  log "pip install -e $name ..."
  pip install -e "$dest"
  ok "$name installed (editable)."
}

install_openvla_libero() {
  mkdir -p "$SRC_DIR"
  clone_and_install "https://github.com/openvla/openvla.git" "openvla"
  clone_and_install "https://github.com/Lifelong-Robot-Learning/LIBERO.git" "LIBERO"
}

# ------------------------------------------------------------------------- smoke test
gpu_smoke_test() {
  log "Running GPU smoke test..."
  python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available to PyTorch!"
dev = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"  torch            : {torch.__version__}")
print(f"  CUDA (torch)     : {torch.version.cuda}")
print(f"  device           : {dev}  (sm_{cap[0]}{cap[1]})")
# Actually exercise the GPU: a matmul + an event-timed kernel.
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize(); start.record()
c = a @ b
end.record(); torch.cuda.synchronize()
assert c.shape == (4096, 4096)
print(f"  4096^3 fp16 matmul: {start.elapsed_time(end):.2f} ms")
print(f"  peak mem allocated : {torch.cuda.max_memory_allocated()/1e6:.1f} MB")
try:
    import bitsandbytes as bnb  # noqa: F401
    print(f"  bitsandbytes     : {bnb.__version__}")
except Exception as e:
    print(f"  bitsandbytes     : NOT importable ({e})")
print("GPU smoke test passed.")
PY
  ok "GPU smoke test passed."
}

# ------------------------------------------------------------------------------- main
main() {
  log "OpenVLA profiling environment setup — starting."
  check_gpu_driver
  check_cuda_toolkit
  check_nsight

  if [[ "$VERIFY_ONLY" -eq 1 ]]; then
    if [[ -d "$VENV_DIR" ]]; then
      # shellcheck disable=SC1091
      source "$VENV_DIR/bin/activate"
      gpu_smoke_test
    else
      warn "No venv yet — run without --verify-only to install."
    fi
    ok "Verify-only run complete."
    return
  fi

  setup_venv
  install_pytorch
  install_python_deps
  install_openvla_libero
  gpu_smoke_test

  cat <<EOF

$(ok "Setup complete.")
  Activate the environment with:  source "$VENV_DIR/bin/activate"
  Pre-cache the checkpoint on first use: openvla/openvla-7b-finetuned-libero-object
  Remember to STOP the EC2 instance when you're not profiling.
EOF
}

main "$@"
