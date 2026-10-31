#!/bin/bash
# One-time environment for the keypoint study on Helios. Must run on aarch64 (a GH200 node), because the
# wheels are architecture specific:  sbatch ciop/kpstudy/helios/setup.sbatch   (from $KP3D)
# If compute nodes have no internet, run the same lines in an interactive GPU shell instead:
#   srun -A plgciopbhp-gpu-gh200 -p plgrid-gpu-gh200 --gres=gpu:1 -c 16 -t 1:00:00 --pty bash
#SBATCH -J kp-setup
#SBATCH -A plgciopbhp-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o kp-setup-%j.out
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" 2>/dev/null || source ciop/kpstudy/helios/env.sh
uname -m; nvidia-smi -L || true
mkdir -p "$KP_VENVS" "$HF_HOME"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# torch-based methods (MoGe-2, NLF, Molmo2/MolmoPoint): CUDA aarch64 wheels from download.pytorch.org
uv venv --python 3.11 "$KP_VENVS/kp"
VIRTUAL_ENV="$KP_VENVS/kp" uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV="$KP_VENVS/kp" uv pip install "transformers==4.57.1" accelerate einops pillow opencv-python-headless scipy \
    huggingface_hub molmo_utils onnx onnxruntime
# MoGe without its viewer/GUI extras (moderngl/gradio are not needed for inference)
VIRTUAL_ENV="$KP_VENVS/kp" uv pip install --no-deps "git+https://github.com/microsoft/MoGe.git" \
    "git+https://github.com/EasternJournalist/utils3d-moge.git"
"$KP_VENVS/kp/bin/python" - <<'EOF'
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
from moge.model.v2 import MoGeModel; print("moge ok")
EOF

# phone-proxy CPU benchmark: llama.cpp prebuilt for Linux arm64 (plain armv8 + dotprod build, no SVE kernels
# chosen explicitly -- the binary picks its CPU backend at runtime; the job records which), ONNX Runtime, LiteRT
mkdir -p "$KP3D/tools" && cd "$KP3D/tools"
if [ ! -x llama_arm64/bin/llama-bench ] && [ ! -x llama_arm64/llama-bench ]; then
  curl -sfL -o llama_arm64.tgz https://github.com/ggml-org/llama.cpp/releases/download/b11756/llama-b11756-bin-ubuntu-arm64.tar.gz
  mkdir -p llama_arm64 && tar xzf llama_arm64.tgz -C llama_arm64 --strip-components=1 && rm llama_arm64.tgz
fi
ls llama_arm64 | head
uv venv --python 3.11 "$KP_VENVS/cpubench"
VIRTUAL_ENV="$KP_VENVS/cpubench" uv pip install numpy onnxruntime ai-edge-litert "mediapipe==0.10.14" pillow
echo setup done
