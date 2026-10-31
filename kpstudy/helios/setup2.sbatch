#!/bin/bash
# Druga tura środowisk (29.10.2026, przebudowa pracy): modele głębi z jednego zdjęcia i z wielu zdjęć
# (UniDepthV2, Depth Anything 3, MapAnything) w osobnym venv kp2, żeby ich zależności nie ruszyły kp
# (transformers 4.57.1 pod Molmo/Qwen3-VL/RoboBrain). Do kp dochodzi tylko bitsandbytes (kwantyzacja 4/8 bit).
#   sbatch ciop/kpstudy/helios/setup2.sbatch     (z $KP3D)
#SBATCH -J kp-setup2
#SBATCH -A plgciopbhp-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH -o kp-setup2-%j.out
set -uo pipefail
source ciop/kpstudy/helios/env.sh
uname -m; nvidia-smi -L || true
"$KP_VENVS/kp/bin/python" -c "import bitsandbytes as b; print('bitsandbytes', b.__version__)"

[ -x "$KP_VENVS/kp2/bin/python" ] || uv venv --python 3.11 "$KP_VENVS/kp2"
export VIRTUAL_ENV="$KP_VENVS/kp2"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install numpy pillow opencv-python-headless scipy huggingface_hub einops timm safetensors
mkdir -p "$KP3D/ext" && cd "$KP3D/ext"
[ -d UniDepth ] || git clone --depth 1 https://github.com/lpiccinelli-eth/UniDepth.git
[ -d map-anything ] || git clone --depth 1 https://github.com/facebookresearch/map-anything.git
[ -d Depth-Anything-3 ] || git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git
uv pip install --no-deps -e UniDepth && uv pip install wandb   # bez xformers: nie ma koła na aarch64, kompilacja trwa godzinami
uv pip install -e map-anything || echo "MAPANYTHING INSTALL FAILED"
uv pip install -e Depth-Anything-3 || echo "DA3 INSTALL FAILED"
"$KP_VENVS/kp2/bin/python" - <<'PY'
import torch; print("torch", torch.__version__, torch.cuda.is_available())
for m in ("unidepth.models", "mapanything.models", "depth_anything_3.api"):
    try:
        __import__(m); print(m, "ok")
    except Exception as e:
        print(m, "BLAD", type(e).__name__, str(e)[:200])
PY
echo setup2 done
# Depth Anything 3: --no-deps, bo zależy od pycolmap bez koła na aarch64; pycolmap jest potrzebny tylko do
# eksportu do COLMAP, a moduł eksportu importuje go przy starcie, więc wystarcza pusty moduł zastępczy.
VIRTUAL_ENV="$KP_VENVS/kp2" uv pip install --no-deps -e "$KP3D/ext/Depth-Anything-3" && \
  VIRTUAL_ENV="$KP_VENVS/kp2" uv pip install moviepy==1.0.3 plyfile pillow_heif imageio evo trimesh omegaconf e3nn addict
SP=$("$KP_VENVS/kp2/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
[ -f "$SP/pycolmap.py" ] || echo '"""Pusty moduł: pycolmap nie ma koła na aarch64, a DA3 używa go tylko do eksportu."""' > "$SP/pycolmap.py"
