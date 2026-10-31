#!/bin/bash
# Trzecia tura środowisk (30.10.2026, wszystkie modele na wspólnym zbiorze), osobno, żeby nie ruszyć kp:
#   kp3  transformers >= 5.7 pod LLaVA-OneVision-2
#   kp4  RoboRefer: pakiet llava z klonu ext/RoboRefer. Ich pyproject przypina pakiety nvidia-* tylko na x86 i torch 2.5.1,
#        a GH200 to aarch64, więc llava idzie bez zależności, a potrzebne biblioteki osobno. Może się nie udać: patrz log.
#   sbatch ciop/kpstudy/helios/setup3.sbatch     (z $KP3D)
#SBATCH -J kp-setup3
#SBATCH -A plgciopbhp-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=60G
#SBATCH -t 01:30:00
#SBATCH -o kp-setup3-%j.out
set -uo pipefail
source ciop/kpstudy/helios/env.sh
uname -m
[ -x "$KP_VENVS/kp3/bin/python" ] || uv venv --python 3.11 "$KP_VENVS/kp3"
export VIRTUAL_ENV="$KP_VENVS/kp3"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install "transformers>=5.7.0" accelerate einops pillow opencv-python-headless scipy decord bitsandbytes || \
  uv pip install "transformers>=5.7.0" accelerate einops pillow opencv-python-headless scipy bitsandbytes
"$VIRTUAL_ENV/bin/python" -c "import transformers, torch; print('kp3', transformers.__version__, torch.__version__, torch.cuda.is_available())"

mkdir -p "$KP3D/ext"
[ -d "$KP3D/ext/RoboRefer" ] || git clone --depth 1 https://github.com/Zhoues/RoboRefer "$KP3D/ext/RoboRefer"
[ -x "$KP_VENVS/kp4/bin/python" ] || uv venv --python 3.10 "$KP_VENVS/kp4"
export VIRTUAL_ENV="$KP_VENVS/kp4"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
grep -n -E '"(transformers|tokenizers|huggingface|accelerate|peft|numpy|timm|einops|sentencepiece)[^"]*"' "$KP3D/ext/RoboRefer/pyproject.toml"
uv pip install "transformers==4.46.0" "accelerate==1.3.0" "peft>=0.9.0" "numpy<2" timm einops einops-exts pillow opencv-python-headless \
  sentencepiece protobuf loguru termcolor fire tyro hydra-core pytorchvideo huggingface_hub safetensors scipy shortuuid flask \
  "git+https://github.com/bfshi/scaling_on_scales" pydantic opencv-python-headless wandb matplotlib
uv pip install --no-deps -e "$KP3D/ext/RoboRefer"
P=$("$VIRTUAL_ENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
[ -d "$KP3D/ext/RoboRefer/llava/train/transformers_replace" ] && cp -rv "$KP3D/ext/RoboRefer/llava/train/transformers_replace/"* "$P/transformers/" | tail -2
"$VIRTUAL_ENV/bin/python" -c "import llava; print('kp4 llava', llava.__file__)"
echo koniec
