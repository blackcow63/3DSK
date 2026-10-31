#!/bin/bash
# TAIHRI (Qwen3-VL-4B fine-tune) end-to-end on the kawalerka photos, in the kp venv (transformers backend, sdpa).
# Clones the repo and downloads the open weights (HF RammusAoLi/TAIHRI, Apache-2.0) on first run.
#   sbatch ciop/kpstudy/helios/taihri.sbatch        (from $KP3D)
#SBATCH -J kp-taihri
#SBATCH -A plgciopbhp-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 02:00:00
#SBATCH -o kp-taihri-%j.out
set -euo pipefail
source ciop/kpstudy/helios/env.sh
source "$KP_VENVS/kp/bin/activate"
VIRTUAL_ENV="$KP_VENVS/kp" uv pip install -q "qwen_vl_utils==0.0.14"
mkdir -p "$KP3D/ext" && cd "$KP3D/ext"
[ -d TAIHRI ] || git clone -q https://github.com/Tencent/TAIHRI.git
cd TAIHRI
# upstream bug (20.10.2026): eval_wrapper modules annotate with typing names they never import (Optional, Any);
# postponed evaluation of annotations makes them harmless
for f in eval_wrapper/*.py; do
  grep -q "^from __future__ import annotations" "$f" || sed -i '1i from __future__ import annotations' "$f"
done
[ -s checkpoints/taihri/config.json ] || hf download RammusAoLi/TAIHRI --local-dir checkpoints/taihri
nvidia-smi --query-gpu=name --format=csv,noheader
python "$KPSTUDY/methods/taihri_e2e.py" --model checkpoints/taihri --name taihri4b
