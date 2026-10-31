#!/bin/bash
# SAM 3D Body (DINOv3-H+) end to end on the kawalerka photos, with the person's box from the label image
# (the built-in ViTDet detector needs detectron2, which has no aarch64 wheel here). Weights are downloaded
# beforehand into ext/sam-3d-body/checkpoints/sam-3d-body-dinov3 (gated; the token is never stored on Helios).
#   sbatch ciop/kpstudy/helios/sam3d_body.sbatch        (from $KP3D)
#SBATCH -J kp-sam3db
#SBATCH -A plgciopbhp-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o kp-sam3db-%j.out
set -euo pipefail
source ciop/kpstudy/helios/env.sh
source "$KP_VENVS/kp/bin/activate"
VIRTUAL_ENV="$KP_VENVS/kp" uv pip install -q pytorch-lightning opencv-python-headless yacs scikit-image einops timm dill pandas rich \
    hydra-core pyrootutils webdataset networkx roma joblib loguru optree fvcore pycocotools huggingface_hub braceexpand \
    omegaconf jsonlines appdirs 2>&1 | tail -2
cd "$KP3D/ext/sam-3d-body"
python "$KPSTUDY/methods/sam3d_body_e2e.py" --repo facebook/sam-3d-body-dinov3 --name sam3db_dinov3 --oracle-box
