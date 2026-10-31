#!/bin/bash
# Jak gpu_job.sbatch, ale w venv kp2 (modele głębi: UniDepthV2, Depth Anything 3, MapAnything).
#   sbatch -J kp-unidepth ciop/kpstudy/helios/gpu_job2.sbatch python methods/depth_models.py unidepthv2_l --root $KP3D/dane/arkitscenes/kp
#SBATCH -A plgciopbhp-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 02:00:00
#SBATCH -o kp-%x-%j.out
set -euo pipefail
source ciop/kpstudy/helios/env.sh
cd "$KPSTUDY"
source "$KP_VENVS/kp2/bin/activate"
echo "job $SLURM_JOB_ID on $(hostname) $(uname -m)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/usr/bin/time -v "$@" 2> >(tee -a "kp-$SLURM_JOB_ID.time" >&2)
