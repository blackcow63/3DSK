#!/bin/bash
# Jak gpu_job.sbatch, ale w venv podanym pierwszym argumentem (kp3: LLaVA-OneVision-2, kp4: RoboRefer).
#   sbatch -J rz-ov2 ciop/kpstudy/helios/gpu_job3.sbatch kp3 python methods/qwen3vl_point.py ...
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
V=$1; shift
source "$KP_VENVS/$V/bin/activate"
echo "job $SLURM_JOB_ID on $(hostname) $(uname -m)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/usr/bin/time -v "$@" 2> >(tee -a "kp-$SLURM_JOB_ID.time" >&2)
