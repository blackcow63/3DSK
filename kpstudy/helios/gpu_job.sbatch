#!/bin/bash
# Run one method of the keypoint study on one GH200 GPU, inside the kp venv, from ciop/kpstudy.
# Submit from $KP3D (the mirrored project root), e.g.:
#   sbatch -J kp-moge-l   ciop/kpstudy/helios/gpu_job.sbatch python methods/moge_depth.py vitl --all
#   sbatch -J kp-nlf-l    ciop/kpstudy/helios/gpu_job.sbatch python methods/nlf_e2e.py models/nlf_l_multi_0.3.2.torchscript nlf_l
#   sbatch -J kp-nlf-l-gt ciop/kpstudy/helios/gpu_job.sbatch python methods/nlf_e2e.py models/nlf_l_multi_0.3.2.torchscript nlf_l --oracle-box
#   sbatch -J kp-molmo    -t 06:00:00 ciop/kpstudy/helios/gpu_job.sbatch python methods/molmo_point.py --model allenai/MolmoPoint-8B --name molmopoint8b
# Wall time defaults to 2 h; override with sbatch -t.
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
source "$KP_VENVS/kp/bin/activate"
echo "job $SLURM_JOB_ID on $(hostname) $(uname -m)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p models
for f in nlf_s_multi_0.2.2.torchscript nlf_l_multi_0.3.2.torchscript; do      # NLF TorchScript releases
  case "$*" in *"$f"*) [ -s "models/$f" ] || curl -sfL -o "models/$f" \
      "https://github.com/isarandi/nlf/releases/download/v$(echo $f | sed -E 's/.*_([0-9.]+)\.torchscript/\1/')/$f";; esac
done
/usr/bin/time -v "$@" 2> >(tee -a "kp-$SLURM_JOB_ID.time" >&2)
