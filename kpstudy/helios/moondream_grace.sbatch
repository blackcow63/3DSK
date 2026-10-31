#!/bin/bash
# Moondream 2 on 4 Grace cores (bf16 PyTorch; no quantized runtime keeps its pointing head), 3 photos x 23
# furniture questions, between two LiteRT BlazePose calibrations (phone factor, as cpu_bench.sbatch).
#   sbatch ciop/kpstudy/helios/moondream_grace.sbatch        (from $KP3D)
#SBATCH -J kp-md-grace
#SBATCH -A plgciopbhp-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o kp-md-grace-%j.out
set -euo pipefail
source ciop/kpstudy/helios/env.sh
cd "$KPSTUDY"
CORES=$(taskset -pc $$ | awk -F': ' '{print $2}' | tr ',' '\n' | head -1 | cut -d- -f1)
PIN="$CORES-$((CORES + 3))"
echo "host $(hostname); pinning to cores $PIN"
set +u; source /etc/profile 2>/dev/null || true; module load GCCcore/14.3.0; set -u
export LD_LIBRARY_PATH="${EBROOTGCCCORE}/lib64:${LD_LIBRARY_PATH:-}"
"$KP_VENVS/cpubench/bin/python" -c "pass" && taskset -c "$PIN" "$KP_VENVS/cpubench/bin/python" feasibility/bench_cpu.py --calib
taskset -c "$PIN" "$KP_VENVS/kp/bin/python" methods/moondream_point.py --name moondream2_grace_t4 --device cpu --threads 4 \
    --anchors furniture --views ring --limit 3
taskset -c "$PIN" "$KP_VENVS/cpubench/bin/python" feasibility/bench_cpu.py --calib --calib-key calibration_end
cat preds/stage1/moondream2_grace_t4/_timing.json
