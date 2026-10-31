#!/bin/bash
# Phone proxy on the Grace CPU of a GH200 node: BlazePose calibration (vs published Pixel 4 times), MoGe-2 ONNX
# (official exports, fp32 + dynamic int8), Qwen3-VL-2B Q4 in llama.cpp (the same GGUF as on the laptop).
# Pinned to 4 cores (a phone has ~4 big/performance cores). The GPU is allocated only because the grant is GPU.
#   sbatch ciop/kpstudy/helios/cpu_bench.sbatch        (from $KP3D)
#SBATCH -J kp-cpubench
#SBATCH -A plgciopbhp-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o kp-cpubench-%j.out
set -euo pipefail
source ciop/kpstudy/helios/env.sh
cd "$KPSTUDY"
CORES=$(taskset -pc $$ | awk -F': ' '{print $2}' | tr ',' '\n' | head -1 | cut -d- -f1)
PIN="$CORES-$((CORES + 3))"
echo "host $(hostname) $(uname -m); pinning to cores $PIN"; lscpu | grep -E "Model name|Flags" | cut -c1-200

# 1) calibration + ONNX models (official MoGe-2 ONNX exports, fetched beforehand on the login node -- some
#    compute nodes have no internet):
#    cd models && curl -sfL -o moge2_vits.onnx https://huggingface.co/Ruicheng/moge-2-vits-normal-onnx/resolve/main/model.onnx
# newer libstdc++ for the LiteRT wheel (needs GLIBCXX_3.4.30); batch shells do not define `module` by default
set +u; source /etc/profile 2>/dev/null || true; module load GCCcore/14.3.0; set -u   # /etc/profile uses unset variables
export LD_LIBRARY_PATH="${EBROOTGCCCORE}/lib64:${LD_LIBRARY_PATH:-}"
[ "$(grep -c GLIBCXX_3.4.30 "${EBROOTGCCCORE}/lib64/libstdc++.so.6")" -gt 0 ] || { echo "no GLIBCXX_3.4.30 in $EBROOTGCCCORE"; exit 1; }
source "$KP_VENVS/cpubench/bin/activate"
for s in vits vitb vitl; do test -s models/moge2_${s}.onnx || { echo "missing models/moge2_${s}.onnx"; exit 1; }; done
# 1440x1080 photo resized as MoGe does for ~1800 tokens (patch 14): 37 x 49 patches -> 518 x 686
KP_NUM_TOKENS=1800 taskset -c "$PIN" python feasibility/bench_cpu.py --calib \
    --onnx moge2_vits=models/moge2_vits.onnx:1x3x518x686 --onnx moge2_vitb=models/moge2_vitb.onnx:1x3x518x686 \
    --onnx moge2_vitl=models/moge2_vitl.onnx:1x3x518x686

# 2) NLF crop model (backbone + localizer at 17 points, one 256 px crop) in the kp venv's PyTorch, 1 and 4 threads
for m in "nlf_s_multi_0.2.2.torchscript nlf_s" "nlf_l_multi_0.3.2.torchscript nlf_l"; do
  set -- $m
  taskset -c "$PIN" "$KP_VENVS/kp/bin/python" feasibility/bench_nlf.py "models/$1" "$2" 1 4
done

# 3) Qwen3-VL-2B Q4_K_M + Q8_0 projector in llama.cpp (Linux arm64 build), 3 photos x 40 questions:
#    full resolution and the phone setting (640 px), 4 threads; 640 px also on 1 thread
LS=$(ls "$KP3D"/tools/llama_arm64/llama-server "$KP3D"/tools/llama_arm64/bin/llama-server 2>/dev/null | head -1)
export LD_LIBRARY_PATH="$(dirname "$LS"):${LD_LIBRARY_PATH:-}"
for cfg in "4 0" "4 640" "1 640"; do
  set -- $cfg
  taskset -c "$PIN" python methods/vlm_point_llamacpp.py --server "$LS" --threads $1 --max-side $2 --views ring --limit 3 \
      --name qwen3vl2b_q4_grace_t$1_s$2 --port $((8600 + $1 + $2))
  grep -iE "image (slice )?encoded|encod.*ms|CPU :|system_info" preds/stage1/qwen3vl2b_q4_grace_t$1_s$2/_server.log | head -20 || true
done
taskset -c "$PIN" python feasibility/bench_cpu.py --calib --calib-key calibration_end   # drift check
mkdir -p bench && for d in preds/stage1/qwen3vl2b_q4_grace_t*; do cp "$d/_server.log" "bench/$(basename $d)_server.log"; done || true
echo done
