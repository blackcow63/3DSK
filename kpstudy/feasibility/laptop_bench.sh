#!/usr/bin/env bash
# Phone-proxy timing on the laptop CPU (AMD Ryzen 5 5500U), all in one sitting so that the calibration and the
# model timings share the same power / thermal state: calibration -> NLF -> MoGe-2 (ONNX) -> Qwen3-VL-2B @640 px
# on 4 threads (3 photos) -> calibration again. Run with nothing else busy on the machine.
#   bash ciop/kpstudy/feasibility/laptop_bench.sh
set -euo pipefail
cd "$(dirname "$0")/.."                                   # ciop/kpstudy
PY=.venv/Scripts/python.exe
MP="$LOCALAPPDATA/ergobench-mp/venv/Lib/site-packages/mediapipe/modules/pose_landmark"
export PYTHONIOENCODING=utf-8 KP_NUM_TOKENS=1800
$PY feasibility/bench_cpu.py --calib --mp-dir "$MP" --calib-key calibration
$PY feasibility/bench_nlf.py models/nlf_s_multi_0.2.2.torchscript nlf_s 1 4
$PY feasibility/bench_nlf.py models/nlf_l_multi_0.3.2.torchscript nlf_l 1 4
$PY feasibility/bench_cpu.py --onnx moge2_vits=models/moge2_vits.onnx:1x3x518x686 \
    --onnx moge2_vitb=models/moge2_vitb.onnx:1x3x518x686 --onnx moge2_vitl=models/moge2_vitl.onnx:1x3x518x686
rm -rf preds/stage1/qwen3vl2b_q4_640_t4
py methods/vlm_point_llamacpp.py --views ring --limit 3 --max-side 640 --threads 4 --name qwen3vl2b_q4_640_t4 --anchors furniture
$PY feasibility/bench_cpu.py --calib --mp-dir "$MP" --calib-key calibration_end
echo laptop bench done
