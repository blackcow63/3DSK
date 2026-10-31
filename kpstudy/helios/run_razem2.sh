#!/usr/bin/env bash
# Druga fala modeli na wspólnym zbiorze (po run_razem.sh i setup3.sbatch). Uruchamiać z $KP3D:
#   bash ciop/kpstudy/helios/run_razem2.sh [kp|kp3|kp4]      # bez argumentu kp, potem pozostałe, gdy ich venv stoi
set -euo pipefail
source ciop/kpstudy/helios/env.sh
R=$KP3D/dane/kp_razem
J=ciop/kpstudy/helios/gpu_job.sbatch
J3=ciop/kpstudy/helios/gpu_job3.sbatch
W="--root $R --views frame,crop"
CO=${1:-kp}
s() { echo "$*"; sbatch "$@"; }
if [ "$CO" = kp ]; then
  # wskaźniki 2D z tych samych wrapperów
  s -J rz-molmo-er -t 08:00:00 $J python methods/molmo_point.py --model allenai/Molmo2-ER --name molmo2_er $W
  s -J rz-molmo4b  -t 08:00:00 $J python methods/molmo_point.py --model allenai/Molmo2-4B --name molmo2_4b $W
  s -J rz-qwen4b   -t 08:00:00 $J python methods/qwen3vl_point.py --model Qwen/Qwen3-VL-4B-Instruct --name qwen3vl4b $W
  s -J rz-qwen2b   -t 08:00:00 $J python methods/qwen3vl_point.py --model Qwen/Qwen3-VL-2B-Instruct --name qwen3vl2b $W
  s -J rz-emb15    -t 12:00:00 $J python methods/qwen3vl_point.py --model IffYuan/Embodied-R1.5 --name embodied_r15 --styl r15 $W
  s -J rz-emb1     -t 12:00:00 $J python methods/qwen3vl_point.py --model IffYuan/Embodied-R1-3B-v1 --name embodied_r1_3b --styl r1 $W
  s -J rz-moon2    -t 04:00:00 $J python methods/moondream_point.py --name moondream2 --root $R --views frame,crop
  # zdjęcie wzorcowe zamiast tekstu
  for m in small:dinov2s base:dinov2b large:dinov2l; do
    s -J rz-${m#*:}-ref -t 02:00:00 $J python methods/ref_match.py --model facebook/dinov2-with-registers-${m%:*} --name ${m#*:}_ref $W
  done
fi
if [ "$CO" = kp3 ]; then
  s -J rz-ov2 -t 12:00:00 $J3 kp3 python methods/qwen3vl_point.py --model lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct --name onevision2_8b --trust-remote-code $W
fi
if [ "$CO" = kp4 ]; then
  s -J rz-rr2b -t 08:00:00 $J3 kp4 python methods/roborefer_point.py --model Zhoues/RoboRefer-2B-SFT --name roborefer2b $W
  s -J rz-rr8b -t 12:00:00 $J3 kp4 python methods/roborefer_point.py --model Zhoues/RoboRefer-8B-SFT --name roborefer8b $W
fi
