#!/usr/bin/env bash
# Tylko modele, które w jednym przejściu zwracają punkt 3D (decyzja 30.10). Uruchamiać z $KP3D:
#   bash ciop/kpstudy/helios/run_jedno.sh
# Wyniki w dane/kp_razem/preds/e2e, pobiera je pull_razem.sh. Wrappery pomijają gotowe widoki. Wskaźniki 2D z głębią
# (run_razem.sh, run_razem2.sh) to potok, nie jedno przejście: ich częściowe wyniki zostają, a skrypty można wznowić.
# Embodied-R1.5 odpada (30.10): pojedynczy punkt zwraca w 2D, a 3D ma tylko w trasach z głębią startu podaną w pytaniu.
set -euo pipefail
source ciop/kpstudy/helios/env.sh
R=$KP3D/dane/kp_razem
J=ciop/kpstudy/helios/gpu_job.sbatch
W="--root $R --views frame,crop --mode 3d"
O=--export=ALL,HF_HUB_OFFLINE=1
s() { echo "$*"; sbatch "$@"; }
s -J j-rb25-4b  -t 08:00:00 $J python methods/robobrain_point.py --model BAAI/RoboBrain2.5-4B --name robobrain25_4b $W
s -J j-rb25-8b  -t 12:00:00 $O $J python methods/robobrain_point.py --model BAAI/RoboBrain2.5-8B-NV --name robobrain25_8b $W
s -J j-rb25-8mt -t 12:00:00 $O $J python methods/robobrain_point.py --model BAAI/RoboBrain2.5-8B-MT --name robobrain25_8b_mt $W
s -J j-rb4-q4   -t 08:00:00 $J python methods/robobrain_point.py --model BAAI/RoboBrain2.5-4B --name robobrain25_4b_q4 --quant 4bit $W
s -J j-rb8-q4   -t 12:00:00 $O $J python methods/robobrain_point.py --model BAAI/RoboBrain2.5-8B-NV --name robobrain25_8b_q4 --quant 4bit $W
s -J j-qwen8    -t 08:00:00 $J python methods/robobrain_point.py --model Qwen/Qwen3-VL-8B-Instruct --name qwen3vl8b_3d $W
s -J j-qwen32   -t 12:00:00 $J python methods/robobrain_point.py --model Qwen/Qwen3-VL-32B-Instruct --name qwen3vl32b_3d $W
s -J j-qwen8b   -t 08:00:00 $J python methods/robobrain_point.py --model Qwen/Qwen3-VL-8B-Instruct --name qwen3vl8b_box --styl bbox3d $W
s -J j-qwen32b  -t 12:00:00 $J python methods/robobrain_point.py --model Qwen/Qwen3-VL-32B-Instruct --name qwen3vl32b_box --styl bbox3d $W
# Typ B, piksel i pudełko w jednej odpowiedzi. Zadania krótkie, drugie rusza po pierwszym i pomija gotowe widoki.
s -J j-razem1 -t 01:30:00 $J python methods/robobrain_point.py --model Qwen/Qwen3-VL-8B-Instruct --name qwen3vl8b_razem --styl razem $W
