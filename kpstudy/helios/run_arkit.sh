#!/usr/bin/env bash
# Wysyła na Heliosie wszystkie przebiegi metod na zbiorze ARKitScenes (po push_arkit.sh). Uruchamiać z $KP3D:
#   bash ciop/kpstudy/helios/run_arkit.sh kp            # albo kp_test
# Każda metoda to osobne zadanie; wyniki w dane/arkitscenes/<zbior>/preds, pobiera je pull_arkit.sh.
set -euo pipefail
source ciop/kpstudy/helios/env.sh     # bez tego $KP3D jest pusty i ścieżka zbioru wychodzi z /
ZB=${1:-kp}
R=$KP3D/dane/arkitscenes/$ZB
[ -s "$R/views.json" ] || { echo "brak $R/views.json"; exit 1; }
J=ciop/kpstudy/helios/gpu_job.sbatch
J2=ciop/kpstudy/helios/gpu_job2.sbatch
s() { echo "$*"; sbatch "$@"; }
# jedno przejście: tekst -> punkt 3D (u, v, głębia) i ten sam model jako wskaźnik 2D
s -J kp-rb25-4b  -t 06:00:00 $J python methods/robobrain_point.py --model BAAI/RoboBrain2.5-4B --name robobrain25_4b --root $R
s -J kp-rb25-8b  -t 08:00:00 $J python methods/robobrain_point.py --model BAAI/RoboBrain2.5-8B-NV --name robobrain25_8b --root $R
# wskaźniki 2D
s -J kp-qwen8b   -t 06:00:00 $J python methods/qwen3vl_point.py --model Qwen/Qwen3-VL-8B-Instruct --name qwen3vl8b --root $R
s -J kp-qwen32b  -t 10:00:00 $J python methods/qwen3vl_point.py --model Qwen/Qwen3-VL-32B-Instruct --name qwen3vl32b --root $R
s -J kp-molmo2   -t 06:00:00 $J python methods/molmo_point.py --model allenai/Molmo2-8B --name molmo2_8b --root $R
s -J kp-molmopt  -t 06:00:00 $J python methods/molmo_point.py --model allenai/MolmoPoint-8B --name molmopoint8b --root $R
# kwantyzacja 4-bit (ta sama ocena, mniejsza pamięć)
s -J kp-rb25-q4  -t 06:00:00 $J python methods/robobrain_point.py --model BAAI/RoboBrain2.5-4B --name robobrain25_4b_q4 --quant 4bit --root $R
s -J kp-qwen8q4  -t 06:00:00 $J python methods/qwen3vl_point.py --model Qwen/Qwen3-VL-8B-Instruct --name qwen3vl8b_q4 --quant 4bit --root $R
# głębia z RGB
s -J kp-moge-l   $J  python methods/moge_depth.py vitl --root $R
s -J kp-unidepth $J2 python methods/depth_models.py unidepthv2_l --root $R
s -J kp-da3      $J2 python methods/depth_models.py da3_nested --root $R
s -J kp-mapany   $J2 python methods/depth_models.py mapanything_scene --root $R
