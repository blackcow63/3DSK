#!/usr/bin/env bash
# Wysyła na Heliosie wszystkie przebiegi metod na wspólnym zbiorze (po push_razem.sh). Uruchamiać z $KP3D:
#   bash ciop/kpstudy/helios/run_razem.sh [wolne|gated]
# Każda metoda to osobne zadanie; wyniki w dane/kp_razem/preds, pobiera je pull_razem.sh. Wrappery pomijają gotowe
# widoki, więc przerwane zadanie wystarczy wysłać ponownie. Modele gated muszą już być w $HF_HOME: zadania nie mają tokenu.
set -euo pipefail
source ciop/kpstudy/helios/env.sh     # bez tego $KP3D jest pusty i ścieżka zbioru wychodzi z /
R=$KP3D/dane/kp_razem
[ -s "$R/views.json" ] || { echo "brak $R/views.json"; exit 1; }
J=ciop/kpstudy/helios/gpu_job.sbatch
J2=ciop/kpstudy/helios/gpu_job2.sbatch
W="--root $R --views frame,crop"       # wskaźniki i modele jednoprzejściowe
G="--root $R --views frame,pelny"      # modele głębi: pełne zdjęcia
CO=${1:-wszystkie}                     # wolne = bez modeli gated, gated = tylko one
s() { echo "$*"; sbatch "$@"; }
w() { [ "$CO" = gated ] || s "$@"; }
g() { [ "$CO" = wolne ] || s "$@"; }
# jedno przejście: tekst -> punkt 3D (u, v, głębia) i ten sam model jako wskaźnik 2D
w -J rz-rb25-4b  -t 08:00:00 $J python methods/robobrain_point.py --model BAAI/RoboBrain2.5-4B --name robobrain25_4b $W
g -J rz-rb25-8b  -t 12:00:00 --export=ALL,HF_HUB_OFFLINE=1 $J python methods/robobrain_point.py --model BAAI/RoboBrain2.5-8B-NV --name robobrain25_8b $W
g -J rz-rb25-8mt -t 12:00:00 --export=ALL,HF_HUB_OFFLINE=1 $J python methods/robobrain_point.py --model BAAI/RoboBrain2.5-8B-MT --name robobrain25_8b_mt $W
# wskaźniki 2D
w -J rz-qwen8b   -t 08:00:00 $J python methods/qwen3vl_point.py --model Qwen/Qwen3-VL-8B-Instruct --name qwen3vl8b $W
w -J rz-qwen32b  -t 12:00:00 $J python methods/qwen3vl_point.py --model Qwen/Qwen3-VL-32B-Instruct --name qwen3vl32b $W
w -J rz-molmo2   -t 08:00:00 $J python methods/molmo_point.py --model allenai/Molmo2-8B --name molmo2_8b $W
w -J rz-molmopt  -t 08:00:00 $J python methods/molmo_point.py --model allenai/MolmoPoint-8B --name molmopoint8b $W
# DepthLM: głębia zadanego piksela; najpierw prawdziwe piksele, wskaźniki 2D po ich wynikach (methods/depthlm_lift.py --stage1 <nazwa>)
g -J rz-dlm-or   -t 08:00:00 --export=ALL,HF_HUB_OFFLINE=1 $J python methods/depthlm_lift.py --stage1 oracle2d $W
# kwantyzacja 4-bit (ta sama ocena, mniejsza pamięć)
w -J rz-rb4-q4   -t 08:00:00 $J python methods/robobrain_point.py --model BAAI/RoboBrain2.5-4B --name robobrain25_4b_q4 --quant 4bit $W
g -J rz-rb8-q4   -t 12:00:00 --export=ALL,HF_HUB_OFFLINE=1 $J python methods/robobrain_point.py --model BAAI/RoboBrain2.5-8B-NV --name robobrain25_8b_q4 --quant 4bit $W
w -J rz-qwen8q4  -t 08:00:00 $J python methods/qwen3vl_point.py --model Qwen/Qwen3-VL-8B-Instruct --name qwen3vl8b_q4 --quant 4bit $W
# głębia z RGB
w -J rz-moge-l   $J  python methods/moge_depth.py vitl $G
w -J rz-unidepth $J2 python methods/depth_models.py unidepthv2_l $G
w -J rz-da3      $J2 python methods/depth_models.py da3_nested $G
w -J rz-mapany   $J2 python methods/depth_models.py mapanything_scene $G
