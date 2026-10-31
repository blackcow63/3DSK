#!/usr/bin/env bash
# Pozostałe modele na wspólnej próbce: co dziewiąte zdjęcie typu frame (33 zdjęcia ze wszystkich źródeł z pomiarem), bez 3DPW.
# Pełny zbiór nie zdążyłby przed terminem. Uruchamiać z $KP3D: bash ciop/kpstudy/helios/run_e3.sh
set -euo pipefail
source ciop/kpstudy/helios/env.sh
R=$KP3D/dane/kp_razem
J=ciop/kpstudy/helios/gpu_job.sbatch
J3=ciop/kpstudy/helios/gpu_job3.sbatch
O=--export=ALL,HF_HUB_OFFLINE=1
P="python methods/robobrain_point.py --root $R --mode 3d --views frame --co 9"
s() { echo "$1 $(sbatch --parsable -J "$1" -t "$2" $O "${@:3}")"; }
s e-qwen  01:30:00 $J  python methods/robobrain_point.py --root $R --mode 3d --views frame --co 3 --model Qwen/Qwen3-VL-8B-Instruct --name qwen3vl8b_razem --styl razem
s e-cos   00:40:00 $J  $P --model nvidia/Cosmos-Reason2-8B --name cosmos_r2_8b_razem --styl razem
s e-emb   00:45:00 $J  $P --model IffYuan/Embodied-R1.5 --name embodied_r15_pkt --styl r15
s e-ry2   00:30:00 $J3 kp3 $P --model Alibaba-DAMO-Academy/RynnBrain1.1-2B --name rynnbrain11_2b --styl rynn
s e-ry9   01:30:00 $J3 kp3 $P --model Alibaba-DAMO-Academy/RynnBrain1.1-9B --name rynnbrain11_9b --styl rynn
s e-mimo0 02:30:00 $J  $P --czesc 0/2 --model XiaomiMiMo/MiMo-Embodied-7B --name mimo7b_razem --styl razem --tokeny 2048
s e-mimo1 02:30:00 $J  $P --czesc 1/2 --model XiaomiMiMo/MiMo-Embodied-7B --name mimo7b_razem --styl razem --tokeny 2048
