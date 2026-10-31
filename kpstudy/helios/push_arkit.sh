#!/usr/bin/env bash
# Laptop (Git Bash) -> Helios: kod badania i zdjęcia zbioru ARKitScenes w formacie kpstudy (data_arkitscenes.py).
# Idą tylko pliki, które czytają metody: views.json, prompts.json oraz rgb.jpg i cam.json (K) każdego zdjęcia.
# Prawda (gt.json, kp.json, lidar.png) zostaje na laptopie, gdzie liczy się ocena.
#   bash ciop/kpstudy/helios/push_arkit.sh kp_test|kp [user@host]
set -euo pipefail
ZB=${1:-kp}
HOST=${2:-plgzsmolen@helios.cyfronet.pl}
cd "$(dirname "$0")/../../.."                       # korzeń projektu (C:/Users/zofsm/3d)
REMOTE=$(ssh "$HOST" 'echo ${KP3D:-$SCRATCH/kp3d}')
echo "-> $HOST:$REMOTE ($ZB)"
ssh "$HOST" "mkdir -p $REMOTE/dane/arkitscenes"
tar czf - \
  --exclude='ciop/kpstudy/.venv' --exclude='ciop/kpstudy/preds' --exclude='ciop/kpstudy/results' \
  --exclude='ciop/kpstudy/tools/llama_win' --exclude='ciop/kpstudy/models' --exclude='__pycache__' \
  --exclude='ciop/kpstudy/annot/sceny' --exclude='ciop/kpstudy/annot/punkty' \
  ciop/kpstudy \
  dane/arkitscenes/$ZB/views.json dane/arkitscenes/$ZB/prompts.json \
  $(cd dane/arkitscenes/$ZB && find . -name rgb.jpg -o -name cam.json | sed "s#^\./#dane/arkitscenes/$ZB/#") \
  | ssh "$HOST" "tar xzf - -C $REMOTE"
echo done
