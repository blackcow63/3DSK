#!/usr/bin/env bash
# Laptop (Git Bash) -> Helios: kod badania i wspólny zbiór testowy (data_razem.py).
# Idą tylko pliki, które czytają metody: views.json, prompts.json, frazy.json oraz rgb.jpg i cam.json każdego widoku.
# Prawda (gt.json, kp.json, lidar.png) zostaje na laptopie, gdzie liczy się ocena.
#   bash ciop/kpstudy/helios/push_razem.sh [user@host]
set -euo pipefail
HOST=${1:-plgzsmolen@helios.cyfronet.pl}
cd "$(dirname "$0")/../../.."                       # korzeń projektu (C:/Users/zofsm/3d)
REMOTE=$(ssh "$HOST" 'echo ${KP3D:-$SCRATCH/kp3d}')
echo "-> $HOST:$REMOTE (kp_razem)"
ssh "$HOST" "mkdir -p $REMOTE/dane"
LISTA=$(mktemp)
(cd dane/kp_razem && find . -name rgb.jpg -o -name cam.json -o -name frazy.json | sed "s#^\./#dane/kp_razem/#") > "$LISTA"
tar czf - \
  --exclude='ciop/kpstudy/.venv' --exclude='ciop/kpstudy/preds' --exclude='ciop/kpstudy/results' \
  --exclude='ciop/kpstudy/tools/llama_win' --exclude='ciop/kpstudy/models' --exclude='__pycache__' \
  --exclude='ciop/kpstudy/annot' --exclude='ciop/kpstudy/_helios_pull' \
  ciop/kpstudy dane/kp_razem/views.json dane/kp_razem/prompts.json -T "$LISTA" \
  | ssh "$HOST" "tar xzf - -C $REMOTE"
rm -f "$LISTA"
echo done
