#!/usr/bin/env bash
# Helios -> laptop: wyniki metod na zbiorze ARKitScenes (preds), bez dużych map głębi, jeśli --bez-glebi.
#   bash ciop/kpstudy/helios/pull_arkit.sh kp [user@host]
set -euo pipefail
ZB=${1:-kp}
HOST=${2:-plgzsmolen@helios.cyfronet.pl}
cd "$(dirname "$0")/../../.."
REMOTE=$(ssh "$HOST" 'echo ${KP3D:-$SCRATCH/kp3d}')
mkdir -p dane/arkitscenes/$ZB
ssh "$HOST" "cd $REMOTE/dane/arkitscenes/$ZB && tar czf - preds" | tar xzf - -C dane/arkitscenes/$ZB
echo done
