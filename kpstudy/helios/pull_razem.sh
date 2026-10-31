#!/usr/bin/env bash
# Helios -> laptop: wyniki metod na wspólnym zbiorze (preds).
#   bash ciop/kpstudy/helios/pull_razem.sh [user@host]
set -euo pipefail
HOST=${1:-plgzsmolen@helios.cyfronet.pl}
cd "$(dirname "$0")/../../.."
REMOTE=$(ssh "$HOST" 'echo ${KP3D:-$SCRATCH/kp3d}')
mkdir -p dane/kp_razem
ssh "$HOST" "cd $REMOTE/dane/kp_razem && tar czf - preds" | tar xzf - -C dane/kp_razem
echo done
