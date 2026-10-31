#!/usr/bin/env bash
# Helios -> laptop: predictions (preds/) and benchmark results (bench/) produced by the jobs.
#   bash ciop/kpstudy/helios/pull.sh [user@host]
set -euo pipefail
HOST=${1:-plgzsmolen@helios.cyfronet.pl}
cd "$(dirname "$0")/.."                             # ciop/kpstudy
REMOTE=$(ssh "$HOST" 'echo ${KP3D:-$SCRATCH/kp3d}')/ciop/kpstudy
ssh "$HOST" "cd $REMOTE && tar czf - preds bench 2>/dev/null || true" | tar xzf - -C .
echo "pulled preds/ and bench/ from $HOST:$REMOTE"
