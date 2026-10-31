#!/usr/bin/env bash
# Laptop (Git Bash) -> Helios: the study code, the phone photos + camera/GT files, and the GGUF models.
# Only what the methods read is sent (rgb.jpg, cam.json, labels.png for the oracle person box, gt/views);
# depth/kp ground truth stays on the laptop, where scoring runs.
#   bash ciop/kpstudy/helios/push.sh [user@host]
set -euo pipefail
HOST=${1:-plgzsmolen@helios.cyfronet.pl}
cd "$(dirname "$0")/../../.."                       # project root (C:/Users/zofsm/3d)
REMOTE=$(ssh "$HOST" 'echo ${KP3D:-$SCRATCH/kp3d}')
echo "-> $HOST:$REMOTE"
ssh "$HOST" "mkdir -p $REMOTE"
tar czf - \
  --exclude='ciop/kpstudy/.venv' --exclude='ciop/kpstudy/preds' --exclude='ciop/kpstudy/results' \
  --exclude='ciop/kpstudy/tools/llama_win' --exclude='ciop/kpstudy/models' --exclude='__pycache__' \
  ciop/kpstudy \
  chairs-dataset/renders/kawalerka_v1/gt.json chairs-dataset/renders/kawalerka_v1/views.json \
  $(cd chairs-dataset/renders/kawalerka_v1 && ls -d ring_* walk* 2>/dev/null | sed 's#^#chairs-dataset/renders/kawalerka_v1/#;s#$#/rgb.jpg#') \
  $(cd chairs-dataset/renders/kawalerka_v1 && ls -d ring_* walk* 2>/dev/null | sed 's#^#chairs-dataset/renders/kawalerka_v1/#;s#$#/cam.json#') \
  $(cd chairs-dataset/renders/kawalerka_v1 && ls -d ring_* walk* 2>/dev/null | sed 's#^#chairs-dataset/renders/kawalerka_v1/#;s#$#/labels.png#') \
  | ssh "$HOST" "tar xzf - -C $REMOTE"
# models are fetched on Helios itself (the login node has internet and a faster link than the laptop):
# the same GGUF files as on the laptop for the Grace CPU benchmark, and the NLF TorchScript releases
ssh "$HOST" "mkdir -p $REMOTE/ciop/kpstudy/models && cd $REMOTE/ciop/kpstudy/models && \
  for f in Qwen3VL-2B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf; do \
    test -s \$f || curl -sfL -o \$f https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/\$f; done; \
  test -s nlf_s_multi_0.2.2.torchscript || curl -sfL -o nlf_s_multi_0.2.2.torchscript https://github.com/isarandi/nlf/releases/download/v0.2.2/nlf_s_multi_0.2.2.torchscript; \
  test -s nlf_l_multi_0.3.2.torchscript || curl -sfL -o nlf_l_multi_0.3.2.torchscript https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript; \
  ls -la"
echo done
