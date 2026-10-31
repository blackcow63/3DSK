# Common settings for the keypoint study on Helios (PLGrid grant plgciopbhp, GH200 = aarch64 + H100).
# Sourced by every script/job here. The project tree is mirrored under $KP3D with the same relative layout as
# on the laptop (ciop/kpstudy, chairs-dataset/renders/kawalerka_v1), so all relative paths in the code work.
export KP_ACCOUNT=${KP_ACCOUNT:-plgciopbhp-gpu-gh200}
export KP_PARTITION=${KP_PARTITION:-plgrid-gpu-gh200}
export KP3D=${KP3D:-$SCRATCH/kp3d}
export KPSTUDY=$KP3D/ciop/kpstudy
export KP_VENVS=${KP_VENVS:-$SCRATCH/kp_venvs}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}          # 8B checkpoints do not fit a home quota
export UV_CACHE_DIR=${UV_CACHE_DIR:-$SCRATCH/uv_cache}
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
