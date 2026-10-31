#!/bin/bash
# Example-photo pointer (methods/dino_match.py, DINOv2 with registers) on one GH200:
#   example photos of a DIFFERENT workstation (renders/kawalerka_ref_v1): S/B/L at 840 px, S at 560 and 1120 px
#   example photos of the SAME workstation (six walk frames no test set uses): S/B/L at 840 px
# then every backbone exported to ONNX and timed on 4 Grace cores with the BlazePose phone calibration, as in
# cpu_bench.sbatch.     sbatch ciop/kpstudy/helios/dino.sbatch        (from $KP3D)
#SBATCH -J kp-dino
#SBATCH -A plgciopbhp-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 01:30:00
#SBATCH -o kp-dino-%j.out
set -euo pipefail
source ciop/kpstudy/helios/env.sh
cd "$KPSTUDY"
source "$KP_VENVS/kp/bin/activate"
echo "job $SLURM_JOB_ID on $(hostname) $(uname -m)"; nvidia-smi --query-gpu=name --format=csv,noheader
ROOT=../../chairs-dataset/renders
if [ "${BENCH_ONLY:-0}" != 1 ]; then              # sbatch --export=ALL,BENCH_ONLY=1 ... skips the matching
SAME=walkC_06,walkC_18,walkB_06,walkB_15,walkA_15,walkA_28
for m in s:small b:base l:large; do
  t=${m%%:*}; M=facebook/dinov2-with-registers-${m#*:}
  python methods/dino_match.py --model $M --side 840 --ref-root $ROOT/kawalerka_ref_v1 --name dinov2${t}_840_other --used-only
  python methods/dino_match.py --model $M --side 840 --ref-root $ROOT/kawalerka_v1 --ref-views $SAME --name dinov2${t}_840_same --used-only
done
for s in 560 1120; do
  python methods/dino_match.py --model facebook/dinov2-with-registers-small --side $s --ref-root $ROOT/kawalerka_ref_v1 \
      --name dinov2s_${s}_other --used-only
done
fi
mkdir -p models
for spec in "small 560" "small 840" "small 1120" "base 840" "large 840"; do
  set -- $spec
  test -s models/dinov2${1:0:1}_$2.onnx || python methods/dino_match.py --model facebook/dinov2-with-registers-$1 --side $2 \
      --export-onnx models/dinov2${1:0:1}_$2.onnx
done
deactivate

# phone proxy: 4 Grace cores, LiteRT calibration before and after (see cpu_bench.sbatch for the module dance)
CORES=$(taskset -pc $$ | awk -F': ' '{print $2}' | tr ',' '\n' | head -1 | cut -d- -f1)
PIN="$CORES-$((CORES + 3))"
set +u; source /etc/profile 2>/dev/null || true; module load GCCcore/14.3.0; set -u
export LD_LIBRARY_PATH="${EBROOTGCCCORE}/lib64:${LD_LIBRARY_PATH:-}"
source "$KP_VENVS/cpubench/bin/activate"
taskset -c "$PIN" python feasibility/bench_cpu.py --calib \
    --onnx dinov2s_560=models/dinov2s_560.onnx:1x3x420x560 --onnx dinov2s_840=models/dinov2s_840.onnx:1x3x630x840 \
    --onnx dinov2s_1120=models/dinov2s_1120.onnx:1x3x840x1120 --onnx dinov2b_840=models/dinov2b_840.onnx:1x3x630x840 \
    --onnx dinov2l_840=models/dinov2l_840.onnx:1x3x630x840
taskset -c "$PIN" python feasibility/bench_cpu.py --calib --calib-key calibration_end
echo done
