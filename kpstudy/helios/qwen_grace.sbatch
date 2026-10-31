#!/bin/bash
# Qwen3-VL-2B Q4 + Q8 projector in llama.cpp on the Grace CPU (Neoverse V2), pinned to 4 cores: 640 px and
# full-size photos on 4 threads, 640 px also on 1 thread; 3 photos x 40 questions each.
# The prebuilt Linux arm64 release needs glibc 2.38 (Ubuntu); Helios nodes have 2.34 (RHEL 9), so llama.cpp is
# built from source here, with the GCC module that also provides the newer libstdc++. Plain armv8.6-a + dotprod
# + i8mm flags, no SVE, like a phone core.
#SBATCH -J kp-qwengrace
#SBATCH -A plgciopbhp-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 02:30:00
#SBATCH -o kp-qwengrace-%j.out
set -euo pipefail
source ciop/kpstudy/helios/env.sh
cd "$KPSTUDY"
set +u; source /etc/profile 2>/dev/null || true; module load GCCcore/14.3.0 CMake; set -u
export LD_LIBRARY_PATH="${EBROOTGCCCORE}/lib64:${LD_LIBRARY_PATH:-}"
echo "glibc $(ldd --version | head -1); cmake $(cmake --version | head -1); gcc $(gcc --version | head -1)"

SRC="$KP3D/ext/llama.cpp"
BIN="$SRC/build/bin/llama-server"
if [ ! -x "$BIN" ]; then
  [ -d "$SRC" ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp.git "$SRC"
  cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
        -DCMAKE_C_FLAGS="-march=armv8.6-a+dotprod+i8mm" -DCMAKE_CXX_FLAGS="-march=armv8.6-a+dotprod+i8mm" \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON > "$SRC/cmake.log" 2>&1 || { tail -30 "$SRC/cmake.log"; exit 1; }
  cmake --build "$SRC/build" --target llama-server -j 16 > "$SRC/build.log" 2>&1 || { tail -30 "$SRC/build.log"; exit 1; }
fi
[ -x "$BIN" ] || { echo "no llama-server at $BIN"; exit 1; }

CORES=$(taskset -pc $$ | awk -F': ' '{print $2}' | tr ',' '\n' | head -1 | cut -d- -f1)
PIN="$CORES-$((CORES + 3))"
source "$KP_VENVS/cpubench/bin/activate"
for cfg in "4 640" "4 0" "1 640"; do
  set -- $cfg
  taskset -c "$PIN" python methods/vlm_point_llamacpp.py --server "$BIN" --threads $1 --max-side $2 --views ring --limit 3 \
      --name qwen3vl2b_q4_grace_t$1_s$2 --port $((8600 + $1 + $2))
done
mkdir -p bench && for d in preds/stage1/qwen3vl2b_q4_grace_t*; do cp "$d/_server.log" "bench/$(basename $d)_server.log"; done
echo done
