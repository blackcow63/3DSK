"""
Phone-proxy CPU timing on the Helios Grace CPU (Arm Neoverse V2, the server sibling of the phone Cortex-X3).

  calibration  BlazePose GHUM lite / full / heavy landmark models (the MediaPipe .tflite files), LiteRT, one
               thread. Published Pixel 4 single-core XNNPACK times are 25 / 40 / 147 ms (arXiv 2206.11678,
               Table 2); phone_ms / grace_ms is the calibration factor applied to every CPU-bound model.
  models       ONNX models (fp32 and ORT dynamic int8), 1 and 4 threads, median and p90 of 30 runs after warm-up.

    taskset -c 0-3 python feasibility/bench_cpu.py --calib --onnx moge2_vits=models/moge2_vits.onnx:1x3x518x686
writes bench/cpu_<hostname>.json
"""
import argparse, json, os, platform, socket, statistics, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KP = os.path.dirname(HERE)
PIXEL4_MS = {"lite": 25.0, "full": 40.0, "heavy": 147.0}
HEAVY_URL = "https://storage.googleapis.com/mediapipe-assets/pose_landmark_heavy.tflite"


def stats(ts):
    ts = sorted(ts)
    return dict(median_ms=statistics.median(ts), p90_ms=ts[int(0.9 * (len(ts) - 1))], n=len(ts))


def time_fn(fn, warm=5, n=30):
    for _ in range(warm):
        fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); out.append((time.perf_counter() - t0) * 1000)
    return stats(out)


def calibrate(mp_dir=None):
    import urllib.request
    try:
        from ai_edge_litert.interpreter import Interpreter          # Linux / macOS
    except ImportError:
        import tensorflow as tf                                       # Windows: same TFLite runtime via TensorFlow
        Interpreter = tf.lite.Interpreter
    if mp_dir:
        base = mp_dir
    else:
        import mediapipe
        base = os.path.join(os.path.dirname(mediapipe.__file__), "modules", "pose_landmark")
    res = {}
    for k in ("lite", "full", "heavy"):
        p = os.path.join(base, f"pose_landmark_{k}.tflite")
        if not os.path.exists(p):
            p = os.path.join(KP, "models", f"pose_landmark_{k}.tflite")
            if not os.path.exists(p):
                urllib.request.urlretrieve(HEAVY_URL.replace("heavy", k), p)
        it = Interpreter(model_path=p, num_threads=1)
        it.allocate_tensors()
        inp = it.get_input_details()[0]
        it.set_tensor(inp["index"], np.random.rand(*inp["shape"]).astype(inp["dtype"]))
        r = time_fn(it.invoke, n=100)
        r["pixel4_ms"] = PIXEL4_MS[k]
        r["phone_over_grace"] = PIXEL4_MS[k] / r["median_ms"]
        res[k] = r
        print(f"BlazePose {k}: this CPU, 1 thread {r['median_ms']:.1f} ms vs Pixel 4 {PIXEL4_MS[k]} ms -> x{r['phone_over_grace']:.2f}", flush=True)
    res["factor"] = statistics.median(res[k]["phone_over_grace"] for k in PIXEL4_MS)
    # the BlazePose person detector (2.8 MB): the light detector a phone app would put in front of NLF
    det = os.path.join(os.path.dirname(base), "pose_detection", "pose_detection.tflite")
    if os.path.exists(det):
        for th in (1, 4):
            it = Interpreter(model_path=det, num_threads=th)
            it.allocate_tensors()
            inp = it.get_input_details()[0]
            it.set_tensor(inp["index"], np.random.rand(*inp["shape"]).astype(inp["dtype"]))
            res[f"person_detector_t{th}"] = time_fn(it.invoke, n=50)
            print(f"BlazePose person detector, {th} thread(s): {res[f'person_detector_t{th}']['median_ms']:.1f} ms", flush=True)
    return res


def bench_onnx(name, path, shape):
    import onnxruntime as ort
    out = {}
    variants = {"fp32": path}
    q = path.replace(".onnx", ".int8.onnx")
    try:
        if not os.path.exists(q):
            from onnxruntime.quantization import quantize_dynamic, QuantType     # needs the onnx package
            quantize_dynamic(path, q, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])   # ConvInteger has no CPU kernel
        variants["int8"] = q
    except Exception as e:                       # some graphs do not quantize; report and go on
        out["int8_error"] = repr(e)[:300]
    for var, p in variants.items():
        for th in (1, 4):
            so = ort.SessionOptions(); so.intra_op_num_threads = th; so.inter_op_num_threads = 1
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
            feeds = {}
            for i in sess.get_inputs():
                if len(i.shape) == 4:
                    feeds[i.name] = np.random.rand(*shape).astype(np.float32)
                else:                              # e.g. MoGe's num_tokens scalar
                    feeds[i.name] = np.array(int(os.environ.get("KP_NUM_TOKENS", 1800)), dtype=np.int64)
            r = time_fn(lambda: sess.run(None, feeds), warm=2, n=10 if "vitl" in name else 30)
            r["model_mb"] = os.path.getsize(p) / 2**20
            out[f"{var}_t{th}"] = r
            print(f"{name} {var} {th} thread(s): {r['median_ms']:.0f} ms ({r['model_mb']:.0f} MB)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--calib-key", default="calibration", help="key to store the calibration under (e.g. calibration_end)")
    ap.add_argument("--mp-dir", help="folder with pose_landmark_{lite,full,heavy}.tflite (default: the mediapipe package)")
    ap.add_argument("--onnx", action="append", default=[], help="name=path:1x3xHxW")
    args = ap.parse_args()
    res = {"host": socket.gethostname(), "machine": platform.machine(), "cpu": platform.processor(),
           "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    if args.calib:
        res[args.calib_key] = calibrate(args.mp_dir)
    for spec in args.onnx:
        name, rest = spec.split("=", 1)
        path, shape = rest.rsplit(":", 1)
        res[name] = bench_onnx(name, path, tuple(int(x) for x in shape.split("x")))
    os.makedirs(os.path.join(KP, "bench"), exist_ok=True)
    dst = os.path.join(KP, "bench", f"cpu_{res['host']}.json")
    old = json.load(open(dst)) if os.path.exists(dst) else {}
    old.update(res)
    json.dump(old, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
