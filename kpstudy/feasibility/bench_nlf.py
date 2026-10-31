"""
CPU timing of the NLF crop model (the part a phone would run: EfficientNetV2-S or -L backbone + localizer field
queried at 17 body points, on one 256 x 256 person crop) for the phone proxy. The bundled YOLOv8x person
detector (68 M parameters) is a traced half-precision graph that does not run on CPU; a phone app would use a
light detector instead, so only its parameter count is reported.

    python feasibility/bench_nlf.py models/nlf_s_multi_0.2.2.torchscript nlf_s [threads ...]
writes bench/nlf_<name>_<hostname>.json
"""
import json, os, platform, socket, statistics, sys, time
import torch
import torchvision  # noqa: F401  (needed to load the TorchScript)

HERE = os.path.dirname(os.path.abspath(__file__))
KP = os.path.dirname(HERE)


def timeit(fn, warm=3, n=20):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return dict(median_ms=statistics.median(ts), p90_ms=ts[int(0.9 * (len(ts) - 1))])


def main():
    path, name = sys.argv[1], sys.argv[2]
    threads = [int(t) for t in sys.argv[3:]] or [1, 4]
    m = torch.jit.load(path, map_location="cpu").float().eval()          # weights ship in fp16 for the GPU
    cm, det = m.crop_model, m.detector
    crop = torch.rand(1, 3, 256, 256)
    K = torch.tensor([[[400.0, 0, 128], [0, 400.0, 128], [0, 0, 1]]])
    canon = torch.rand(17, 3) * 0.4 - 0.2                         # 17 query points in the canonical body volume
    res = {"model": os.path.basename(path), "host": socket.gethostname(), "machine": platform.machine(),
           "crop_model_params_M": sum(p.numel() for p in cm.parameters()) / 1e6,
           "detector_params_M": sum(p.numel() for p in det.parameters()) / 1e6}
    for t in threads:
        torch.set_num_threads(t)
        with torch.inference_mode():
            res[f"crop_t{t}"] = timeit(lambda: cm.predict_multi_same_canonicals(crop, K, canon))
        print(t, "threads:", {k: round(v["median_ms"]) for k, v in res.items() if k.endswith(f"_t{t}")}, flush=True)
    os.makedirs(os.path.join(KP, "bench"), exist_ok=True)
    json.dump(res, open(os.path.join(KP, "bench", f"nlf_{name}_{res['host']}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
