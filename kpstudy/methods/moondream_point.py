"""
Text -> 2D point with Moondream 2 (vikhyatk/moondream2, 1.9B), via transformers on a GPU (Helios). Moondream has
a dedicated pointing skill: model.point(image, "<object>") returns every matching point, normalized to 0..1; we
keep the first. The image is encoded once per photo and reused for every question; the phrase is the same one
every other pointer gets (prompts.json), without the chat template, since point() takes an object description.

    python methods/moondream_point.py --name moondream2 [--revision 2025-06-21] [--used-only]
    phone proxy on the CPU (bf16 PyTorch, pinned cores):
    taskset -c 0-3 python methods/moondream_point.py --name moondream2_grace_t4 --device cpu --threads 4 \
        --anchors furniture --views ring --limit 3
writes preds/stage1/<name>/<view>.json (points_2d, per-question timing, raw points) and _timing.json
Needs transformers (trust_remote_code), torch, pillow, einops; pyvips is optional (PIL fallback).
"""
import argparse, json, os, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
KP = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vikhyatk/moondream2")
    ap.add_argument("--revision", default="2025-06-21")
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default=os.path.abspath(os.path.join(KP, "..", "..", "chairs-dataset", "renders", "kawalerka_v1")))
    ap.add_argument("--views", default="all", help="all albo rodzaje widoków po przecinku")
    ap.add_argument("--anchors", default="all", choices=["furniture", "body", "all"])
    ap.add_argument("--used-only", action="store_true", help="only photos used by the view sets (view_sets.json)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (with --device cpu)")
    ap.add_argument("--limit", type=int, default=0, help="first N photos only (timing runs)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    prompts = json.load(open(os.path.join(KP, "prompts.json"), encoding="utf-8"))
    phrases = {**(prompts["furniture"] if args.anchors in ("furniture", "all") else {}),
               **(prompts["body"] if args.anchors in ("body", "all") else {})}
    views = [v["id"] for v in json.load(open(os.path.join(args.root, "views.json")))
             if (args.views == "all" or v["kind"] in args.views.split(",")) and os.path.exists(os.path.join(args.root, v["id"], "rgb.jpg"))]
    if args.used_only:
        used = {v for s in json.load(open(os.path.join(KP, "view_sets.json"))) for v in s["views"]}
        views = [v for v in views if v in used]
    if args.limit:
        views = views[:args.limit]
    razem = os.path.exists(os.path.join(args.root, "gt.json")) is False and os.path.basename(args.root.rstrip("/")) == "kp_razem"
    out = os.path.join(args.root if razem else KP, "preds", "stage1", args.name)
    os.makedirs(out, exist_ok=True)

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, trust_remote_code=True,
                                                 device_map={"": args.device},
                                                 **({"dtype": torch.bfloat16} if args.device == "cpu" else {}))
    load_ms = (time.perf_counter() - t0) * 1000
    n_params = sum(p.numel() for p in model.parameters())
    enc_times, q_times = [], []
    for vid in views:
        dst = os.path.join(out, vid + ".json")
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        img = Image.open(os.path.join(args.root, vid, "rgb.jpg")).convert("RGB")
        W, H = img.size
        sync(); t1 = time.perf_counter()
        with torch.inference_mode():
            enc = model.encode_image(img)
        sync(); enc_ms = (time.perf_counter() - t1) * 1000
        enc_times.append(enc_ms)
        pts, timing, raw = {}, {}, {}
        fr = os.path.join(args.root, vid.split("/")[0], "frazy.json")   # zbiór razem: każda scena ma własne punkty
        for name, phrase in (json.load(open(fr, encoding="utf-8")) if os.path.exists(fr) else phrases).items():
            sync(); t1 = time.perf_counter()
            with torch.inference_mode():
                cand = model.point(enc, phrase)["points"]
            sync(); dt = (time.perf_counter() - t1) * 1000
            raw[name] = [[float(p["x"]), float(p["y"])] for p in cand]
            if cand:
                pts[name] = [float(cand[0]["x"]) * W, float(cand[0]["y"]) * H, 1.0]
            timing[name] = dict(ms=dt, n_points=len(cand))
            q_times.append(dt)
        json.dump({"method": args.name, "view": vid, "points_2d": pts, "timing_ms": timing, "encode_ms": enc_ms, "raw": raw},
                  open(dst, "w"))
        print(f"{vid}: {len(pts)}/{len(phrases)}  encode {enc_ms:.0f} ms", flush=True)
    json.dump({"model": args.model, "revision": args.revision, "params": n_params, "load_ms": load_ms,
               "ms_encode_median": float(np.median(enc_times)) if enc_times else None,
               "ms_per_question_median": float(np.median(q_times)) if q_times else None,
               "device": args.device, "threads": torch.get_num_threads(),
               **({"gpu": torch.cuda.get_device_name(0), "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20}
                  if args.device == "cuda" else {})},
              open(os.path.join(out, "_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
