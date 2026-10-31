"""
TAIHRI (Tencent, 2026; Qwen3-VL-4B fine-tune) end-to-end on the kawalerka photos: photo + intrinsics + a text
question naming the joints -> 2D and 3D keypoints. Uses the repository's own wrapper and its crop/rescale
convention (images rescaled so the focal length becomes 1000 px), exactly as demo/simple_demo_kpt2d_3d_mllm.py.

keypoints_3d is stored as returned; the demo does not document its convention, so compare one photo with
ground truth (camera-frame metres vs normalised x/z, y/z + depth) before scoring with run_lift.py.

Runs on a GPU from a clone of https://github.com/Tencent/TAIHRI with checkpoints/taihri from HF RammusAoLi/TAIHRI:
    cd $KP3D/ext/TAIHRI && python $KPSTUDY/methods/taihri_e2e.py --model checkpoints/taihri
"""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(KP, "..", "..", "chairs-dataset", "renders", "kawalerka_v1"))
NAMES = ["left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist",
         "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"]
PROMPT = "Could you tell me the coordinates of " + ", ".join(NAMES[:-1]) + " and " + NAMES[-1] + "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/taihri")
    ap.add_argument("--name", default="taihri4b")
    ap.add_argument("--backend", default="transformers")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, os.getcwd())
    from eval_wrapper import TAIHRIKpt2D3DMLLMWrapper
    import importlib.util                                    # demo/demo.py (a folder of the same name shadows it)
    spec = importlib.util.spec_from_file_location("taihri_demo", os.path.join(os.getcwd(), "demo", "demo.py"))
    demo = importlib.util.module_from_spec(spec); spec.loader.exec_module(demo)
    t0 = time.perf_counter()
    # sdpa instead of the default flash_attention_2 (no flash-attn build for aarch64 here); same outputs
    model = TAIHRIKpt2D3DMLLMWrapper(model_path=args.model, adapter_model_path=None, backend=args.backend, max_tokens=2048,
                                     temperature=0.0, top_p=0.05, top_k=1, repetition_penalty=1.05,
                                     min_pixels=16 * 28 * 28, max_pixels=2560 * 28 * 28, quantization=None,
                                     attn_implementation="sdpa")
    load_ms = (time.perf_counter() - t0) * 1000
    gt = json.load(open(os.path.join(ROOT, "gt.json")))
    fx, fy, cx, cy = gt["intrinsics"]["K"]
    views = [v["id"] for v in json.load(open(os.path.join(ROOT, "views.json"))) if os.path.exists(os.path.join(ROOT, v["id"], "rgb.jpg"))]
    if args.limit:
        views = views[:args.limit]
    d3, d2 = os.path.join(KP, "preds", "e2e", args.name), os.path.join(KP, "preds", "stage1", args.name)
    os.makedirs(d3, exist_ok=True); os.makedirs(d2, exist_ok=True)
    times = []
    for vid in views:
        img = Image.open(os.path.join(ROOT, vid, "rgb.jpg")).convert("RGB")
        imgs, crop_boxes, scales = demo.crop_batch_images([img], crop_size=(1400, 1400), focal_length=fx, crop=False)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        with torch.no_grad():
            res = model.inference(images=imgs, task="human_prompt", prompts=PROMPT)[0]
        torch.cuda.synchronize(); dt = (time.perf_counter() - t1) * 1000
        times.append(dt)
        raw, p2, p3 = res, {}, {}
        if res.get("success"):
            pr = res["extracted_predictions"]
            k2, _ = demo.unproj(pr.get("keypoints_2d", {}), pr.get("bbox", [0, 0, 0, 0]), crop_boxes[0], scales[0])
            for n, uv in k2.items():
                p2["body_" + n.replace(" ", "_")] = [float(uv[0]), float(uv[1]), 1.0]
            for n, xyz in pr.get("keypoints_3d", {}).items():
                p3["body_" + n.replace(" ", "_")] = [float(c) for c in xyz]
        json.dump({"method": args.name, "view": vid, "points_cam": p3, "timing_ms": {"gpu": dt},
                   "convention": "as returned by TAIHRI (verify against GT before scoring)", "raw": str(raw)[:4000]},
                  open(os.path.join(d3, vid + ".json"), "w"))
        json.dump({"method": args.name, "view": vid, "points_2d": p2, "timing_ms": {"gpu": dt}}, open(os.path.join(d2, vid + ".json"), "w"))
        print(f"{vid}: {len(p3)} points, {dt:.0f} ms", flush=True)
    json.dump({"model": args.model, "load_ms": load_ms, "gpu_ms_median": float(np.median(times)), "prompt": PROMPT,
               "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20}, open(os.path.join(d3, "_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
