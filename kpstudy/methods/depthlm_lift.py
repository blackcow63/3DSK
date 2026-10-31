"""
Metric depth of one given pixel with DepthLM (facebook/DepthLM: Pixtral 12B fine-tuned, FAIR Noncommercial licence):
the second stage after a 2D pointer, or after the true pixel (--stage1 oracle2d) for the pure lifting error.
Same input as eval.py in DepthLM_Official: the photo resized so that fx = 750 px, a small red arrow ending at the pixel,
image before text, greedy decoding, one question per point. Crops are answered on their full photo (cam.json: pelny).

    python methods/depthlm_lift.py --stage1 molmopoint8b --root <zbior> --views frame,crop
reads <preds>/stage1/<stage1>/<view>.json, writes <preds>/lift/depthlm/<stage1>/<view>.json (depth_m: metres along the axis)
"""
import argparse, json, os, re, sys, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

MODEL = "facebook/DepthLM"
F_NORM = 750.0
RAMIE = 5
PYTANIE = ("Given this image, how far is the point pointed by the red arrow from the camera? Output the thinking process in "
           "<think> </think> and final answer (the meter number only, without the unit) in <answer> </answer> tags.")
LICZBA = r"-?\d+(?:\.\d+)?"


def strzalka(img, x, y):
    """Znacznik --> z datasets.py DepthLM; None, gdy nie mieści się w kadrze (tak samo pomijają to autorzy)."""
    if not (RAMIE <= x < img.width - RAMIE and RAMIE <= y < img.height - RAMIE):
        return None
    im = img.copy()
    for dx in range(1, RAMIE + 1):
        im.putpixel((x - dx, y), (255, 0, 0))
    for dy in range(1, RAMIE // 2 + 1):
        im.putpixel((x - dy - 1, y + dy), (255, 0, 0))
        im.putpixel((x - dy - 1, y - dy), (255, 0, 0))
    return im


def odpowiedz(t):
    m = re.search(r"<answer>\s*(" + LICZBA + ")", t) or None
    if m:
        return float(m.group(1))
    x = re.findall(LICZBA, t)
    return float(x[-1]) if x else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True, help="wskaźnik 2D z preds/stage1 albo oracle2d (prawdziwe piksele)")
    common.dodaj_opcje(ap)
    args = ap.parse_args()
    t0 = time.perf_counter()
    model = LlavaForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL)
    load_ms = (time.perf_counter() - t0) * 1000
    times = []

    def ask(img):
        chat = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PYTANIE, "content": PYTANIE}]}]
        inputs = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt").to(model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        with torch.inference_mode():
            o = model.generate(**inputs, max_new_tokens=64, do_sample=False, top_p=None, top_k=None)
        torch.cuda.synchronize(); times.append((time.perf_counter() - t1) * 1000)
        return processor.batch_decode(o[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    wyj = os.path.join(common.katalog_wynikow(args), "lift", "depthlm", args.stage1)
    for vid in common.widoki(args):
        src = os.path.join(common.katalog_wynikow(args), "stage1", args.stage1, vid + ".json")
        dst = os.path.join(wyj, vid + ".json")
        if not os.path.exists(src) or os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        pts = json.load(open(src))["points_2d"]
        cam = json.load(open(os.path.join(args.root, vid, "cam.json")))
        cel = cam.get("pelny", vid)
        K, Kp = np.array(cam["K"]), common.K_widoku(args, cel)
        img = Image.open(os.path.join(args.root, cel, "rgb.jpg")).convert("RGB")
        sk = F_NORM / Kp[0, 0]
        img = img.resize((int(img.width * sk), int(img.height * sk)))
        z, raw = {}, {}
        for name, p in pts.items():
            u = (p[0] - K[0, 2]) / K[0, 0] * Kp[0, 0] + Kp[0, 2]
            v = (p[1] - K[1, 2]) / K[1, 1] * Kp[1, 1] + Kp[1, 2]
            im = strzalka(img, int(u * sk), int(v * sk))
            if im is None:
                continue
            raw[name] = ask(im)
            d = odpowiedz(raw[name])
            if d is not None:
                z[name] = d
        json.dump({"method": "depthlm", "stage1": args.stage1, "view": vid, "depth_m": z, "raw": raw}, open(dst, "w"))
        print(f"{vid}: {len(z)}/{len(pts)}", flush=True)
    os.makedirs(wyj, exist_ok=True)
    json.dump({"model": MODEL, "params": sum(p.numel() for p in model.parameters()), "load_ms": load_ms,
               "gpu_ms_per_question_median": float(np.median(times)) if times else None,
               "gpu": torch.cuda.get_device_name(0), "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20},
              open(os.path.join(wyj, "_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
