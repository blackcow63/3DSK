"""
Text -> 2D point with Molmo2 (allenai/Molmo2-4B, -8B) or MolmoPoint (allenai/MolmoPoint-8B), via transformers on
a GPU (Helios). One question per named point, same phrases as every other pointer (prompts.json).

  Molmo2     answers <points coords="1 1 x y"> with x, y scaled to 0..1000 (decoded as in the model card)
  MolmoPoint answers with grounding tokens, decoded by model.extract_image_points (pixel coordinates)

    python methods/molmo_point.py --model allenai/MolmoPoint-8B --name molmopoint8b [--root <zbior>] [--views all]
writes preds/stage1/<name>/<view>.json (points_2d, per-question timing, raw text)
Needs transformers==4.57.1 (model cards), torch, pillow, einops, accelerate.
"""
import argparse, json, os, re, sys, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

HERE = os.path.dirname(os.path.abspath(__file__))
KP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import common  # noqa: E402
COORD_REGEX = re.compile(r"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
FRAME_REGEX = re.compile(r"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")


def molmo2_points(text, w, h):
    out = []
    for coord in COORD_REGEX.finditer(text):
        for grp in FRAME_REGEX.finditer(coord.group(1)):
            for p in POINTS_REGEX.finditer(grp.group(2)):
                x, y = float(p.group(2)) / 1000 * w, float(p.group(3)) / 1000 * h
                if 0 <= x <= w and 0 <= y <= h:
                    out.append((x, y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--used-only", action="store_true", help="kawalerka: only photos used by the view sets (view_sets.json)")
    common.dodaj_opcje(ap)
    args = ap.parse_args()
    is_point = "MolmoPoint" in args.model
    template, phrases = common.frazy(args)
    views = common.widoki(args)
    if args.used_only:
        used = {v for s in json.load(open(os.path.join(KP, "view_sets.json"))) for v in s["views"]}
        views = [v for v in views if v in used]
    out = os.path.join(common.katalog_wynikow(args), "stage1", args.name)
    os.makedirs(out, exist_ok=True)

    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(args.model, trust_remote_code=True, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    load_ms = (time.perf_counter() - t0) * 1000
    n_params = sum(p.numel() for p in model.parameters())
    times = []
    for vid in views:
        dst = common.plik_wyniku(args, "stage1", args.name, vid)
        if os.path.exists(dst):
            continue
        img = Image.open(os.path.join(args.root, vid, "rgb.jpg")).convert("RGB")
        W, H = img.size
        pts, timing, raw = {}, {}, {}
        for name, phrase in common.frazy_widoku(args, vid, phrases).items():
            msgs = [{"role": "user", "content": [{"type": "text", "text": template.format(phrase=phrase)},
                                                 {"type": "image", "image": img}]}]
            inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
                                                   return_dict=True, padding=True, **({"return_pointing_metadata": True} if is_point else {}))
            meta = inputs.pop("metadata") if is_point else None
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            torch.cuda.synchronize(); t1 = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                kw = {"logits_processor": model.build_logit_processor_from_inputs(inputs)} if is_point else {}
                o = model.generate(**inputs, max_new_tokens=64, do_sample=False, **kw)
            torch.cuda.synchronize(); dt = (time.perf_counter() - t1) * 1000
            gen = o[:, inputs["input_ids"].size(1):]
            if is_point:
                text = processor.post_process_image_text_to_text(gen, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
                p = model.extract_image_points(text, meta["token_pooling"], meta["subpatch_mapping"], meta["image_sizes"])
                cand = [(float(q[2]), float(q[3])) for q in p]
            else:
                text = processor.tokenizer.decode(gen[0], skip_special_tokens=True)
                cand = molmo2_points(text, W, H)
            raw[name] = text
            if cand:
                pts[name] = [cand[0][0], cand[0][1], 1.0]
            timing[name] = dict(gpu_ms=dt, prompt_tokens=int(inputs["input_ids"].shape[1]), new_tokens=int(gen.shape[1]))
            times.append(dt)
        json.dump({"method": args.name, "view": vid, "points_2d": pts, "timing_ms": timing, "raw": raw}, open(dst, "w"))
        print(f"{vid}: {len(pts)}/{len(phrases)}", flush=True)
    json.dump({"model": args.model, "params": n_params, "load_ms": load_ms, "gpu_ms_per_question_median": float(np.median(times)) if times else None,
               "gpu": torch.cuda.get_device_name(0), "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20},
              open(os.path.join(out, "_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
