"""
Text -> 2D point with Qwen3-VL (Qwen/Qwen3-VL-8B-Instruct, -32B-Instruct, ...) or another model with the same
chat format and 0..1000 point output (Embodied-R1 answers the same question), via transformers on a GPU (Helios).
Same question per named point as the llama.cpp runs of Qwen3-VL-2B (vlm_point_llamacpp.py), so the two agree:

  "<template with the phrase> Answer with JSON only: {"point_2d": [x, y]}, coordinates relative to the image on a 0-1000 scale."

--styl r15 and r1 ask in the models' own format instead (Embodied-R1.5 and Embodied-R1, as in their inference examples); the
answer inside <answer> is read the same way. --trust-remote-code for models with custom code (LLaVA-OneVision-2, venv kp3).

    python methods/qwen3vl_point.py --model Qwen/Qwen3-VL-8B-Instruct --name qwen3vl8b --root <zbior> [--quant 4bit]
writes <preds>/stage1/<name>/<view>.json (points_2d, raw, timing)
"""
import argparse, json, os, re, sys, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SUFFIX = ' Answer with JSON only: {"point_2d": [x, y]}, coordinates relative to the image on a 0-1000 scale.'
XY = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
R15 = ('Provide one or more points coordinate of objects region this sentence describes: {phrase}. '
       'The answer should be presented in JSON format as follows: [{{"point_2d": [x, y]}}].')
R1 = ("Provide one or more points coordinate of objects region {phrase}. "
      "The results are presented in a format <point>[[x1,y1], [x2,y2], ...]</point>. "
      "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
      "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags. "
      "The answer consists only of several coordinate points, with the overall format being: "
      "<think> reasoning process here </think><answer><point>[[x1, y1], [x2, y2], ...]</point></answer>")
STYL = {"qwen": 40, "r15": 512, "r1": 512}      # max_new_tokens: Embodied myśli na głos przed odpowiedzią


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--quant", default="none", choices=["none", "4bit", "8bit"])
    ap.add_argument("--styl", default="qwen", choices=list(STYL))
    ap.add_argument("--trust-remote-code", action="store_true")
    common.dodaj_opcje(ap)
    args = ap.parse_args()
    template, phrases = common.frazy(args)
    views = common.widoki(args)
    kw = {}
    if args.quant != "none":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=args.quant == "4bit", load_in_8bit=args.quant == "8bit",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.perf_counter()
    trc = args.trust_remote_code
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=trc, **kw)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=trc)

    def pytanie(phrase):
        if args.styl == "r15":
            return R15.format(phrase=phrase)
        if args.styl == "r1":
            return R1.format(phrase=phrase)
        return template.format(phrase=phrase) + SUFFIX
    load_ms = (time.perf_counter() - t0) * 1000
    times = []
    for vid in views:
        dst = common.plik_wyniku(args, "stage1", args.name, vid)
        if os.path.exists(dst):
            continue
        img = Image.open(os.path.join(args.root, vid, "rgb.jpg")).convert("RGB")
        W, H = img.size
        pts, raw, timing = {}, {}, {}
        for name, phrase in common.frazy_widoku(args, vid, phrases).items():
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": pytanie(phrase)}]}]
            inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                   return_tensors="pt").to(model.device)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            with torch.inference_mode():
                o = model.generate(**inputs, max_new_tokens=STYL[args.styl], do_sample=False)
            torch.cuda.synchronize(); dt = (time.perf_counter() - t1) * 1000
            times.append(dt)
            text = processor.batch_decode(o[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            raw[name] = text
            timing[name] = dict(gpu_ms=dt, prompt_tokens=int(inputs["input_ids"].shape[1]))
            m = XY.search(text.split("<answer>")[-1])
            if m:
                pts[name] = [float(m.group(1)) / 1000 * W, float(m.group(2)) / 1000 * H, 1.0]
        json.dump({"method": args.name, "view": vid, "points_2d": pts, "timing_ms": timing, "raw": raw}, open(dst, "w"))
        print(f"{vid}: {len(pts)}/{len(phrases)}", flush=True)
    json.dump({"model": args.model, "quant": args.quant, "params": sum(p.numel() for p in model.parameters()), "load_ms": load_ms,
               "gpu_ms_per_question_median": float(np.median(times)) if times else None,
               "gpu": torch.cuda.get_device_name(0), "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20},
              open(os.path.join(common.katalog_wynikow(args), "stage1", args.name, "_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
