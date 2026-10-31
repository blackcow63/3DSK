"""
Text -> point with RoboBrain 2.5 (BAAI/RoboBrain2.5-4B, -8B-NV, -8B-MT; Qwen3-VL architecture), via transformers
on a GPU (Helios). One question per named point, same phrases as every other pointer.

Two prompts per point. The authors publish 3D prompts only for traces, so the single-point one is ours:
  3d  their 2D wording with the answer format of the RoboTracer evaluation (Evaluation/test_benchmark.py), [(x, y, d)],
      x and y 0-1000, depth in meters -> one-pass 3D: the point in the camera frame comes from (x, y), the depth d
      and the photo's intrinsics (cam.json)
  2d  the model's "pointing" prompt (inference.py), [(x, y)] in 0-1000 -> a 2D pointer like Molmo or Qwen

Other one-pass models answer through --styl, always one forward pass with no depth model around it:
  r15     Embodied-R1.5: the authors' reasoning and trace_3d templates (inference/hf_example.py) with the wording of their
          trace_3d training questions, for one point; answer {"point_2d": [x, y], "depth": d} (0-1000, metres)
  bbox3d  Qwen3-VL: its 3D grounding answer, {"bbox_3d": [x, y, z, ...]}; the centre is taken as the point
  razem   the same answer with "point_2d" added: pixel from point_2d, depth from z_center, in one query
  rynn    RynnBrain 1.1: the single-box prompt of the RynnScale benchmark (detect3d.py), intrinsics in the question; the centre is taken as the point
Plain Qwen3-VL with the default style is the untrained baseline for the RoboBrain question.

    python methods/robobrain_point.py --model BAAI/RoboBrain2.5-8B-NV --name robobrain25_8b --root <zbior> [--mode 3d|2d|both]
writes <preds>/e2e/<name>/<view>.json (points_cam, points_2d, raw) and <preds>/stage1/<name>_2d/<view>.json
"""
import argparse, json, os, re, sys, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

P3 = ("{phrase}. Please provide its 3D coordinates. Your answer should be formatted as a list of tuples, i.e. [(x, y, d)], "
      "where x and y are the coordinates of the point in the image and d is the depth of the point. The x and y should be "
      "0-1000 to indicate a point in the image. The unit of depth is meters. But you never output unit, just output number.")
P2 = ("{phrase}. Please provide its 2D coordinates. Your answer should be formatted as a tuple, i.e. [(x, y)], where the tuple "
      "contains the x and y coordinates of a point satisfying the conditions above.")
P15 = ('You are a robot performing robotic manipulation tasks. Locate the {phrase}. '
       'For this point, provide: (1) point_2d: 2D coordinates in normalized range [0, 1000]; (2) depth: distance from camera in meters. '
       'The number of points must be 1.\n'
       'Please answer this question based on the visual content.'
       'You FIRST think about the reasoning process as an internal monologue and then provide the final answer.'
       'At the end, you must output the final answer in the format:\n'
       '<answer><your_answer_here></answer>\n'
       "Please provide only the ordered 2D waypoints with depth (in meters) as JSON with key 'point_2d' and 'depth' within the <answer>...</answer> tags.\n"
       'Example:\n<answer>```json\n[{{"point_2d": [463, 599], "depth": 1.08}}]\n```</answer>')
PBOX = ('Locate the {phrase} in the provided image and output its position and dimensions using a 3D bounding box. The result must be '
        'in JSON format: {{"bbox_3d":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],"label":"point"}}.')
PRAZEM = ('The {phrase} is visible in the image. ' + PBOX + ' Add the exact point in the image as "point_2d": [x, y], with x and y 0-1000, in the same JSON object.')
PRYNN = ("Find the {phrase}.\n\nThe camera intrinsics matrix is:\n{K}\n\nPredict one 3D bounding box in the camera coordinate system, where:\n"
         "- x points to the right\n- y points downward\n- z points forward\n\nReturn:\n"
         "<3D Grounding> cx, cy, cz, x_size, y_size, z_size, pitch, yaw, roll </3D Grounding>\n\nDefinitions:\n"
         "- cx, cy, cz: 3D coordinates of the box center in the camera coordinate system, in meters\n"
         "- x_size, y_size, z_size: box dimensions in the box local coordinate system, in meters\n"
         "  - x_size: extent along the local x-axis\n  - y_size: extent along the local y-axis\n  - z_size: extent along the local z-axis\n"
         "- pitch: rotation around the camera x-axis, in radians (range: -1 to 1, corresponding to -180\u00b0 to 180\u00b0)\n"
         "- yaw: rotation around the camera y-axis, in radians (range: -1 to 1, corresponding to -180\u00b0 to 180\u00b0)\n"
         "- roll: rotation around the camera z-axis, in radians (range: -1 to 1, corresponding to -180\u00b0 to 180\u00b0)\n\nConstraints:\n"
         "- x_size >= z_size\n- Use meters for cx, cy, cz, x_size, y_size, z_size\n- Use radians for pitch, yaw, roll, with each value in the range [-1, 1]\n"
         "- Output exactly one 3D bounding box\n<think>\n\n</think>\n\n")
TOKENY = {"rb": 64, "r15": 1024, "bbox3d": 128, "razem": 200, "rynn": 300}
NUM, BOX, P2D = common.NUM, common.BOX, common.P2D
R15 = re.compile(r'"point_2d"\s*:\s*\[\s*' + NUM + r'\s*,\s*' + NUM + r'\s*\]\s*\}?\s*,\s*"depth"\s*:\s*' + NUM)
RBOX = re.compile(r'<3D Grounding>\s*' + NUM + r'\s*,\s*' + NUM + r'\s*,\s*' + NUM)


def parse(text, n):
    m = re.search(r"[\(\[]\s*" + r"\s*,\s*".join([NUM] * n) + r"\s*[\)\]]", text)
    return [float(x) for x in m.groups()] if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="both", choices=["3d", "2d", "both"])
    ap.add_argument("--quant", default="none", choices=["none", "4bit", "8bit"])
    ap.add_argument("--styl", default="rb", choices=list(TOKENY))
    ap.add_argument("--tokeny", type=int, default=0, help="limit długości odpowiedzi; domyślnie z TOKENY")
    common.dodaj_opcje(ap)
    args = ap.parse_args()
    _, phrases = common.frazy(args)
    views = common.widoki(args)
    t0 = time.perf_counter()
    kw = {}
    if args.quant != "none":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=args.quant == "4bit", load_in_8bit=args.quant == "8bit",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto", **kw)
    processor = AutoProcessor.from_pretrained(args.model)
    load_ms = (time.perf_counter() - t0) * 1000
    times = []

    def ask(img, text):
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": text}]}]
        inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        with torch.inference_mode():
            o = model.generate(**inputs, max_new_tokens=args.tokeny or TOKENY[args.styl], do_sample=False)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t1) * 1000
        times.append(dt)
        return processor.batch_decode(o[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0], dt

    for vid in views:
        dst = common.plik_wyniku(args, "e2e", args.name, vid)
        dst2 = common.plik_wyniku(args, "stage1", args.name + "_2d", vid)
        if os.path.exists(dst) and os.path.exists(dst2):
            continue
        img = Image.open(os.path.join(args.root, vid, "rgb.jpg")).convert("RGB")
        W, H = img.size
        K = common.K_widoku(args, vid)
        kk = "[" + ", ".join("[" + ", ".join("%.2f" % x for x in w) + "]" for w in K) + "]"
        cam, p2d, p2d_only, raw, timing = {}, {}, {}, {}, {}
        for name, phrase in common.frazy_widoku(args, vid, phrases).items():
            if args.mode in ("3d", "both"):
                text, dt = ask(img, {"rb": P3, "r15": P15, "bbox3d": PBOX, "razem": PRAZEM, "rynn": PRYNN}[args.styl].format(phrase=phrase, K=kk))
                raw[name + ":3d"] = text
                timing[name + ":3d"] = dt
                if args.styl in ("bbox3d", "rynn"):
                    m = (BOX if args.styl == "bbox3d" else RBOX).search(text)
                    if m:
                        c = [float(x) for x in m.groups()]
                        cam[name] = c
                        if c[2] > 0:
                            p2d[name] = [K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2], 1.0]
                    continue
                if args.styl == "razem":
                    r = common.odczyt_razem(text, W, H, K)
                    if r:
                        p2d[name], cam[name] = r
                    continue
                if args.styl == "r15":
                    m = R15.search(text.split("<answer>")[-1])
                    r = [float(x) for x in m.groups()] if m else None
                else:
                    r = parse(text, 3)
                if r:
                    u, v, d = r[0] / 1000 * W, r[1] / 1000 * H, r[2]
                    p2d[name] = [u, v, 1.0]
                    cam[name] = [(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d]
            if args.mode in ("2d", "both"):
                text, dt = ask(img, P2.format(phrase=phrase))
                raw[name + ":2d"] = text
                timing[name + ":2d"] = dt
                r = parse(text, 2)
                if r:
                    p2d_only[name] = [r[0] / 1000 * W, r[1] / 1000 * H, 1.0]
        json.dump({"method": args.name, "view": vid, "points_cam": cam, "points_2d": p2d, "raw": raw, "timing_ms": timing},
                  open(dst, "w"))
        json.dump({"method": args.name + "_2d", "view": vid, "points_2d": p2d_only, "raw": {k: v for k, v in raw.items() if k.endswith(":2d")}},
                  open(dst2, "w"))
        print(f"{vid}: 3d {len(cam)}/{len(phrases)}, 2d {len(p2d_only)}/{len(phrases)}", flush=True)
    for rodzaj, nazwa in (("e2e", args.name), ("stage1", args.name + "_2d")):
        json.dump({"model": args.model, "quant": args.quant, "params": sum(p.numel() for p in model.parameters()), "load_ms": load_ms,
                   "gpu_ms_per_question_median": float(np.median(times)) if times else None,
                   "gpu": torch.cuda.get_device_name(0), "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20},
                  open(os.path.join(common.katalog_wynikow(args), rodzaj, nazwa, "_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
