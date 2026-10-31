"""
Text -> 2D point with RoboRefer (Zhoues/RoboRefer-2B-SFT, -8B-SFT, Apache-2.0), RGB only (no depth input), through the
authors' llava package from a clone (ext/RoboRefer) in the kp4 venv. Same question as API/use_api.py: the sentence plus
their suffix; the answer is a list of tuples in 0..1.

    python methods/roborefer_point.py --model Zhoues/RoboRefer-8B-SFT --name roborefer8b --root <zbior> --views frame,crop
writes <preds>/stage1/<name>/<view>.json (points_2d, raw, timing)
"""
import argparse, json, os, re, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SUFFIX = (" Your answer should be formatted as a list of tuples, i.e. [(x1, y1)],  where each tuple contains the x and y coordinates "
          "of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel "
          "locations of the points in the image.")
XY = re.compile(r"\(\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    common.dodaj_opcje(ap)
    args = ap.parse_args()
    import llava
    from llava import conversation as clib
    from llava.media import Image as LImage
    from huggingface_hub import snapshot_download
    from PIL import Image
    template, phrases = common.frazy(args)
    t0 = time.perf_counter()
    model = llava.load(snapshot_download(args.model))
    clib.default_conversation = clib.conv_templates["auto"].copy()
    load_ms = (time.perf_counter() - t0) * 1000
    times = []
    for vid in common.widoki(args):
        dst = common.plik_wyniku(args, "stage1", args.name, vid)
        if os.path.exists(dst):
            continue
        plik = os.path.join(args.root, vid, "rgb.jpg")
        W, H = Image.open(plik).size
        pts, raw = {}, {}
        for name, phrase in common.frazy_widoku(args, vid, phrases).items():
            torch.cuda.synchronize(); t1 = time.perf_counter()
            text = model.generate_content([LImage(plik), template.format(phrase=phrase) + SUFFIX])
            torch.cuda.synchronize(); times.append((time.perf_counter() - t1) * 1000)
            raw[name] = text
            m = XY.search(text)
            if m:
                pts[name] = [float(m.group(1)) * W, float(m.group(2)) * H, 1.0]
        json.dump({"method": args.name, "view": vid, "points_2d": pts, "raw": raw}, open(dst, "w"))
        print(f"{vid}: {len(pts)}/{len(raw)}", flush=True)
    json.dump({"model": args.model, "load_ms": load_ms, "gpu_ms_per_question_median": float(np.median(times)) if times else None,
               "gpu": torch.cuda.get_device_name(0), "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20},
              open(os.path.join(common.katalog_wynikow(args), "stage1", args.name, "_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
