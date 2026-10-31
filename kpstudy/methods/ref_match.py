"""
Point marked on another photo -> the same point here, by DINOv2 feature matching, on the combined set (data_razem.py):
the reference-image baseline. Same matching as dino_match.py (cosine similarity to the marked point's feature, the best of
the reference photos wins, argmax of the upsampled map), with per-scene references from ref.json (ref_pary.py).

    python methods/ref_match.py --model facebook/dinov2-with-registers-large --side 840 --name dinov2l_ref --root <zbior> --views frame,crop
writes <preds>/stage1/<name>/<view>.json (points_2d with the similarity as confidence)
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
from dino_match import Backbone, load_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--side", type=int, default=840)
    common.dodaj_opcje(ap)
    args = ap.parse_args()
    dev = "cuda"
    net = Backbone(AutoModel.from_pretrained(args.model).eval()).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    pary = json.load(open(os.path.join(args.root, "ref.json")))
    cache, times = {}, []

    def cechy(vid):
        if vid not in cache:
            x, W0, H0 = load_image(os.path.join(args.root, vid, "rgb.jpg"), args.side, dev)
            cache[vid] = (net(x).cpu(), W0, H0)
        return cache[vid]

    for vid in common.widoki(args):
        dst = common.plik_wyniku(args, "stage1", args.name, vid)
        if os.path.exists(dst):
            continue
        s = vid.split("/")[0]
        pts = {}
        with torch.inference_mode():
            x, W0, H0 = load_image(os.path.join(args.root, vid, "rgb.jpg"), args.side, dev)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            fm = net(x)
            torch.cuda.synchronize(); times.append((time.perf_counter() - t1) * 1000)
            for n, refs in pary.get(s, {}).items():
                best = None
                for r in refs:
                    if r["view"] == vid:
                        continue
                    f, w, h = cechy(r["view"])
                    g = torch.tensor([[[[(r["u"] + 0.5) / w * 2 - 1, (r["v"] + 0.5) / h * 2 - 1]]]])
                    q = F.normalize(F.grid_sample(f, g, mode="bilinear", align_corners=False)[0, :, 0, 0], dim=0).to(dev)
                    S = torch.einsum("c,chw->hw", q, fm[0])
                    if best is None or S.max() > best.max():
                        best = S
                if best is None:
                    continue
                up = F.interpolate(best[None, None], size=(H0, W0), mode="bilinear", align_corners=False)[0, 0]
                k = int(torch.argmax(up))
                pts[n] = [float(k % W0), float(k // W0), float(best.max())]
        json.dump({"method": args.name, "view": vid, "points_2d": pts, "backbone_ms": times[-1]}, open(dst, "w"))
        print(f"{vid}: {len(pts)}", flush=True)
        if len(cache) > 400:
            cache.clear()
    json.dump({"model": args.model, "side": args.side, "params": n_params, "backbone_ms_median": float(np.median(times)) if times else None,
               "gpu": torch.cuda.get_device_name(0), "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20},
              open(os.path.join(common.katalog_wynikow(args), "stage1", args.name, "_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
