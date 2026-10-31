"""
MoGe-2 monocular metric geometry: depth along the optical axis for the depth_<model> lift (run_lift.py for the
kawalerka, eval_arkit.py for ARKitScenes). The phone knows its own field of view (camera metadata), so it is
passed in (fov_x) -- as an app would; for ARKitScenes it comes from each photo's K (cam.json).

    python methods/moge_depth.py [vits|vitb|vitl] [--all] [--root <zbior>] [--limit N]
writes <preds>/depth/moge2_<size>/<view>.npy (float32 metres, 0 = masked) + _timing.json
(the kawalerka keeps its old default: ring photos only, unless --all)
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

REPO = {"vits": "Ruicheng/moge-2-vits-normal", "vitb": "Ruicheng/moge-2-vitb-normal", "vitl": "Ruicheng/moge-2-vitl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("size", nargs="?", default="vits", choices=list(REPO))
    ap.add_argument("--all", action="store_true", help="kawalerka: all photos, not only the ring")
    common.dodaj_opcje(ap)
    args = ap.parse_args()
    if os.path.abspath(args.root) == common.KAWALERKA and not args.all and args.views == "all":
        args.views = "ring"
    from moge.model.v2 import MoGeModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    model = MoGeModel.from_pretrained(REPO[args.size]).to(dev).eval()
    load_ms = (time.perf_counter() - t0) * 1000
    n_params = sum(p.numel() for p in model.parameters())
    times = []
    for vid in common.widoki(args):
        dst = common.plik_wyniku(args, "depth", f"moge2_{args.size}", vid, ".npy")
        if os.path.exists(dst):
            continue
        img = np.asarray(Image.open(os.path.join(args.root, vid, "rgb.jpg")).convert("RGB"))
        K = common.K_widoku(args, vid)
        fov_x = float(np.degrees(2 * np.arctan(img.shape[1] / 2 / K[0, 0])))
        x = torch.tensor(img / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        t0 = time.perf_counter()
        with torch.no_grad():
            o = model.infer(x, fov_x=fov_x, use_fp16=(dev == "cuda"))
        times.append((time.perf_counter() - t0) * 1000)
        d = o["depth"].float().cpu().numpy()
        m = o["mask"].cpu().numpy().astype(bool)
        common.zapisz_glebie(args, dst, np.where(m & np.isfinite(d), d, 0))
        print(f"{vid}: {times[-1]:.0f} ms", flush=True)
    json.dump({"model": REPO[args.size], "params": n_params, "device": dev, "load_ms": load_ms,
               "infer_ms_median": float(np.median(times)) if times else None, "infer_ms": times},
              open(os.path.join(common.katalog_wynikow(args), "depth", f"moge2_{args.size}", "_timing.json"), "w"), indent=1)
    print(f"moge2_{args.size}: {n_params / 1e6:.1f} M params, median {np.median(times) if times else 0:.0f} ms/photo on {dev}")


if __name__ == "__main__":
    main()
