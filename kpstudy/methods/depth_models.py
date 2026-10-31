"""
Metric depth from RGB for the depth_<model> lift (eval_arkit.py): UniDepthV2, Depth Anything 3 (nested, metric) and
MapAnything. Runs in the kp2 venv on Helios (helios/setup2.sbatch).

  unidepthv2_l     lpiccinelli/unidepth-v2-vitl14, one photo, given the photo's intrinsics (the phone knows them)
  da3_nested       depth-anything/DA3NESTED-GIANT-LARGE-1.1, one photo, metric output without intrinsics
  mapanything_scene facebook/map-anything-apache on all photos of a scene at once (no poses, no intrinsics):
                   the multi-view, RGB-only alternative to VIO; depth of each photo from the joint reconstruction

    python methods/depth_models.py unidepthv2_l --root <zbior> [--limit N]
writes <preds>/depth/<name>/<view>.npy (float32 metres along the optical axis, photo resolution, 0 = unknown)
"""
import argparse, collections, json, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def resize(d, W, H):
    return np.array(Image.fromarray(d.astype(np.float32)).resize((W, H), Image.BILINEAR))


def run_unidepth(args, views, out):
    from unidepth.models import UniDepthV2
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").cuda().eval()
    times = []
    for vid in views:
        dst = os.path.join(out, vid + ".npy")
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        img = np.array(Image.open(os.path.join(args.root, vid, "rgb.jpg")).convert("RGB"))
        rgb = torch.from_numpy(img).permute(2, 0, 1).cuda()
        K = torch.from_numpy(common.K_widoku(args, vid)).float().cuda()
        t0 = time.perf_counter()
        with torch.no_grad():
            try:
                from unidepth.utils.camera import Pinhole
                pred = model.infer(rgb, Pinhole(K=K.unsqueeze(0)))
            except ImportError:
                pred = model.infer(rgb, K)
        times.append((time.perf_counter() - t0) * 1000)
        d = pred["depth"].squeeze().float().cpu().numpy()
        common.zapisz_glebie(args, dst, resize(d, img.shape[1], img.shape[0]) if d.shape != img.shape[:2] else d)
        print(f"{vid}: {times[-1]:.0f} ms", flush=True)
    return model, times


def run_da3(args, views, out):
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1").cuda().eval()
    times = []
    for vid in views:
        dst = os.path.join(out, vid + ".npy")
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        p = os.path.join(args.root, vid, "rgb.jpg")
        W, H = Image.open(p).size
        t0 = time.perf_counter()
        pred = model.inference([p])
        times.append((time.perf_counter() - t0) * 1000)
        common.zapisz_glebie(args, dst, resize(np.asarray(pred.depth[0]), W, H))
        print(f"{vid}: {times[-1]:.0f} ms", flush=True)
    return model, times


def run_mapanything(args, views, out):
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    model = MapAnything.from_pretrained("facebook/map-anything-apache").cuda().eval()
    sceny = collections.defaultdict(list)
    for vid in views:
        sceny[vid.split("/")[0]].append(vid)
    times = []
    for scena, vids in sceny.items():
        if all(os.path.exists(os.path.join(out, v + ".npy")) for v in vids):
            continue
        paths = [os.path.join(args.root, v, "rgb.jpg") for v in vids]
        t0 = time.perf_counter()
        with torch.no_grad():
            preds = model.infer(load_images(paths), memory_efficient_inference=True, use_amp=True, amp_dtype="bf16")
        times.append((time.perf_counter() - t0) * 1000 / len(vids))
        for v, p, pr in zip(vids, paths, preds):
            W, H = Image.open(p).size
            d = pr["depth_z"].squeeze().float().cpu().numpy()
            common.zapisz_glebie(args, os.path.join(out, v + ".npy"), resize(d, W, H))
        print(f"{scena}: {len(vids)} photos, {times[-1]:.0f} ms/photo", flush=True)
    return model, times


RUN = {"unidepthv2_l": run_unidepth, "da3_nested": run_da3, "mapanything_scene": run_mapanything}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(RUN))
    common.dodaj_opcje(ap)
    args = ap.parse_args()
    views = common.widoki(args)
    out = os.path.join(common.katalog_wynikow(args), "depth", args.model)
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    model, times = RUN[args.model](args, views, out)
    json.dump({"model": args.model, "params": sum(p.numel() for p in model.parameters()), "total_s": time.perf_counter() - t0,
               "infer_ms_median": float(np.median(times)) if times else None, "gpu": torch.cuda.get_device_name(0),
               "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20}, open(os.path.join(out, "_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
