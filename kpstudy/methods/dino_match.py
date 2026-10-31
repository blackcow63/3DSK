"""
Point marked on an example photo -> the same point in a new photo, by DINOv2 feature matching (the plain
semantic-correspondence baseline: nearest neighbour in feature space, no training).

The example photos and their marked points are a rendered view folder (rgb.jpg + kp.json); every point visible
in an example photo is a marked point. For each new photo and named point: cosine similarity of the photo's
patch features to the marked point's feature (bilinearly sampled), in every example photo; the example with the
highest peak wins; its similarity map is upsampled to full resolution and its maximum is the answer.

  example photos of a DIFFERENT workstation (render/render_reference.py):
    python methods/dino_match.py --model facebook/dinov2-with-registers-small --side 840 \
        --ref-root ../../chairs-dataset/renders/kawalerka_ref_v1 --name dinov2s_840_other --used-only
  example photos of the SAME workstation, from places no test set uses:
    ... --ref-root ../../chairs-dataset/renders/kawalerka_v1 --ref-views ring_r130_h120_a-035,... --name ..._same
  ONNX export of the backbone at that input size, for the phone-proxy CPU benchmark:
    python methods/dino_match.py --model ... --side 840 --export-onnx models/dinov2s_840.onnx

writes preds/stage1/<name>/<view>.json (points_2d with the similarity as confidence) and _timing.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
KP = os.path.dirname(HERE)
sys.path.insert(0, KP)
from data import FURNITURE                                  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Backbone(torch.nn.Module):
    """image (1,3,H,W), already normalized -> L2-normalized patch features (1,C,H/14,W/14)"""
    def __init__(self, model):
        super().__init__()
        self.m = model
        self.skip = 1 + getattr(model.config, "num_register_tokens", 0)

    def forward(self, x):
        h, w = x.shape[2] // 14, x.shape[3] // 14
        t = self.m(pixel_values=x).last_hidden_state[:, self.skip:]
        return F.normalize(t, dim=-1).transpose(1, 2).reshape(1, -1, h, w)


def load_image(path, side, device):
    img = Image.open(path).convert("RGB")
    W0, H0 = img.size
    w = side; h = int(round(side * H0 / W0 / 14)) * 14       # 4:3 -> 840 x 630 = 60 x 45 patches
    x = torch.from_numpy(np.asarray(img.resize((w, h), Image.BICUBIC), np.float32) / 255).permute(2, 0, 1)[None]
    return ((x - MEAN) / STD).to(device), W0, H0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--side", type=int, default=840, help="input width in pixels (multiple of 14)")
    ap.add_argument("--name")
    ap.add_argument("--root", default=os.path.abspath(os.path.join(KP, "..", "..", "chairs-dataset", "renders", "kawalerka_v1")))
    ap.add_argument("--ref-root")
    ap.add_argument("--ref-views", help="comma-separated example photos (default: every view in ref-root)")
    ap.add_argument("--used-only", action="store_true", help="only photos used by the view sets (view_sets.json)")
    ap.add_argument("--export-onnx")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = Backbone(AutoModel.from_pretrained(args.model).eval()).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    if args.export_onnx:
        import onnxruntime as ort
        net = net.cpu()
        x = torch.zeros(1, 3, int(round(args.side * 3 / 4 / 14)) * 14, args.side)
        # the position embedding is resized with antialiased bicubic interpolation, which ONNX lacks: the input
        # size is fixed anyway, so resize it once here and export the result as a constant
        emb = net.m.embeddings
        with torch.no_grad():
            pe = emb.interpolate_pos_encoding(torch.zeros(1, 1 + (x.shape[2] // 14) * (x.shape[3] // 14),
                                                          emb.position_embeddings.shape[-1]), x.shape[2], x.shape[3])
        emb.interpolate_pos_encoding = lambda embeddings, height, width: pe
        torch.onnx.export(net, x, args.export_onnx, input_names=["image"], output_names=["features"],
                          opset_version=17, dynamo=False)
        xr = torch.randn_like(x)
        with torch.no_grad():
            a = net(xr).numpy()
        b = ort.InferenceSession(args.export_onnx, providers=["CPUExecutionProvider"]).run(None, {"image": xr.numpy()})[0]
        print("exported", args.export_onnx, tuple(x.shape), f"{n_params / 1e6:.1f}M params; max |onnx - torch| =",
              float(np.abs(a - b).max()))
        return

    # marked points: features sampled at the true pixel of every point visible in an example photo
    ref_ids = args.ref_views.split(",") if args.ref_views else [v["id"] for v in json.load(open(os.path.join(args.ref_root, "views.json")))]
    feats, owner = [], []                                   # one row per (example photo, point)
    with torch.inference_mode():
        for rid in ref_ids:
            x, W0, H0 = load_image(os.path.join(args.ref_root, rid, "rgb.jpg"), args.side, dev)
            fm = net(x)
            kp = json.load(open(os.path.join(args.ref_root, rid, "kp.json")))
            for n in FURNITURE:
                if n in kp and kp[n]["visible"]:
                    g = torch.tensor([[[[(kp[n]["u"] + 0.5) / W0 * 2 - 1, (kp[n]["v"] + 0.5) / H0 * 2 - 1]]]], device=dev)
                    feats.append(F.normalize(F.grid_sample(fm, g, mode="bilinear", align_corners=False)[0, :, 0, 0], dim=0))
                    owner.append(n)
    R = torch.stack(feats)                                   # (M, C)
    names = sorted(set(owner), key=FURNITURE.index)
    print(f"{len(ref_ids)} example photos, {len(R)} marked points, {len(names)} named points; {n_params / 1e6:.1f}M params", flush=True)

    views = [v["id"] for v in json.load(open(os.path.join(args.root, "views.json")))
             if os.path.exists(os.path.join(args.root, v["id"], "rgb.jpg")) and v["id"] not in ref_ids]
    if args.used_only:
        used = {v for s in json.load(open(os.path.join(KP, "view_sets.json"))) for v in s["views"]}
        views = [v for v in views if v in used]
    out = os.path.join(KP, "preds", "stage1", args.name)
    os.makedirs(out, exist_ok=True)
    times = []
    for vid in views:
        with torch.inference_mode():
            x, W0, H0 = load_image(os.path.join(args.root, vid, "rgb.jpg"), args.side, dev)
            if dev == "cuda": torch.cuda.synchronize()
            t1 = time.perf_counter()
            fm = net(x)
            if dev == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t1) * 1000)
            C, h, w = fm.shape[1:]
            S = torch.einsum("mc,chw->mhw", R, fm[0])         # (M, h, w)
            peak = S.flatten(1).max(1).values
            pts = {}
            for n in names:
                idx = [i for i, o in enumerate(owner) if o == n]
                best = idx[int(torch.argmax(peak[idx]))]
                up = F.interpolate(S[best][None, None], size=(H0, W0), mode="bilinear", align_corners=False)[0, 0]
                k = int(torch.argmax(up))
                pts[n] = [float(k % W0), float(k // W0), float(peak[best])]
        json.dump({"method": args.name, "view": vid, "points_2d": pts, "backbone_ms": times[-1],
                   "examples": ref_ids}, open(os.path.join(out, vid + ".json"), "w"))
    json.dump({"model": args.model, "side": args.side, "params": n_params, "examples": ref_ids,
               "ms_per_photo_median": float(np.median(times[1:] if len(times) > 1 else times)), "device": dev,
               "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None},
              open(os.path.join(out, "_timing.json"), "w"), indent=1)
    print(f"{args.name}: {len(views)} photos, backbone {np.median(times):.1f} ms/photo on {dev}", flush=True)


if __name__ == "__main__":
    main()
