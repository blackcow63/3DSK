"""
NLF (Neural Localizer Fields, Sárándi & Pons-Moll, NeurIPS'24) end-to-end on the kawalerka photos: one
photo + the phone's intrinsics -> metric 3D body points in the camera frame. NLF is queried for SMPL-X, so
its points have exactly the ground-truth definitions (skeleton joints + SMPL-X landmark vertices). The skeleton
joints come from NLF's nonparametric localizer; eyes, ears and nose from the full SMPL-X mesh of its own
parametric fit (the bundled vertex output is a 1024-point subsample). Needs the smplx package and the SMPL-X
model at models/body_models/smplx/SMPLX_NEUTRAL.npz.

  e2e:     preds/e2e/<name>/<view>.json        nonparametric localizer output (NLF's own "any point" path)
           preds/e2e/<name>_param/<view>.json   after its SMPL-X fit
  stage 1: preds/stage1/<name>/<view>.json     the same points projected to 2D

Models (noncommercial research licence) from https://github.com/isarandi/nlf/releases:
  nlf_s_multi_0.2.2.torchscript (EfficientNetV2-S, the phone candidate), nlf_l_multi_0.3.2.torchscript
The "multi" TorchScript bundles a YOLOv8 person detector.

    python ciop/kpstudy/methods/nlf_e2e.py <model.torchscript> <name> [--oracle-box]
--oracle-box: skip the bundled detector and give NLF the person's box from the label image (isolates NLF
from detection; on the phone a light detector would supply the box). Needs a GPU: on CPU the TorchScript
returns NaNs (tested 20.10.2026).
"""
import json, os, sys, time
import numpy as np
import torch
import torchvision  # noqa: F401  (must be imported for the TorchScript model to load)
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
KP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(KP, "..", "..", "chairs-dataset", "renders", "kawalerka_v1"))
JOINTS = {"body_left_shoulder": 16, "body_right_shoulder": 17, "body_left_elbow": 18, "body_right_elbow": 19,
          "body_left_wrist": 20, "body_right_wrist": 21, "body_left_hip": 1, "body_right_hip": 2,
          "body_left_knee": 4, "body_right_knee": 5, "body_left_ankle": 7, "body_right_ankle": 8}
VERTS = {"body_nose": 9120, "body_right_eye": 9929, "body_left_eye": 9448, "body_right_ear": 616, "body_left_ear": 6}


_BM = {}
def smplx_from_params(pred, i):
    """full SMPL-X vertices + joints (metres, camera frame) from NLF's parametric output"""
    import smplx
    pose = pred["pose"][0][i].reshape(-1).float().cpu()[None]
    betas = pred["betas"][0][i].reshape(-1).float().cpu()[None]
    trans = pred["trans"][0][i].reshape(-1).float().cpu()[None]
    if trans.abs().max() > 50:                                         # millimetres -> metres
        trans = trans / 1000
    nb = betas.shape[1]
    if nb not in _BM:
        _BM[nb] = smplx.create(os.path.join(KP, "models", "body_models"), model_type="smplx", gender="neutral", ext="npz",
                               num_betas=nb, use_pca=False, flat_hand_mean=True, batch_size=1)
    with torch.no_grad():
        o = _BM[nb](global_orient=pose[:, :3], body_pose=pose[:, 3:66], jaw_pose=pose[:, 66:69], leye_pose=pose[:, 69:72],
                    reye_pose=pose[:, 72:75], left_hand_pose=pose[:, 75:120], right_hand_pose=pose[:, 120:165],
                    betas=betas, transl=trans, expression=torch.zeros(1, 10))
    return o.vertices[0].numpy(), o.joints[0].numpy()


def main():
    path, name = sys.argv[1], sys.argv[2]
    oracle_box = "--oracle-box" in sys.argv
    if oracle_box:
        name += "_gtbox"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    model = torch.jit.load(path, map_location=dev).eval()
    load_ms = (time.perf_counter() - t0) * 1000
    gt = json.load(open(os.path.join(ROOT, "gt.json")))
    fx, fy, cx, cy = gt["intrinsics"]["K"]
    Km = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32, device=dev)
    views = [v["id"] for v in json.load(open(os.path.join(ROOT, "views.json"))) if os.path.exists(os.path.join(ROOT, v["id"], "rgb.jpg"))]
    outs = {k: os.path.join(KP, "preds", *k.split("/")) for k in (f"e2e/{name}", f"e2e/{name}_param", f"stage1/{name}")}
    for d in outs.values():
        os.makedirs(d, exist_ok=True)
    times, found = [], 0
    for vid in views:
        img = torchvision.io.read_image(os.path.join(ROOT, vid, "rgb.jpg")).to(dev)
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.device(dev):
            if oracle_box:
                lab = cv2.imread(os.path.join(ROOT, vid, "labels.png"), cv2.IMREAD_UNCHANGED)
                ys, xs = np.where(lab == 6)                                   # kpscene.LABELS["person"]
                box = torch.tensor([[xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()]], dtype=torch.float32, device=dev)
                pred = model.estimate_smpl_batched(img.unsqueeze(0), [box], intrinsic_matrix=Km, model_name="smplx")
            else:
                pred = model.detect_smpl_batched(img.unsqueeze(0), intrinsic_matrix=Km, model_name="smplx")
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
        res = {k: {} for k in outs}
        boxes = pred["boxes"][0].cpu().numpy() if "boxes" in pred else None
        n = len(pred["joints3d"][0])
        if n:
            found += 1
            i = int(np.argmax((boxes[:, 2] * boxes[:, 3]) if boxes is not None and len(boxes) == n else np.zeros(n)))
            # the bundled vertex outputs are a 1024-point subsample, so the face landmarks come from the full
            # SMPL-X mesh of NLF's own parametric fit (pose / betas / trans), as in the authors' demo
            Vp, Jp = smplx_from_params(pred, i)
            Jn = pred["joints3d_nonparam"][0][i].cpu().numpy()
            s = 0.001 if np.abs(Jn[:, 2]).mean() > 50 else 1.0             # NLF reports millimetres
            for nm, k in JOINTS.items():
                res[f"e2e/{name}"][nm] = (Jn[k] * s).tolist()           # nonparametric localizer output
                res[f"e2e/{name}_param"][nm] = Jp[k].tolist()
            for nm, k in VERTS.items():
                res[f"e2e/{name}"][nm] = Vp[k].tolist()
                res[f"e2e/{name}_param"][nm] = Vp[k].tolist()
            for nm, p in res[f"e2e/{name}"].items():
                p = np.array(p)
                res[f"stage1/{name}"][nm] = [float(fx * p[0] / p[2] + cx), float(fy * p[1] / p[2] + cy), 1.0]
        for key, d in outs.items():
            field = "points_2d" if key.startswith("stage1") else "points_cam"
            json.dump({"method": os.path.basename(key), "view": vid, field: res[key], "timing_ms": {"infer": dt, "device": str(dev)}},
                      open(os.path.join(d, vid + ".json"), "w"))
        print(f"{vid}: {dt:.0f} ms, {n} people", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    json.dump({"model": os.path.basename(path), "params_incl_detector": n_params, "device": str(dev), "load_ms": load_ms,
               "infer_ms_median": float(np.median(times))}, open(os.path.join(outs[f"e2e/{name}"], "_timing.json"), "w"), indent=1)
    print(f"{name}: person found in {found}/{len(views)}; {n_params / 1e6:.1f} M params incl. detector; median {np.median(times):.0f} ms on {dev}")


if __name__ == "__main__":
    main()
