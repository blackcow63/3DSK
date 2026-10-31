"""
SAM 3D Body (Meta, 2025) end-to-end on the kawalerka photos: one photo + the phone's intrinsics -> MHR70 body
keypoints in 3D (camera frame = pred_keypoints_3d + pred_cam_t). Its built-in ViTDet person detector is used
(or --oracle-box: the person's box from the label image). The FOV estimator is bypassed by passing cam_int.

Runs on a GPU from a clone of https://github.com/facebookresearch/sam-3d-body (gated checkpoints on Hugging
Face: request access, then `huggingface-cli login` on Helios):
    cd $KP3D/ext/sam-3d-body && python $KPSTUDY/methods/sam3d_body_e2e.py --repo facebook/sam-3d-body-dinov3 --name sam3db_dinov3
writes preds/e2e/<name>/<view>.json and preds/stage1/<name>/<view>.json (its own 2D keypoints)
"""
import argparse, json, os, sys, time
import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(KP, "..", "..", "chairs-dataset", "renders", "kawalerka_v1"))
# MHR70 indices (sam_3d_body/metadata/mhr70.py) -> our body points; wrists sit after the hand joints
MHR = {"body_nose": 0, "body_left_eye": 1, "body_right_eye": 2, "body_left_ear": 3, "body_right_ear": 4,
       "body_left_shoulder": 5, "body_right_shoulder": 6, "body_left_elbow": 7, "body_right_elbow": 8,
       "body_left_hip": 9, "body_right_hip": 10, "body_left_knee": 11, "body_right_knee": 12,
       "body_left_ankle": 13, "body_right_ankle": 14, "body_right_wrist": 41, "body_left_wrist": 62}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="facebook/sam-3d-body-dinov3")
    ap.add_argument("--name", default="sam3db_dinov3")
    ap.add_argument("--oracle-box", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, os.getcwd())                       # run from the sam-3d-body clone
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    name = args.name + ("_gtbox" if args.oracle_box else "")
    t0 = time.perf_counter()
    ck = os.path.join(os.getcwd(), "checkpoints", args.repo.split("/")[-1])      # downloaded beforehand (gated)
    model, model_cfg = load_sam_3d_body(checkpoint_path=os.path.join(ck, "model.ckpt"), device="cuda",
                                        mhr_path=os.path.join(ck, "assets", "mhr_model.pt"))
    detector = None
    if not args.oracle_box:                                  # ViTDet needs detectron2 (no aarch64 wheel)
        from tools.build_detector import HumanDetector
        detector = HumanDetector(name="vitdet", device="cuda")
    est = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg, human_detector=detector,
                             human_segmentor=None, fov_estimator=None)
    load_ms = (time.perf_counter() - t0) * 1000
    gt = json.load(open(os.path.join(ROOT, "gt.json")))
    fx, fy, cx, cy = gt["intrinsics"]["K"]
    cam_int = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32)
    views = [v["id"] for v in json.load(open(os.path.join(ROOT, "views.json"))) if os.path.exists(os.path.join(ROOT, v["id"], "rgb.jpg"))]
    d3, d2 = os.path.join(KP, "preds", "e2e", name), os.path.join(KP, "preds", "stage1", name)
    os.makedirs(d3, exist_ok=True); os.makedirs(d2, exist_ok=True)
    times = []
    for vid in views:
        img = cv2.cvtColor(cv2.imread(os.path.join(ROOT, vid, "rgb.jpg")), cv2.COLOR_BGR2RGB)
        boxes = None
        if args.oracle_box:
            lab = cv2.imread(os.path.join(ROOT, vid, "labels.png"), cv2.IMREAD_UNCHANGED)
            ys, xs = np.where(lab == 6)
            boxes = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], np.float32)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        outs = est.process_one_image(img, bboxes=boxes, cam_int=cam_int)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t1) * 1000
        times.append(dt)
        p3, p2 = {}, {}
        if outs:
            o = max(outs, key=lambda o: (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]))
            k3 = np.asarray(o["pred_keypoints_3d"]) + np.asarray(o["pred_cam_t"])[None]
            k2 = np.asarray(o["pred_keypoints_2d"])
            for n, i in MHR.items():
                p3[n] = k3[i].tolist(); p2[n] = [float(k2[i, 0]), float(k2[i, 1]), 1.0]
        json.dump({"method": name, "view": vid, "points_cam": p3, "timing_ms": {"gpu": dt}}, open(os.path.join(d3, vid + ".json"), "w"))
        json.dump({"method": name, "view": vid, "points_2d": p2, "timing_ms": {"gpu": dt}}, open(os.path.join(d2, vid + ".json"), "w"))
        print(f"{vid}: {len(outs)} people, {dt:.0f} ms", flush=True)
    n_params = sum(p.numel() for p in est.model.parameters()) if hasattr(est, "model") else None
    json.dump({"repo": args.repo, "params": n_params, "load_ms": load_ms, "gpu_ms_median": float(np.median(times)),
               "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20}, open(os.path.join(d3, "_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
