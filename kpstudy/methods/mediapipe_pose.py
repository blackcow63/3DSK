"""
MediaPipe Pose (BlazePose lite / full / heavy) on the kawalerka photos -- the phone-native reference.

  stage 1 (2D):  33 landmarks in pixels -> BODY points            preds/stage1/mediapipe_c<k>/<view>.json
  end-to-end:    world landmarks (metres, hip-centred, camera-aligned axes) placed in the camera frame
                 by fitting the translation that best reprojects them onto the 2D landmarks
                                                                 preds/e2e/mediapipe_c<k>/<view>.json
Runs in the MediaPipe venv (mediapipe==0.10.14 keeps the legacy solutions API):
    %LOCALAPPDATA%/ergobench-mp/venv/Scripts/python.exe ciop/kpstudy/methods/mediapipe_pose.py
"""
import json, os, sys, time
import numpy as np
import cv2
import mediapipe as mp

HERE = os.path.dirname(os.path.abspath(__file__))
KP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(KP, "..", "..", "chairs-dataset", "renders", "kawalerka_v1"))
MAP = {"body_nose": 0, "body_left_eye": 2, "body_right_eye": 5, "body_left_ear": 7, "body_right_ear": 8,
       "body_left_shoulder": 11, "body_right_shoulder": 12, "body_left_elbow": 13, "body_right_elbow": 14,
       "body_left_wrist": 15, "body_right_wrist": 16, "body_left_hip": 23, "body_right_hip": 24,
       "body_left_knee": 25, "body_right_knee": 26, "body_left_ankle": 27, "body_right_ankle": 28}


def fit_translation(Pw, uv, w, K):
    """t minimising the weighted reprojection error of Pw + t (Gauss-Newton)."""
    fx, fy, cx, cy = K
    ext3 = np.ptp(Pw[:, :2], axis=0).max()
    ext2 = np.ptp(uv, axis=0).max() / fx
    t = np.array([0.0, 0.0, max(0.5, ext3 / max(ext2, 1e-6))])
    t[:2] = ((uv - [cx, cy]) / [fx, fy] * t[2]).mean(0) - Pw[:, :2].mean(0)
    for _ in range(30):
        P = Pw + t
        r = np.concatenate([(fx * P[:, 0] / P[:, 2] + cx - uv[:, 0]) * w, (fy * P[:, 1] / P[:, 2] + cy - uv[:, 1]) * w])
        J = np.zeros((2 * len(P), 3))
        J[:len(P), 0] = fx / P[:, 2] * w; J[:len(P), 2] = -fx * P[:, 0] / P[:, 2] ** 2 * w
        J[len(P):, 1] = fy / P[:, 2] * w; J[len(P):, 2] = -fy * P[:, 1] / P[:, 2] ** 2 * w
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        t += step
        if np.linalg.norm(step) < 1e-6:
            break
    return t


def main():
    gt = json.load(open(os.path.join(ROOT, "gt.json")))
    K, W, H = gt["intrinsics"]["K"], gt["intrinsics"]["W"], gt["intrinsics"]["H"]
    views = [v["id"] for v in json.load(open(os.path.join(ROOT, "views.json"))) if os.path.exists(os.path.join(ROOT, v["id"], "rgb.jpg"))]
    for c in (0, 1, 2):
        name = f"mediapipe_c{c}"
        d2, d3 = os.path.join(KP, "preds", "stage1", name), os.path.join(KP, "preds", "e2e", name)
        os.makedirs(d2, exist_ok=True); os.makedirs(d3, exist_ok=True)
        times, found = [], 0
        with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=c, enable_segmentation=False) as pose:
            for vid in views:
                img = cv2.cvtColor(cv2.imread(os.path.join(ROOT, vid, "rgb.jpg")), cv2.COLOR_BGR2RGB)
                t0 = time.perf_counter()
                res = pose.process(img)
                dt = (time.perf_counter() - t0) * 1000
                times.append(dt)
                p2, p3 = {}, {}
                if res.pose_landmarks:
                    found += 1
                    lm, wl = res.pose_landmarks.landmark, res.pose_world_landmarks.landmark
                    uv = np.array([[l.x * W, l.y * H] for l in lm]); vis = np.array([l.visibility for l in lm])
                    Pw = np.array([[l.x, l.y, l.z] for l in wl])
                    for n, i in MAP.items():
                        p2[n] = [float(uv[i, 0]), float(uv[i, 1]), float(vis[i])]
                    sel = vis > 0.5
                    if sel.sum() >= 4:
                        t = fit_translation(Pw[sel], uv[sel], np.sqrt(vis[sel]), K)
                        for n, i in MAP.items():
                            p3[n] = (Pw[i] + t).tolist()
                json.dump({"method": name, "view": vid, "points_2d": p2, "timing_ms": {"infer_cpu_x86": dt}}, open(os.path.join(d2, vid + ".json"), "w"))
                json.dump({"method": name, "view": vid, "points_cam": p3, "timing_ms": {"infer_cpu_x86": dt},
                           "note": "world landmarks + fitted translation"}, open(os.path.join(d3, vid + ".json"), "w"))
        print(f"{name}: person found in {found}/{len(views)} photos, median {np.median(times):.0f} ms/photo (this PC, CPU)")


if __name__ == "__main__":
    main()
