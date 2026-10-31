"""
Loading the rendered kawalerka dataset (chairs-dataset/renders/kawalerka_v1) for the keypoint study.

Anchor groups:
  FURNITURE  named points on the desk, monitor, keyboard, mouse, A4 sheet, chair, floor and wall
  BODY       17 COCO-style body points of the seated person (SMPL-X joints / surface landmarks)
  ERGO_BODY  ergobench body anchors, derived from BODY points the same way for every method
"""
import json, os
import numpy as np
from PIL import Image
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chairs-dataset", "renders", "kawalerka_v1"))

FURNITURE = ["desk_top_front_left", "desk_top_front_right", "desk_top_back_left", "desk_top_back_right",
             "desk_front_edge_center", "monitor_screen_center", "monitor_top_edge_center", "monitor_top_left",
             "monitor_top_right", "monitor_bottom_left", "monitor_bottom_right", "keyboard_center",
             "keyboard_near_edge_center", "mouse_center", "a4_front_left", "a4_front_right", "a4_back_left",
             "a4_back_right", "seat_top_center", "armrest_top_right", "chair_back_outer", "floor_under_seat",
             "wall_behind_point"]
BODY = ["body_nose", "body_left_eye", "body_right_eye", "body_left_ear", "body_right_ear", "body_left_shoulder",
        "body_right_shoulder", "body_left_elbow", "body_right_elbow", "body_left_wrist", "body_right_wrist",
        "body_left_hip", "body_right_hip", "body_left_knee", "body_right_knee", "body_left_ankle", "body_right_ankle"]
DESK_PLANE = ["desk_top_front_left", "desk_top_front_right", "desk_top_back_left", "desk_top_back_right",
              "desk_front_edge_center", "a4_front_left", "a4_front_right", "a4_back_left", "a4_back_right"]


def derive_ergo_body(P):
    """ergobench body anchors from BODY points (same rule for ground truth and every method)"""
    g = lambda n: np.asarray(P[n]) if n in P and P[n] is not None else None
    out = {}
    pairs = {"shoulder_r": ["body_right_shoulder"], "elbow_r": ["body_right_elbow"], "wrist_r": ["body_right_wrist"],
             "ear_r": ["body_right_ear"], "hip_mid": ["body_left_hip", "body_right_hip"],
             "shoulder_mid": ["body_left_shoulder", "body_right_shoulder"], "eye_center": ["body_left_eye", "body_right_eye"]}
    for k, src in pairs.items():
        v = [g(s) for s in src]
        if all(x is not None for x in v):
            out[k] = np.mean(v, axis=0)
    return out


class Dataset:
    def __init__(self, root=ROOT):
        self.root = root
        self.gt = json.load(open(os.path.join(root, "gt.json")))
        self.views = {v["id"]: v for v in json.load(open(os.path.join(root, "views.json")))}
        self.K = self.gt["intrinsics"]["K"]
        self.W, self.H = self.gt["intrinsics"]["W"], self.gt["intrinsics"]["H"]
        self.P_room = {k: np.asarray(v) for k, v in self.gt["points_room"].items()}

    def ids(self, kind=None):
        return [i for i, v in self.views.items() if (kind is None or v["kind"] == kind)
                and os.path.exists(os.path.join(self.root, i, "kp.json"))]

    def cam(self, vid):
        c = json.load(open(os.path.join(self.root, vid, "cam.json")))
        c["T_wc"] = np.asarray(c["T_wc"])
        return c

    def kp(self, vid):
        return json.load(open(os.path.join(self.root, vid, "kp.json")))

    def image(self, vid, phone=True):
        return np.asarray(Image.open(os.path.join(self.root, vid, "rgb.jpg" if phone else "rgb.png")).convert("RGB"))

    def image_path(self, vid, phone=True):
        return os.path.join(self.root, vid, "rgb.jpg" if phone else "rgb.png")

    def depth(self, vid):
        return cv2.imread(os.path.join(self.root, vid, "depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    def labels(self, vid):
        return cv2.imread(os.path.join(self.root, vid, "labels.png"), cv2.IMREAD_UNCHANGED)

    def gt_cam(self, vid, names=None):
        """ground-truth anchors in the camera frame of view `vid` (metres)"""
        T = self.cam(vid)["T_wc"]
        Tcw = np.linalg.inv(T)
        names = names or list(self.P_room)
        return {n: Tcw[:3, :3] @ self.P_room[n] + Tcw[:3, 3] for n in names}

    def baseline_deg(self, a, b):
        """angle between the two viewing directions toward the workstation centre"""
        c = np.array([-0.15, 0.65, -1.0])
        pa, pb = self.cam(a)["T_wc"][:3, 3], self.cam(b)["T_wc"][:3, 3]
        u, v = pa - c, pb - c
        return float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
