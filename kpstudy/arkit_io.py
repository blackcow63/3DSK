# -*- coding: utf-8 -*-
"""Czytanie nagrań ARKitScenes pobranych przez dane/arkitscenes/pobierz.py.

Nagranie (dane/arkitscenes/raw/<video_id>/):
  upsampling/<id>/wide/<id>_<ts>.png           klatki 1920x1440 z kamery szerokiej iPada
  upsampling/<id>/highres_depth/<id>_<ts>.png  głębia z lasera Faro wyrównana do klatki, uint16, mm
  lowres_wide.traj                             pozy ARKit (VIO): ts, oś-kąt (3), przesunięcie (3), świat -> kamera
  lowres_wide_intrinsics/**/<id>_<ts>.pincam   w h fx fy cx cy dla 256x192; ta sama kamera co wide
  mesh.ply                                     siatka ARKit sceny, w tym samym układzie świata co pozy
  annotation.json                              pudełka 3D obiektów (obbAligned, oś z w górę)

Konwencje jak w kodzie ARKitScenes (threedod/benchmark_scripts/utils/tenFpsDataLoader.py):
oś-kąt i przesunięcie z .traj to przekształcenie świat -> kamera; kamera jak w OpenCV.
"""
import glob
import os

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dane', 'arkitscenes', 'raw')
WIDE = (1920, 1440)


def read_ply(path):
    """Binarna siatka z VCGLIB: wierzchołki (x, y, z, r, g, b, a), ściany trójkątne."""
    with open(path, 'rb') as f:
        head = b''
        while not head.endswith(b'end_header\n'):
            head += f.readline()
        h = head.decode('ascii')
        nv = int(h.split('element vertex ')[1].split()[0])
        nf = int(h.split('element face ')[1].split()[0])
        vt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')])
        v = np.frombuffer(f.read(nv * vt.itemsize), dtype=vt)
        ft = np.dtype([('n', 'u1'), ('i', '<i4', (3,))])
        fc = np.frombuffer(f.read(nf * ft.itemsize), dtype=ft)
    assert (fc['n'] == 3).all(), 'siatka ma ściany inne niż trójkąty'
    xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)
    rgb = np.stack([v['r'], v['g'], v['b']], 1)
    return xyz, rgb, fc['i'].astype(np.int64)


def read_traj(path):
    """Zwraca (ts[N], T_cw[N,4,4]) – przekształcenie świat -> kamera."""
    ts, T = [], []
    for line in open(path):
        p = line.split()
        if len(p) != 7:
            continue
        R, _ = cv2.Rodrigues(np.array([float(x) for x in p[1:4]]))
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = [float(x) for x in p[4:7]]
        ts.append(float(p[0]))
        T.append(M)
    return np.array(ts), np.array(T)


def read_intrinsics(scene):
    """ts -> K dla klatki wide (przeskalowane z lowres_wide 256x192)."""
    out = {}
    for p in glob.glob(os.path.join(scene, 'lowres_wide_intrinsics', '**', '*.pincam'), recursive=True):
        w, h, fx, fy, cx, cy = [float(x) for x in open(p).read().split()]
        s = WIDE[0] / w
        out[float(os.path.basename(p)[:-7].split('_')[1])] = np.array(
            [[fx * s, 0, (cx + 0.5) * s - 0.5], [0, fy * s, (cy + 0.5) * s - 0.5], [0, 0, 1]])
    return out


def nearest(keys, t):
    k = np.asarray(keys)
    i = int(np.argmin(np.abs(k - t)))
    return i, abs(k[i] - t)


def sky_direction(video_id):
    """Kierunek nieba na klatce (Up/Down/Left/Right) z metadanych ARKitScenes."""
    import csv
    meta = os.path.join(ROOT, '..', 'raw_metadata.csv')
    for r in csv.DictReader(open(meta)):
        if r['video_id'] == str(video_id):
            return r['sky_direction']
    return 'Up'


# Obrót obrazu tak, żeby niebo było u góry, i odpowiadający mu obrót kamery wokół osi optycznej.
# Dla obrotu o 90° w prawo piksel (u, v) przechodzi w (H-1-v, u), a współrzędne kamery
# (x, y, z) w (-y, x, z); stąd K' i R_rot. Sprawdzone rzutem punktów siatki (test_arkit_rot).
def upright(rgb, K, T_cw, sky):
    H, W = rgb.shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if sky == 'Up':
        return rgb, K.copy(), T_cw.copy()
    if sky == 'Left':      # niebo po lewej -> obrót o 90° w prawo
        img = np.rot90(rgb, k=-1)
        K2 = np.array([[fy, 0, H - 1 - cy], [0, fx, cx], [0, 0, 1.0]])
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    elif sky == 'Right':   # niebo po prawej -> obrót o 90° w lewo
        img = np.rot90(rgb, k=1)
        K2 = np.array([[fy, 0, cy], [0, fx, W - 1 - cx], [0, 0, 1.0]])
        R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])
    else:                  # Down -> 180°
        img = np.rot90(rgb, k=2)
        K2 = np.array([[fx, 0, W - 1 - cx], [0, fy, H - 1 - cy], [0, 0, 1.0]])
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1.0]])
    M = np.eye(4)
    M[:3, :3] = R
    return np.ascontiguousarray(img), K2, M @ T_cw


class Scene:
    def __init__(self, video_id, root=ROOT):
        self.id = str(video_id)
        self.dir = os.path.join(root, self.id)
        self.sky = sky_direction(self.id)
        self.ts_traj, self.T_cw = read_traj(os.path.join(self.dir, 'lowres_wide.traj'))
        self.K_by_ts = read_intrinsics(self.dir)
        self.frames = sorted(glob.glob(os.path.join(self.dir, 'upsampling', '*', 'wide', '*.png')))

    def ts(self, frame):
        return float(os.path.basename(frame)[:-4].split('_')[1])

    def pose(self, frame, interpoluj=True):
        """T_cw dla klatki i odstęp czasu do najbliższej pozy z ARKit (s). Pozy są co ok. 0,08 s,
        więc domyślnie interpolujemy między sąsiednimi: obrót sferycznie, przesunięcie liniowo."""
        return self.pose_at(self.ts(frame), interpoluj)

    def pose_at(self, t, interpoluj=True):
        i, dt = nearest(self.ts_traj, t)
        if not interpoluj or dt < 1e-4:
            return self.T_cw[i], dt
        j = i + 1 if self.ts_traj[i] < t else i - 1
        if j < 0 or j >= len(self.ts_traj):
            return self.T_cw[i], dt
        a, b = sorted((i, j))
        w = (t - self.ts_traj[a]) / (self.ts_traj[b] - self.ts_traj[a])
        Ta, Tb = np.linalg.inv(self.T_cw[a]), np.linalg.inv(self.T_cw[b])   # kamera -> świat
        from scipy.spatial.transform import Rotation, Slerp
        R = Slerp([0, 1], Rotation.from_matrix([Ta[:3, :3], Tb[:3, :3]]))([w]).as_matrix()[0]
        T_wc = np.eye(4)
        T_wc[:3, :3] = R
        T_wc[:3, 3] = (1 - w) * Ta[:3, 3] + w * Tb[:3, 3]
        return np.linalg.inv(T_wc), dt

    def K(self, frame):
        return self.K_at(self.ts(frame))

    def K_at(self, t):
        keys = sorted(self.K_by_ts)
        i, dt = nearest(keys, t)
        return self.K_by_ts[keys[i]], dt

    def video_ts(self):
        """Znacznik czasu klatki n pliku video.mov: ts[n + 1] z listy intrinsics (60 kl./s).
        Sprawdzone: klatka n wideo (ffmpeg -noautorotate) jest identyczna z klatką wide o tym ts."""
        return sorted(self.K_by_ts)[1:]

    def rgb(self, frame):
        return np.array(Image.open(frame).convert('RGB'))

    def depth(self, frame):
        """Głębia z lasera w metrach, 0 = brak."""
        p = frame.replace(os.sep + 'wide' + os.sep, os.sep + 'highres_depth' + os.sep)
        return np.array(Image.open(p)).astype(np.float64) / 1000.0

    def mesh(self):
        return read_ply(os.path.join(self.dir, 'mesh.ply'))


def project(X, T_cw, K):
    """Punkty świata (N,3) -> piksele (N,2) i głębokość z w kamerze (N,)."""
    Xc = (T_cw[:3, :3] @ X.T + T_cw[:3, 3:4]).T
    z = Xc[:, 2]
    uv = (K @ Xc.T).T
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = uv[:, :2] / uv[:, 2:3]
    return uv, z


def render_depth(X, T_cw, K, size=WIDE, scale=4):
    """Przybliżona głębia siatki: najbliższy wierzchołek na piksel, w rozdzielczości 1/scale."""
    w, h = size[0] // scale, size[1] // scale
    uv, z = project(X, T_cw, K)
    uv = uv / scale
    ok = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    u, v, zz = uv[ok, 0].astype(int), uv[ok, 1].astype(int), z[ok]
    D = np.full((h, w), np.inf)
    np.minimum.at(D, (v, u), zz)
    D[np.isinf(D)] = 0
    return D


def backproject(D, K, T_cw, step=8):
    """Głębia (m) -> punkty świata (N,3), co step pikseli."""
    h, w = D.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = D[v, u]
    ok = z > 0
    u, v, z = u[ok], v[ok], z[ok]
    Xc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    T_wc = np.linalg.inv(T_cw)
    return (T_wc[:3, :3] @ Xc.T + T_wc[:3, 3:4]).T
