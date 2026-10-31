"""
Stage 2 of the two-stage pipeline: lift named 2D points to 3D with the simplest geometry that a phone has.

All functions return points in the CAMERA frame of a reference view (OpenCV axes, metres), because that is
what a phone app actually knows; evaluation compares against ground truth in the same frame.

  triangulate_known_poses   >= 2 photos + camera poses from the phone's motion tracking (VIO), DLT + refinement
  triangulate_ransac        the same, but answers that disagree with the other photos are dropped first
  two_view_from_images      2 photos, no poses: SIFT -> essential matrix -> relative pose -> triangulation,
                            metric scale from the A4 sheet (its 4 corners must be among the 2D points)
  depth_backproject         1 photo + a depth map (monocular depth model, or the rendered depth for an oracle)
  a4_scale_from_points      metric scale from the A4 corners of an up-to-scale reconstruction
  desk_plane_from_a4        1 photo: pose of the desk plane from the A4 corners (PnP), then ray / plane
                            intersection -- valid only for points lying on the desk top
"""
import numpy as np
import cv2

A4_W, A4_D = 0.210, 0.297
MAX_DESK_TILT = 10.0                        # desk-plane self-check against the phone's gravity, degrees
A4_TOL = 1.15                              # sheet self-check: edge ratios within 15 %
MIN_INLIERS = 30                           # relative pose from fewer essential-matrix inliers is not trusted
RANSAC_PX = 40.0                           # robust triangulation: an answer further than this from the consensus is dropped
A4_NAMES = ("a4_front_left", "a4_front_right", "a4_back_right", "a4_back_left")   # ring order


def K_matrix(K):
    fx, fy, cx, cy = K
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def _P(K, T_wc):
    Tcw = np.linalg.inv(T_wc)
    return K_matrix(K) @ Tcw[:3, :]


def triangulate_known_poses(K, poses, obs, refine=True):
    """
    poses: list of T_wc (camera->world) as the phone believes them; obs: {name: [(view_idx, u, v), ...]}
    Returns {name: X_cam_ref} in the frame of poses[0] (the reference view), plus per-point diagnostics.
    """
    Ps = [_P(K, T) for T in poses]
    Tref_cw = np.linalg.inv(poses[0])
    out, diag = {}, {}
    for name, ob in obs.items():
        if len(ob) < 2:
            continue
        A = []
        for vi, u, v in ob:
            P = Ps[vi]
            A.append(u * P[2] - P[0]); A.append(v * P[2] - P[1])
        _, _, Vt = np.linalg.svd(np.array(A))
        X = Vt[-1, :3] / Vt[-1, 3]
        if refine:
            X = _refine_point(X, [(Ps[vi], u, v) for vi, u, v in ob])
        reproj = [np.hypot(*(_proj(Ps[vi], X) - [u, v])) for vi, u, v in ob]
        depths = [(np.linalg.inv(poses[vi]) @ np.r_[X, 1])[2] for vi, _, _ in ob]
        if min(depths) <= 0:
            continue
        out[name] = (Tref_cw[:3, :3] @ X + Tref_cw[:3, 3])
        diag[name] = dict(reproj_px=float(np.mean(reproj)), n_views=len(ob))
    return out, diag


def triangulate_ransac(K, poses, obs, thr_px=RANSAC_PX):
    """
    As triangulate_known_poses, but a pointer's wrong answer in one photo (a swapped corner, a point on the
    wrong object) is dropped: every pair of photos proposes a point, the one that reprojects within thr_px into
    the most photos wins (ties: smaller total error), and the point is re-triangulated from those photos only.
    """
    Ps = [_P(K, T) for T in poses]
    keep = {}
    for name, ob in obs.items():
        if len(ob) < 3:
            keep[name] = ob
            continue
        best = None
        for a in range(len(ob)):
            for b in range(a + 1, len(ob)):
                A = []
                for vi, u, v in (ob[a], ob[b]):
                    A.append(u * Ps[vi][2] - Ps[vi][0]); A.append(v * Ps[vi][2] - Ps[vi][1])
                X = np.linalg.svd(np.array(A))[2][-1]
                if abs(X[3]) < 1e-12:
                    continue
                X = X[:3] / X[3]
                if min((np.linalg.inv(poses[vi]) @ np.r_[X, 1])[2] for vi, _, _ in ob) <= 0:
                    continue
                err = np.array([np.hypot(*(_proj(Ps[vi], X) - [u, v])) for vi, u, v in ob])
                key = (int((err < thr_px).sum()), -float(err[err < thr_px].sum()))
                if best is None or key > best[0]:
                    best = (key, [o for o, e in zip(ob, err) if e < thr_px])
        if best is not None and len(best[1]) >= 2:
            keep[name] = best[1]
    return triangulate_known_poses(K, poses, keep)


def _proj(P, X):
    x = P @ np.r_[X, 1]
    return x[:2] / x[2]


def _refine_point(X, obs, iters=10):
    """Gauss-Newton on reprojection error."""
    for _ in range(iters):
        J, r = [], []
        for P, u, v in obs:
            x = P @ np.r_[X, 1]
            p = x[:2] / x[2]
            r.extend(p - [u, v])
            dx = (P[:2, :3] * x[2] - np.outer(x[:2], P[2, :3])) / x[2] ** 2
            J.append(dx)
        J, r = np.vstack(J), np.array(r)
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        X = X + step
        if np.linalg.norm(step) < 1e-7:
            break
    return X


def perturb_pose(T_wc, trans_sigma_m, rot_sigma_deg, rng):
    """simulated VIO error: random rotation (axis uniform, angle ~ N(0, sigma)) + translation offset"""
    T = np.array(T_wc, float).copy()
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    ang = np.radians(rng.normal(0, rot_sigma_deg))
    R = cv2.Rodrigues(axis * ang)[0]
    T[:3, :3] = R @ T[:3, :3]
    T[:3, 3] = T[:3, 3] + rng.normal(0, trans_sigma_m, 3)
    return T


def sift_matches(img1, img2, ratio=0.75, max_features=8000):
    sift = cv2.SIFT_create(nfeatures=max_features)
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.ndim == 3 else img2
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < ratio * b.distance]
    return (np.float64([k1[g.queryIdx].pt for g in good]), np.float64([k2[g.trainIdx].pt for g in good]))


def two_view_from_images(K, img1, img2, pts1, pts2):
    """
    pts1/pts2: {name: (u, v)} named 2D points in each image (from stage 1). Relative pose from SIFT +
    essential matrix (no poses from the phone), triangulation, then metric scale from the A4 corners.
    Returns ({name: X_cam1}, info) or ({}, info) when the pose or the A4 scale cannot be found.
    """
    Km = K_matrix(K)
    p1, p2 = sift_matches(img1, img2)
    info = dict(n_matches=int(len(p1)))
    if len(p1) < 15:
        return {}, dict(info, fail="too few matches")
    E, mask = cv2.findEssentialMat(p1, p2, Km, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or E.shape != (3, 3):
        return {}, dict(info, fail="no essential matrix")
    n_in, R, t, mask_pose = cv2.recoverPose(E, p1, p2, Km, mask=mask)
    info.update(n_inliers=int(n_in))
    if n_in < MIN_INLIERS:                  # an app would reject the pair and ask for another photo
        return {}, dict(info, fail="too few inliers")
    T1 = np.eye(4)
    T2 = np.eye(4); T2[:3, :3] = R; T2[:3, 3] = t.ravel()                 # cam1 -> cam2 (unit baseline)
    poses = [T1, np.linalg.inv(T2)]                                       # camera -> "world" = cam1
    common = sorted(set(pts1) & set(pts2))
    obs = {n: [(0, *pts1[n]), (1, *pts2[n])] for n in common}
    X, diag = triangulate_known_poses(K, poses, obs)
    s = a4_scale_from_points(X)
    info.update(a4_scale=s)
    if s is None:
        return {}, dict(info, fail="A4 corners not triangulated")
    return {n: x * s for n, x in X.items()}, info


def a4_scale_from_points(X):
    """metric scale for an up-to-scale reconstruction: median ratio of true to measured A4 edge lengths"""
    if not all(n in X for n in A4_NAMES):
        return None
    c = [np.asarray(X[n]) for n in A4_NAMES]
    true = [A4_W, A4_D, A4_W, A4_D]
    meas = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
    if min(meas) <= 0:
        return None
    # self-check an app can do without ground truth: the four corners must look like the sheet (opposite
    # edges equal, long/short = 297/210); a hidden corner lifted onto whatever covers it fails this
    if max(meas[0], meas[2]) / min(meas[0], meas[2]) > A4_TOL or max(meas[1], meas[3]) / min(meas[1], meas[3]) > A4_TOL:
        return None
    if abs((meas[1] + meas[3]) / (meas[0] + meas[2]) / (A4_D / A4_W) - 1) > A4_TOL - 1:
        return None
    return float(np.median(np.array(true) / np.array(meas)))


def depth_backproject(K, pts, depth, z_is_range=False, win=2):
    """
    pts: {name: (u, v)}; depth: (H,W) metres along the optical axis (0/NaN = unknown).
    Takes the nearest surface in a (2*win+1)^2 window: corner points sit on the edge of their object, and
    the window minimum picks the object rather than whatever lies behind the edge.
    """
    fx, fy, cx, cy = K
    H, W = depth.shape
    out = {}
    for n, (u, v) in pts.items():
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        d = depth[max(0, vi - win):vi + win + 1, max(0, ui - win):ui + win + 1].ravel()
        d = d[np.isfinite(d) & (d > 0)]
        if not len(d):
            continue
        z = float(np.min(d))
        x = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        out[n] = x * (z / np.linalg.norm(x) if z_is_range else z)
    return out


def scale_points(X, s):
    return {n: np.asarray(x) * s for n, x in X.items()}


def desk_plane_from_a4(K, pts, names_on_plane, gravity_cam=None, max_tilt_deg=MAX_DESK_TILT):
    """
    1 photo: PnP of the A4 corners (known 29.7 x 21 cm rectangle) gives the desk-plane pose; each named point
    that lies on the desk top is intersected with that plane. Returns ({name: X_cam}, info).
    With the phone's gravity direction, a plane tilted more than max_tilt_deg from horizontal is rejected
    (a desk is level; wrongly placed corners give a tilted plane).
    """
    if not all(n in pts for n in A4_NAMES):
        return {}, dict(fail="A4 corners missing")
    obj = np.array([[0, 0, 0], [A4_W, 0, 0], [A4_W, A4_D, 0], [0, A4_D, 0]], np.float64)   # FL, FR, BR, BL
    img = np.array([pts[n] for n in A4_NAMES], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K_matrix(K), None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return {}, dict(fail="PnP failed")
    R = cv2.Rodrigues(rvec)[0]
    n_cam = R[:, 2]                     # plane normal in camera coordinates
    p0 = tvec.ravel()                   # a point on the plane (A4 front-left corner)
    if gravity_cam is not None:
        g = np.asarray(gravity_cam, float) / np.linalg.norm(gravity_cam)
        tilt = np.degrees(np.arccos(min(1.0, abs(float(n_cam @ g)))))
        if tilt > max_tilt_deg:
            return {}, dict(fail=f"desk plane tilted {tilt:.0f} deg")
    fx, fy, cx, cy = K
    out = {}
    for n in names_on_plane:
        if n not in pts:
            continue
        u, v = pts[n]
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        den = ray @ n_cam
        if abs(den) < 1e-9:
            continue
        s = (p0 @ n_cam) / den
        if s > 0:
            out[n] = ray * s
    return out, dict(plane_normal_cam=n_cam.tolist(), a4_origin_cam=p0.tolist())
