"""
Two-stage pipeline on the kawalerka renders: stage-1 2D points (preds/stage1/<method>/<view>.json) lifted to
3D by the simple geometry in lift.py; every 3D point is scored against ground truth in the camera frame of
the reference view (the first view of a set).

View sets
  single   each ring photo                      lifts: depth_gt (true depth = oracle), desk_plane_a4,
                                                       depth_<model> for every preds/depth/<model>/ found
  pair     two photos at a chosen angle apart    lifts: tri_vio (poses from the phone's tracker, 4 noise
           (5/15/30/60/90 deg)                         levels), sfm_a4 (poses from the images alone + A4 scale)
  multi    3 or 5 photos spread over a sweep     lifts: tri_vio, tri_vio_ransac (wrong answers dropped first)
Rows go to results/lift_<stage1>.jsonl (one per anchor) and results/meas_<stage1>.jsonl (ergonomic
measurements from requirements.py, computed in a gravity-aligned frame from the predicted anchors).

    py ciop/kpstudy/run_lift.py oracle oracle_px3 ...
"""
import itertools, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ergobench"))
from data import Dataset, FURNITURE, BODY, DESK_PLANE, derive_ergo_body
import lift
from ergobench import requirements as REQ

PRED = os.path.join(HERE, "preds")
RES = os.path.join(HERE, "results")
NOISE = [(0.0, 0.0), (0.005, 0.2), (0.01, 0.5), (0.02, 1.0), (0.04, 2.0)]   # VIO pose error per photo: metres, degrees
SEEDS = 3
BINS = [5, 15, 30, 60, 90]
MEAS = [m for r in REQ.REQUIREMENTS for m in r.measurements if m.formula != "angle_vectors"]   # B6 needs directions


def view_sets(ds):
    sets = [dict(kind="single", views=[v]) for v in ds.ids("ring")]
    walks = {}
    for v in ds.ids("walk"):
        walks.setdefault(ds.views[v]["walk"], []).append(v)
    for w in walks.values():
        w.sort(key=lambda v: ds.views[v]["frame"])
    ring = ds.ids("ring")
    cand = []
    for w in walks.values():
        cand += list(itertools.combinations(w, 2))
    for a, b in itertools.combinations(ring, 2):
        va, vb = ds.views[a], ds.views[b]
        if va["radius"] == vb["radius"] and va["height"] == vb["height"]:
            cand.append((a, b))
    base = {p: ds.baseline_deg(*p) for p in cand}
    for b in BINS:
        tol = max(2.0, 0.15 * b)
        near = sorted((p for p in cand if abs(base[p] - b) <= tol), key=lambda p: abs(base[p] - b))
        used, picked = set(), []
        for p in near:                       # spread over different photos
            if p[0] in used or p[1] in used:
                continue
            picked.append(p); used |= set(p)
            if len(picked) == 6:
                break
        sets += [dict(kind="pair", views=list(p), baseline_bin=b, baseline_deg=round(base[p], 1)) for p in picked]
    for w in walks.values():
        for n in (3, 5):
            idx = np.linspace(0, len(w) - 1, n).round().astype(int)
            sets.append(dict(kind="multi", views=[w[i] for i in idx]))
    by_rh = {}
    for v in ring:
        by_rh.setdefault((ds.views[v]["radius"], ds.views[v]["height"]), []).append(v)
    for vs in by_rh.values():
        vs.sort(key=lambda v: ds.views[v]["az_deg"])
        idx = np.linspace(0, len(vs) - 1, 5).round().astype(int)
        sets.append(dict(kind="multi", views=[vs[i] for i in idx]))
    for i, s in enumerate(sets):
        s["set_id"] = f"{s['kind']}{i:03d}"
    return sets


def load_stage1(method, vid):
    p = os.path.join(PRED, "stage1", method, vid + ".json")
    if not os.path.exists(p):
        return None
    return {n: (u, v) for n, (u, v, c) in json.load(open(p))["points_2d"].items()}


def gravity_frame(g_cam, rng, noise_deg=1.0):
    """rotation camera -> a frame with +Z up (yaw arbitrary), from the phone's accelerometer (+-1 deg)"""
    g = np.asarray(g_cam, float)
    if noise_deg:
        g = lift.perturb_pose(np.eye(4), 0, noise_deg, rng)[:3, :3] @ g
    up = -g / np.linalg.norm(g)
    x = np.cross(up, [0, 0, 1.0]); x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
    y = np.cross(up, x)
    return np.stack([x, y, up])          # rows: new axes in camera coordinates


def measurements(P_cam, R_up):
    P = dict(P_cam)
    P.update(derive_ergo_body(P_cam))
    E = {n: (100.0 * R_up @ np.asarray(x)).tolist() for n, x in P.items()}
    return {m.id: REQ.compute_measurement(m, E) for m in MEAS}


def score(rows, mrows, base, pred, gt_cam, kp_ref, n_vis, R_up):
    for n, x in pred.items():
        if n not in gt_cam:
            continue
        rows.append(dict(base, anchor=n, group="body" if n.startswith("body_") else "furniture",
                         visible_ref=kp_ref[n]["visible"], n_visible=n_vis.get(n, 0),
                         err_cm=round(float(np.linalg.norm(np.asarray(x) - gt_cam[n]) * 100), 3),
                         pred=np.round(np.asarray(x), 5).tolist()))
    gm, pm = measurements(gt_cam, R_up), measurements(pred, R_up)
    for m in MEAS:
        g, p = gm[m.id], pm[m.id]
        mrows.append(dict(base, meas=m.id, unit=m.unit, gt=round(g, 3) if np.isfinite(g) else None,
                          pred=round(p, 3) if np.isfinite(p) else None,
                          err=round(p - g, 3) if np.isfinite(g) and np.isfinite(p) else None,
                          verdict_gt=m.evaluate(g), verdict_pred=m.evaluate(p)))


def depth_models():
    d = os.path.join(PRED, "depth")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def run(method, ds, sets):
    rows, mrows = [], []
    rng = np.random.default_rng(11)
    names = FURNITURE + BODY
    for s in sets:
        views = s["views"]
        obs2d = [load_stage1(method, v) for v in views]
        if any(o is None for o in obs2d):
            continue
        ref = views[0]
        cam = ds.cam(ref)
        gt_cam = ds.gt_cam(ref, names)
        kps = [ds.kp(v) for v in views]
        n_vis = {n: sum(k[n]["visible"] for k in kps) for n in names}
        R_up = gravity_frame(cam["gravity_cam"], rng)
        common = dict(stage1=method, set_id=s["set_id"], kind=s["kind"], views=views,
                      baseline_deg=s.get("baseline_deg"))
        if s["kind"] == "single":
            score(rows, mrows, dict(common, lift="depth_gt"), lift.depth_backproject(ds.K, obs2d[0], ds.depth(ref)),
                  gt_cam, kps[0], n_vis, R_up)
            for dm in depth_models():
                p = os.path.join(PRED, "depth", dm, ref + ".npy")
                if os.path.exists(p):
                    X = lift.depth_backproject(ds.K, obs2d[0], np.load(p))
                    score(rows, mrows, dict(common, lift="depth_" + dm), X, gt_cam, kps[0], n_vis, R_up)
                    s_a4 = lift.a4_scale_from_points(X)              # the sheet fixes the model's metric scale
                    if s_a4 is not None:
                        score(rows, mrows, dict(common, lift="depth_" + dm + "_a4", a4_scale=s_a4),
                              lift.scale_points(X, s_a4), gt_cam, kps[0], n_vis, R_up)
            X, info = lift.desk_plane_from_a4(ds.K, obs2d[0], DESK_PLANE, gravity_cam=-R_up[2])   # accelerometer, +-1 deg
            score(rows, mrows, dict(common, lift="desk_plane_a4"), X, gt_cam, kps[0], n_vis, R_up)
            continue
        poses_true = [ds.cam(v)["T_wc"] for v in views]
        obs = {}
        for vi, o in enumerate(obs2d):
            for n, (u, v) in o.items():
                obs.setdefault(n, []).append((vi, u, v))
        for (tm, rd) in NOISE:
            for seed in range(SEEDS if tm else 1):
                prng = np.random.default_rng(1000 * seed + 17)
                poses = [lift.perturb_pose(T, tm, rd, prng) if tm else T for T in poses_true]
                X, _ = lift.triangulate_known_poses(ds.K, poses, obs)
                # result is in the reference camera frame as the phone believes it (its own, perturbed pose)
                score(rows, mrows, dict(common, lift="tri_vio", pose_noise_cm=tm * 100, pose_noise_deg=rd, seed=seed),
                      X, gt_cam, kps[0], n_vis, R_up)
                if s["kind"] == "multi":
                    X, _ = lift.triangulate_ransac(ds.K, poses, obs)
                    score(rows, mrows, dict(common, lift="tri_vio_ransac", pose_noise_cm=tm * 100, pose_noise_deg=rd,
                                            seed=seed), X, gt_cam, kps[0], n_vis, R_up)
        if s["kind"] == "pair":
            X, info = lift.two_view_from_images(ds.K, ds.image(views[0]), ds.image(views[1]), obs2d[0], obs2d[1])
            score(rows, mrows, dict(common, lift="sfm_a4", sfm=info), X, gt_cam, kps[0], n_vis, R_up)
    return rows, mrows


def load_e2e(method, vid):
    p = os.path.join(PRED, "e2e", method, vid + ".json")
    if not os.path.exists(p):
        return None
    return {n: np.asarray(x) for n, x in json.load(open(p))["points_cam"].items()}


def run_e2e(method, ds, sets):
    """end-to-end methods: one photo -> named 3D points in its camera frame; several photos are fused by
    moving each prediction into the room frame with the phone's (perturbed) poses and averaging"""
    rows, mrows = [], []
    rng = np.random.default_rng(11)
    names = FURNITURE + BODY
    for s in sets:
        views = s["views"]
        preds = [load_e2e(method, v) for v in views]
        if any(p is None for p in preds):
            continue
        ref = views[0]
        cam = ds.cam(ref)
        gt_cam = ds.gt_cam(ref, names)
        kps = [ds.kp(v) for v in views]
        n_vis = {n: sum(k[n]["visible"] for k in kps) for n in names}
        R_up = gravity_frame(cam["gravity_cam"], rng)
        common = dict(stage1=method, set_id=s["set_id"], kind=s["kind"], views=views, baseline_deg=s.get("baseline_deg"))
        if s["kind"] == "single":
            score(rows, mrows, dict(common, lift="e2e"), preds[0], gt_cam, kps[0], n_vis, R_up)
            continue
        poses_true = [ds.cam(v)["T_wc"] for v in views]
        for (tm, rd) in NOISE:
            for seed in range(SEEDS if tm else 1):
                prng = np.random.default_rng(1000 * seed + 17)
                poses = [lift.perturb_pose(T, tm, rd, prng) if tm else T for T in poses_true]
                acc = {}
                for T, p in zip(poses, preds):
                    for n, x in p.items():
                        acc.setdefault(n, []).append(T[:3, :3] @ x + T[:3, 3])
                Tcw = np.linalg.inv(poses[0])
                fused = {n: Tcw[:3, :3] @ np.mean(v, 0) + Tcw[:3, 3] for n, v in acc.items()}
                score(rows, mrows, dict(common, lift="e2e_fuse", pose_noise_cm=tm * 100, pose_noise_deg=rd, seed=seed),
                      fused, gt_cam, kps[0], n_vis, R_up)
    return rows, mrows


def main():
    ds = Dataset()
    sets = view_sets(ds)
    json.dump(sets, open(os.path.join(HERE, "view_sets.json"), "w"), indent=0)
    print({k: sum(s["kind"] == k for s in sets) for k in ("single", "pair", "multi")}, "view sets")
    for m in sys.argv[1:]:
        t0 = time.time()
        if m.startswith("e2e:"):
            m = m[4:]
            rows, mrows = run_e2e(m, ds, sets)
            tag = "e2e_" + m
        else:
            rows, mrows = run(m, ds, sets)
            tag = m
        os.makedirs(RES, exist_ok=True)
        with open(os.path.join(RES, f"lift_{tag}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(RES, f"meas_{tag}.jsonl"), "w") as f:
            for r in mrows:
                f.write(json.dumps(r) + "\n")
        print(f"{tag}: {len(rows)} anchor rows, {len(mrows)} measurement rows ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
