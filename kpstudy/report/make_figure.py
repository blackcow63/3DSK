"""
Qualitative figure for the paper: reference photos with ground truth (circle) and the prediction of a
configuration (cross), joined by a line and labelled with the 3D error in cm.

Each panel = one configuration on one view set; the prediction (3D, reference camera frame) is projected
into the reference photo, so a long line can mean a wrong pixel OR a wrong depth -- the label tells which
(3D error in cm), the line only shows the image-plane part.

    py ciop/kpstudy/report/make_figure.py --panel "tag=e2e_nlf_s;lift=e2e;set=single000" \
        --panel "tag=oracle;lift=tri_vio;set=pair040;noise=1" ... --cols 3 --out fig.pdf [--group body]
    Several configurations on one photo: --panel "tag=e2e_nlf_s;lift=e2e;set=single034;group=body|tag=..."
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
KP = os.path.dirname(HERE)
sys.path.insert(0, KP)
from data import Dataset

SHORT = {"desk_top_front_left": "desk FL", "desk_top_front_right": "desk FR", "desk_top_back_left": "desk BL",
         "desk_top_back_right": "desk BR", "desk_front_edge_center": "desk edge", "monitor_screen_center": "screen",
         "monitor_top_edge_center": "mon. top", "monitor_top_left": "mon. TL", "monitor_top_right": "mon. TR",
         "monitor_bottom_left": "mon. BL", "monitor_bottom_right": "mon. BR", "keyboard_center": "keyb.",
         "keyboard_near_edge_center": "keyb. edge", "mouse_center": "mouse", "seat_top_center": "seat",
         "armrest_top_right": "armrest", "chair_back_outer": "backrest", "floor_under_seat": "floor",
         "wall_behind_point": "wall"}


def parse(spec):
    d = dict(kv.split("=", 1) for kv in spec.split(";"))
    return d


def rows_for(spec):
    tag = spec["tag"]
    rows = []
    for line in open(os.path.join(KP, "results", f"lift_{tag}.jsonl")):
        r = json.loads(line)
        if r["lift"] != spec["lift"] or r["set_id"] != spec["set"]:
            continue
        if "noise" in spec and float(r.get("pose_noise_cm") or 0) != float(spec["noise"]):
            continue
        if int(r.get("seed") or 0) != int(spec.get("seed", 0)):
            continue
        rows.append(r)
    return rows


# points shown in the paper figure (all points are scored; these keep the panels readable)
SHOW = {"body": ["body_nose", "body_left_shoulder", "body_right_shoulder", "body_left_elbow", "body_right_elbow",
                 "body_left_wrist", "body_right_wrist", "body_left_hip", "body_right_hip", "body_left_knee",
                 "body_right_knee", "body_left_ankle", "body_right_ankle"],
        "furniture": ["desk_top_front_left", "desk_top_front_right", "desk_top_back_left", "desk_top_back_right",
                      "monitor_screen_center", "monitor_top_left", "monitor_top_right", "monitor_bottom_left",
                      "monitor_bottom_right", "keyboard_center", "mouse_center", "seat_top_center", "armrest_top_right"]}


def panel(ax, ds, specs, group):
    """specs: one or more configurations drawn on the same reference photo (e.g. people from one method and
    furniture from another); a spec may restrict itself with group=body or group=furniture"""
    spec = specs[0]
    rows, scored = [], []
    for s in specs:
        g = s.get("group", group)
        keep = SHOW["body"] + SHOW["furniture"] if g == "all" else SHOW[g]
        rs = rows_for(s)
        rows += [r for r in rs if r["anchor"] in keep]
        scored += [r for r in rs if (g == "all" or r["group"] == g) and r["visible_ref"]]   # as in the tables
    if not rows:
        ax.set_title(f"{spec.get('title', spec['tag'])}: no output", fontsize=9); ax.axis("off"); return
    assert len({r["views"][0] for r in rows}) == 1, "all configurations of a panel must share the reference photo"
    ref = rows[0]["views"][0]
    img = np.asarray(Image.open(ds.image_path(ref)).convert("RGB"))
    kp = ds.kp(ref)
    fx, fy, cx, cy = ds.K
    gt_uv = np.array([[kp[r["anchor"]]["u"], kp[r["anchor"]]["v"]] for r in rows])
    P = np.array([r["pred"] for r in rows])
    pr_uv = np.stack([fx * P[:, 0] / P[:, 2] + cx, fy * P[:, 1] / P[:, 2] + cy], 1)
    pts = np.vstack([gt_uv, pr_uv])
    pts = pts[np.all(np.isfinite(pts), 1)]
    x0, y0 = np.clip(pts.min(0) - 60, 0, None)
    x1, y1 = np.minimum(pts.max(0) + 60, [ds.W, ds.H])
    ax.imshow(img)
    for r, g, p in zip(rows, gt_uv, pr_uv):
        col = "#1b9e77" if r["err_cm"] < 5 else ("#d95f02" if r["err_cm"] < 15 else "#c0392b")
        ax.plot([g[0], p[0]], [g[1], p[1]], "-", color=col, lw=0.8)
        ax.plot(*g, "o", mfc="none", mec="white", ms=5, mew=1.0)
        ax.plot(*p, "x", color=col, ms=4, mew=1.2)
        lab = SHORT.get(r["anchor"], r["anchor"].replace("body_", "").replace("left_", "L ").replace("right_", "R "))
        ax.annotate(f"{lab} {r['err_cm']:.0f}", p, fontsize=5.5, color="white", xytext=(3, -3), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.1", fc=col, ec="none", alpha=0.75))
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    meds = []                                   # over every scored point visible in the photo, not only those drawn
    for grp, lab in (("body", "people"), ("furniture", "furniture")):
        e = [r["err_cm"] for r in scored if r["group"] == grp]
        if e:
            meds.append(f"{lab} {np.median(e):.0f} cm")
    ax.set_title(f"{spec.get('title', spec['tag'])}\nmedian 3D error: " + ", ".join(meds), fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--group", default="all", choices=["all", "body", "furniture"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ds = Dataset()
    specs = [[parse(s) for s in p.split("|")] for p in args.panel]
    n = len(specs); cols = min(args.cols, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.7 * rows), squeeze=False)
    for ax, s in zip(axes.ravel(), specs):
        panel(ax, ds, s, args.group)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    plt.tight_layout(pad=0.3)
    plt.savefig(args.out, dpi=250)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
