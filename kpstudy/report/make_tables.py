"""
Tables for the paper from results/*.jsonl, written to report/out/:

  errors.md / errors_tex.tex / errors.csv   appendix: 3D error of named points (cm) for the configurations of
      both tracks -- median and p90 over points visible in the reference photo, median over occluded points,
      coverage (share of points returned) and two ergonomic measurements (median |error|): B3.a elbow angle and
      E3.t trunk flexion (deg) for people, B1.w desk width and C2.d eye-to-screen distance (cm) for furniture
Multi-photo sets use the phone's tracker at 0.5 cm / 0.2 deg per photo.

    py ciop/kpstudy/report/make_tables.py
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KP = os.path.dirname(HERE)
sys.path.insert(0, KP)
from summarize import load, config_cols                     # noqa: E402

OUT = os.path.join(HERE, "out")
NOISE = "0.5cm"
# (track, label, tag, lift, kind, bins) -- bins as in summarize.config_cols
ROWS = [
    ("body", "MediaPipe heavy, 1 photo", "e2e_mediapipe_c2", "e2e", "single", "1"),
    ("body", "NLF-S, 1 photo", "e2e_nlf_s", "e2e", "single", "1"),
    ("body", "NLF-L, 1 photo", "e2e_nlf_l", "e2e", "single", "1"),
    ("body", "TAIHRI-4B, 1 photo", "e2e_taihri4b", "e2e", "single", "1"),
    ("body", "SAM 3D Body, 1 photo (true person box)", "e2e_sam3db_dinov3_gtbox", "e2e", "single", "1"),
    ("body", "NLF-S, 1 photo (true person box)", "e2e_nlf_s_gtbox", "e2e", "single", "1"),
    ("body", "NLF-S, 5 photos fused", "e2e_nlf_s", "e2e_fuse", "multi", "5v"),
    ("body", "MediaPipe 2D + triangulation, 5 photos", "mediapipe_c2", "tri_vio", "multi", "5v"),
    ("body", "NLF-S 2D + triangulation, 5 photos", "nlf_s", "tri_vio", "multi", "5v"),
    ("body", "NLF-S 2D + robust triangulation, 5 photos", "nlf_s", "tri_vio_ransac", "multi", "5v"),
    ("body", "NLF-S 2D + triangulation, 2 photos 90 deg", "nlf_s", "tri_vio", "pair", "~90deg"),
    ("body", "SAM 3D Body 2D + triangulation, 5 photos (true person box)", "sam3db_dinov3_gtbox", "tri_vio", "multi", "5v"),
    ("body", "NLF-S 2D + triangulation, 5 photos (true person box)", "nlf_s_gtbox", "tri_vio", "multi", "5v"),
    ("body", "Qwen3-VL-2B 2D + triangulation, 5 photos", "qwen3vl2b_q4", "tri_vio", "multi", "5v"),
    ("body", "Molmo2-4B 2D + triangulation, 5 photos", "molmo2_4b", "tri_vio", "multi", "5v"),
    ("body", "Moondream 2 2D + triangulation, 5 photos", "moondream2", "tri_vio", "multi", "5v"),
    ("body", "true 2D + triangulation, 5 photos (geometry only)", "oracle", "tri_vio", "multi", "5v"),
    ("furniture", "Qwen3-VL-2B 2D + MoGe-2-S depth, 1 photo", "qwen3vl2b_q4", "depth_moge2_vits", "single", "1"),
    ("furniture", "Qwen3-VL-2B 2D + MoGe-2-L depth, 1 photo", "qwen3vl2b_q4", "depth_moge2_vitl", "single", "1"),
    ("furniture", "Qwen3-VL-2B @640 px 2D + MoGe-2-S depth, 1 photo", "qwen3vl2b_q4_640", "depth_moge2_vits", "single", "1"),
    ("furniture", "Qwen3-VL-2B 2D + triangulation, 5 photos", "qwen3vl2b_q4", "tri_vio", "multi", "5v"),
    ("furniture", "Qwen3-VL-2B @640 px 2D + triangulation, 5 photos", "qwen3vl2b_q4_640", "tri_vio", "multi", "5v"),
    ("furniture", "Molmo2-4B 2D + MoGe-2-L depth, 1 photo", "molmo2_4b", "depth_moge2_vitl", "single", "1"),
    ("furniture", "Molmo2-4B 2D + triangulation, 5 photos", "molmo2_4b", "tri_vio", "multi", "5v"),
    ("furniture", "MolmoPoint-8B 2D + triangulation, 5 photos", "molmopoint8b", "tri_vio", "multi", "5v"),
    ("furniture", "Moondream 2 2D + triangulation, 5 photos", "moondream2", "tri_vio", "multi", "5v"),
    ("furniture", "Qwen3-VL-2B 2D + robust triangulation, 5 photos", "qwen3vl2b_q4", "tri_vio_ransac", "multi", "5v"),
    ("furniture", "Molmo2-4B 2D + robust triangulation, 5 photos", "molmo2_4b", "tri_vio_ransac", "multi", "5v"),
    ("furniture", "DINOv2-S, other desk, 2D + MoGe-2-S depth, 1 photo", "dinov2s_840_other", "depth_moge2_vits", "single", "1"),
    ("furniture", "DINOv2-S, other desk, 2D + triangulation, 5 photos", "dinov2s_840_other", "tri_vio", "multi", "5v"),
    ("furniture", "DINOv2-S, other desk, 2D + robust triangulation, 5 photos", "dinov2s_840_other", "tri_vio_ransac", "multi", "5v"),
    ("furniture", "DINOv2-L, other desk, 2D + robust triangulation, 5 photos", "dinov2l_840_other", "tri_vio_ransac", "multi", "5v"),
    ("furniture", "DINOv2-S, same desk, 2D + triangulation, 5 photos", "dinov2s_840_same", "tri_vio", "multi", "5v"),
    ("furniture", "DINOv2-S, same desk, 2D + robust triangulation, 5 photos", "dinov2s_840_same", "tri_vio_ransac", "multi", "5v"),
    ("furniture", "whole assessment: NLF-S + DINOv2-S, other desk, robust triangulation, 5 photos", "full_nlfs_dinov2s",
     "tri_vio_ransac", "multi", "5v"),
    ("furniture", "true 2D + MoGe-2-S depth, 1 photo", "oracle", "depth_moge2_vits", "single", "1"),
    ("furniture", "true 2D + MoGe-2-L depth, 1 photo", "oracle", "depth_moge2_vitl", "single", "1"),
    ("furniture", "true 2D + MoGe-2-L depth + A4 scale, 1 photo", "oracle", "depth_moge2_vitl_a4", "single", "1"),
    ("furniture", "true 2D + desk plane from A4, 1 photo", "oracle", "desk_plane_a4", "single", "1"),
    ("furniture", "true 2D + poses from images + A4 scale, 2 photos 15 deg", "oracle", "sfm_a4", "pair", "~15deg"),
    ("furniture", "true 2D + triangulation, 5 photos (geometry only)", "oracle", "tri_vio", "multi", "5v"),
]
MEAS = {"body": ("B3.a", "E3.t"), "furniture": ("B1.w", "C2.d")}


def num(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1f}"


def main():
    tags = sorted({r[2] for r in ROWS})
    da, dm = load("lift", set(tags)), load("meas", set(tags))
    da, dm = config_cols(da), config_cols(dm)
    out = []
    for track, label, tag, lift, kind, bins in ROWS:
        sel = (da["tag"] == tag) & (da["lift"] == lift) & (da["kind"] == kind) & (da["bins"] == bins)
        msel = (dm["tag"] == tag) & (dm["lift"] == lift) & (dm["kind"] == kind) & (dm["bins"] == bins)
        if lift in ("tri_vio", "tri_vio_ransac", "e2e_fuse"):   # configurations that use the phone's poses
            sel &= da["noise"] == NOISE
            msel &= dm["noise"] == NOISE
        g = da[sel & (da["group"] == track)]
        m = dm[msel]
        n_att = m.groupby(["set_id", "seed"] if m["seed"].notna().any() else ["set_id"]).ngroups if len(m) else 0
        vis, occ = g[g["visible_ref"] == True]["err_cm"].values, g[g["visible_ref"] == False]["err_cm"].values  # noqa: E712
        row = dict(track=track, configuration=label,
                   median_cm=float(np.median(vis)) if len(vis) else np.nan,
                   p90_cm=float(np.percentile(vis, 90)) if len(vis) else np.nan,
                   occluded_median_cm=float(np.median(occ)) if len(occ) else np.nan,
                   coverage=len(g) / (n_att * (17 if track == "body" else 23)) if n_att else 0.0)
        for mid in MEAS[track]:
            e = m[m["meas"] == mid]["err"].dropna().abs().values
            row[mid] = float(np.median(e)) if len(e) else np.nan
        out.append(row)
    df = pd.DataFrame(out)
    os.makedirs(OUT, exist_ok=True)
    lines, tex = [], []
    for track in ("body", "furniture"):
        t = df[df["track"] == track].drop(columns="track").dropna(axis=1, how="all")
        cols = list(t.columns)
        lines += [f"## {track}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        tex.append(r"\multicolumn{7}{l}{\emph{" + ("People (body points)" if track == "body" else "Furniture points") + r"}}\\")
        for r in t.to_dict("records"):
            cells = [r["configuration"]] + [f"{r[c]:.0%}" if c == "coverage" else num(r[c]) for c in cols[1:]]
            lines.append("| " + " | ".join(cells) + " |")
            tex.append(" & ".join("--" if c == "-" else c.replace("%", r"\%").replace("~", r"$\sim$") for c in cells) + r" \\")
        lines.append("")
    open(os.path.join(OUT, "errors.md"), "w", encoding="utf-8").write("\n".join(lines))
    open(os.path.join(OUT, "errors_tex.tex"), "w", encoding="utf-8").write("\n".join(tex))
    df.to_csv(os.path.join(OUT, "errors.csv"), index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
