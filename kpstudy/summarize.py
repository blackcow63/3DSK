"""
Summary tables from results/lift_*.jsonl and results/meas_*.jsonl -> results/summary.md (+ printed).

Per configuration (stage-1 method x lifting x view set x pose noise):
  3D error of named points in cm (median / p90), split furniture / body and visible / occluded in the
  reference photo, and coverage (share of anchors the configuration returned at all);
  ergonomic measurements: median |error| and how often the pass/fail verdict matches ground truth.

    py ciop/kpstudy/summarize.py [stage1 ...]
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
from data import FURNITURE, BODY


def load(prefix, only):
    frames = []
    for p in sorted(glob.glob(os.path.join(RES, f"{prefix}_*.jsonl"))):
        tag = os.path.basename(p)[len(prefix) + 1:-6]
        if only and tag not in only:
            continue
        frames.append(pd.read_json(p, lines=True).assign(tag=tag))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def config_cols(df):
    df = df.copy()
    for c in ("pose_noise_cm", "baseline_deg"):
        if c not in df:
            df[c] = np.nan
    df["noise"] = df["pose_noise_cm"].fillna(-1).map(lambda x: "-" if x < 0 else f"{x:g}cm")
    df["bins"] = df.apply(lambda r: f"{len(r['views'])}v" if r["kind"] == "multi" else
                          (f"{r['baseline_deg']:.0f}deg" if r["kind"] == "pair" else "1"), axis=1)
    if "pair" in set(df["kind"]):
        bb = {5: "5", 15: "15", 30: "30", 60: "60", 90: "90"}
        df.loc[df["kind"] == "pair", "bins"] = df.loc[df["kind"] == "pair", "baseline_deg"].map(
            lambda b: min(bb, key=lambda k: abs(k - b))).map(lambda k: f"~{k}deg")
    return df


def md(df):
    """markdown table without the tabulate dependency"""
    def cell(v):
        if isinstance(v, float):
            return "-" if np.isnan(v) else f"{v:.1f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else np.nan


def anchor_table(df, dm):
    """coverage counts every attempted configuration (one measurement block per attempt, also when the
    method returned nothing), so a detector that misses the person lowers coverage instead of vanishing"""
    out = []
    keys = ["tag", "lift", "kind", "bins", "noise"]
    seedcol = lambda g: ["set_id", "seed"] if "seed" in g and g["seed"].notna().any() else ["set_id"]
    attempts = {k: g.groupby(seedcol(g)).ngroups for k, g in dm.groupby(keys, dropna=False)}
    for k, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, k))
        nsets = attempts.get(k, g.groupby(seedcol(g)).ngroups)
        for grp, names in (("furn", FURNITURE), ("body", BODY)):
            gg = g[g["group"] == ("furniture" if grp == "furn" else "body")]
            for vis, lab in ((True, "vis"), (False, "occ")):
                e = gg[gg["visible_ref"] == vis]["err_cm"].values
                row[f"{grp}_{lab}_med"] = q(e, 50); row[f"{grp}_{lab}_p90"] = q(e, 90)
            row[f"{grp}_cov"] = len(gg) / max(1, nsets * len(names))
        row["n_sets"] = nsets
        out.append(row)
    return pd.DataFrame(out)


def meas_table(dm):
    out = []
    keys = ["tag", "lift", "kind", "bins", "noise"]
    for k, g in dm.groupby(keys, dropna=False):
        row = dict(zip(keys, k))
        for mid, gg in g.groupby("meas"):
            e = gg["err"].dropna().abs().values
            ok = gg[gg["verdict_pred"] != "unknown"]
            row[f"{mid}"] = f"{np.median(e):.1f}" if len(e) else "-"
            row[f"{mid}_v"] = f"{(ok['verdict_pred'] == ok['verdict_gt']).mean() * 100:.0f}%" if len(ok) else "-"
        out.append(row)
    return pd.DataFrame(out)


def main():
    only = set(sys.argv[1:])
    da, dm = load("lift", only), load("meas", only)
    if da.empty:
        sys.exit("no results")
    da, dm = config_cols(da), config_cols(dm)
    ta, tm = anchor_table(da, dm), meas_table(dm)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 400)
    fmt = {c: "{:.1f}".format for c in ta.columns if c.endswith(("_med", "_p90"))}
    fmt.update({c: "{:.0%}".format for c in ta.columns if c.endswith("_cov")})
    txt = ["# Keypoint study: 3D error of named points (cm)", "",
           "med/p90 = median / 90th percentile; vis/occ = visible / occluded in the reference photo; cov = share of anchors returned", "",
           md(ta), "",
           "# Ergonomic measurements: median |error| (cm or deg) and verdict agreement (_v)", "",
           md(tm)]
    open(os.path.join(RES, "summary.md"), "w", encoding="utf-8").write("\n".join(txt))
    print(ta.to_string(index=False, formatters=fmt))
    print()
    print(tm.to_string(index=False))


if __name__ == "__main__":
    main()
