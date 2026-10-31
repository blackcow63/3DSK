"""
Oracle stage-1 "methods": the true 2D projection of every anchor that falls inside the frame, optionally
with Gaussian pixel noise. Lifting these isolates the error of stage 2 (the geometry) from stage 1.

    py ciop/kpstudy/stage1_oracle.py
writes ciop/kpstudy/preds/stage1/{oracle, oracle_px3, oracle_px8}/<view>.json
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import Dataset, FURNITURE, BODY

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preds", "stage1")


def main():
    ds = Dataset()
    for name, sigma in (("oracle", 0.0), ("oracle_px3", 3.0), ("oracle_px8", 8.0)):
        rng = np.random.default_rng(7)
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        for vid in ds.ids():
            kp = ds.kp(vid)
            pts = {}
            for n in FURNITURE + BODY:
                a = kp[n]
                if a["in_frame"]:
                    u, v = np.array([a["u"], a["v"]]) + (rng.normal(0, sigma, 2) if sigma else 0)
                    pts[n] = [float(u), float(v), 1.0]
            json.dump({"method": name, "view": vid, "points_2d": pts, "timing_ms": {}}, open(os.path.join(d, vid + ".json"), "w"))
        print(name, "->", d)


if __name__ == "__main__":
    main()
