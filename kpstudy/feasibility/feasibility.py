"""
Phone feasibility of whole ergonomic assessments, from the CPU measurements in bench/ and preds/.

Proxy: a CPU whose single-thread speed is calibrated against a phone. BlazePose GHUM lite / full / heavy run on
one thread (TFLite, XNNPACK) and are compared with the published Pixel 4 single-core times (25 / 40 / 147 ms,
arXiv 2206.11678, Table 2); phone_ms = proxy_ms x factor. Model times are 4-thread medians (a phone has about four
performance cores). A configuration is feasible when it needs no cloud, its models fit in 2 GB, and the whole
assessment takes at most 60 s.

The assessment needs every body anchor (one network pass gives all of them) and the 14 furniture anchors used by
the ergonomic requirements; a text-to-point VLM answers one point per question.

    py ciop/kpstudy/feasibility/feasibility.py [proxy_host]
writes results/feasibility.md and results/feasibility.json
"""
import glob, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
KP = os.path.dirname(HERE)
LIMIT_S, LIMIT_MB = 60.0, 2048.0
N_FURN = 14                         # furniture anchors used by the requirements (requirements.ANCHOR_PRODUCER, CV3.1.3)


def load(host):
    cpu = json.load(open(os.path.join(KP, "bench", f"cpu_{host}.json")))
    nlf = {n: json.load(open(os.path.join(KP, "bench", f"nlf_{n}_{host}.json"))) for n in ("nlf_s", "nlf_l")
           if os.path.exists(os.path.join(KP, "bench", f"nlf_{n}_{host}.json"))}
    return cpu, nlf


def vlm_times(name):
    """(image ms, per-question ms) from a llama.cpp pointing run: first question carries the image"""
    first, rest = [], []
    for f in glob.glob(os.path.join(KP, "preds", "stage1", name, "*.json")):
        t = [v["wall_ms"] for k, v in json.load(open(f))["timing_ms"].items() if k != "_batch"]
        if len(t) > 1:
            first.append(t[0] - statistics.median(t[1:])); rest += t[1:]
    return (statistics.median(first), statistics.median(rest)) if first else (None, None)


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "odkurzacz"
    vlm_run = sys.argv[2] if len(sys.argv) > 2 else "qwen3vl2b_q4_640_t4"
    cpu, nlf = load(host)
    cal = cpu["calibration"]
    # calibration before and after the model timings (laptop CPUs change speed with power / thermal state)
    f = statistics.mean([cal["factor"]] + ([cpu["calibration_end"]["factor"]] if "calibration_end" in cpu else []))
    det = cal["person_detector_t4"]["median_ms"]
    heavy = cal["heavy"]["median_ms"]                               # 1 thread (upper bound)
    comp = {   # name: (ms per photo on the proxy, MB of model files at shipping precision)
        "person detector (BlazePose)": (det, 2.8),
        "MediaPipe Pose heavy": (det + heavy, 2.8 + 26.0),
        "NLF-S": (det + nlf["nlf_s"]["crop_t4"]["median_ms"], 2.8 + nlf["nlf_s"]["crop_model_params_M"] * 4),
        "NLF-L": (det + nlf["nlf_l"]["crop_t4"]["median_ms"], 2.8 + nlf["nlf_l"]["crop_model_params_M"] * 4),
    }
    for m in ("moge2_vits", "moge2_vitb", "moge2_vitl"):
        if m in cpu:
            comp["MoGe-2-" + m[-1].upper()] = (cpu[m]["fp32_t4"]["median_ms"], cpu[m]["fp32_t4"]["model_mb"])
    img_ms, q_ms = vlm_times(vlm_run)
    gguf_mb = (os.path.getsize(os.path.join(KP, "models", "Qwen3VL-2B-Instruct-Q4_K_M.gguf")) +
               os.path.getsize(os.path.join(KP, "models", "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"))) / 2**20
    vlm = (img_ms + N_FURN * q_ms, gguf_mb) if img_ms else None

    configs = [   # (name, photos, components per photo, furniture lift)
        ("people only: NLF-S, 1 photo", 1, ["NLF-S"]),
        ("people only: NLF-S 2D + triangulation, 5 photos", 5, ["NLF-S"]),
        ("people only: NLF-L, 1 photo", 1, ["NLF-L"]),
        ("people only: MediaPipe heavy 2D + triangulation, 5 photos", 5, ["MediaPipe Pose heavy"]),
        ("full: NLF-S + Qwen3-VL-2B@640 + MoGe-2-S depth, 1 photo", 1, ["NLF-S", "VLM", "MoGe-2-S"]),
        ("full: NLF-S + Qwen3-VL-2B@640 + MoGe-2-L depth, 1 photo", 1, ["NLF-S", "VLM", "MoGe-2-L"]),
        ("full: NLF-S + Qwen3-VL-2B@640 + triangulation, 2 photos", 2, ["NLF-S", "VLM"]),
        ("full: NLF-S + Qwen3-VL-2B@640 + triangulation, 5 photos", 5, ["NLF-S", "VLM"]),
    ]
    rows = []
    for name, photos, parts in configs:
        ms, mb, missing = 0.0, 0.0, []
        for p in parts:
            c = vlm if p == "VLM" else comp.get(p)
            if c is None:
                missing.append(p); continue
            ms += photos * c[0]; mb += c[1]
        if missing:
            rows.append(dict(configuration=name, missing=missing)); continue
        phone_s = ms * f / 1000
        rows.append(dict(configuration=name, photos=photos, proxy_s=round(ms / 1000, 1), phone_s=round(phone_s, 1),
                         models_mb=round(mb), feasible=bool(phone_s <= LIMIT_S and mb <= LIMIT_MB)))
    res = {"proxy_host": host, "calibration_factor_phone_over_proxy": f, "vlm_run": vlm_run,
           "vlm_image_ms": img_ms, "vlm_question_ms": q_ms, "n_furniture_questions": N_FURN,
           "components_proxy_ms_mb": comp, "configurations": rows}
    os.makedirs(os.path.join(KP, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(KP, "results", "feasibility.json"), "w"), indent=1)
    lines = [f"Proxy CPU: {host}; phone = proxy x {f:.2f} (BlazePose single-thread vs Pixel 4)", "",
             "| component | proxy ms / photo | models MB |", "|---|---|---|"]
    lines += [f"| {k} | {v[0]:.0f} | {v[1]:.0f} |" for k, v in comp.items()]
    if vlm:
        lines.append(f"| Qwen3-VL-2B Q4 @640 px, {N_FURN} questions | {vlm[0]:.0f} (image {img_ms:.0f} + {q_ms:.0f}/question) | {vlm[1]:.0f} |")
    lines += ["", "| configuration | photos | proxy s | est. phone s | models MB | feasible (<= 60 s, <= 2 GB) |", "|---|---|---|---|---|---|"]
    for r in rows:
        if "missing" in r:
            lines.append(f"| {r['configuration']} | missing {', '.join(r['missing'])} | | | | |")
        else:
            lines.append(f"| {r['configuration']} | {r['photos']} | {r['proxy_s']} | {r['phone_s']} | {r['models_mb']} | {'yes' if r['feasible'] else 'no'} |")
    open(os.path.join(KP, "results", "feasibility.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
