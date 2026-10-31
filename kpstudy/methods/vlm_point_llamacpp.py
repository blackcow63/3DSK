"""
Text -> 2D point with a small vision-language model run by llama.cpp (the runtime a phone would use):
Qwen3-VL-2B-Instruct, Q4_K_M weights + Q8_0 vision projector. One question per named point, the photo
first so llama-server can reuse the image prefix between questions. Qwen3-VL answers in coordinates
relative to the image, scaled to 0..1000.

    py ciop/kpstudy/methods/vlm_point_llamacpp.py [--views ring|walk|all] [--anchors furniture|body|all]
writes preds/stage1/<name>/<view>.json with points_2d + per-question timings from llama-server
"""
import argparse, base64, json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
KP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(KP, "..", "..", "chairs-dataset", "renders", "kawalerka_v1"))
# the first [x, y] pair in the answer (a bare number regex would catch the "2" of "point_2d")
XY = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")


ITEM = re.compile(r"(\d+)\s*[:.)]\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")


def encode_image(path, max_side):
    """base64 JPEG; optionally downscaled (relative 0-1000 answers map back to the full-size photo unchanged)"""
    if not max_side:
        return base64.b64encode(open(path, "rb").read()).decode()
    from PIL import Image
    import io
    im = Image.open(path).convert("RGB")
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def parse_xy(txt, W, H):
    m = XY.search(txt)
    return [float(m.group(1)) / 1000 * W, float(m.group(2)) / 1000 * H, 1.0] if m else None


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=os.path.join(KP, "tools", "llama_win", "llama-server.exe"))
    ap.add_argument("--model", default=os.path.join(KP, "models", "Qwen3VL-2B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--mmproj", default=os.path.join(KP, "models", "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"))
    ap.add_argument("--name", default="qwen3vl2b_q4")
    ap.add_argument("--views", default="ring", choices=["ring", "walk", "all"])
    ap.add_argument("--anchors", default="all", choices=["furniture", "body", "all"])
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--port", type=int, default=8571)
    ap.add_argument("--limit", type=int, default=0, help="only the first N photos (timing runs)")
    ap.add_argument("--used-only", action="store_true", help="only photos used by the view sets (view_sets.json)")
    ap.add_argument("--max-side", type=int, default=0, help="downscale the photo so its longer side is this (phone setting)")
    ap.add_argument("--batch", action="store_true", help="one question listing every point instead of one per point")
    args = ap.parse_args()

    prompts = json.load(open(os.path.join(KP, "prompts.json"), encoding="utf-8"))
    phrases = {**(prompts["furniture"] if args.anchors in ("furniture", "all") else {}),
               **(prompts["body"] if args.anchors in ("body", "all") else {})}
    gt = json.load(open(os.path.join(ROOT, "gt.json")))
    W, H = gt["intrinsics"]["W"], gt["intrinsics"]["H"]
    views = [v["id"] for v in json.load(open(os.path.join(ROOT, "views.json")))
             if (args.views == "all" or v["kind"] == args.views) and os.path.exists(os.path.join(ROOT, v["id"], "rgb.jpg"))]
    if args.used_only:                      # only photos that some view set of run_lift.py uses
        used = {v for s in json.load(open(os.path.join(KP, "view_sets.json"))) for v in s["views"]}
        views = [v for v in views if v in used]
    if args.limit:
        views = views[:args.limit]
    out = os.path.join(KP, "preds", "stage1", args.name)
    os.makedirs(out, exist_ok=True)

    srv = subprocess.Popen([args.server, "-m", args.model, "--mmproj", args.mmproj, "--port", str(args.port),
                            "-t", str(args.threads), "-c", "8192", "--temp", "0", "-np", "1", "--no-warmup"],
                           stdout=subprocess.DEVNULL, stderr=open(os.path.join(out, "_server.log"), "w"))
    base = f"http://127.0.0.1:{args.port}"
    for _ in range(600):
        try:
            if json.load(urllib.request.urlopen(base + "/health", timeout=5)).get("status") == "ok":
                break
        except Exception:
            time.sleep(1)
    try:
        for vid in views:
            dst = os.path.join(out, vid + ".json")
            if os.path.exists(dst):
                continue
            img = encode_image(os.path.join(ROOT, vid, "rgb.jpg"), args.max_side)
            pts, timing, raw = {}, {}, {}
            if args.batch:                          # all points in one answer, one line per numbered item
                names = list(phrases)
                q = ("Point to each of the following in this image. Coordinates are relative to the image on a "
                     "0-1000 scale. Answer with one line per item, exactly in the form `number: [x, y]`, nothing else.\n" +
                     "\n".join(f"{i + 1}. {phrases[n]}" for i, n in enumerate(names)))
                t0 = time.perf_counter()
                r = post(base + "/v1/chat/completions", {
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}},
                        {"type": "text", "text": q}]}],
                    "temperature": 0, "max_tokens": 20 * len(names), "cache_prompt": True})
                wall = (time.perf_counter() - t0) * 1000
                txt = r["choices"][0]["message"]["content"]
                raw["_batch"] = txt
                for m in ITEM.finditer(txt):
                    i = int(m.group(1)) - 1
                    if 0 <= i < len(names):
                        pts[names[i]] = [float(m.group(2)) / 1000 * W, float(m.group(3)) / 1000 * H, 1.0]
                timing["_batch"] = dict(wall_ms=wall, **{k: r.get("timings", {}).get(k) for k in
                                                         ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "cache_n")})
            for name, phrase in ({} if args.batch else phrases).items():
                q = (prompts["template"].format(phrase=phrase) +
                     ' Answer with JSON only: {"point_2d": [x, y]}, coordinates relative to the image on a 0-1000 scale.')
                t0 = time.perf_counter()
                r = post(base + "/v1/chat/completions", {
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}},
                        {"type": "text", "text": q}]}],
                    "temperature": 0, "max_tokens": 40, "cache_prompt": True})
                wall = (time.perf_counter() - t0) * 1000
                txt = r["choices"][0]["message"]["content"]
                raw[name] = txt
                xy = parse_xy(txt, W, H)
                if xy:
                    pts[name] = xy
                timing[name] = dict(wall_ms=wall, **{k: r.get("timings", {}).get(k) for k in
                                                      ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "cache_n")})
            json.dump({"method": args.name, "view": vid, "points_2d": pts, "timing_ms": timing, "raw": raw}, open(dst, "w"))
            tot = sum(t["wall_ms"] for t in timing.values()) / 1000
            print(f"{vid}: {len(pts)}/{len(phrases)} answered, {tot:.0f} s", flush=True)
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
