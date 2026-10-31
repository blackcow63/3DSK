# kpstudy — named 3D points from phone photos, tested on the synthetic kawalerka

This is the study behind §6 of `ciop/docs/paper/main.tex`. The full plan is in `C:\Users\zofsm\.claude\plans\korzystaj-c-z-sysntetycznego-modelu-spicy-crab.md`.

It answers two questions about each method: how accurate it is (3D error of named points in cm) and whether it is feasible on a phone (the whole assessment ≤ 60 s, ≤ 2 GB of models, no cloud). Two configurations are compared:

- **E2E** (people only): one model goes from a photo straight to named 3D points. Candidates are NLF, SAM 3D Body, TAIHRI and MediaPipe world landmarks.
- **Two-stage** (people and furniture): stage 1 finds the named point in 2D, using MediaPipe, NLF-2D, SAM 3D Body, MolmoPoint/Molmo2, Qwen3-VL-2B or Moondream 2 (a point named in words), or DINOv2 feature matching (a point marked on an example photo, `methods/dino_match.py`; example photos of a different workstation in `renders/kawalerka_ref_v1`, from `render/render_reference.py`, or of the same one). Sapiens and LocLLM were planned but not run. Stage 2 lifts it to 3D with the simplest geometry a phone has (`lift.py`):
  - triangulation with the phone's tracked poses (`tri_vio`, at four noise levels), and the same after dropping answers that disagree with the other photos (`tri_vio_ransac`, pairs of photos vote, 40 px);
  - poses from the images alone plus scale from the A4 sheet (`sfm_a4`);
  - monocular depth (`depth_<model>`, e.g. MoGe-2);
  - the desk plane from the A4 sheet (`desk_plane_a4`).

  Oracle stage 1 (`oracle`, `oracle_px3`, `oracle_px8`) gives the error of the geometry alone. Oracle depth (`depth_gt`) gives the error of stage 1 alone.

## Data
`chairs-dataset/renders/kawalerka_v1/` holds:
- 36 single photos taken around the workstation;
- 3 hand-held sweeps of 30 frames each;
- a virtual phone camera at 1440×1080 with a 26 mm-equivalent lens;
- per view: depth, instance labels, camera pose, and every anchor's pixel and visibility.

To rebuild it:

```
py chairs-dataset/render/sync_scene.py                       # snapshot of the live scene -> chairs-dataset/scene/
chairs-dataset/smplx_venv/Scripts/python.exe chairs-dataset/render/body_gt.py   # SMPL-X ground truth
py chairs-dataset/render/render_protocols.py                 # ~20 min, software WebGL
py chairs-dataset/render/test_gt.py [--render]                # acceptance tests
```

## Running methods and scoring
```
py ciop/kpstudy/stage1_oracle.py
%LOCALAPPDATA%/ergobench-mp/venv/Scripts/python.exe ciop/kpstudy/methods/mediapipe_pose.py
ciop/kpstudy/.venv/Scripts/python.exe ciop/kpstudy/methods/moge_depth.py vits
ciop/kpstudy/.venv/Scripts/python.exe ciop/kpstudy/methods/nlf_e2e.py ciop/kpstudy/models/nlf_s_multi_0.2.2.torchscript nlf_s
py ciop/kpstudy/methods/vlm_point_llamacpp.py --views all
py chairs-dataset/render/render_reference.py                 # example photos of a different workstation
# on Helios (from $KP3D): Moondream 2, DINOv2 S/B/L matching + ONNX export + Grace timing
sbatch -J kp-moondream ciop/kpstudy/helios/gpu_job.sbatch python methods/moondream_point.py --name moondream2 --used-only
sbatch ciop/kpstudy/helios/dino.sbatch;  sbatch ciop/kpstudy/helios/moondream_grace.sbatch
py ciop/kpstudy/run_lift.py oracle oracle_px3 mediapipe_c1 e2e:mediapipe_c1 e2e:nlf_s ...
py ciop/kpstudy/summarize.py
py ciop/kpstudy/report/make_figure.py --panel "tag=...;lift=...;set=..." --out fig.pdf
```

Predictions go to `preds/{stage1,e2e,depth}/<method>/<view>.json|npy`. Per-anchor and per-measurement rows go to `results/`.
