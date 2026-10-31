# 3D semantic keypoints - eksperymenty

Tylko kod i wyniki eksperymentów (`kpstudy/`). Papier, docs i dane są poza repo.

- `kpstudy/README.md` - opis badania i komendy uruchomienia
- `kpstudy/methods/` - metody etapu 1 (Qwen3-VL, Molmo, Moondream, RoboBrain, NLF, DINOv2, MoGe...)
- `kpstudy/lift.py`, `run_lift.py`, `eval_arkit.py` - podniesienie do 3D i ocena
- `kpstudy/data_razem.py` - wspólny loader 5 źródeł (`dane/kp_razem`)
- `kpstudy/annot/` - narzędzia do anotacji (`annotate*.html`) i same anotacje (json); zdjęcia/chmury pominięte
- `kpstudy/helios/` - zadania slurm na Helios
- `kpstudy/results/` - wyniki per metoda (`lift_*.jsonl`); tabele: `report/tabela_eval.py`

Nie ma w repo: zdjęć i chmur do anotacji (`annot/*/*.jpg|ply|bin`), wag modeli, predykcji (`preds/`), .venv.
