# Dane do anotacji - jak je pobrać

W repo są kod, anotacje (`annot/punkty*/`, `annot/punkty_ludzie.json`) i skrypty pobierania (`dane_skrypty/`).
Zdjęć, chmur i modeli obiektów nie ma - trzeba je pobrać (licencje niekomercyjne, dlatego nie w gicie).

## Układ katalogów (skrypty zakładają go na sztywno)
```
<ROOT>/ciop/        <- ten repozytorium (git clone ... ciop)
<ROOT>/dane/        <- dane; skopiuj tu zawartość ciop/kpstudy/dane_skrypty/  (arkitscenes/, av2/)
```
Windows/Git Bash, Python 3 z numpy, scipy, opencv-python, pillow, pandas/pyarrow (AV2 feather); `curl` i `ffmpeg` w PATH.

## 1. ARKitScenes (biurka, 26 scen) - Apple, licencja niekomercyjna
Nic się nie rejestruje, pliki idą po HTTPS z docs-assets.developer.apple.com.
```
cd <ROOT>/dane/arkitscenes
python pobierz.py                  # siatka, trajektoria, intrinsics, adnotacje 3D (lista: biurka_po_wizycie.json)
python klatki.py                   # klatki z video.mov + LiDAR (lista: pokrycie_biurek.json), potem kasuje wideo
cd <ROOT>/ciop/kpstudy/annot
python przygotuj.py                # -> annot/sceny/<id>/ (podglądy + meta.json)
```

## 2. Argoverse 2 Sensor (ulica, 8 logów z `av2/wybrane.txt`) - CC BY-NC-SA 4.0
Publiczny bucket s3://argoverse, bez konta.
```
cd <ROOT>/dane/av2
python pobierz.py meta
python pobierz.py klatki <log> 4   # dla każdego logu z wybrane.txt
cd <ROOT>/ciop/kpstudy/annot
python -P -W ignore przygotuj_av2.py        # -> annot/kontekst/<log>_<track>/
python -P -W ignore dodaj_hd_lidar.py       # zdjęcia HD i lidar.bin do zbliżeń
```

## 3. HOPE i HANDAL (obiekty, BOP) - ręcznie
Ze strony BOP (https://bop.felk.cvut.cz/datasets/) pobierz dla HOPE i HANDAL: modele (`*_models.zip`) oraz zdjęcia
split `val` (`*_base.zip` + paczka val). Rozpakuj do `<ROOT>/dane/bop/hope/` i `<ROOT>/dane/bop/handal/`
(muszą być `models/`, `val/`). Potem:
```
cd <ROOT>/ciop/kpstudy/annot
python przygotuj_obj.py            # -> annot/obiekty/<zbior>_<id>/
```

## 4. 3DPW (ludzie) - ręcznie, rejestracja
Z https://virtualhumans.mpi-inf.mpg.de/3DPW/ pobierz `imageFiles.zip`, `sequenceFiles.zip`, `readme_and_demo.zip`
i rozpakuj do `<ROOT>/dane/3dpw/`. Potrzebne są też modele SMPL v1.0.0 (rejestracja na https://smpl.is.tue.mpg.de/).
```
cd <ROOT>/ciop/kpstudy/ludzie
python -P przygotuj_3dpw.py        # -> annot/ludzie/
```

## 5. Anotowanie
```
cd <ROOT>/ciop/kpstudy/annot
python serwer.py                   # http://127.0.0.1:8765  (/ ARKit, /obiekty, /kontekst, /ludzie)
```
Zapis idzie do `annot/punkty/`, `punkty_obj/`, `punkty_ctx/` i `punkty_ludzie.json` - te pliki commituj.

## 6. Zbiór wspólny do metod i ocen
```
cd <ROOT>/ciop/kpstudy
python data_arkitscenes.py && python -P -W ignore data_razem.py    # -> <ROOT>/dane/kp_razem/
```
Wyniki metod: `results/*.jsonl`, tabele: `report/tabela_eval.py`.
