# Dane do anotacji - jak je pobrać

W repo są kod, anotacje (`annot/punkty*/`, `annot/punkty_ludzie.json`) i skrypty pobierania (`dane_skrypty/`).
Zdjęć, chmur i modeli obiektów nie ma - trzeba je pobrać (licencje niekomercyjne, dlatego nie w gicie).

## Gdzie lądują dane
Domyślnie w `dane/` obok `kpstudy/` (czyli `<repo>/dane/`, git go ignoruje). Inne miejsce: ustaw zmienną `KP_DANE`
(np. `export KP_DANE=/dysk/dane`). Repo można sklonować pod dowolną nazwą. Skrypty pobierania są w `kpstudy/dane_skrypty/`
i uruchamia się je stamtąd, zapisują do `$KP_DANE`.

Wszystkie polecenia poniżej wykonuj w katalogu `kpstudy/` repo. Python 3 z numpy, scipy, opencv-python, pillow, pandas, pyarrow;
`curl` i `ffmpeg` w PATH.

## 1. ARKitScenes (biurka, 26 scen) - Apple, licencja niekomercyjna
Nic się nie rejestruje, pliki idą po HTTPS z docs-assets.developer.apple.com.
```
cd dane_skrypty/arkitscenes
python pobierz.py                  # siatka, trajektoria, intrinsics, adnotacje 3D (lista: biurka_po_wizycie.json) -> $KP_DANE/arkitscenes/raw
python klatki.py                   # klatki z video.mov + LiDAR (lista: pokrycie_biurek.json), potem kasuje wideo
cd ../../annot
python przygotuj.py                # -> annot/sceny/<id>/ (podglądy + meta.json)
```

## 2. Argoverse 2 Sensor (ulica, 8 logów z `av2/wybrane.txt`) - CC BY-NC-SA 4.0
Publiczny bucket s3://argoverse, bez konta.
```
cd dane_skrypty/av2
python pobierz.py meta
python pobierz.py klatki <log> 4   # dla każdego logu z wybrane.txt
cd ../../annot
python -P -W ignore przygotuj_av2.py        # -> annot/kontekst/<log>_<track>/
python -P -W ignore dodaj_hd_lidar.py       # zdjęcia HD i lidar.bin do zbliżeń
```

## 3. HOPE i HANDAL (obiekty, BOP) - ręcznie
Ze strony BOP (https://bop.felk.cvut.cz/datasets/) pobierz dla HOPE i HANDAL: modele (`*_models.zip`) oraz zdjęcia
split `val` (`*_base.zip` + paczka val). Rozpakuj do `$KP_DANE/bop/hope/` i `$KP_DANE/bop/handal/`
(muszą być `models/`, `val/`). Potem:
```
cd annot
python przygotuj_obj.py            # -> annot/obiekty/<zbior>_<id>/
```

## 4. 3DPW (ludzie) - ręcznie, rejestracja
Z https://virtualhumans.mpi-inf.mpg.de/3DPW/ pobierz `imageFiles.zip`, `sequenceFiles.zip`, `readme_and_demo.zip`
i rozpakuj do `$KP_DANE/3dpw/`. Potrzebne są też modele SMPL v1.0.0 (rejestracja na https://smpl.is.tue.mpg.de/) - pliki `.pkl` do `$KP_DANE/smpl/` (albo `KP_SMPL=<katalog>`).
```
cd ludzie
python -P przygotuj_3dpw.py        # -> annot/ludzie/
```

## 5. Anotowanie
```
cd annot
python serwer.py                   # http://127.0.0.1:8765  (/ ARKit, /obiekty, /kontekst, /ludzie)
```
Zapis idzie do `annot/punkty/`, `punkty_obj/`, `punkty_ctx/` i `punkty_ludzie.json` - te pliki commituj.

## 6. Zbiór wspólny do metod i ocen
```
python data_arkitscenes.py && python -P -W ignore data_razem.py    # -> $KP_DANE/kp_razem/
```
Wyniki metod: `results/*.jsonl`, tabele: `report/tabela_eval.py`.
