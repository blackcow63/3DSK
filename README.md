# 3D semantic keypoints - eksperymenty

Czy da się ze zwykłego zdjęcia (telefon, RGB, bez LiDAR-u) i nazwy punktu po angielsku ("front left corner of the desk top",
"center of the left headlight of a car") dostać położenie tego punktu w 3D, w metrach, w układzie kamery? To repo zawiera
kod, ręczne anotacje i wyniki eksperymentów do tej pracy. Papier i docs są poza repo.

Zasada oceny: **jedno zapytanie do jednego modelu**. Wejście to zdjęcie 2D i opis tekstem, wyjście to punkt 3D. Testujemy tylko modele
open source. Modele dwuetapowe (wskaźnik 2D + głębia) są wariantem porównawczym, a nie głównym celem.

## Spis treści
1. [Co jest w repo, a czego nie ma](#1-co-jest-w-repo-a-czego-nie-ma)
2. [Szybki start](#2-szybki-start)
3. [Układ katalogów](#3-układ-katalogów)
4. [Zbiór danych `kp_razem`](#4-zbiór-danych-kp_razem)
5. [Anotacje: co, jak i w jakim formacie](#5-anotacje)
6. [Metody](#6-metody)
7. [Jak liczona jest ocena](#7-ocena)
8. [Wyniki](#8-wyniki)
9. [Jak dodać nowy eksperyment](#9-jak-dodać-nowy-eksperyment)
10. [Helios (klaster GPU)](#10-helios)
11. [Starsze badanie na syntetycznej kawalerce](#11-starsze-badanie-na-syntetycznej-kawalerce)
12. [Licencje](#12-licencje)

## 1. Co jest w repo, a czego nie ma
**Jest:** kod metod i oceny, narzędzia do anotacji (strony HTML + serwer), same anotacje (json), wyniki starszego badania
(`kpstudy/results/*.jsonl`), skrypty na Heliosa, skrypty pobierania danych.

**Nie ma** (za duże albo licencja niekomercyjna), trzeba pobrać:

| Co | Rozmiar | Gdzie |
|---|---|---|
| zdjęcia i chmury do anotacji (`annot/sceny`, `kontekst`, `obiekty`, `ludzie`) | ok. 0,3 GB | paczka `1_annot_zdjecia.zip` |
| zbiór wspólny `kp_razem` (zdjęcia, kamery, prawda) | ok. 0,15 GB | paczka `2_kp_razem.zip` |
| siatki ARKitScenes i modele obiektów BOP (do klikania w narzędziu) | ok. 0,4 GB | paczka `3_siatki_arkit_i_modele_bop.zip` |
| predykcje wszystkich modeli na `kp_razem` | ok. 0,2 GB | paczka `4_kp_razem_preds.zip` |
| wagi modeli, `.venv`, `kpstudy/preds` ze starego badania | kilka GB | pobierają się same (HuggingFace) albo niepotrzebne |

Paczki zip dostajesz od autorki (link poza repo). Rozpakowanie opisane w [kpstudy/DANE.md](kpstudy/DANE.md). Można też zbudować dane
od zera z oryginalnych źródeł, to tamże.

## 2. Szybki start
```bash
git clone https://github.com/blackcow63/3DSK.git        # nazwa katalogu nie ma znaczenia
cd 3DSK/kpstudy
export KP_DANE=/gdzieś/dane                              # domyślnie <repo>/dane
unzip 2_kp_razem.zip 4_kp_razem_preds.zip -d $KP_DANE    # -> $KP_DANE/kp_razem/...
unzip 1_annot_zdjecia.zip -d .                           # -> kpstudy/annot/sceny, kontekst, obiekty, ludzie
unzip 3_siatki_arkit_i_modele_bop.zip -d $KP_DANE        # -> $KP_DANE/arkitscenes/raw, $KP_DANE/bop
python annot/serwer.py                                   # anotator: http://127.0.0.1:8765
cd report && python -P -W ignore tabela_eval.py --root $KP_DANE/kp_razem   # tabele wyników -> report/out/
```
Wymagania: Python 3 z numpy, scipy, opencv-python, pillow, pandas, pyarrow. Modele uruchamia się na GPU (Helios, rozdz. 10).

## 3. Układ katalogów
```
kpstudy/
  README.md            opis starszego badania na syntetycznej kawalerce (po angielsku)
  DANE.md              skąd wziąć dane i jak je przygotować
  paths.py             gdzie leżą dane: $KP_DANE, potem <repo>/dane, potem <repo>/../dane
  prompts.json         frazy dla punktów kawalerki (stare badanie)
  annot/               NARZĘDZIA I ANOTACJE (rozdz. 5)
    annotate*.html     strony do klikania punktów (ARKit, obiekty BOP, ulica AV2, człowiek SMPL)
    serwer.py          lokalny serwer anotatora, zapisuje do punkty*/ 
    przygotuj*.py      przygotowanie scen do anotacji z danych źródłowych
    slownik*.json      nazwy punktów i frazy dla modeli
    punkty/ punkty_obj/ punkty_ctx/ punkty_ludzie.json     ANOTACJE (prawda)
  data_arkitscenes.py  buduje zbiór z ARKitScenes (klatki, kamery, widoczność)
  data_razem.py        buduje wspólny zbiór `kp_razem` ze wszystkich 5 źródeł
  ref_pary.py          pary punktów do metod ze zdjęciem wzorcowym
  methods/             wrappery modeli (rozdz. 6); wynik trafia do <zbiór>/preds/
  lift.py, run_lift.py podniesienie punktów 2D do 3D (triangulacja, głębia, płaszczyzna biurka), stare badanie
  eval_arkit.py        ocena na samym ARKitScenes
  summarize.py         tabele ze starego badania -> results/summary.md
  results/             wyniki starego badania (jsonl na anchor i na pomiar)
  report/tabela_eval.py   tabele wyników na kp_razem (rozdz. 8)
  feasibility/, bench/ pomiary czasu i pamięci (CPU, laptop, Grace)
  helios/              zadania slurm i skrypty wysyłania/pobierania (rozdz. 10)
  ludzie/              przygotowanie 3DPW (ciało SMPL) do anotacji
  dane_skrypty/        pobieranie ARKitScenes i Argoverse 2
```

## 4. Zbiór danych `kp_razem`
Jeden zbiór testowy z pięciu źródeł, zbudowany z ręcznych anotacji (`python data_arkitscenes.py && python -P -W ignore data_razem.py`).
Powstaje w `$KP_DANE/kp_razem/`.

| Źródło | Co to | Układ punktu | Scen w `kp_razem` | Anotacji ręcznych |
|---|---|---|---|---|
| **ARKitScenes** | biurka z monitorem i krzesłem, nagrania iPada (RGB 1920x1440, poza z VIO, LiDAR) | świat ARKit, metry | 6 | 6 scen, razem 80 punktów (11-15 na scenę: nie każdy punkt da się zaznaczyć) |
| **HOPE** | małe obiekty (BOP), zdjęcia z kamery z pozą 6D | układ modelu obiektu | 3 | 3 obiekty |
| **HANDAL** | obiekty ręczne (BOP) | jw. | 6 | 6 obiektów (HOPE + HANDAL razem 12 punktów) |
| **Argoverse 2** | ulica: ludzie, auta, rowery, znaki, pachołki (17 kategorii) | ramka obiektu | 19 (+19 w wycinkach `av2c_`) | 19 jednostek, 110 punktów |
| **3DPW** | ludzie, ciało SMPL | siatka SMPL, każda klatka osobno | 25 | 14 punktów na wzorcowym ciele |

Razem 78 scen i 434 widoki: 291 typu `frame` (pełne zdjęcie), 118 `crop` (wycinek wokół obiektu), 25 `pelny`.

**Układ plików:**
```
kp_razem/
  views.json            lista widoków: {"id": "<scena>/<widok>", "scene", "kind": frame|crop|pelny, "zbior"}
  gt.json               prawda: scenes[<scena>] = {points_world: {nazwa: [x,y,z]}, zbior, zrodlo, (jakosc)}
  prompts.json          szablon pytania "Point to the {phrase} in this image."
  ref.json              (ref_pary.py) punkty zaznaczone na innym zdjęciu, dla metod ze wzorcem
  <scena>/frazy.json    nazwa punktu -> fraza po angielsku dla tej sceny (każda scena ma własny zestaw)
  <scena>/<widok>/
     rgb.jpg            zdjęcie, które dostaje model
     cam.json           K (3x3) i T_wc (4x4, kamera->świat), W, H
     kp.json            dla punktu: u, v (piksel), z (m), in_frame, visible
     lidar.png          głębia LiDAR (mm), tylko do widoczności i wyroczni; modele jej NIE dostają
  preds/                wyniki modeli (rozdz. 9), osobna paczka
```
Świat sceny zależy od źródła (tabela wyżej). 3DPW ma świat = kamera, bo ciało się rusza, a 3DPW jest dopasowaniem, nie pomiarem,
więc liczy się tylko odległość między punktami ciała. Szczegóły i uzasadnienia w docstringu [data_razem.py](kpstudy/data_razem.py).

Widoczność punktu: w kadrze, głębiej niż 0,1 m i nie głębiej niż LiDAR w otoczeniu 5x5 plus max(5 cm, 3% z).
Błąd liczy się tylko dla punktów widocznych.

## 5. Anotacje
Ręczne, w przeglądarce, przez lokalny serwer (`python annot/serwer.py`, tylko 127.0.0.1:8765). Cztery tryby:

| Adres | Strona | Co się klika | Zapis |
|---|---|---|---|
| `/` | `annotate.html` | punkty 3D na siatce sceny ARKit, podgląd na 4 zdjęciach | `annot/punkty/<scena>.json` |
| `/obiekty` | `annotate_obj.html` | punkty na modelu 3D obiektu BOP (HOPE/HANDAL) | `annot/punkty_obj/<zbior>_<id>.json` |
| `/kontekst` | `annotate_ctx.html` | punkt na 2-3 zdjęciach tego samego obiektu z ulicy (AV2); serwer trianguluje; zbliżenia HD i kontrola LiDAR | `annot/punkty_ctx/av2_<log>_<track>.json` |
| `/ludzie` | `annotate_ludzie.html` | punkt na wzorcowym ciele SMPL (jeden raz dla wszystkich ludzi) | `annot/punkty_ludzie.json` |

**Nazwy punktów** są w słownikach (`annot/slownik.json`, `slownik_ctx.json`): każdy punkt ma stałą nazwę i jedną angielską frazę, której
używają wszystkie modele tekst->punkt. Dla biurka (17 punktów): `desk_top_front_left/right`, `desk_top_back_left/right`, `desk_front_edge_center`,
`monitor_screen_center`, `monitor_top_edge_center`, `monitor_top_left/right`, `monitor_bottom_left/right`, `keyboard_center`, `mouse_center`,
`seat_top_center`, `chair_back_top_center`, `armrest_top_left/right`. **Lewo/prawo i przód/tył zawsze z punktu widzenia osoby siedzącej
przy biurku, twarzą do monitora.** Dla AV2 jest 17 kategorii (person, car, bicycle, stop sign, bollard...) z własnymi listami punktów.

**Format pliku z punktami** (np. `punkty/43649609.json`): `{"video_id": "...", "punkty": {"<nazwa>": [x, y, z], ...}}` w metrach, w świecie
ARKit. Pliki `punkty_obj` i `punkty_ctx` mają dodatkowo `klucz`, `kategoria`, `opis_obiektu`, `uklad`, `czas`; `punkty_ludzie.json` zapisuje punkt jako
trójkąt siatki SMPL plus współrzędne barycentryczne.

**Ile jest:** 6 scen ARKit (80 punktów), 9 obiektów BOP (3 HOPE + 6 HANDAL, 12 punktów), 19 jednostek AV2 (110 punktów), 14 punktów ciała.
Anotacje są w repo, zdjęcia osobno (paczka 1). **Po anotowaniu commituj pliki z `annot/punkty*`.**

Kontrola jakości: dla każdego punktu `gt.json` trzyma `jakosc` (odległość do LiDAR-u w cm, residuum w pikselach, kąt bazy dla AV2),
a ocena liczy też podzbiór potwierdzony LiDAR-em.

## 6. Metody
Dwie rodziny, obie dostają zdjęcie i frazę:

**Jedno przejście (główny cel): zdjęcie + opis -> punkt 3D w kamerze** (`preds/e2e/<nazwa>/<widok>.json`, pole `points_cam`)
- *Typ A, punkt i głębia w jednej odpowiedzi:* RoboBrain 2.5 (4B, 8B-NV, 8B-MT, także 4-bit). Wrapper [methods/robobrain_point.py](kpstudy/methods/robobrain_point.py)
  `--mode 3d` (format odpowiedzi `(x, y, d)`).
- *Typ B, piksel i pudełko 3D w jednej odpowiedzi:* Qwen3-VL 8B/32B (`--styl razem` albo `bbox3d`), Cosmos-Reason2-8B, RynnBrain 1.1 (2B, 9B),
  MiMo-Embodied-7B. Środek pudełka daje punkt.
- Embodied-R1.5 odpadł: pojedynczy punkt daje tylko w 2D.

**Dwa etapy: wskaźnik 2D + głębia** (`preds/stage1/<nazwa>/` + `preds/depth/<model>/`, podniesienie w `lift.py`, `eval_arkit.py`)
- Wskaźniki 2D: MolmoPoint 8B, Molmo2 (4B, 8B), Qwen3-VL (2B...32B), Moondream 2, dopasowanie cech DINOv2 do zdjęcia wzorcowego (`dino_match.py`, `ref_match.py`).
- Głębia: MoGe-2, UniDepthV2, Depth Anything 3, MapAnything, DepthLM.
- Podniesienie: triangulacja z pozami VIO telefonu (`tri_arkit`, z odrzucaniem błędnych wskazań), `depth_<model>`, `depth_lidar` (wyrocznia), `oracle2d` (prawdziwe 2D, mierzy sam błąd podniesienia).

**Tylko ludzie (stare badanie):** NLF, SAM 3D Body, TAIHRI, MediaPipe world landmarks (`nlf_e2e.py`, `sam3d_body_e2e.py`, `taihri_e2e.py`, `mediapipe_pose.py`).

Wszystkie wrappery mają ten sam interfejs: `--root <zbiór> --name <nazwa> [--views frame,crop]`, pomijają widoki, które już mają wynik, i zapisują do
`<zbiór>/preds/`. Przykłady uruchomień: [kpstudy/helios/run_jedno.sh](kpstudy/helios/run_jedno.sh).

## 7. Ocena
- Błąd punktu = odległość 3D między odpowiedzią modelu a prawdą w układzie kamery, **w procentach odległości punktu od kamery**
  (dzięki temu biurko i ulica dają się połączyć w jeden wynik). Starsze badanie podaje cm.
- Liczą się tylko punkty widoczne na zdjęciu. **Brak odpowiedzi to chybienie** w progach (<5%, <10%); mediana liczona z punktów, na które model odpowiedział.
- Próbka: co dziewiąte zdjęcie typu `frame` ze źródeł arkit, av2, handal, hope. Przedziały ufności: bootstrap po scenach, 2000 losowań, ziarno 0.
- Rozkład błędu: (prawdziwy piksel + głębia modelu) i (piksel modelu + prawdziwa głębia), żeby było widać, czy myli się wskazanie, czy głębia.
- Kod: [kpstudy/report/tabela_eval.py](kpstudy/report/tabela_eval.py), wyniki do `report/out/` (albo `$KP_OUT`).

## 8. Wyniki
Mediana błędu 3D jako procent odległości od kamery, na wspólnej próbce (`tabela_eval.py`, `report/out/tab_eval.tex`):

| Model | Mediana błędu | Punktów z odpowiedzią |
|---|---|---|
| **Typ A: punkt i głębia** | | |
| RoboBrain 2.5 8B-NV | 16% | 99% |
| RoboBrain 2.5 4B | 18% | 90% |
| RoboBrain 2.5 8B-NV, 4-bit | 18% | 100% |
| RoboBrain 2.5 8B-MT | 20% | 100% |
| RoboBrain 2.5 4B, 4-bit | 22% | 68% |
| **Typ B: piksel i pudełko 3D** | | |
| Qwen3-VL-8B | 16% | 97% |
| Cosmos-Reason2-8B | 16% | 66% |
| RynnBrain 1.1 2B | 73% | 100% |
| RynnBrain 1.1 9B | 89% | 93% |
| MiMo-Embodied-7B | 5120% | 75% |
| **Qwen3-VL zapytany inaczej** | | |
| 8B, tylko środek pudełka | 16% | 90% |
| 32B, tylko środek pudełka | 18% | 96% |
| 32B, format $(x, y, d)$ | 23% | 92% |
| 8B, format $(x, y, d)$ | 25% | 99% |

Wniosek na dziś: najlepsze jednoprzejściowe modele mają medianę ok. 16% odległości, czyli ok. 0,5 m dla punktu 3 m od kamery. RynnBrain i MiMo
nie nadają się w tym zadaniu bez dopasowania (autorzy trenowali je do innych pytań).

Starsze badanie na syntetycznej kawalerce (błąd w cm, według metody i sposobu podniesienia): `kpstudy/results/summary.md`.

## 9. Jak dodać nowy eksperyment
1. **Model**: skopiuj najbliższy wrapper z `kpstudy/methods/` (np. `robobrain_point.py`) i zmień ładowanie modelu oraz prompt.
2. **Wejście**: dla każdego widoku z `views.json` (typy `frame`, `crop`) weź `<scena>/<widok>/rgb.jpg`, `cam.json` i frazy z `<scena>/frazy.json`
   (szablon z `prompts.json`).
3. **Wyjście**: zapisz `<kp_razem>/preds/e2e/<nazwa>/<scena>/<widok>.json`:
   `{"method": "<nazwa>", "view": "<scena>/<widok>", "points_cam": {"<punkt>": [x, y, z]}}` (metry, układ kamery OpenCV: x w prawo, y w dół, z do przodu).
   Dla wskaźnika 2D: `preds/stage1/<nazwa>/<scena>/<widok>.json` z `points_2d: {"<punkt>": [u, v, ...]}` w pikselach.
4. **Ocena**: dopisz model do listy `MODELE` w `report/tabela_eval.py` (nazwa katalogu, etykieta, typ A/B/control) i uruchom skrypt.
5. Wrappery pomijają gotowe widoki, więc przerwane zadanie wystarczy puścić ponownie.

Dobre kontrole: wariant z innym promptem tego samego modelu (jak `qwen3vl8b_box` vs `_3d`), model w 4-bit, większy model z tej samej rodziny.

## 10. Helios
Obliczenia idą na klastrze Helios (PLGrid, grant `plgciopbhp`, GPU GH200 = aarch64 + H100). To jest konfiguracja autorki; kolega bez dostępu do grantu może
uruchamiać wrappery na własnym GPU, a skrypty slurm traktować jako przykład.
- `helios/env.sh` ustawia konto, partycję i ścieżki (`$KP3D`, venv-y, cache HuggingFace). Drzewo projektu jest odbite pod `$KP3D` w tym samym układzie (`ciop/kpstudy`, `dane/kp_razem`).
- `helios/gpu_job.sbatch` uruchamia dowolną komendę na jednym GPU: `sbatch -J nazwa -t 08:00:00 ciop/kpstudy/helios/gpu_job.sbatch python methods/<wrapper>.py ...`.
- `push_razem.sh` / `pull_razem.sh` wysyłają zbiór (bez prawdy) i pobierają `preds`; `run_jedno.sh`, `run_razem*.sh`, `run_e3.sh` to przepisy na kolejne serie.
- Modele gated (np. SAM 3D Body) trzeba pobrać wcześniej do `$HF_HOME`, zadania nie mają tokenu.
- Pomiar wykonalności na telefonie (czas całego pomiaru, pamięć): `feasibility/`, `bench/`, `results/feasibility.md`.

## 11. Starsze badanie na syntetycznej kawalerce
Pierwsza część: syntetyczny pokój ("kawalerka") z biurkiem i osobą, 36 zdjęć pojedynczych i 3 przejścia po 30 klatek, z wirtualnym telefonem
(1440x1080, obiektyw 26 mm). Tam porównywane były potoki E2E dla ludzi (NLF, SAM 3D Body, TAIHRI, MediaPipe) i dwuetapowe (wskaźnik + proste podniesienie
z pozami telefonu, z szumem 0-4 cm). Opis i komendy: [kpstudy/README.md](kpstudy/README.md); wyniki: `kpstudy/results/*.jsonl`, `summary.md`.
Dane syntetyczne (`chairs-dataset/`) nie są w repo i do bieżących eksperymentów nie są potrzebne.

## 12. Licencje
- ARKitScenes (Apple): niekomercyjna. Argoverse 2: CC BY-NC-SA 4.0. BOP HOPE/HANDAL i 3DPW: warunki na stronach zbiorów (3DPW wymaga rejestracji, modele SMPL też).
- Dlatego zdjęć i danych nie ma w gicie. Nie wrzucaj ich do publicznego repo ani nie udostępniaj poza zespołem.
