# -*- coding: utf-8 -*-
"""Przygotowuje obiekty z BOP (HOPE, HANDAL: modele zrobione skanerem albo przez grafików, prawdziwe zdjęcia z
dokładnymi pozami 6D) do ręcznej anotacji w annotate_obj.html.

    python przygotuj_obj.py              # wszystkie obiekty ze zbiorów, które są już w dane/bop/<zbior>/val

Dla każdego obiektu zapisuje annot/obiekty/<zbior>_<id>/meta.json i do 4 podglądów: wycinki prawdziwych zdjęć
wokół tego obiektu (różne sceny, obiekt widoczny w co najmniej 60%), z K przeliczonym na wycinek oraz pozą
obiektu (R, t w mm, model -> kamera), żeby strona rzutowała kliknięte punkty. Punkty klika się na modelu,
więc każdy egzemplarz na każdym zdjęciu dostaje je automatycznie. Lista: annot/obiekty.json.
"""
import functools
import json
import os
import sys
import re

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import paths  # noqa: E402
BOP = os.path.join(paths.DANE, 'bop')
OUT = os.path.join(HERE, 'obiekty')
BOK = 480          # bok podglądu (px)
MIN_WIDOCZNOSC = 0.6
MIN_OSTROSC = 8     # wariancja laplasjanu; poniżej tego wycinek jest wyraźnie rozmazany

# HANDAL nie podaje nazw obiektów; rozpoznane z rzutu modelu na zdjęcia (29.10.2026). Tylko obiekty z val.
NAZWY_HANDAL = {
    2: 'hammer, black-teal grip', 6: 'hammer, red-black grip', 8: 'hammer, wooden handle', 9: 'hammer, all black',
    10: 'slotted spoon, green, wooden handle', 11: 'spoon, red, wooden handle', 12: 'spoon, green, wooden handle',
    16: 'measuring cup, blue', 17: 'measuring cup, yellow', 20: 'power drill, green-black',
    27: 'spatula, pink, wooden handle', 28: 'slotted turner, orange, wooden handle', 29: 'turner, blue, wooden handle',
    31: 'spider strainer, wire, black handle', 33: 'strainer, teal rim', 34: 'colander, green, collapsible',
    36: 'whisk, black handle', 37: 'whisk, mint handle', 39: 'whisk, red wire, black handle',
    40: 'whisk, yellow wire, black handle',
}
# Podpowiedź kategorii na stronie (można nadpisać); po niej strona podsuwa nazwy punktów z innych obiektów.
KATEGORIE = {
    'hope': {**{n: 'can' for n in ('AlphabetSoup', 'Cherries', 'Corn', 'GreenBeans', 'Mushrooms', 'Peaches',
                                    'Peas&Carrots', 'Pineapple', 'TomatoSauce', 'Tuna', 'Parmesan')},
             **{n: 'box' for n in ('Butter', 'ChocolatePudding', 'Cookies', 'CreamCheese', 'GranolaBars',
                                    'Macaroni&Cheese', 'Popcorn', 'Raisins', 'Spaghetti')},
             **{n: 'bottle' for n in ('BBQSauce', 'Ketchup', 'Mayo', 'Mustard', 'SaladDressing')},
             'Milk': 'carton', 'OrangeJuice': 'carton', 'Yogurt': 'cup'},
    'handal': {n: v.split(',')[0].replace('slotted ', '').replace('turner', 'spatula').replace('colander', 'strainer')
                  .replace('spider ', '') for n, v in NAZWY_HANDAL.items()},
}


def nazwy(zbior):
    """Nazwy obiektów z tabeli w dataset_info.md (HOPE ma ją; HANDAL nie, wtedy 'obiekt N')."""
    out = {}
    for p in (os.path.join(BOP, zbior, 'dataset_info.md'), os.path.join(BOP, zbior, zbior, 'dataset_info.md')):
        if os.path.exists(p):
            for m in re.finditer(r'^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|', open(p, encoding='utf-8').read(), re.M):
                out[int(m.group(1))] = m.group(2)
    return out


def wystapienia(zbior):
    """obj_id -> lista (scena, obraz, indeks egzemplarza, widoczność, bbox)."""
    val = os.path.join(BOP, zbior, 'val')
    out = {}
    for sc in sorted(os.listdir(val)):
        d = os.path.join(val, sc)
        if not os.path.exists(os.path.join(d, 'scene_gt.json')):
            continue
        gt = json.load(open(os.path.join(d, 'scene_gt.json')))
        info = json.load(open(os.path.join(d, 'scene_gt_info.json')))
        for im in gt:
            for i, (g, inf) in enumerate(zip(gt[im], info[im])):
                out.setdefault(g['obj_id'], []).append((sc, int(im), i, inf['visib_fract'], inf['bbox_obj']))
    return out


@functools.lru_cache(maxsize=None)
def wczytaj_json(sciezka):
    return json.load(open(sciezka))


def podglad(zbior, sc, im, i, bbox, k):
    d = os.path.join(BOP, zbior, 'val', sc)
    cam = wczytaj_json(os.path.join(d, 'scene_camera.json'))[str(im)]
    g = wczytaj_json(os.path.join(d, 'scene_gt.json'))[str(im)][i]
    rgb_dir = os.path.join(d, 'rgb')
    plik = next(f for f in os.listdir(rgb_dir) if f.startswith('%06d.' % im))
    img = Image.open(os.path.join(rgb_dir, plik)).convert('RGB')
    x, y, w, h = bbox
    s = max(w, h) * 1.3
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    s = int(round(s))
    crop = img.crop((x0, y0, x0 + s, y0 + s)).resize((BOK, BOK), Image.BICUBIC)
    K = np.array(cam['cam_K']).reshape(3, 3)
    sk = BOK / s
    K2 = K.copy()
    K2[0, 2] -= x0
    K2[1, 2] -= y0
    K2[:2] *= sk
    return crop, dict(K=K2.tolist(), R=np.array(g['cam_R_m2c']).reshape(3, 3).tolist(), t=g['cam_t_m2c'],
                      zrodlo='%s/val/%s/%06d' % (zbior, sc, im), egzemplarz=i)


def ostrosc(crop):
    """Wariancja laplasjanu na wycinku w skali szarości; HANDAL to klatki z wideo z ręki i część jest rozmazana."""
    g = np.asarray(crop.convert('L'), dtype=np.float32)
    lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def kopia_modelu(zbior, obj, d):
    """Kopia modelu z nazwami współrzędnych tekstury, które czyta PLYLoader three.js r128 (s, t zamiast
    texture_u, texture_v); bez tego model ładuje się bez tekstury, na czarno."""
    with open(os.path.join(BOP, zbior, 'models', 'obj_%06d.ply' % obj), 'rb') as f:
        dane = f.read()
    i = dane.index(b'end_header')
    naglowek = dane[:i].replace(b'property float texture_u', b'property float s').replace(b'property float texture_v', b'property float t')
    with open(os.path.join(d, 'model.ply'), 'wb') as f:
        f.write(naglowek + dane[i:])


def main():
    os.makedirs(OUT, exist_ok=True)
    lista = []
    for zbior in ('hope', 'handal'):
        if not os.path.isdir(os.path.join(BOP, zbior, 'val')):
            print(zbior, ': brak zdjęć val, pomijam')
            continue
        nz = nazwy(zbior) if zbior != 'handal' else NAZWY_HANDAL
        for obj, wyst in sorted(wystapienia(zbior).items()):
            dobre = sorted([w for w in wyst if w[3] >= MIN_WIDOCZNOSC], key=lambda w: -w[3])
            # kandydaci: do 3 najlepiej widocznych z każdej sceny; odpadają rozmazane; z reszty 4 najostrzejsze, najpierw z różnych scen
            na_scene, kand = {}, []
            for w in dobre:
                if na_scene.get(w[0], 0) < 3:
                    kand.append(w)
                    na_scene[w[0]] = na_scene.get(w[0], 0) + 1
            if not kand:
                continue
            kand = sorted(((ostrosc(c), w, c, m) for w in kand for c, m in [podglad(zbior, w[0], w[1], w[2], w[4], 0)]),
                          key=lambda x: -x[0])
            kand = [x for x in kand if x[0] >= max(MIN_OSTROSC, 0.25 * kand[0][0])] or kand[:1]
            sceny, wyb = set(), []
            for x in kand:
                if x[1][0] not in sceny:
                    wyb.append(x)
                    sceny.add(x[1][0])
            wyb = (wyb + [x for x in kand if all(x is not y for y in wyb)])[:4]
            klucz = '%s_%06d' % (zbior, obj)
            d = os.path.join(OUT, klucz)
            os.makedirs(d, exist_ok=True)
            podg = []
            for k, (ost, w, crop, m) in enumerate(wyb):
                crop.save(os.path.join(d, 'podglad_%d.jpg' % k), quality=90)
                podg.append(dict(m, plik='podglad_%d.jpg' % k, ostrosc=round(ost, 1)))
            nazwa = nz.get(obj, 'obiekt %d' % obj)
            kopia_modelu(zbior, obj, d)
            kat = KATEGORIE[zbior].get(nazwa if zbior == 'hope' else obj, '')
            meta = dict(klucz=klucz, zbior=zbior, obj_id=obj, nazwa=nazwa, kategoria=kat,
                        model='obiekty/%s/model.ply' % klucz, tekstura='/bop/%s/models/obj_%06d.png' % (zbior, obj),
                        jednostki='mm', podglady=podg, wystapien=len(wyst), dobrze_widocznych=len(dobre))
            json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'), indent=1)
            lista.append(dict(klucz=klucz, zbior=zbior, obj_id=obj, nazwa=nazwa, wystapien=len(wyst), dobrze_widocznych=len(dobre)))
            print(klucz, nazwa, '| wystąpień', len(wyst), 'dobrze widocznych', len(dobre), flush=True)
    json.dump(lista, open(os.path.join(HERE, 'obiekty.json'), 'w'), indent=1)
    print('obiektów:', len(lista))


if __name__ == '__main__':
    main()
