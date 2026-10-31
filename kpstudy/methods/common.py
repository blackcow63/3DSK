# -*- coding: utf-8 -*-
"""Wspólne opcje wrapperów metod: zbiór (kawalerka albo ARKitScenes), frazy, lista zdjęć, ścieżki wyników.

Zbiór ARKitScenes (dane/arkitscenes/kp, data_arkitscenes.py) ma identyfikatory zdjęć "<scena>/<klatka>",
własne frazy w <root>/prompts.json i K w cam.json każdego zdjęcia; kawalerka ma płaskie identyfikatory,
frazy w kpstudy/prompts.json i jedno K w gt.json.
"""
import json
import os
import re

import numpy as np

NUM = r"(-?\d+(?:\.\d+)?)"
BOX = re.compile(r'"bbox_3d"\s*:\s*\[\s*' + NUM + r'\s*,\s*' + NUM + r'\s*,\s*' + NUM)
P2D = re.compile(r'"point_2d"\s*:\s*\[\s*' + NUM + r'\s*,\s*' + NUM)
KP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAWALERKA = os.path.abspath(os.path.join(KP, '..', '..', 'chairs-dataset', 'renders', 'kawalerka_v1'))


def dodaj_opcje(ap):
    ap.add_argument('--root', default=KAWALERKA, help='katalog zbioru (views.json, <id>/rgb.jpg)')
    ap.add_argument('--prompts', default=None, help='frazy; domyślnie <root>/prompts.json, a gdy go nie ma, kpstudy/prompts.json')
    ap.add_argument('--preds', default=None, help='katalog wyników; domyślnie kpstudy/preds dla kawalerki i <root>/preds dla innych zbiorów')
    ap.add_argument('--views', default='all', help='all albo rodzaje widoków z views.json po przecinku (ring, walk, frame, crop, pelny)')
    ap.add_argument('--anchors', default='all', choices=['furniture', 'body', 'all'])
    ap.add_argument('--limit', type=int, default=0, help='tylko pierwsze N zdjęć (test dymny)')
    ap.add_argument('--co', type=int, default=1, help='co N-te zdjęcie, dla modeli zbyt wolnych na cały zbiór')
    ap.add_argument('--zrodlo', default='', help='tylko widoki o identyfikatorze z tym początkiem, np. 3dpw')
    ap.add_argument('--czesc', default='0/1', help='k/n: k-ta z n rozłącznych części listy, do zadań równoległych')


def frazy(args):
    p = args.prompts or os.path.join(args.root, 'prompts.json')
    if not os.path.exists(p):
        p = os.path.join(KP, 'prompts.json')
    pr = json.load(open(p, encoding='utf-8'))
    out = {}
    if args.anchors in ('furniture', 'all'):
        out.update(pr.get('furniture', {}))
    if args.anchors in ('body', 'all'):
        out.update(pr.get('body', {}))
    return pr['template'], out


def odczyt_razem(text, W, H, K):
    """Odpowiedź stylu razem: piksel z point_2d, głębia z z_center. Tylko tekst po </think>, bo rozumowanie zawiera szkice;
    ucięte rozumowanie to brak odpowiedzi. Zwraca ([u, v, 1], [X, Y, Z]) albo None."""
    if '</think>' in text:
        text = text.split('</think>')[-1]
    elif '<think>' in text:
        return None
    a, b = P2D.search(text), BOX.search(text)
    if not (a and b and float(b.group(3)) > 0):
        return None
    u, v, d = float(a.group(1)) / 1000 * W, float(a.group(2)) / 1000 * H, float(b.group(3))
    return [u, v, 1.0], [(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d]


def frazy_widoku(args, vid, domyslne):
    """Zbiór razem (data_razem.py): każda scena ma własne punkty w <root>/<scena>/frazy.json."""
    p = os.path.join(args.root, vid.split('/')[0], 'frazy.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else domyslne


def widoki(args):
    v = [x['id'] for x in json.load(open(os.path.join(args.root, 'views.json')))
         if (args.views == 'all' or x['kind'] in args.views.split(',')) and os.path.exists(os.path.join(args.root, x['id'], 'rgb.jpg'))]
    v = [x for x in v if x.startswith(args.zrodlo)][::args.co]
    k, n = (int(x) for x in args.czesc.split('/'))
    v = v[k::n]
    return v[:args.limit] if args.limit else v


def katalog_wynikow(args):
    if args.preds:
        return args.preds
    return os.path.join(KP, 'preds') if os.path.abspath(args.root) == KAWALERKA else os.path.join(args.root, 'preds')


def plik_wyniku(args, rodzaj, nazwa, vid, ext='.json'):
    p = os.path.join(katalog_wynikow(args), rodzaj, nazwa, vid + ext)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def K_widoku(args, vid):
    c = json.load(open(os.path.join(args.root, vid, 'cam.json')))
    if 'K' in c:
        return np.array(c['K'])
    return np.array(json.load(open(os.path.join(args.root, 'gt.json')))['intrinsics']['K'])


def zapisz_glebie(args, dst, d):
    """Mapa głębi (m). Dla zbiorów innych niż kawalerka zapis w 1/4 rozdzielczości i float16 (~0,35 MB zamiast
    11 MB na zdjęcie 1920x1440), bo wyniki z Heliosa idą łączem ~0,25 MB/s; eval_arkit.py skaluje ją z powrotem.
    Kawalerka zostaje w pełnej rozdzielczości (tego oczekuje run_lift.py)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    d = np.asarray(d, dtype=np.float32)
    if os.path.abspath(args.root) != KAWALERKA:
        from PIL import Image
        h, w = d.shape
        d = np.array(Image.fromarray(d).resize((max(1, w // 4), max(1, h // 4)), Image.NEAREST)).astype(np.float16)
    np.save(dst, d)
