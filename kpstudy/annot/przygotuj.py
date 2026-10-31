# -*- coding: utf-8 -*-
"""Przygotowuje sceny ARKitScenes do ręcznej anotacji w annotate.html.

    python przygotuj.py              # wszystkie pobrane nagrania z dane/arkitscenes/biurka_po_wizycie.json

Bierze sceny, dla których dane/arkitscenes/klatki.py wyciągnął klatki z wideo. Dla każdej zapisuje
annot/sceny/<id>/meta.json oraz 4 podglądy (jpg 960x720, obrócone niebem do góry) z tych klatek:
  * pudełka 3D biurka (stół z monitorem na blacie), monitorów na nim i krzeseł obok;
  * dla podglądów: K przeskalowane do 960x720 i T_cw (pozy ARKit, interpolowane), żeby strona
    mogła rzutować kliknięte punkty na zdjęcia;
  * ścieżkę do siatki (serwowanej przez serwer.py z dane/arkitscenes/raw).
Lista scen trafia do annot/sceny.json.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import arkit_io as A  # noqa: E402

LISTA = os.path.join(A.ROOT, '..', 'biurka_po_wizycie.json')
OUT = os.path.join(HERE, 'sceny')
KLATKI = os.path.join(A.ROOT, '..', 'klatki')   # klatki z wideo (dane/arkitscenes/klatki.py)
PODGLAD = (960, 720)


def pudelka(scena):
    d = json.load(open(os.path.join(scena.dir, 'annotation.json'), encoding='utf-8'))
    out = []
    for o in d['data']:
        s = o['segments']['obbAligned']
        out.append(dict(id=o['objectId'], label=o['label'], c=np.array(s['centroid']), L=np.array(s['axesLengths']),
                        N=np.array(s['normalizedAxes']).reshape(3, 3)))
    return out


def wewnatrz(X, b, R):
    loc = (X - b['c']) @ R          # kolumny R to osie pudełka
    return (np.abs(loc) <= b['L'] / 2 + 0.02).all(1)


def konwencja(X, boxes):
    """normalizedAxes jako wiersze czy kolumny macierzy osi: wybieramy tę, przy której więcej
    wierzchołków siatki leży w pudełkach (dla obrotów wokół pionu różnią się znakiem kąta)."""
    sub = X[::20]
    n_kol = sum(wewnatrz(sub, b, b['N']).sum() for b in boxes)
    n_wier = sum(wewnatrz(sub, b, b['N'].T).sum() for b in boxes)
    return 'kolumny' if n_kol >= n_wier else 'wiersze', int(n_kol), int(n_wier)


def biurko(boxes):
    """Stół z monitorem na blacie i krzesłem obok (ta sama reguła co przy wyborze nagrań)."""
    T = [b for b in boxes if b['label'] == 'table']
    M = [b for b in boxes if b['label'] == 'tv_monitor']
    C = [b for b in boxes if b['label'] in ('chair', 'stool')]
    najlepsze = None
    for t in T:
        gora = t['c'][2] + t['L'][2] / 2
        na = [m for m in M if abs(m['c'][2] - m['L'][2] / 2 - gora) < 0.15
              and np.all(np.abs((m['c'] - t['c']) @ t['R'])[:2] <= t['L'][:2] / 2 + 0.1)]
        if not na:
            continue
        krz = sorted(C, key=lambda c: np.linalg.norm((c['c'] - t['c'])[:2]))
        krz = [c for c in krz if np.linalg.norm((c['c'] - t['c'])[:2]) < 1.5][:2]
        if krz and (najlepsze is None or len(na) > len(najlepsze[1])):
            najlepsze = (t, na, krz)
    return najlepsze


def main():
    os.makedirs(OUT, exist_ok=True)
    sceny = []
    for x in json.load(open(LISTA)):
        v = x['video_id']
        if not os.path.exists(os.path.join(KLATKI, v, 'klatki.json')):
            continue
        s = A.Scene(v)
        X, _, _ = s.mesh()
        boxes = pudelka(s)
        konw, nk, nw = konwencja(X, boxes)
        # na 19 scenach (29.10) wiersze wygrywały wszędzie poza jednym remisem (10623 vs 10622),
        # więc przyjmujemy wiersze zawsze; liczby zostają w wydruku jako kontrola
        konw = 'wiersze'
        for b in boxes:
            b['R'] = b['N'] if konw == 'kolumny' else b['N'].T
        wyb = biurko(boxes)
        if wyb is None:
            print(v, 'brak biurka z monitorem i krzesłem po ponownym sprawdzeniu, pomijam')
            continue
        t, mon, krz = wyb
        kl = json.load(open(os.path.join(KLATKI, v, 'klatki.json')))
        wybrane = [kl['klatki'][i] for i in np.linspace(0, len(kl['klatki']) - 1, 4).round().astype(int)]
        d = os.path.join(OUT, v)
        os.makedirs(d, exist_ok=True)
        klatki = []
        for k, nazwa in enumerate(wybrane):
            t_ = float(nazwa.split('_')[1])
            T, _ = s.pose_at(t_)
            K, _ = s.K_at(t_)
            sk = PODGLAD[0] / A.WIDE[0]
            img = np.array(Image.open(os.path.join(KLATKI, v, 'rgb', nazwa + '.jpg')).convert('RGB').resize(PODGLAD))
            Ks = K.copy()
            Ks[:2] *= sk
            img, Ks, Tu = A.upright(img, Ks, T, s.sky)      # niebo u góry, jak zobaczy to człowiek i model
            Image.fromarray(img).save(os.path.join(d, 'podglad_%d.jpg' % k), quality=85)
            klatki.append(dict(plik='podglad_%d.jpg' % k, klatka=nazwa, K=Ks.tolist(), T_cw=Tu.tolist(),
                               szer=img.shape[1], wys=img.shape[0]))
        kandydaci = kl['klatki']
        obiekty = [t] + mon + krz
        meta = dict(video_id=v, siatka='/raw/%s/mesh.ply' % v, konwencja_osi=konw,
                    biurko=t['id'], pudelka=[dict(id=b['id'], label=b['label'], c=b['c'].tolist(), L=b['L'].tolist(),
                                                  R=b['R'].tolist()) for b in obiekty],
                    klatki=klatki, widoczne_na=len(kandydaci), wszystkich_klatek=len(s.video_ts()))
        json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'), indent=1)
        sceny.append(dict(video_id=v, monitorow=len(mon), krzesel=len(krz), widoczne_na=len(kandydaci)))
        print(v, 'osie:', konw, nk, nw, '| monitorów', len(mon), 'krzeseł', len(krz), '| klatek z biurkiem', len(kandydaci), flush=True)
    json.dump(sceny, open(os.path.join(HERE, 'sceny.json'), 'w'), indent=1)
    print('scen:', len(sceny))


if __name__ == '__main__':
    main()
