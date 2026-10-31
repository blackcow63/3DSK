# -*- coding: utf-8 -*-
"""Dokłada do jednostek z przygotuj_av2.py to, co jest potrzebne do klikania z przybliżaniem i sprawdzania odległości.

    python -P -W ignore dodaj_hd_lidar.py

hd_k.jpg: ten sam wycinek co podglad_k.jpg, ale w rozdzielczości zdjęcia (do 1600 px). K i kliknięcia zostają
we współrzędnych wycinka 640x640, strona rysuje tylko ostrzejszy obraz.
lidar.bin: punkty LiDAR z ramki obiektu (float32 xyz, m, siatka 1 cm), także podłoże pod obiektem, żeby punkty przy
ziemi też miały pomiar. Ludzie itp.: tylko okno 1,2 s.
Stała poza: ramka z anotacji pływa na całym logu (znaki stop 13,5 cm w medianie, choć stoją), więc obiekt, który
przez czas ujęć naprawdę stoi, dostaje jedną pozę w układzie miasta i pozy ujęć liczone z samych ego-póz. Punkty SIFT
ze zdjęć są wtedy bliżej LiDAR-u (do 5 cm: 90% vs 85% na tych samych obiektach, 30.10). O tym, czy stoi, decyduje LiDAR: grubość powierzchni
w stałej pozie nie może być większa niż w ramce z anotacji. Pozy w meta.json są nadpisywane, a meta['poza'] mówi,
która została. Kliknięcia są w pikselach, więc przeliczenie ich nie psuje.
Przemiatania AV2 są już skompensowane ruchem auta: kompensacja per punkt z offset_ns daje grubszą powierzchnię
(auto 1,26 vs 1,17 cm; 4 z 4 obiektów, sprawdzone 30.10).
"""
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from przygotuj_av2 import BOK, HERE, Log, OUT   # noqa: E402

HD_MAX = 1600
ZAPAS = 0.12                # m wokół ramki obiektu
SIATKA = 0.01               # m
TOLERANCJA_CM = 0.15        # o tyle grubsza powierzchnia w stałej pozie jeszcze uchodzi za to samo


def hd(L, p):
    K = L.K[p['kam']]
    Kc = np.array(p['K'])
    n = int(np.clip(round(BOK * K[0, 0] / Kc[0, 0]), BOK, HD_MAX))    # bok wycinka w pikselach zdjęcia
    g = (np.arange(n, dtype=np.float32) + 0.5) * BOK / n              # środki pikseli we współrzędnych wycinka
    uu, vv = np.meshgrid(g, g)
    mx = ((uu - Kc[0, 2]) / Kc[0, 0] * K[0, 0] + K[0, 2] - 0.5).astype(np.float32)
    my = ((vv - Kc[1, 2]) / Kc[1, 1] * K[1, 1] + K[1, 2] - 0.5).astype(np.float32)
    return cv2.remap(L.obraz(p['kam'], p['t']), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


def siatka(P):
    if not len(P):
        return P
    _, i = np.unique(np.floor(P / SIATKA).astype(np.int32), axis=0, return_index=True)
    return P[np.sort(i)]


def grubosc(P):
    """Mediana grubości powierzchni w cm: najmniejsza oś PCA 25 sąsiadów po rozrzedzeniu."""
    P = siatka(P)
    if len(P) < 60:
        return float('nan')
    i = np.random.default_rng(0).choice(len(P), min(1500, len(P)), replace=False)
    _, s = cKDTree(P).query(P[i], k=25)
    return float(np.median([np.sqrt(np.linalg.eigvalsh(np.cov(P[q].T))[0]) for q in s]) * 100)


def poza_stala(tor, t0, t1):
    w = (tor.ts >= t0) & (tor.ts <= t1)
    if w.sum() < 2:
        w = np.abs(tor.ts - (t0 + t1) // 2) == np.abs(tor.ts - (t0 + t1) // 2).min()
    O = np.eye(4)
    O[:3, :3] = tor.rot[np.where(w)[0]].mean().as_matrix()
    O[:3, 3] = np.median(tor.t[w], 0)
    return O


def zbierz(L, jedn):
    for j in jedn:
        j.update(tor_cala=[], tor_okno=[], stala=[])
    for tl in L.t_lidar:
        tl = int(tl)
        P = L.lidar_city(tl)
        for j in jedn:
            if j['okno'] and not (j['okno'][0] - 60_000_000 <= tl <= j['okno'][1] + 60_000_000):
                continue
            w_oknie = j['O0'] is not None and j['t0'] - 500_000_000 <= tl <= j['t1'] + 500_000_000
            O = j['tor'](tl, 100_000_000)                 # poza śledzeniem obiektu nie wiadomo, gdzie był
            if O is not None:
                Xo = (P - O[:3, 3]) @ O[:3, :3]
                m = (np.abs(Xo) <= j['wym'] / 2 + ZAPAS).all(1)
                if m.any():
                    j['tor_cala'].append(Xo[m])
                    if w_oknie:
                        j['tor_okno'].append(Xo[m])
            if w_oknie:
                Xo = (P - j['O0'][:3, 3]) @ j['O0'][:3, :3]
                m = (np.abs(Xo) <= j['wym'] / 2 + ZAPAS).all(1)
                if m.any():
                    j['stala'].append(Xo[m])


def main():
    lista = json.load(open(os.path.join(HERE, 'kontekst.json')))
    decyzje = []
    for log in sorted({x['log'] for x in lista}):
        metas = {x['klucz']: json.load(open(os.path.join(OUT, x['klucz'], 'meta.json'))) for x in lista if x['log'] == log}
        L = Log(next(iter(metas.values()))['log'])
        J = []
        for k, m in metas.items():
            tor, _, _ = L.tor(m['track'])
            ts = [u['t'] for u in m['ujecia']]
            sztywny = m['okno_ns'] is None
            J.append(dict(klucz=k, tor=tor, wym=np.array(m['wymiary_m']), okno=m['okno_ns'], t0=min(ts), t1=max(ts),
                          O0=poza_stala(tor, min(ts), max(ts)) if sztywny else None))
        zbierz(L, J)
        for j in J:
            k, m = j['klucz'], metas[j['klucz']]
            d = os.path.join(OUT, k)
            for i, p in enumerate(m['podglady']):
                p['hd'] = 'hd_%d.jpg' % i
                if not os.path.exists(os.path.join(d, p['hd'])):
                    cv2.imwrite(os.path.join(d, p['hd']), hd(L, p), [cv2.IMWRITE_JPEG_QUALITY, 92])
            stala = False
            if j['O0'] is not None and j['stala'] and j['tor_okno']:
                g_tor, g_st = grubosc(np.vstack(j['tor_okno'])), grubosc(np.vstack(j['stala']))
                stala = g_st <= g_tor + TOLERANCJA_CM
                m['grubosc_cm'] = dict(ramka=round(g_tor, 2), stala=round(g_st, 2))
                decyzje.append((stala, g_tor, g_st))
            for u, zaokr in [(p, None) for p in m['podglady']] + [(u, 1) for u in m['ujecia']]:
                T = L.cam_T_city(u['kam'], u['t']) @ (j['O0'] if stala else j['tor'](u['t']))
                u['R'] = T[:3, :3].tolist() if zaokr is None else T[:3, :3].round(6).tolist()
                u['t_m'] = T[:3, 3].tolist() if zaokr is None else T[:3, 3].round(4).tolist()
            m['poza'] = 'stala' if stala else 'ze_sledzenia'
            zr = j['stala'] if stala else j['tor_cala']
            P = siatka(np.vstack(zr)).astype(np.float32) if zr else np.zeros((0, 3), np.float32)
            P.tofile(os.path.join(d, 'lidar.bin'))
            m['lidar'] = dict(plik='lidar.bin', punktow=len(P))
            with open(os.path.join(d, 'meta.tmp'), 'w') as f:
                json.dump(m, f, indent=1)
            os.replace(os.path.join(d, 'meta.tmp'), os.path.join(d, 'meta.json'))
        print(log, len(metas), 'jednostek, stała poza:', sum(metas[j['klucz']]['poza'] == 'stala' for j in J), flush=True)
    d = np.array(decyzje, dtype=float)
    s = d[:, 0] > 0
    print('obiekty sztywne: %d, stała poza %d, ramka %d' % (len(d), s.sum(), (~s).sum()))
    print('grubość cm przy stałej pozie: ramka %.2f -> stała %.2f' % (np.nanmedian(d[s, 1]), np.nanmedian(d[s, 2])))
    if (~s).any():
        print('grubość cm przy ramce: ramka %.2f, stała byłaby %.2f' % (np.nanmedian(d[~s, 1]), np.nanmedian(d[~s, 2])))


if __name__ == '__main__':
    main()
