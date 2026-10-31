# -*- coding: utf-8 -*-
"""Przygotowuje 3DPW do anotacji w annotate_ludzie.html (http://127.0.0.1:8765/ludzie).

    python -P przygotuj_3dpw.py

Wszyscy ludzie w 3DPW mają tę samą siatkę SMPL, więc punkt zaznacza się raz, na jednym wzorcowym ciele, jako
miejsce na siatce (trójkąt + współrzędne barycentryczne) i przechodzi na każdą osobę w każdej klatce.
Zapisuje w annot/ludzie/: cialo.ply (średnia z modelu męskiego i żeńskiego, opuszczone ręce, metry), lustro.json (lustrzany
wierzchołek dla każdego), meta.json oraz 8 podglądów prawdziwych osób (podglad_k.jpg + podglad_k.json: rzut
wszystkich wierzchołków na wycinek i ich widoczność), żeby strona od razu pokazywała, gdzie punkt wypada.

Kamera: cam_poses = świat -> kamera, K z cam_intrinsics. Ciało jak w demo 3DPW: model płci, betas[:10], pose, trans.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pkl_bezpieczny import wczytaj     # noqa: E402
from smpl_np import SMPL               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import paths  # noqa: E402
D3PW = os.path.join(paths.DANE, '3dpw')
OUT = os.path.abspath(os.path.join(HERE, '..', 'annot', 'ludzie'))
BOK = 640
MIN_OSTROSC = 20
CO_KLATKE = 15            # przegląd sekwencji co 0,5 s


def rzut(X, K):
    return X[:, :2] / X[:, 2:] * [K[0, 0], K[1, 1]] + K[:2, 2]


def kandydaci():
    """(ocena, kierunek, sekwencja, plik, osoba, klatka) dla dużych, całych osób; kierunek: przod/tyl/lewy/prawy."""
    out = []
    for p in sorted(glob.glob(os.path.join(D3PW, 'sequenceFiles', '*', '*.pkl'))):
        d = wczytaj(p)
        sek = str(d['sequence'])
        K = np.asarray(d['cam_intrinsics'])
        ob = glob.glob(os.path.join(D3PW, 'imageFiles', sek, 'image_00000.jpg'))
        if not ob:
            continue
        H, W = cv2.imread(ob[0]).shape[:2]
        for os_i in range(len(d['poses'])):
            J = np.asarray(d['jointPositions'][os_i]).reshape(-1, 24, 3)
            for kl in range(0, len(J), CO_KLATKE):
                if not d['campose_valid'][os_i][kl]:
                    continue
                C = np.asarray(d['cam_poses'][kl])
                X = J[kl] @ C[:3, :3].T + C[:3, 3]
                if X[:, 2].min() < 0.5:
                    continue
                uv = rzut(X, K)
                x0, y0 = uv.min(0)
                x1, y1 = uv.max(0)
                if x0 < 30 or y0 < 60 or x1 > W - 30 or y1 > H - 30 or y1 - y0 < 0.5 * H:
                    continue
                L = X[1] - X[2]                        # lewe biodro - prawe, w układzie kamery
                c = L[0] / (np.linalg.norm(L[[0, 2]]) + 1e-9)
                # przodem do kamery lewe biodro osoby jest po prawej stronie obrazu
                kier = 'przod' if c > 0.85 else 'tyl' if c < -0.85 else None
                if kier is None and abs(c) < 0.35:
                    kier = 'lewy' if L[2] < 0 else 'prawy'  # bliżej kamery jest lewy / prawy bok
                if kier:
                    out.append((float(y1 - y0), kier, sek, p, os_i, kl))
    return out


def wycinek(img, uv):
    x0, y0 = uv.min(0)
    x1, y1 = uv.max(0)
    s = max(x1 - x0, y1 - y0) * 1.2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x_, y_ = cx - s / 2, cy - s / 2
    sk = BOK / s
    M = np.array([[sk, 0, -x_ * sk], [0, sk, -y_ * sk]])
    return cv2.warpAffine(img, M, (BOK, BOK), flags=cv2.INTER_CUBIC), M


def widocznosc(uvs, zs, faces, cel):
    """Bufor numerów trójkątów (malowanie od najdalszych); wierzchołek osoby `cel` jest widoczny, gdy w jego okolicy
    wygrał trójkąt tej osoby, który go zawiera albo leży tuż obok (ta sama głębokość)."""
    ids, glebia = [], []
    for o, (uv, z) in enumerate(zip(uvs, zs)):
        ids.append(np.c_[np.full(len(faces), o), np.arange(len(faces))])
        glebia.append(z[faces].mean(1))
    ids, glebia = np.vstack(ids), np.concatenate(glebia)
    bufor = np.full((BOK, BOK), -1, np.int32)
    for i in np.argsort(-glebia):
        o, f = ids[i]
        cv2.fillConvexPoly(bufor, np.round(uvs[o][faces[f]]).astype(np.int32), int(i))
    uv, z = uvs[cel], zs[cel]
    wid = np.zeros(len(uv), bool)
    for v in range(len(uv)):
        x, y = int(round(uv[v, 0])), int(round(uv[v, 1]))
        if not (0 <= x < BOK and 0 <= y < BOK):
            continue
        ok = bufor[max(0, y - 1):y + 2, max(0, x - 1):x + 2].ravel()
        ok = ok[ok >= 0]
        wid[v] = len(ok) > 0 and bool(np.any((ids[ok, 0] == cel) & (np.abs(glebia[ok] - z[v]) < 0.04)))
    return wid


def main():
    os.makedirs(OUT, exist_ok=True)
    n, n2 = SMPL('m'), SMPL('f')
    # ciało wzorcowe: średnia z modelu męskiego i żeńskiego (ta sama siatka, modele v1.0.0 jak w demo 3DPW),
    # ręce opuszczone o ok. 50 stopni, nogi lekko rozstawione
    p = np.zeros(72)
    p[16 * 3 + 2], p[17 * 3 + 2], p[1 * 3 + 2], p[2 * 3 + 2] = -0.9, 0.9, 0.08, -0.08
    V = (n(np.zeros(10), p) + n2(np.zeros(10), p)) / 2
    V = V - V.mean(0)
    with open(os.path.join(OUT, 'cialo.ply'), 'w', newline='\n') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n'
                'element face %d\nproperty list uchar int vertex_indices\nend_header\n' % (len(V), len(n.faces)))
        f.writelines('%.5f %.5f %.5f\n' % tuple(v) for v in V)
        f.writelines('3 %d %d %d\n' % tuple(t) for t in n.faces)
    # lustro: najbliższy wierzchołek po odbiciu x -> -x na szablonie bez pozy (jest symetryczny)
    T = n.v_template - n.v_template.mean(0)
    lustro = [int(np.argmin(((T - v * [-1, 1, 1]) ** 2).sum(1))) for v in T]
    json.dump(lustro, open(os.path.join(OUT, 'lustro.json'), 'w'))

    kand = sorted(kandydaci(), key=lambda k: -k[0])
    print('kandydatów:', len(kand), flush=True)
    mod = {}
    wyb, sekwencje, podg = {'przod': 0, 'tyl': 0, 'lewy': 0, 'prawy': 0}, set(), []
    for ocena, kier, sek, plik, os_i, kl in kand:
        if wyb[kier] >= 2 or sek in sekwencje:
            continue
        d = wczytaj(plik)
        K = np.asarray(d['cam_intrinsics'])
        C = np.asarray(d['cam_poses'][kl])
        img = cv2.imread(os.path.join(D3PW, 'imageFiles', sek, 'image_%05d.jpg' % kl))
        V_kam = []
        for o in range(len(d['poses'])):
            g = str(d['genders'][o])
            if g not in mod:
                mod[g] = SMPL(g)
            v = mod[g](np.asarray(d['betas'][o]), np.asarray(d['poses'][o][kl]), np.asarray(d['trans'][o][kl]))
            V_kam.append(v @ C[:3, :3].T + C[:3, 3])
        crop, M = wycinek(img, rzut(V_kam[os_i], K))
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ost = float(cv2.Laplacian(g, cv2.CV_32F).var())
        if ost < MIN_OSTROSC:
            continue
        uvs = [rzut(v, K) @ M[:2, :2].T + M[:2, 2] for v in V_kam]
        wid = widocznosc(uvs, [v[:, 2] for v in V_kam], n.faces, os_i)
        k = len(podg)
        cv2.imwrite(os.path.join(OUT, 'podglad_%d.jpg' % k), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        json.dump(dict(uv=np.round(uvs[os_i], 1).tolist(), wid=wid.astype(int).tolist()), open(os.path.join(OUT, 'podglad_%d.json' % k), 'w'))
        podg.append(dict(plik='podglad_%d.jpg' % k, dane='podglad_%d.json' % k, sekwencja=sek, osoba=os_i, klatka=kl, kierunek=kier, ostrosc=round(ost, 1)))
        wyb[kier] += 1
        sekwencje.add(sek)
        print(k, sek, 'osoba', os_i, 'klatka', kl, kier, 'ostrość %.0f' % ost, 'widocznych %.0f%%' % (100 * wid.mean()), flush=True)
        if len(podg) == 8:
            break
    json.dump(dict(podglady=podg, wierzcholkow=len(V), trojkatow=len(n.faces)), open(os.path.join(OUT, 'meta.json'), 'w'), indent=1)
    print('podglądów:', len(podg))


if __name__ == '__main__':
    main()
