# -*- coding: utf-8 -*-
"""Wspólny zbiór testowy ze wszystkich ręcznych anotacji: ARKitScenes, HOPE, HANDAL, Argoverse 2, 3DPW.

    python data_arkitscenes.py && python -P -W ignore data_razem.py

Układ jak w data_arkitscenes.py (dane/kp_razem/): views.json, gt.json, prompts.json, <scena>/frazy.json (każda scena
ma własne punkty), <scena>/<widok>/{rgb.jpg, cam.json, kp.json, lidar.png jeśli jest głębia}.
Widoki: frame = pełne zdjęcie dla wszystkich metod; crop = wycinek wokół obiektu dla wskaźników i modeli
jednoprzejściowych, głębię bierze z pełnego zdjęcia (cam.json: pelny); pelny = pełne zdjęcie tylko dla modeli głębi.

Świat sceny:
  arkit_<id>     świat ARKit; punkty z siatki, klatki z data_arkitscenes.py
  hope_, handal_ układ modelu obiektu w metrach; T_wc z pozy 6D, więc zdjęcia z różnych scen BOP się składają.
                 Tylko zdjęcia z jednym egzemplarzem obiektu. Widoczny = w masce mask_visib i pierwszy na promieniu
                 w z-buforze modelu
  av2_, av2c_    ramka obiektu; punkt liczony od nowa z kliknięć (pozy mogły się zmienić po zapisie), widoczny tam,
                 gdzie kliknęła anotatorka. av2c_ to te same ujęcia w wycinkach. Opis obiektu w nawiasie frazy
  3dpw_          każda klatka osobno, świat = kamera (ciało się rusza). Prawa strona z lustra lewej. 3DPW to
                 dopasowanie, nie pomiar, więc liczy się tylko odległość między punktami ciała
Wszystkie punkty zostają; gt.json: jakosc[punkt] ma lidar_cm, residuum_px, baza_st (AV2), a ocena liczy też podzbiór
potwierdzony LiDAR-em.
"""
import collections
import glob
import json
import os
import re
import shutil
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ANNOT = os.path.join(HERE, 'annot')
import paths  # noqa: E402
DANE = paths.DANE
OUT = os.path.join(DANE, 'kp_razem')
sys.path[:0] = [ANNOT, os.path.join(HERE, 'ludzie')]
import przygotuj_av2 as AV2        # noqa: E402
import przygotuj_obj as PO         # noqa: E402

N_BOP = 12                 # zdjęć na obiekt
N_3DPW = 14                # sekwencji
POMIN_3DPW = {'flat_packBags_00'}   # osoba zasłania się trzymanym ubraniem, a widoczność liczy tylko ludzi
TEMPLATE = 'Point to the {phrase} in this image.'

# opisy obiektów z arkusza wycinków (30.10); pole opis_obiektu zostało puste, a w kadrze bywa kilka aut i rowerów
OPISY_AV2 = {
    '070bbf42_4390': 'the light blue bike-share bicycle at the bike rack',
    '070bbf42_59f2': 'the black sedan parked at the curb',
    '070bbf42_83cc': 'the black mountain bicycle locked to the signpost',
    '070bbf42_8bae': 'the silver sedan parked at the curb',
    '070bbf42_a035': 'the silver station wagon parked at the curb',
    '51bbdd4d_e549': 'the bicycle parked behind the dark car',
    'a060c4c1_080d': 'the bicycle with a black bag on its handlebar',
    'a060c4c1_17db': 'the black bicycle in the shade',
    'a060c4c1_c515': 'the black cylindrical bollard with a ribbed top',
    'a060c4c1_d089': 'the man in a black suit talking on the phone',
    'b213af37_02a8': 'the white pickup work truck',
    'b213af37_2cdb': 'the white postal box truck',
    'b213af37_6ded': 'the orange and white construction barrel on the black base',
    'b213af37_74c4': 'the road worker in a yellow safety vest',
    'b213af37_f165': 'the yellow school bus',
    'd5fa4d54_1e79': 'the red DO NOT ENTER sign on a metal post',
    'd5fa4d54_aaff': 'the man with a backpack, black t-shirt and denim shorts',
    'd5fa4d54_b05b': 'the man in a black t-shirt riding the black bicycle',
    'f2576c8a_69f1': 'the orange and white construction barrel with the yellow cable on top',
}
# etykieta Argoverse jest zła: to znak zakazu wjazdu, nie stop
ZAMIANA_FRAZY = {'d5fa4d54_1e79': ('stop sign', 'sign')}


def slug(s):
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')


def bez_przedimka(s):
    return re.sub(r'^(a|an|the)\s+', '', s.strip(), flags=re.I)


def T_wc(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return np.linalg.inv(T)


def rzut(R, t, K, X):
    c = R @ np.asarray(X) + t
    return (K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2], float(c[2]))


class Zbior:
    def __init__(self):
        self.views, self.gt = [], {}

    def scena(self, scena, zbior, P, frazy, **inne):
        os.makedirs(os.path.join(OUT, scena), exist_ok=True)
        self.gt[scena] = dict(points_world={k: [float(x) for x in v] for k, v in P.items()}, zbior=zbior, **inne)
        json.dump(frazy, open(os.path.join(OUT, scena, 'frazy.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def widok(self, scena, nazwa, kind, zrodlo, K, T, W, H, kp, pelny=None):
        """zrodlo: ścieżka do JPEG-a albo obraz BGR."""
        vid = '%s/%s' % (scena, nazwa)
        d = os.path.join(OUT, scena, nazwa)
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, 'rgb.jpg')
        if isinstance(zrodlo, np.ndarray):
            cv2.imwrite(dst, zrodlo, [cv2.IMWRITE_JPEG_QUALITY, 93])
        elif zrodlo.lower().endswith(('.jpg', '.jpeg')):
            shutil.copyfile(zrodlo, dst)
        else:
            cv2.imwrite(dst, cv2.imread(zrodlo), [cv2.IMWRITE_JPEG_QUALITY, 93])
        cam = dict(K=np.asarray(K).tolist(), T_wc=np.asarray(T).tolist(), W=int(W), H=int(H))
        if pelny:
            cam['pelny'] = pelny
        json.dump(cam, open(os.path.join(d, 'cam.json'), 'w'), indent=1)
        json.dump(kp, open(os.path.join(d, 'kp.json'), 'w'), indent=1)
        self.views.append(dict(id=vid, scene=scena, kind=kind, zbior=self.gt[scena]['zbior']))
        return vid


def wpis(q, in_frame, visible):
    return dict(u=float(q[0]), v=float(q[1]), z=float(q[2]), in_frame=bool(in_frame), visible=bool(visible))


def arkit(z):
    kp = os.path.join(DANE, 'arkitscenes', 'kp')
    gt = json.load(open(os.path.join(kp, 'gt.json')))['scenes']
    fr = json.load(open(os.path.join(kp, 'prompts.json'), encoding='utf-8'))['furniture']
    for v in json.load(open(os.path.join(kp, 'views.json'))):
        s = 'arkit_' + v['scene']
        if s not in z.gt:
            P = gt[v['scene']]['points_world']
            z.scena(s, 'arkit', P, {n: fr[n] for n in P}, zrodlo='siatka ARKit, ręcznie')
        nazwa = v['id'].split('/')[1]
        d, src = os.path.join(OUT, s, nazwa), os.path.join(kp, v['id'])
        os.makedirs(d, exist_ok=True)
        for f in ('rgb.jpg', 'cam.json', 'kp.json', 'lidar.png'):
            shutil.copyfile(os.path.join(src, f), os.path.join(d, f))
        z.views.append(dict(id='%s/%s' % (s, nazwa), scene=s, kind='frame', zbior='arkit'))
    print('arkit: scen', sum(k.startswith('arkit_') for k in z.gt), 'punktów', sum(len(x['points_world']) for k, x in z.gt.items() if k.startswith('arkit_')), flush=True)


def ply_ascii(p):
    with open(p, 'rb') as f:
        n = f = None or f
        linie = f.read().decode('latin1').split('\n')
    i = linie.index('end_header') if 'end_header' in linie else [k for k, x in enumerate(linie) if x.strip() == 'end_header'][0]
    nv = int(next(x for x in linie[:i] if x.startswith('element vertex')).split()[2])
    nf = int(next(x for x in linie[:i] if x.startswith('element face')).split()[2])
    V = np.array([x.split()[:3] for x in linie[i + 1:i + 1 + nv]], dtype=np.float64)
    F = np.array([x.split()[1:4] for x in linie[i + 1 + nv:i + 1 + nv + nf]], dtype=np.int64)
    return V / 1000, F


def zbufor(V, F, R, t, K, W, H):
    c = V @ R.T + t
    uv = c[:, :2] / c[:, 2:] * [K[0, 0], K[1, 1]] + K[:2, 2]
    zf = c[F, 2].mean(1)
    pk = np.round(uv[F]).astype(np.int32)
    bz = np.full((H, W), np.inf, np.float32)
    for i in np.argsort(-zf):
        cv2.fillConvexPoly(bz, pk[i], float(zf[i]))
    return bz


def wybierz_bop(zbior, obj, wyst):
    """Do N_BOP ostrych zdjęć z jednym egzemplarzem obiektu, po równo ze scen, rozłożone w czasie nagrania."""
    ile = collections.Counter((o[0], o[1]) for o in wyst)
    sc = collections.defaultdict(list)
    for o in wyst:
        if ile[(o[0], o[1])] == 1 and o[3] >= PO.MIN_WIDOCZNOSC:
            sc[o[0]].append(o)
    kand = []
    for s, lst in sorted(sc.items()):
        lst = sorted(lst, key=lambda o: o[1])
        for j in np.linspace(0, len(lst) - 1, min(len(lst), N_BOP)).astype(int):
            crop, _ = PO.podglad(zbior, lst[j][0], lst[j][1], lst[j][2], lst[j][4], 0)
            kand.append((s, PO.ostrosc(crop), lst[j]))
    if not kand:
        return []
    prog = max(PO.MIN_OSTROSC, 0.25 * max(k[1] for k in kand))
    grupy = collections.defaultdict(list)
    for s, o, x in kand:
        if o >= prog:
            grupy[s].append(x)
    out = []
    while len(out) < N_BOP and any(grupy.values()):
        for s in sorted(grupy):
            if grupy[s] and len(out) < N_BOP:
                out.append(grupy[s].pop(len(grupy[s]) // 2))
    return out


def bop(z):
    wyst = {}
    for f in sorted(glob.glob(os.path.join(ANNOT, 'punkty_obj', '*.json'))):
        d = json.load(open(f, encoding='utf-8'))
        zb, obj, s = d['zbior'], d['obj_id'], d['klucz']
        if not d['punkty']:
            continue
        wyst.setdefault(zb, None)
        if wyst[zb] is None:
            wyst[zb] = PO.wystapienia(zb)
        P = {slug(p['nazwa']): np.array(p['xyz_mm']) / 1000 for p in d['punkty']}
        z.scena(s, zb, P, {slug(p['nazwa']): bez_przedimka(p['opis']) for p in d['punkty']}, zrodlo='model BOP, ręcznie', obiekt=d['nazwa'])
        V, F = ply_ascii(os.path.join(ANNOT, 'obiekty', s, 'model.ply'))
        nw = 0
        for sc, im, i, _, _ in wybierz_bop(zb, obj, wyst[zb][obj]):
            dd = os.path.join(PO.BOP, zb, 'val', sc)
            cam = PO.wczytaj_json(os.path.join(dd, 'scene_camera.json'))[str(im)]
            g = PO.wczytaj_json(os.path.join(dd, 'scene_gt.json'))[str(im)][i]
            K = np.array(cam['cam_K']).reshape(3, 3)
            R, t = np.array(g['cam_R_m2c']).reshape(3, 3), np.array(g['cam_t_m2c']) / 1000
            plik = next(x for x in sorted(glob.glob(os.path.join(dd, 'rgb', '%06d.*' % im))))
            H, W = cv2.imread(plik).shape[:2]
            maska = cv2.dilate(cv2.imread(os.path.join(dd, 'mask_visib', '%06d_%06d.png' % (im, i)), 0), np.ones((7, 7), np.uint8))
            bz = zbufor(V, F, R, t, K, W, H)
            kp = {}
            for n, X in P.items():
                q = rzut(R, t, K, X)
                inf = q[2] > 0.05 and 0 <= q[0] < W and 0 <= q[1] < H
                wid = False
                if inf:
                    x, y = int(q[0]), int(q[1])
                    wid = maska[y, x] > 0 and q[2] - bz[max(0, y - 1):y + 2, max(0, x - 1):x + 2].min() <= 0.008
                kp[n] = wpis(q, inf, wid)
            vid = z.widok(s, '%s_%06d' % (sc, im), 'frame', plik, K, T_wc(R, t), W, H, kp)
            gl = os.path.join(dd, 'depth', '%06d.png' % im)
            if os.path.exists(gl):
                D = np.array(Image.open(gl)).astype(np.float32) * float(cam.get('depth_scale', 1.0))
                Image.fromarray(np.clip(D, 0, 65535).astype(np.uint16)).save(os.path.join(OUT, vid, 'lidar.png'))
            nw += 1
        print(s, 'punktów', len(P), 'zdjęć', nw, flush=True)


def trianguluj(m, pt):
    kl = [c for c in pt['klikniecia'] if c['podglad'] < len(m['podglady'])]
    wid = [(np.array(m['podglady'][c['podglad']]['K']), np.array(m['podglady'][c['podglad']]['R']), np.array(m['podglady'][c['podglad']]['t_m']), c) for c in kl]
    A, b, dd = np.zeros((3, 3)), np.zeros(3), []
    for K, R, t, c in wid:
        o = -R.T @ t
        v = R.T @ [(c['u'] - K[0, 2]) / K[0, 0], (c['v'] - K[1, 2]) / K[1, 1], 1]
        v /= np.linalg.norm(v)
        dd.append(v)
        Q = np.eye(3) - np.outer(v, v)
        A += Q
        b += Q @ o
    X = np.linalg.solve(A, b)

    def res(Y):
        return np.ravel([np.array(rzut(R, t, K, Y)[:2]) - [c['u'], c['v']] for K, R, t, c in wid])
    X = least_squares(res, X, x_scale=0.1).x
    r = res(X).reshape(-1, 2)
    kat = max(np.degrees(np.arccos(np.clip(abs(a @ c), 0, 1))) for i, a in enumerate(dd) for c in dd[i + 1:])
    return X, float(np.hypot(r[:, 0], r[:, 1]).max()), float(kat)


def av2(z):
    logi = {}
    for f in sorted(glob.glob(os.path.join(ANNOT, 'punkty_ctx', '*.json'))):
        d = json.load(open(f, encoding='utf-8'))
        k = d['klucz']
        if not d['punkty']:
            continue
        kr = k[4:17]
        m = json.load(open(os.path.join(ANNOT, 'kontekst', k, 'meta.json')))
        L = logi.setdefault(m['log'], None) or AV2.Log(m['log'])
        logi[m['log']] = L
        ch = np.fromfile(os.path.join(ANNOT, 'kontekst', k, 'lidar.bin'), dtype=np.float32).reshape(-1, 3)
        T = cKDTree(ch) if len(ch) else None
        P, jak, fr, klik = {}, {}, {}, {}
        for pt in d['punkty']:
            n = slug(pt['opis'])
            X, res, kat = trianguluj(m, pt)
            P[n] = X
            jak[n] = dict(lidar_cm=float(T.query(X)[0] * 100) if T else None, residuum_px=res, baza_st=kat, klikniec=len(pt['klikniecia']))
            o = pt['opis']
            if kr in ZAMIANA_FRAZY:
                o = o.replace(*ZAMIANA_FRAZY[kr])
            fr[n] = '%s (%s)' % (bez_przedimka(o), OPISY_AV2[kr])
            klik[n] = {c['podglad'] for c in pt['klikniecia']}
        sf, sc = 'av2_' + kr, 'av2c_' + kr
        for s, zb in ((sf, 'av2'), (sc, 'av2c')):
            z.scena(s, zb, P, fr, zrodlo='triangulacja kliknięć', jakosc=jak, sztywny=m['okno_ns'] is None, kategoria=d['kategoria'],
                    poza=m.get('poza'), opis_obiektu=OPISY_AV2[kr])
        for i, p in enumerate(m['podglady']):
            R, t = np.array(p['R']), np.array(p['t_m'])
            Tw = T_wc(R, t)
            K = L.K[p['kam']]
            W, H = L.wh[p['kam']]
            kp = {}
            for n, X in P.items():
                q = rzut(R, t, K, X)
                kp[n] = wpis(q, q[2] > 0 and 0 <= q[0] < W and 0 <= q[1] < H, i in klik[n])
            pel = z.widok(sf, str(i), 'frame', os.path.join(L.d, 'sensors', 'cameras', p['kam'], '%d.jpg' % p['t']), K, Tw, W, H, kp)
            hd = os.path.join(ANNOT, 'kontekst', k, p['hd'])
            nb = Image.open(hd).size[0]
            Kc = np.array(p['K'])
            Kc[:2] *= nb / AV2.BOK
            kp = {}
            for n, X in P.items():
                q = rzut(R, t, Kc, X)
                kp[n] = wpis(q, q[2] > 0 and 0 <= q[0] < nb and 0 <= q[1] < nb, i in klik[n])
            z.widok(sc, str(i), 'crop', hd, Kc, Tw, nb, nb, kp, pelny=pel)
        print(k, d['kategoria'], 'punktów', len(P), 'ujęć', len(m['podglady']), flush=True)


def tdpw(z):
    import przygotuj_3dpw as L3
    from pkl_bezpieczny import wczytaj
    from smpl_np import SMPL
    pk = json.load(open(os.path.join(ANNOT, 'punkty_ludzie.json'), encoding='utf-8'))['punkty']
    lustro = json.load(open(os.path.join(ANNOT, 'ludzie', 'lustro.json')))
    wzor = []
    for p in pk:
        tr, ba = np.array(p['trojkat']), np.array(p['bary'])
        wzor.append((p['nazwa'], p['opis'], tr, ba))
        if 'left' in p['nazwa']:
            wzor.append((p['nazwa'].replace('left', 'right'), p['opis'].replace('left', 'right'), np.array([lustro[v] for v in tr]), ba))
    fr = {n: bez_przedimka(o) + ' (the person in the center of the image)' for n, o, _, _ in wzor}
    wyb = collections.OrderedDict()
    for _, kier, sek, plik, os_i, kl in sorted(L3.kandydaci(), key=lambda k: -k[0]):
        if sek in POMIN_3DPW:
            continue
        lst = wyb.setdefault(sek, [])
        if len(lst) < 3 and all(x[0] != kier and abs(x[3] - kl) >= 90 for x in lst):
            lst.append((kier, plik, os_i, kl))
    mod, n_sek = {}, 0
    for sek, lst in wyb.items():
        if n_sek == N_3DPW:
            break
        d = wczytaj(lst[0][1])
        K = np.asarray(d['cam_intrinsics'])
        nk = 0
        for kier, _, os_i, kl in lst:
            C = np.asarray(d['cam_poses'][kl])
            zr = os.path.join(L3.D3PW, 'imageFiles', sek, 'image_%05d.jpg' % kl)
            img = cv2.imread(zr)
            H, W = img.shape[:2]
            V_kam = []
            for o in range(len(d['poses'])):
                g = str(d['genders'][o])
                if g not in mod:
                    mod[g] = SMPL(g)
                v = mod[g](np.asarray(d['betas'][o]), np.asarray(d['poses'][o][kl]), np.asarray(d['trans'][o][kl]))
                V_kam.append(v @ C[:3, :3].T + C[:3, 3])
            crop, M = L3.wycinek(img, L3.rzut(V_kam[os_i], K))
            if cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var() < L3.MIN_OSTROSC:
                continue
            uvs = [L3.rzut(v, K) @ M[:2, :2].T + M[:2, 2] for v in V_kam]
            wid = L3.widocznosc(uvs, [v[:, 2] for v in V_kam], mod[str(d['genders'][0])].faces, os_i)
            V = V_kam[os_i]
            P = {n: (V[tr] * ba[:, None]).sum(0) for n, _, tr, ba in wzor}
            s = '3dpw_%s_%d_%05d' % (sek, os_i, kl)
            z.scena(s, '3dpw', P, fr, zrodlo='dopasowanie SMPL, nie pomiar', kierunek=kier)
            Kc = np.array([[K[0, 0] * M[0, 0], 0, K[0, 2] * M[0, 0] + M[0, 2]], [0, K[1, 1] * M[1, 1], K[1, 2] * M[1, 1] + M[1, 2]], [0, 0, 1]])
            kp_c, kp_p = {}, {}
            for n, _, tr, ba in wzor:
                q = rzut(np.eye(3), np.zeros(3), Kc, P[n])
                w = 0 <= q[0] < L3.BOK and 0 <= q[1] < L3.BOK and wid[tr].sum() >= 2
                kp_c[n] = wpis(q, 0 <= q[0] < L3.BOK and 0 <= q[1] < L3.BOK, w)
                kp_p[n] = wpis(rzut(np.eye(3), np.zeros(3), K, P[n]), True, w)
            pel = z.widok(s, 'pelny', 'pelny', zr, K, np.eye(4), W, H, kp_p)
            z.widok(s, 'crop', 'crop', crop, Kc, np.eye(4), L3.BOK, L3.BOK, kp_c, pelny=pel)
            nk += 1
        if nk:
            n_sek += 1
            print(sek, 'klatek', nk, flush=True)
    print('3dpw: punktów na klatkę', len(wzor), flush=True)


def main():
    if os.path.isdir(OUT):
        for x in os.listdir(OUT):                      # wyniki modeli zostają, reszta powstaje od nowa
            if x not in ('preds', 'results'):
                p = os.path.join(OUT, x)
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    os.makedirs(OUT, exist_ok=True)
    z = Zbior()
    arkit(z)
    bop(z)
    av2(z)
    tdpw(z)
    json.dump(dict(_doc='Frazy są w <scena>/frazy.json.', template=TEMPLATE, furniture={}, body={}),
              open(os.path.join(OUT, 'prompts.json'), 'w', encoding='utf-8'), indent=1)
    json.dump(z.views, open(os.path.join(OUT, 'views.json'), 'w'), indent=0)
    json.dump(dict(dataset='ARKitScenes, HOPE, HANDAL, Argoverse 2, 3DPW; ręczna anotacja', scenes=z.gt),
              open(os.path.join(OUT, 'gt.json'), 'w', encoding='utf-8'), indent=1)
    c = collections.Counter((v['zbior'], v['kind']) for v in z.views)
    print('scen', len(z.gt), 'widoków', len(z.views), dict(c))


if __name__ == '__main__':
    main()
