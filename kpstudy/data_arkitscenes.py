# -*- coding: utf-8 -*-
"""Zbiór testowy z ARKitScenes w formacie kpstudy: prawdziwe sceny z biurkiem, pozy z ARKit (VIO iPada).

    python data_arkitscenes.py                 # sceny z ręczną anotacją (annot/punkty/<id>.json)
    python data_arkitscenes.py --z-pudelek     # próbne punkty z pudełek 3D (tylko do testu potoku)

Wejście (dane/arkitscenes/): klatki/<id>/{rgb,lidar}/ i klatki.json z klatki.py, raw/<id>/ z pobierz.py,
punkty z narzędzia annot/annotate.html (układ świata ARKitScenes, metry, oś z w górę).

Wyjście (dane/arkitscenes/kp/, albo kp_test/ dla --z-pudelek):
  views.json             [{"id": "<scena>/<klatka>", "scene", "ts", "kind": "frame"}]
  prompts.json           frazy dla modeli tekst->punkt (z annot/slownik.json), format jak kpstudy/prompts.json
  gt.json                punkty prawdy w świecie dla każdej sceny
  <scena>/<klatka>/rgb.jpg     zdjęcie obrócone niebem do góry (to dostają metody)
  <scena>/<klatka>/cam.json    K i T_wc tego obróconego zdjęcia, W, H, ts
  <scena>/<klatka>/kp.json     dla każdego punktu: u, v, z (m), in_frame, visible
  <scena>/<klatka>/lidar.png   głębia LiDAR iPada (mm, 256x192, obrócona) - tylko do widoczności i wyroczni
Widoczność: punkt w kadrze, z > 0,1 m i nie głębiej niż LiDAR w otoczeniu 5x5 plus max(5 cm, 3% z).
"""
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'annot'))
import arkit_io as A  # noqa: E402

DANE = os.path.abspath(os.path.join(A.ROOT, '..'))
SLOWNIK = os.path.join(HERE, 'annot', 'slownik.json')


def punkty_z_pudelek(s):
    """Próbne punkty (tylko test potoku): narożniki blatu i środki monitora i krzesła z pudełek 3D."""
    from przygotuj import pudelka, biurko
    boxes = pudelka(s)
    for b in boxes:
        b['R'] = b['N'].T
    t, mon, krz = biurko(boxes)
    P = {}
    for nazwa, sx, sy in (('desk_top_front_left', -1, -1), ('desk_top_front_right', 1, -1),
                          ('desk_top_back_left', -1, 1), ('desk_top_back_right', 1, 1)):
        P[nazwa] = (t['c'] + t['R'] @ (np.array([sx, sy, 1]) * t['L'] / 2)).tolist()
    P['monitor_screen_center'] = mon[0]['c'].tolist()
    P['seat_top_center'] = krz[0]['c'].tolist()
    return P


def lidar_png(scena_dir, nazwa):
    p = os.path.join(scena_dir, 'lidar', nazwa + '.png')
    return np.array(Image.open(p)).astype(np.float64) / 1000.0


def widocznosc(uv, z, D, W, H):
    """D: głębia LiDAR (m) w tej samej orientacji co obraz, dowolna rozdzielczość."""
    h, w = D.shape
    out = []
    for (u, v), zz in zip(uv, z):
        inf = bool(zz > 0.1 and 0 <= u < W and 0 <= v < H)
        vis = False
        if inf:
            x, y = int(u * w / W), int(v * h / H)
            okno = D[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
            okno = okno[okno > 0]
            vis = bool(okno.size == 0 or zz <= okno.max() + max(0.05, 0.03 * zz))
        out.append((inf, vis))
    return out


def main():
    test = '--z-pudelek' in sys.argv
    out_root = os.path.join(DANE, 'kp_test' if test else 'kp')
    slownik = json.load(open(SLOWNIK, encoding='utf-8'))
    os.makedirs(out_root, exist_ok=True)
    json.dump({'_doc': 'Frazy z ciop/kpstudy/annot/slownik.json dla scen ARKitScenes (bez ludzi).',
               'template': slownik['template'], 'furniture': {p['nazwa']: p['phrase'] for p in slownik['punkty']}, 'body': {}},
              open(os.path.join(out_root, 'prompts.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    views, gt = [], {}
    for v in sorted(os.listdir(os.path.join(DANE, 'klatki'))):
        kj = os.path.join(DANE, 'klatki', v, 'klatki.json')
        pj = os.path.join(HERE, 'annot', 'punkty', v + '.json')
        if not os.path.exists(kj) or (not test and not os.path.exists(pj)):
            continue
        s = A.Scene(v)
        if test:
            P = punkty_z_pudelek(s)
        else:
            P = {k: x for k, x in json.load(open(pj, encoding='utf-8'))['punkty'].items() if isinstance(x, list)}
        gt[v] = dict(points_world={k: list(map(float, x)) for k, x in P.items()}, sky=s.sky,
                     zrodlo='pudełka 3D (test)' if test else 'ręczna anotacja na siatce ARKit')
        names = list(P)
        X = np.array([P[n] for n in names])
        kl = json.load(open(kj))
        for nazwa, t in zip(kl['klatki'], kl['ts']):
            T, dt = s.pose_at(t)
            K, _ = s.K_at(t)
            rgb = np.array(Image.open(os.path.join(DANE, 'klatki', v, 'rgb', nazwa + '.jpg')).convert('RGB'))
            D = lidar_png(os.path.join(DANE, 'klatki', v), nazwa)
            D_full = np.array(Image.fromarray(D.astype(np.float32)).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))
            rgb_u, K_u, T_u = A.upright(rgb, K, T, s.sky)
            D_u = A.upright(D_full, K, T, s.sky)[0]
            H_, W_ = rgb_u.shape[:2]
            uv, z = A.project(X, T_u, K_u)
            vis = widocznosc(uv, z, D_u, W_, H_)
            vid = '%s/%s' % (v, nazwa)
            d = os.path.join(out_root, v, nazwa)
            os.makedirs(d, exist_ok=True)
            Image.fromarray(rgb_u).save(os.path.join(d, 'rgb.jpg'), quality=95)
            rozm = (192, 256) if W_ < H_ else (256, 192)          # natywna rozdzielczość LiDAR, w orientacji zdjęcia
            small = np.array(Image.fromarray(D_u.astype(np.float32)).resize(rozm, Image.NEAREST))
            Image.fromarray((small * 1000).astype(np.uint16)).save(os.path.join(d, 'lidar.png'))
            json.dump(dict(K=K_u.tolist(), T_wc=np.linalg.inv(T_u).tolist(), W=W_, H=H_, ts=t, dt_pozy=dt, sky=s.sky),
                      open(os.path.join(d, 'cam.json'), 'w'), indent=1)
            json.dump({n: dict(u=float(a[0]), v=float(a[1]), z=float(b), in_frame=f, visible=w)
                       for n, a, b, (f, w) in zip(names, uv, z, vis)}, open(os.path.join(d, 'kp.json'), 'w'), indent=1)
            views.append(dict(id=vid, scene=v, ts=t, kind='frame'))
        print(v, 'klatek', len(kl['klatki']), 'punktów', len(names), flush=True)
    json.dump(views, open(os.path.join(out_root, 'views.json'), 'w'), indent=0)
    json.dump(dict(dataset='ARKitScenes (Apple, licencja niekomercyjna)', uklad='świat ARKitScenes, m, oś z w górę',
                   scenes=gt), open(os.path.join(out_root, 'gt.json'), 'w'), indent=1)
    print('scen', len(gt), 'widoków', len(views))


if __name__ == '__main__':
    main()
