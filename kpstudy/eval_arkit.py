# -*- coding: utf-8 -*-
"""Ocena metod na scenach ARKitScenes (zbiór z data_arkitscenes.py): błąd punktów 3D w cm.

    python eval_arkit.py --root <dane/arkitscenes/kp> [--stage1 molmopoint8b,qwen3vl8b ...] [--e2e robobrain25_8b ...]
                         [--depth moge2_vitl,unidepthv2_l]

Konfiguracje:
  e2e        metoda jednoprzejściowa daje punkt 3D w kamerze (preds/e2e/<m>/<widok>.json, points_cam);
             z kilku zdjęć: punkty przeniesione do świata pozami ARKit i uśrednione
  2 etapy    punkt 2D (preds/stage1/<m>/...) + podniesienie:
               depth_<model>   głębia z jednego zdjęcia (preds/depth/<model>/<widok>.npy, metry, jak rgb.jpg)
               depth_lidar     głębia LiDAR iPada: wyrocznia geometrii (metody jej nie dostają)
               tri_arkit       triangulacja z 2, 3 albo 5 zdjęć z pozami ARKit (prawdziwe VIO)
               tri_arkit_ransac  jw., odporna na pojedyncze złe wskazania
  oracle2d   prawdziwe 2D z kp.json zamiast metody: czysty błąd podniesienia
Błąd liczony tylko dla punktów widocznych na zdjęciu odniesienia (kp.json, visible), w układzie świata. Mediana i p90 z punktów
z odpowiedzią, progi względem wszystkich punktów: brak odpowiedzi 3D to chybienie.
Wynik: <root>/results/rows.jsonl i <root>/results/summary.md.
"""
import argparse
import collections
import itertools
import json
import os

import numpy as np
from PIL import Image

import lift

HERE = os.path.dirname(os.path.abspath(__file__))
PROGI_CM = (5, 10, 20)
LIDAR_PEWNE_CM = 5


class Zbior:
    def __init__(self, root):
        self.root = root
        self.views = json.load(open(os.path.join(root, 'views.json')))
        self.gt = json.load(open(os.path.join(root, 'gt.json')))['scenes']
        self.by_scene = collections.defaultdict(list)
        for v in self.views:
            if v['kind'] != 'pelny':                 # pełne zdjęcia dla modeli głębi, nie są osobnym wskazaniem
                self.by_scene[v['scene']].append(v['id'])

    def cam(self, vid):
        c = json.load(open(os.path.join(self.root, vid, 'cam.json')))
        c['K'] = np.array(c['K'])
        c['T_wc'] = np.array(c['T_wc'])
        return c

    def kp(self, vid):
        return json.load(open(os.path.join(self.root, vid, 'kp.json')))

    def ma_lidar(self, vid):
        return os.path.exists(os.path.join(self.root, vid, 'lidar.png'))

    def lidar(self, vid, shape):
        D = np.array(Image.open(os.path.join(self.root, vid, 'lidar.png'))).astype(np.float32) / 1000
        return np.array(Image.fromarray(D).resize((shape[1], shape[0]), Image.NEAREST))

    def P_world(self, scene):
        return {k: np.array(v) for k, v in self.gt[scene]['points_world'].items()}


def ktuple(K):
    return (K[0, 0], K[1, 1], K[0, 2], K[1, 2])


def do_K(uv, K_src, K_dst):
    """Piksel z kamery o K_src na piksel tej samej wiązki w kamerze o K_dst (te same pozy, inne intrinsics)."""
    x = (uv[0] - K_src[0, 2]) / K_src[0, 0]
    y = (uv[1] - K_src[1, 2]) / K_src[1, 1]
    return x * K_dst[0, 0] + K_dst[0, 2], y * K_dst[1, 1] + K_dst[1, 2]


def do_swiata(X_cam, T_wc):
    return T_wc[:3, :3] @ X_cam + T_wc[:3, 3]


def zestawy(ds, scene, rng):
    """Pojedyncze zdjęcia, pary w przedziałach rozstawu kątowego (patrząc na środek punktów prawdy) i zestawy 3 i 5."""
    ids = ds.by_scene[scene]
    c = np.mean(list(ds.P_world(scene).values()), axis=0)
    poz = {v: ds.cam(v)['T_wc'][:3, 3] for v in ids}
    out = [dict(kind='single', views=[v]) for v in ids]

    def kat(a, b):
        u, w = poz[a] - c, poz[b] - c
        return float(np.degrees(np.arccos(np.clip(u @ w / np.linalg.norm(u) / np.linalg.norm(w), -1, 1))))
    pary = collections.defaultdict(list)
    for a, b in itertools.combinations(ids, 2):
        k = kat(a, b)
        for lo, hi in ((5, 10), (10, 20), (20, 40), (40, 90)):
            if lo <= k < hi:
                pary['%d-%d' % (lo, hi)].append((a, b, k))
    for b, lst in pary.items():
        for i in rng.choice(len(lst), size=min(8, len(lst)), replace=False):
            a, bb, k = lst[i]
            out.append(dict(kind='pair', views=[a, bb], baseline_bin=b, baseline_deg=round(k, 1)))
    for n in (3, 4, 5):                               # 4 = układ kamer ZeroDex
        if len(ids) >= n:
            for _ in range(6):
                out.append(dict(kind='multi%d' % n, views=sorted(rng.choice(ids, size=n, replace=False).tolist())))
    return out


def wczytaj2d(root, m, vid):
    p = os.path.join(root, 'preds', 'stage1', m, vid + '.json')
    if m == 'oracle2d' or not os.path.exists(p):
        return None
    return {k: v[:2] for k, v in json.load(open(p))['points_2d'].items()}


def wczytaj_e2e(root, m, vid):
    p = os.path.join(root, 'preds', 'e2e', m, vid + '.json')
    if not os.path.exists(p):
        return None
    return {k: np.array(v) for k, v in json.load(open(p))['points_cam'].items()}


def wiersz(rows, jak, common, pred_world, Pw, widoczne, obs2d=None, kp_ref=None):
    for n in widoczne:
        r = dict(common, point=n)
        if jak.get(n, {}).get('lidar_cm') is not None:
            r['lidar_cm'] = jak[n]['lidar_cm']
        if n in pred_world:
            r['err_cm'] = float(np.linalg.norm(pred_world[n] - Pw[n]) * 100)
        if obs2d is not None and kp_ref is not None:
            if n in obs2d:
                r['err_px'] = float(np.hypot(obs2d[n][0] - kp_ref[n]['u'], obs2d[n][1] - kp_ref[n]['v']))
        rows.append(r)
    # odległość między dwoma punktami: jeden wiersz na zestaw, mediana po parach
    e = [abs(np.linalg.norm(pred_world[a] - pred_world[b]) - np.linalg.norm(Pw[a] - Pw[b])) * 100
         for a, b in itertools.combinations(widoczne, 2) if a in pred_world and b in pred_world]
    if e:
        rows.append(dict(common, para=len(e), err_para_cm=float(np.median(e))))


def ocen(ds, args):
    rng = np.random.default_rng(0)
    rows = []
    for scene in sorted(ds.by_scene):
        Pw = ds.P_world(scene)
        jak = ds.gt[scene].get('jakosc', {})
        for s in zestawy(ds, scene, rng):
            views = s['views']
            ref = views[0]
            cr = ds.cam(ref)
            kp_ref = ds.kp(ref)
            widoczne = [n for n, k in kp_ref.items() if k['visible'] and n in Pw]
            if not widoczne:
                continue
            common = dict(scene=scene, zbior=ds.gt[scene].get('zbior', 'arkit'), kind=s['kind'], views=views, baseline_bin=s.get('baseline_bin'), baseline_deg=s.get('baseline_deg'))
            # e2e: punkty 3D wprost z modelu
            for m in args.e2e:
                preds = [wczytaj_e2e(ds.root, m, v) for v in views]
                if any(p is None for p in preds):
                    continue
                acc = collections.defaultdict(list)
                for v, p in zip(views, preds):
                    T = ds.cam(v)['T_wc']
                    for n, X in p.items():
                        acc[n].append(do_swiata(X, T))
                wiersz(rows, jak, dict(common, method=m, lift='e2e'), {n: np.mean(x, 0) for n, x in acc.items()}, Pw, widoczne)
            # dwa etapy
            for m in args.stage1:
                if m == 'oracle2d':
                    obs = [{n: (k['u'], k['v']) for n, k in ds.kp(v).items() if k['visible']} for v in views]
                else:
                    obs = [wczytaj2d(ds.root, m, v) for v in views]
                    if any(o is None for o in obs):
                        continue
                c2 = dict(common, method=m)
                if s['kind'] == 'single':
                    # wycinek: głębia pochodzi z pełnego zdjęcia tej samej kamery (cam.json: pelny)
                    vd = cr.get('pelny', ref)
                    cd = ds.cam(vd)
                    K = cd['K']
                    uv = {n: do_K(x, cr['K'], K) for n, x in obs[0].items()}
                    for dm in ['lidar'] + args.depth:
                        if dm == 'lidar':
                            if not ds.ma_lidar(vd):
                                continue
                            D = ds.lidar(vd, (cd['H'], cd['W']))
                        else:
                            p = os.path.join(ds.root, 'preds', 'depth', dm, vd + '.npy')
                            if not os.path.exists(p):
                                continue
                            D = np.load(p).astype(np.float32)
                            if D.shape != (cd['H'], cd['W']):      # wrappery zapisują 1/4 rozdzielczości (common.zapisz_glebie)
                                D = np.array(Image.fromarray(D).resize((cd['W'], cd['H']), Image.NEAREST))
                        Xc = lift.depth_backproject(ktuple(K), uv, D)
                        wiersz(rows, jak, dict(c2, lift='depth_' + dm), {n: do_swiata(X, cr['T_wc']) for n, X in Xc.items()},
                               Pw, widoczne, obs[0], kp_ref)
                    p = os.path.join(ds.root, 'preds', 'lift', 'depthlm', m, ref + '.json')
                    if os.path.exists(p):                  # głębia wskazanego piksela wprost z DepthLM
                        z = json.load(open(p))['depth_m']
                        Xc = {n: np.array([(x[0] - K[0, 2]) / K[0, 0] * z[n], (x[1] - K[1, 2]) / K[1, 1] * z[n], z[n]])
                              for n, x in uv.items() if n in z}
                        wiersz(rows, jak, dict(c2, lift='depthlm'), {n: do_swiata(X, cr['T_wc']) for n, X in Xc.items()},
                               Pw, widoczne, obs[0], kp_ref)
                else:
                    Kref = cr['K']
                    poses = [ds.cam(v)['T_wc'] for v in views]
                    ob = collections.defaultdict(list)
                    for i, (v, o) in enumerate(zip(views, obs)):
                        Kv = ds.cam(v)['K']
                        for n, uv in o.items():
                            ob[n].append((i, *do_K(uv, Kv, Kref)))
                    for nazwa, fn in (('tri_arkit', lift.triangulate_known_poses), ('tri_arkit_ransac', lift.triangulate_ransac)):
                        if nazwa == 'tri_arkit_ransac' and len(views) < 3:
                            continue
                        Xc, _ = fn(ktuple(Kref), poses, ob)
                        wiersz(rows, jak, dict(c2, lift=nazwa), {n: do_swiata(X, poses[0]) for n, X in Xc.items()}, Pw, widoczne, obs[0], kp_ref)
    return rows


def podsumuj(rows):
    g, pary = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        klucz = (r['method'], r['lift'], r['kind'])
        if 'para' in r:
            pary[klucz].append(r['err_para_cm'])
        else:
            g[klucz].append(r)
    out = ['| Metoda | Podniesienie | Zestaw | Punktów | Pokrycie | Mediana [cm] | p90 [cm] | ' +
           ' | '.join('<%d cm' % p for p in PROGI_CM) + ' | Mediana 2D [px] | Odległość między punktami [cm] |', '|' + '---|' * (9 + len(PROGI_CM))]
    for k in sorted(g):
        rr = g[k]
        e = np.array([r['err_cm'] for r in rr if 'err_cm' in r])
        px = np.array([r['err_px'] for r in rr if 'err_px' in r])
        if not len(e):
            continue
        out.append('| %s | %s | %s | %d | %.0f%% | %.1f | %.1f | %s | %s | %s |' % (
            k[0], k[1], k[2], len(rr), 100 * len(e) / len(rr), np.median(e), np.percentile(e, 90),
            ' | '.join('%.0f%%' % (100 * (e < p).sum() / len(rr)) for p in PROGI_CM), '%.0f' % np.median(px) if len(px) else '–',
            '%.1f' % np.median(pary[k]) if pary[k] else '–'))
    return '\n'.join(out)


def sekcje(rows):
    """Tabele wg zbioru; 3DPW osobno, bo to dopasowanie i liczy się tam tylko odległość między punktami."""
    zb = sorted({r['zbior'] for r in rows})
    out = [('Wszystkie zbiory z pomiarem (bez 3DPW i wycinków)', [r for r in rows if r['zbior'] not in ('3dpw', 'av2c')])]
    out += [(z, [r for r in rows if r['zbior'] == z]) for z in zb]
    av = [r for r in rows if r['zbior'] == 'av2' and r.get('lidar_cm', 1e9) <= LIDAR_PEWNE_CM]
    if av:
        out.append(('av2: tylko punkty do %d cm od LiDAR-u' % LIDAR_PEWNE_CM, av))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--stage1', default='oracle2d')
    ap.add_argument('--e2e', default='')
    ap.add_argument('--depth', default='')
    args = ap.parse_args()
    args.stage1 = [x for x in args.stage1.split(',') if x]
    args.e2e = [x for x in args.e2e.split(',') if x]
    args.depth = [x for x in args.depth.split(',') if x]
    ds = Zbior(args.root)
    rows = ocen(ds, args)
    out = os.path.join(args.root, 'results')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'rows.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    s = '\n\n'.join('## %s\n\n%s' % (t, podsumuj(r)) for t, r in sekcje(rows))
    open(os.path.join(out, 'summary.md'), 'w', encoding='utf-8').write(s + '\n')
    print(s)


if __name__ == '__main__':
    main()
