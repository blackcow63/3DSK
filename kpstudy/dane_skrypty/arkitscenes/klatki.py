# -*- coding: utf-8 -*-
"""Klatki testowe z pełnego wideo ARKitScenes (1920x1440, 60 kl./s) tam, gdzie widać biurko.

    python klatki.py                 # sceny z pokrycie_biurek.json, które mają >= 30 póz z biurkiem

Pakiet 'upsampling' ma tylko ~21 klatek na nagranie i na większości biurko prawie nie występuje,
więc klatki bierzemy z video.mov. Czas klatki n to ts[n+1] z listy intrinsics (arkit_io.Scene.video_ts);
pozy ARKit interpolujemy na ten czas. Do sprawdzania widoczności punktów (nie dla metod) bierzemy
głębię LiDAR iPada (lowres_depth, 256x192, mm) z klatki o najbliższym czasie.

Dla każdej sceny zapisuje klatki/<id>/rgb/<id>_<ts>.jpg, klatki/<id>/lidar/<id>_<ts>.png i klatki.json,
a potem usuwa video.mov i lowres_depth.zip (można je pobrać ponownie).
"""
import json
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'annot'))
import paths  # noqa: E402
import arkit_io as A  # noqa: E402
from przygotuj import pudelka, biurko  # noqa: E402

U = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw'
MAX_KLATEK = 48
ODSTEP_S = 0.5


def pobierz(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    subprocess.run(['curl', '-sfL', '--retry', '3', '-o', dst + '.tmp', url], check=True)
    os.replace(dst + '.tmp', dst)


def punkty_biurka(s):
    boxes = pudelka(s)
    for b in boxes:
        b['R'] = b['N'].T
    t, mon, krz = biurko(boxes)
    P = [t['c'] + t['R'] @ (np.array([sx, sy, 1]) * t['L'] / 2) for sx in (-1, 1) for sy in (-1, 1)]
    return np.array(P + [m['c'] for m in mon] + [c['c'] for c in krz])


def wybierz(s, P):
    ts = s.video_ts()
    widoczne = []
    for n in range(0, len(ts), 6):                       # co 0,1 s wystarczy do wyboru
        T, _ = s.pose_at(ts[n])
        K, _ = s.K_at(ts[n])
        uv, z = A.project(P, T, K)
        k = ((z > 0.5) & (z < 3.5) & (uv[:, 0] >= 0) & (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] < 1440)).sum()
        if k >= 4:
            widoczne.append(n)
    wyb, ost = [], -1e9
    for n in widoczne:
        if ts[n] - ost >= ODSTEP_S:
            wyb.append(n)
            ost = ts[n]
    if len(wyb) > MAX_KLATEK:
        wyb = [wyb[i] for i in np.linspace(0, len(wyb) - 1, MAX_KLATEK).round().astype(int)]
    return wyb, ts


def main():
    """--czesc k/n: tylko co n-ta scena od k (żeby puścić kilka procesów równolegle)."""
    k, n = 0, 1
    for a in sys.argv[1:]:
        if a.startswith('--czesc'):
            k, n = [int(x) for x in a.split('=')[1].split('/')]
    pokrycie = json.load(open(os.path.join(HERE, 'pokrycie_biurek.json')))
    pokrycie = [x for x in pokrycie if x['poz_z_biurkiem'] >= 30][k::n]
    fold = {x['video_id']: x['fold'] for x in json.load(open(os.path.join(HERE, 'biurka_po_wizycie.json')))}
    for x in pokrycie:
        v = x['video_id']
        if x['poz_z_biurkiem'] < 30:
            continue
        out = os.path.join(paths.DANE, 'arkitscenes', 'klatki', v)
        if os.path.exists(os.path.join(out, 'klatki.json')):
            print(v, 'już jest', flush=True)
            continue
        s = A.Scene(v)
        wyb, ts = wybierz(s, punkty_biurka(s))
        mov = os.path.join(s.dir, 'video.mov')
        pobierz('%s/%s/%s/%s.mov' % (U, fold[v], v, v), mov)
        os.makedirs(os.path.join(out, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(out, 'lidar'), exist_ok=True)
        tmp = os.path.join(out, 'tmp')
        os.makedirs(tmp, exist_ok=True)
        expr = '+'.join('eq(n\\,%d)' % n for n in wyb)
        subprocess.run(['ffmpeg', '-v', 'error', '-noautorotate', '-y', '-i', mov, '-vf', 'select=' + expr,
                        '-fps_mode', 'passthrough', '-q:v', '2', os.path.join(tmp, 'k_%03d.jpg')], check=True)
        pliki = sorted(os.listdir(tmp))
        assert len(pliki) == len(wyb), (v, len(pliki), len(wyb))
        nazwy = []
        for n, p in zip(wyb, pliki):
            nazwa = '%s_%.3f' % (v, ts[n])
            os.replace(os.path.join(tmp, p), os.path.join(out, 'rgb', nazwa + '.jpg'))
            nazwy.append(nazwa)
        shutil.rmtree(tmp)
        zp = os.path.join(s.dir, 'lowres_depth.zip')
        pobierz('%s/%s/%s/lowres_depth.zip' % (U, fold[v], v), zp)
        z = zipfile.ZipFile(zp)
        glebie = {float(os.path.basename(m)[:-4].split('_')[1]): m for m in z.namelist() if m.endswith('.png')}
        klucze = np.array(sorted(glebie))
        for n, nazwa in zip(wyb, nazwy):
            k = klucze[np.argmin(np.abs(klucze - ts[n]))]
            with open(os.path.join(out, 'lidar', nazwa + '.png'), 'wb') as f:
                f.write(z.read(glebie[k]))
        z.close()
        os.remove(zp)
        os.remove(mov)
        json.dump(dict(video_id=v, klatki=nazwy, ts=[ts[n] for n in wyb], indeksy_wideo=wyb), open(os.path.join(out, 'klatki.json'), 'w'), indent=0)
        print(v, 'klatek', len(nazwy), flush=True)


if __name__ == '__main__':
    main()
