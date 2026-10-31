# -*- coding: utf-8 -*-
"""Pobiera z ARKitScenes (Apple, licencja niekomercyjna) tylko to, czego potrzebuje test na biurkach:
21 klatek wide 1920x1440 z głębią z lasera (pakiet upsampling), siatkę ARKit, trajektorię
(pozy z ARKit = VIO iPada), intrinsics strumienia lowres_wide (ta sama kamera) i adnotację 3D.

    python pobierz.py            # wszystkie nagrania z biurka_po_wizycie.json
"""
import json
import os
import subprocess
import zipfile

U = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1'
HERE = os.path.dirname(os.path.abspath(__file__))


def pobierz(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    r = subprocess.run(['curl', '-sfL', '--retry', '3', '-o', dst + '.tmp', url])
    if r.returncode != 0:
        print('  BŁĄD', url)
        return False
    os.replace(dst + '.tmp', dst)
    return True


def main():
    for x in json.load(open(os.path.join(HERE, 'biurka_po_wizycie.json'))):
        v, fold = x['video_id'], x['fold']
        d = os.path.join(HERE, 'raw', v)
        os.makedirs(d, exist_ok=True)
        pliki = {f'{U}/upsampling/{fold}/{v}.zip': 'upsampling.zip',
                 f'{U}/raw/{fold}/{v}/{v}_3dod_mesh.ply': 'mesh.ply',
                 f'{U}/raw/{fold}/{v}/{v}_3dod_annotation.json': 'annotation.json',
                 f'{U}/raw/{fold}/{v}/lowres_wide.traj': 'lowres_wide.traj',
                 f'{U}/raw/{fold}/{v}/lowres_wide_intrinsics.zip': 'lowres_wide_intrinsics.zip'}
        for url, nazwa in pliki.items():
            pobierz(url, os.path.join(d, nazwa))
        for z in ('upsampling.zip', 'lowres_wide_intrinsics.zip'):
            p = os.path.join(d, z)
            if os.path.exists(p) and not os.path.isdir(p[:-4]):
                zipfile.ZipFile(p).extractall(p[:-4])
                os.remove(p)
        print(v, 'ok', flush=True)


if __name__ == '__main__':
    main()
