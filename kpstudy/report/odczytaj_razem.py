# -*- coding: utf-8 -*-
"""Odczytuje od nowa odpowiedzi stylu razem z zapisanego raw, bez GPU: common.odczyt_razem, W i H z rgb.jpg.

    python report/odczytaj_razem.py --root ../../dane/kp_razem --name mimo7b_razem [--zapisz]

Bez --zapisz tylko liczy zmiany. Potrzebne po pobraniu z Heliosa wyników sprzed odczytu końcowej odpowiedzi.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'methods'))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--zapisz', action='store_true')
    args = ap.parse_args()
    bylo = jest = zmienione = 0
    for p in glob.glob(os.path.join(args.root, 'preds', 'e2e', args.name, '*', '*.json')):
        d = json.load(open(p, encoding='utf-8'))
        K = np.array(json.load(open(os.path.join(args.root, d['view'], 'cam.json')))['K'])
        W, H = Image.open(os.path.join(args.root, d['view'], 'rgb.jpg')).size
        cam, p2d = {}, {}
        for k, text in d['raw'].items():
            r = common.odczyt_razem(text, W, H, K) if k.endswith(':3d') else None
            if r:
                p2d[k[:-3]], cam[k[:-3]] = r
        bylo += len(d['points_cam'])
        jest += len(cam)
        zmienione += sum(n not in cam or not np.allclose(d['points_cam'][n], cam[n]) for n in d['points_cam'])
        if args.zapisz:
            d['points_cam'], d['points_2d'] = cam, p2d
            json.dump(d, open(p, 'w'))
    print('%s: odpowiedzi było %d, jest %d, zmienione lub usunięte %d%s' % (args.name, bylo, jest, zmienione, ', zapisane' if args.zapisz else ''))


if __name__ == '__main__':
    main()
