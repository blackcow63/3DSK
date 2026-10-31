# -*- coding: utf-8 -*-
"""Argoverse 2 Sensor (CC BY-NC-SA 4.0) z publicznego bucketu s3://argoverse przez zwykłe HTTPS, bez konta.

    python pobierz.py meta                 # adnotacje, pozy i kalibracje wszystkich logów val (~0,5 MB na log)
    python pobierz.py klatki <log> [co=4]  # co 4. klatkę z 7 kamer pierścienia (5 Hz) + najbliższe przemiatania LiDAR-u

Układ jak w buckecie: av2/val/<log>/{annotations.feather, city_SE3_egovehicle.feather, calibration/*,
sensors/cameras/<kamera>/<ns>.jpg, sensors/lidar/<ns>.feather}.
"""
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = 'https://argoverse.s3.amazonaws.com/'
PREF = 'datasets/av2/sensor/val/'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import paths  # noqa: E402
OUT = os.path.join(paths.DANE, 'av2', 'val')
KAMERY = ['ring_front_center', 'ring_front_left', 'ring_front_right', 'ring_rear_left', 'ring_rear_right',
          'ring_side_left', 'ring_side_right']


def lista(prefix, delimiter=None):
    """Klucze (albo podkatalogi, gdy delimiter) pod prefiksem, z obsługą stronicowania."""
    out, token = [], None
    while True:
        q = 'list-type=2&prefix=' + prefix + ('&delimiter=' + delimiter if delimiter else '')
        if token:
            q += '&continuation-token=' + urllib.request.quote(token, safe='')
        x = urllib.request.urlopen(URL + '?' + q, timeout=60).read().decode()
        out += re.findall(r'<Prefix>([^<]*)</Prefix></CommonPrefixes>' if delimiter else r'<Key>([^<]*)</Key>', x)
        m = re.search(r'<NextContinuationToken>([^<]*)</NextContinuationToken>', x)
        if not m:
            return out
        token = m.group(1)


def pobierz(klucz):
    cel = os.path.join(OUT, klucz[len(PREF):].replace('/', os.sep))
    if os.path.exists(cel):
        return
    os.makedirs(os.path.dirname(cel), exist_ok=True)
    for proba in range(4):
        try:
            dane = urllib.request.urlopen(URL + klucz, timeout=120).read()
            with open(cel + '.tmp', 'wb') as f:
                f.write(dane)
            os.replace(cel + '.tmp', cel)
            return
        except Exception as e:
            if proba == 3:
                print('BŁĄD', klucz, e, flush=True)


def meta():
    logi = [p[len(PREF):-1] for p in lista(PREF, '/')]
    klucze = []
    for log in logi:
        klucze += [PREF + log + '/annotations.feather', PREF + log + '/city_SE3_egovehicle.feather',
                   PREF + log + '/calibration/egovehicle_SE3_sensor.feather', PREF + log + '/calibration/intrinsics.feather']
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(pobierz, klucze))
    print('logów:', len(logi))


def klatki(log, co=4):
    klucze = []
    lidar = sorted(lista(PREF + log + '/sensors/lidar/'))
    t_lidar = [int(os.path.basename(k)[:-8]) for k in lidar]
    potrzebne = set()
    for kam in KAMERY:
        kl = sorted(lista(PREF + log + '/sensors/cameras/' + kam + '/'))[::co]
        klucze += kl
        for k in kl:                                  # przemiatanie LiDAR-u najbliższe w czasie każdej klatce
            t = int(os.path.basename(k)[:-4])
            potrzebne.add(min(range(len(t_lidar)), key=lambda i: abs(t_lidar[i] - t)))
    klucze += [lidar[i] for i in sorted(potrzebne)]
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(pobierz, klucze))
    print(log, 'plików:', len(klucze))


if __name__ == '__main__':
    if sys.argv[1] == 'meta':
        meta()
    else:
        klatki(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
