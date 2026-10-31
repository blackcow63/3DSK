# -*- coding: utf-8 -*-
"""Przygotowuje obiekty z ulic (Argoverse 2 Sensor, CC BY-NC-SA 4.0: ludzie, auta, rowery, znaki, pachołki...)
do anotacji w annotate_ctx.html (http://127.0.0.1:8765/kontekst).

    python przygotuj_av2.py              # logi z dane/av2/wybrane.txt, które mają już pobrane klatki

Modelu obiektu tu nie ma, więc punkt klika się na 2-3 zdjęciach tego samego obiektu z różnych stron, a strona
trianguluje go w układzie obiektu (ramka 3D śledzona przez cały log, ego-pozy metryczne). Stąd każda jednostka
anotacji to jeden egzemplarz: dla obiektów sztywnych cały log, dla ludzi, rowerzystów i psów okno 1,2 s
(ciało się rusza). Dla każdej jednostki zapisuje annot/kontekst/<klucz>/meta.json z 6 wycinkami do klikania
(wyprostowane, pinhole, K i poza obiekt->kamera) i listą wszystkich ujęć z pozą (do oceny). Lista: kontekst.json.

Konwencje AV2: city_SE3_egovehicle i egovehicle_SE3_sensor to pozy (punkt w układzie potomnym -> rodzic),
kwaterniony wxyz; ramka obiektu w układzie auta w chwili przemiatania LiDAR-u, środek prostopadłościanu,
x do przodu (length), y w lewo (width), z w górę. Kamera: x w prawo, y w dół, z do przodu.
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import paths  # noqa: E402
AV2 = os.path.join(paths.DANE, 'av2')
OUT = os.path.join(HERE, 'kontekst')
BOK = 640
KAMERY = ['ring_front_center', 'ring_front_left', 'ring_front_right', 'ring_rear_left', 'ring_rear_right',
          'ring_side_left', 'ring_side_right']
ZNIEKSZTALCENIE = False     # obrazy AV2 są już wyprostowane: oficjalne API rzutuje samym K, a LiDAR rzutowany bez
                            # k1-k3 trafia w krawędzie w rogach kadru (z k1-k3 rozjeżdża się o ~20 px; sprawdzone 29.10)

KATEGORIE = {
    'REGULAR_VEHICLE': 'car', 'LARGE_VEHICLE': 'large vehicle', 'BUS': 'bus', 'SCHOOL_BUS': 'bus', 'ARTICULATED_BUS': 'bus',
    'BOX_TRUCK': 'box truck', 'TRUCK': 'truck', 'TRUCK_CAB': 'truck', 'VEHICULAR_TRAILER': 'trailer',
    'PEDESTRIAN': 'person', 'OFFICIAL_SIGNALER': 'person', 'BICYCLIST': 'cyclist', 'MOTORCYCLIST': 'motorcyclist',
    'WHEELED_RIDER': 'rider', 'BICYCLE': 'bicycle', 'MOTORCYCLE': 'motorcycle', 'WHEELED_DEVICE': 'scooter',
    'STROLLER': 'stroller', 'WHEELCHAIR': 'wheelchair', 'DOG': 'dog', 'ANIMAL': 'animal',
    'STOP_SIGN': 'stop sign', 'SIGN': 'sign', 'MOBILE_PEDESTRIAN_CROSSING_SIGN': 'pedestrian crossing sign',
    'BOLLARD': 'bollard', 'CONSTRUCTION_CONE': 'traffic cone', 'CONSTRUCTION_BARREL': 'construction barrel',
    'MESSAGE_BOARD_TRAILER': 'message board trailer', 'RAILED_VEHICLE': 'tram',
}
NIESZTYWNE = {'PEDESTRIAN', 'OFFICIAL_SIGNALER', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER', 'DOG', 'ANIMAL', 'WHEELCHAIR'}
OKNO_NS = 1_200_000_000
LIMIT = {'person': 8, 'car': 5}          # jednostek na log; pozostałe kategorie po 3
MIN_WYS_PX = 160                         # wysokość ramki obiektu na pełnym zdjęciu
MIN_OSTROSC = 25
# ramka obiektu dotyczy chwili, gdy LiDAR go minął, a nie nominalnego czasu przemiatania (do 50 ms różnicy), więc przy
# jadących obiektach rozjeżdża się ze zdjęciem; ich tu nie bierzemy (m/s, przesunięcie netto między ujęciami)
MAKS_V_SZTYWNE = 0.3
MAKS_V_NIESZTYWNE = 1.0


def se3(q, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = t
    return T


class Pozy:
    """Poza (rodzic <- dziecko) w czasie, z interpolacją slerp; poza zakresem None."""

    def __init__(self, ts, q_wxyz, t):
        ts = np.asarray(ts, dtype=np.int64)
        _, o = np.unique(ts, return_index=True)          # posortowane, bez powtórzonych chwil
        self.ts = ts[o]
        self.t0 = int(self.ts[0])
        q = np.asarray(q_wxyz)[o]
        self.t = np.asarray(t, dtype=float)[o]
        self.rot = Rotation.from_quat(q[:, [1, 2, 3, 0]])
        # czas względny: ns od 1980 r. jako float tracą precyzję
        self.slerp = Slerp((self.ts - self.t0).astype(float), self.rot) if len(self.ts) > 1 else None

    def __call__(self, t, zapas=60_000_000):
        if t < self.ts[0] - zapas or t > self.ts[-1] + zapas:
            return None
        tc = float(np.clip(t, self.ts[0], self.ts[-1]) - self.t0)
        T = np.eye(4)
        if self.slerp is None:
            T[:3, :3] = self.rot.as_matrix()[0]
            T[:3, 3] = self.t[0]
        else:
            T[:3, :3] = self.slerp([tc]).as_matrix()[0]
            T[:3, 3] = [np.interp(tc, (self.ts - self.t0).astype(float), self.t[:, i]) for i in range(3)]
        return T


def czytaj(p):
    return pd.read_feather(p)


def znieksztalc(x, y, k):
    """Znormalizowane współrzędne pinhole -> zniekształcone (model radialny AV2: k1 r^2 + k2 r^4 + k3 r^6)."""
    if not ZNIEKSZTALCENIE:
        return x, y
    r2 = x * x + y * y
    f = 1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3
    return x * f, y * f


class Log:
    def __init__(self, log):
        self.log = log
        d = os.path.join(AV2, 'val', log)
        self.d = d
        e = czytaj(os.path.join(d, 'city_SE3_egovehicle.feather'))
        self.ego = Pozy(e.timestamp_ns.values, e[['qw', 'qx', 'qy', 'qz']].values, e[['tx_m', 'ty_m', 'tz_m']].values)
        c = czytaj(os.path.join(d, 'calibration', 'egovehicle_SE3_sensor.feather')).set_index('sensor_name')
        self.ego_T_cam = {k: se3(c.loc[k, ['qw', 'qx', 'qy', 'qz']].values, c.loc[k, ['tx_m', 'ty_m', 'tz_m']].values) for k in KAMERY}
        i = czytaj(os.path.join(d, 'calibration', 'intrinsics.feather')).set_index('sensor_name')
        self.K, self.dist, self.wh = {}, {}, {}
        for k in KAMERY:
            r = i.loc[k]
            self.K[k] = np.array([[r.fx_px, 0, r.cx_px], [0, r.fy_px, r.cy_px], [0, 0, 1.0]])
            self.dist[k] = (r.k1, r.k2, r.k3)
            self.wh[k] = (int(r.width_px), int(r.height_px))
        self.ann = czytaj(os.path.join(d, 'annotations.feather'))
        self.klatki = {}
        for k in KAMERY:
            kd = os.path.join(d, 'sensors', 'cameras', k)
            self.klatki[k] = sorted(int(f[:-4]) for f in os.listdir(kd) if f.endswith('.jpg')) if os.path.isdir(kd) else []
        ld = os.path.join(d, 'sensors', 'lidar')
        self.t_lidar = np.array(sorted(int(f[:-8]) for f in os.listdir(ld) if f.endswith('.feather'))) if os.path.isdir(ld) else np.array([])
        self._lidar = {}

    def cam_T_city(self, kam, t):
        E = self.ego(t)
        return None if E is None else np.linalg.inv(E @ self.ego_T_cam[kam])

    def lidar_city(self, t):
        """Przemiatanie LiDAR-u najbliższe chwili t, w układzie miasta (bez kompensacji w obrębie przemiatania)."""
        i = int(np.argmin(np.abs(self.t_lidar - t)))
        tl = int(self.t_lidar[i])
        if tl not in self._lidar:
            p = czytaj(os.path.join(self.d, 'sensors', 'lidar', '%d.feather' % tl))[['x', 'y', 'z']].values
            E = self.ego(tl)
            self._lidar[tl] = p @ E[:3, :3].T + E[:3, 3]
            if len(self._lidar) > 6:
                self._lidar.pop(next(iter(self._lidar)))
        return self._lidar[tl]

    def tor(self, uuid):
        a = self.ann[self.ann.track_uuid == uuid].sort_values('timestamp_ns')
        city = []
        for r in a.itertuples():
            E = self.ego(r.timestamp_ns)
            city.append(E @ se3((r.qw, r.qx, r.qy, r.qz), (r.tx_m, r.ty_m, r.tz_m)))
        city = np.array(city)
        q = Rotation.from_matrix(city[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        wym = a[['length_m', 'width_m', 'height_m']].median().values
        return Pozy(a.timestamp_ns.values, q, city[:, :3, 3]), wym, a

    def obraz(self, kam, t):
        return cv2.imread(os.path.join(self.d, 'sensors', 'cameras', kam, '%d.jpg' % t))


def naroza(wym):
    l, w, h = wym / 2
    return np.array([[sx * l, sy * w, sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def ujecia(L, tor, wym):
    """Wszystkie ujęcia (kamera, chwila), na których obiekt jest duży, w kadrze i nie zasłonięty."""
    out = []
    P = naroza(wym)
    for kam in KAMERY:
        W, H = L.wh[kam]
        for t in L.klatki[kam]:
            O = tor(t)
            C = L.cam_T_city(kam, t)
            if O is None or C is None:
                continue
            T = C @ O                                     # obiekt -> kamera
            X = P @ T[:3, :3].T + T[:3, 3]
            if X[:, 2].min() < 1.0 or X[:, 2].min() > 35:
                continue
            uv = X[:, :2] / X[:, 2:] * [L.K[kam][0, 0], L.K[kam][1, 1]] + L.K[kam][:2, 2]
            x0, y0 = uv.min(0)
            x1, y1 = uv.max(0)
            pole = (x1 - x0) * (y1 - y0)
            wewn = (min(x1, W) - max(x0, 0)) * (min(y1, H) - max(y0, 0)) / max(pole, 1)
            if y1 - y0 < MIN_WYS_PX or wewn < 0.9 or max(x1 - x0, y1 - y0) > 0.9 * min(W, H):
                continue
            # zasłonięcie z LiDAR-u: punkty w ramce 2D bliżej niż najbliższy narożnik obiektu
            pts = L.lidar_city(t)
            Xc = pts @ C[:3, :3].T + C[:3, 3]
            Xo = (pts - O[:3, 3]) @ O[:3, :3]
            m = Xc[:, 2] > 0.5
            u = Xc[m, 0] / Xc[m, 2] * L.K[kam][0, 0] + L.K[kam][0, 2]
            v = Xc[m, 1] / Xc[m, 2] * L.K[kam][1, 1] + L.K[kam][1, 2]
            w2 = (u > x0) & (u < x1) & (v > y0) & (v < y1)
            wobj = (np.abs(Xo[m]) <= wym / 2 + 0.1).all(1) & w2
            zasl = w2 & ~wobj & (Xc[m, 2] < X[:, 2].min() - 0.3)
            n_obj, n_zasl = int(wobj.sum()), int(zasl.sum())
            if n_obj < 15 or n_zasl / (n_obj + n_zasl) > 0.2:
                continue
            kier = -T[:3, :3].T @ T[:3, 3]                # środek kamery w układzie obiektu
            out.append(dict(kam=kam, t=t, T=T, bbox=(x0, y0, x1, y1), wys=y1 - y0, kier=kier / np.linalg.norm(kier),
                            n_obj=n_obj, zasl=round(n_zasl / (n_obj + n_zasl), 3)))
    return out


def wycinek(L, u):
    """Wyprostowany (pinhole) kwadratowy wycinek wokół obiektu; zwraca obraz i K wycinka."""
    x0, y0, x1, y1 = u['bbox']
    s = max(x1 - x0, y1 - y0) * 1.35
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    K = L.K[u['kam']]
    sk = BOK / s
    Kc = np.array([[K[0, 0] * sk, 0, (K[0, 2] - (cx - s / 2)) * sk], [0, K[1, 1] * sk, (K[1, 2] - (cy - s / 2)) * sk], [0, 0, 1]])
    g = np.arange(BOK, dtype=np.float32) + 0.5
    uu, vv = np.meshgrid(g, g)
    x = (uu - Kc[0, 2]) / Kc[0, 0]
    y = (vv - Kc[1, 2]) / Kc[1, 1]
    xd, yd = znieksztalc(x, y, L.dist[u['kam']])
    mx = (xd * K[0, 0] + K[0, 2] - 0.5).astype(np.float32)
    my = (yd * K[1, 1] + K[1, 2] - 0.5).astype(np.float32)
    img = cv2.remap(L.obraz(u['kam'], u['t']), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return img, Kc


def ostrosc(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def okno(ujs, nieszt):
    """Dla obiektów niesztywnych: okno 1,2 s z największą liczbą ujęć i rozrzutem kierunków."""
    if not nieszt or not ujs:
        return ujs, None
    best = None
    for u in ujs:
        w = [x for x in ujs if 0 <= x['t'] - u['t'] <= OKNO_NS]
        D = np.array([x['kier'] for x in w])
        rozrzut = np.degrees(np.arccos(np.clip(D @ D.T, -1, 1))).max()
        ocena = min(rozrzut, 30) * np.log1p(len(w))
        if best is None or ocena > best[0]:
            best = (ocena, w, (u['t'], u['t'] + OKNO_NS))
    return best[1], best[2]


def predkosc(tor, ujs):
    ts = sorted(u['t'] for u in ujs)
    return float(np.linalg.norm(tor(ts[-1], 10**9)[:3, 3] - tor(ts[0], 10**9)[:3, 3]) / max((ts[-1] - ts[0]) / 1e9, 1e-3))


def wybierz(ujs, n=6):
    """Najpierw największe ujęcie, potem kolejno najbardziej odległe kierunkiem patrzenia."""
    ujs = sorted(ujs, key=lambda u: -u['wys'])
    wyb = [ujs[0]]
    while len(wyb) < n:
        reszta = [u for u in ujs if all(u is not w for w in wyb)]
        if not reszta:
            break
        kat = [min(np.degrees(np.arccos(np.clip(u['kier'] @ w['kier'], -1, 1))) for w in wyb) for u in reszta]
        i = int(np.argmax(kat))
        if kat[i] < 1.0 and len(wyb) >= 3:
            break
        wyb.append(reszta[i])
    return wyb


def jednostki_logu(log):
    L = Log(log)
    if not L.klatki['ring_front_center'] or not len(L.t_lidar):
        print(log[:8], 'brak klatek, pomijam')
        return []
    a = L.ann.copy()
    a['d'] = np.hypot(a.tx_m, a.ty_m)
    kand = a[(a.d < 25) & (a.num_interior_pts >= 40)].groupby('track_uuid').agg(kat_av2=('category', 'first'), n=('d', 'size'), dmin=('d', 'min'))
    kand = kand[(kand.n >= 3) & kand.kat_av2.isin(KATEGORIE)].sort_values('dmin')
    ile, out = {}, []
    for uuid, r in kand.iterrows():
        kat = KATEGORIE[r.kat_av2]
        if ile.get(kat, 0) >= LIMIT.get(kat, 3):
            continue
        tor, wym, _ = L.tor(uuid)
        ujs = ujecia(L, tor, wym)
        ujs, ok = okno(ujs, r.kat_av2 in NIESZTYWNE)
        if len(ujs) < 3:
            continue
        v = predkosc(tor, ujs)
        if v > (MAKS_V_NIESZTYWNE if r.kat_av2 in NIESZTYWNE else MAKS_V_SZTYWNE):
            continue
        wyb = wybierz(ujs)
        D = np.array([u['kier'] for u in wyb])
        rozrzut = float(np.degrees(np.arccos(np.clip(D @ D.T, -1, 1))).max())
        if rozrzut < 6:                     # za mała baza do triangulacji
            continue
        wyc = []
        for u in wyb:
            img, Kc = wycinek(L, u)
            o = ostrosc(img)
            if o >= MIN_OSTROSC:
                wyc.append((u, img, Kc, o))
        if len(wyc) < 3:
            continue
        klucz = 'av2_%s_%s' % (log[:8], uuid[:8])
        d = os.path.join(OUT, klucz)
        os.makedirs(d, exist_ok=True)
        podg = []
        for k, (u, img, Kc, o) in enumerate(wyc):
            cv2.imwrite(os.path.join(d, 'podglad_%d.jpg' % k), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            podg.append(dict(plik='podglad_%d.jpg' % k, kam=u['kam'], t=u['t'], K=Kc.tolist(),
                             R=u['T'][:3, :3].tolist(), t_m=u['T'][:3, 3].tolist(), ostrosc=round(o, 1)))
        meta = dict(klucz=klucz, zbior='av2', log=log, track=uuid, kategoria_av2=r.kat_av2, kategoria=kat,
                    nazwa='%s %s' % (kat, uuid[:4]), wymiary_m=[round(float(x), 3) for x in wym], okno_ns=ok,
                    jednostki='m', podglady=podg, predkosc_ms=round(v, 2),
                    ujecia=[dict(kam=u['kam'], t=u['t'], R=u['T'][:3, :3].round(6).tolist(), t_m=u['T'][:3, 3].round(4).tolist(),
                                 bbox=[round(float(x), 1) for x in u['bbox']], zasl=u['zasl']) for u in ujs])
        json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'), indent=1)
        ile[kat] = ile.get(kat, 0) + 1
        out.append(dict(klucz=klucz, zbior='av2', nazwa=meta['nazwa'], kategoria=kat, log=log[:8],
                        ujec=len(ujs), podgladow=len(podg), rozrzut_st=round(rozrzut, 1), predkosc_ms=round(v, 2)))
        print(klucz, kat, '| ujęć', len(ujs), 'podglądów', len(podg), 'rozrzut %.0f°' % rozrzut, flush=True)
    return out


def main():
    logi = [l.strip() for l in open(os.path.join(HERE, '..', 'dane_skrypty', 'av2', 'wybrane.txt')) if l.strip()]
    if len(sys.argv) > 1:
        logi = [l for l in logi if l.startswith(sys.argv[1])]
    lista_p = os.path.join(HERE, 'kontekst.json')
    lista = json.load(open(lista_p)) if os.path.exists(lista_p) else []
    for log in logi:
        nowe = jednostki_logu(log)
        lista = [x for x in lista if x['log'] != log[:8]] + nowe
        json.dump(lista, open(lista_p, 'w'), indent=1)
    print('jednostek:', len(lista))


if __name__ == '__main__':
    main()
