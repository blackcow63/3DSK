# -*- coding: utf-8 -*-
"""SMPL w samym numpy: wczytanie pliku modelu z smpl.is.tue.mpg.de (pickle z obiektami chumpy) bez instalowania
chumpy i ułożenie siatki (skórowanie LBS). Tak jak demo 3DPW: model zależny od płci, betas[:10], pose 72, trans.

    from smpl_np import SMPL
    m = SMPL('m'); v = m(betas, pose, trans)        # (6890, 3) w układzie świata 3DPW
"""
import glob
import os
import pickle

import numpy as np

import sys  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import paths  # noqa: E402
KATALOG = os.environ.get('KP_SMPL') or os.path.join(paths.DANE, 'smpl')   # modele SMPL v1.0.0 (*.pkl)


class _Ch:
    """Zaślepka obiektu chumpy: zostawia tylko tablicę z jego stanu."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        self.x = np.asarray(s.get('x', s.get('a', None)) if isinstance(s, dict) else s)


class _Unpickler(pickle.Unpickler):
    def find_class(self, modul, nazwa):
        if modul.startswith('chumpy'):
            return _Ch
        if modul.startswith('scipy.sparse'):
            return super().find_class(modul, nazwa)
        if modul.split('.')[0] in ('numpy', 'copy_reg', 'copyreg', '__builtin__', 'builtins'):
            return super().find_class(modul, nazwa)
        raise pickle.UnpicklingError('niedozwolona klasa w pliku modelu: %s.%s' % (modul, nazwa))


def _tab(x):
    if isinstance(x, _Ch):
        return np.asarray(x.x, dtype=np.float64)
    if hasattr(x, 'toarray'):
        return x.toarray().astype(np.float64)
    return np.asarray(x)


def plik_modelu(plec):
    """'m', 'f' albo 'n'; szuka basicmodel_{m,f,neutral}_lbs_10_207_0_v1.*.pkl (także w podkatalogach)."""
    wzor = {'m': '*asic*odel_m_lbs*.pkl', 'f': '*asic*odel_f_lbs*.pkl', 'n': '*asic*odel_neutral_lbs*.pkl'}[plec]
    p = sorted(glob.glob(os.path.join(KATALOG, '**', wzor), recursive=True))
    if not p:
        raise FileNotFoundError('brak modelu SMPL (%s) w %s' % (plec, KATALOG))
    return p[-1]


class SMPL:
    def __init__(self, plec='n'):
        with open(plik_modelu(plec), 'rb') as f:
            d = _Unpickler(f, encoding='latin1').load()
        self.v_template = _tab(d['v_template'])
        self.shapedirs = _tab(d['shapedirs'])
        self.posedirs = _tab(d['posedirs'])
        self.J_regressor = _tab(d['J_regressor'])
        self.weights = _tab(d['weights'])
        self.faces = np.asarray(d['f'], dtype=np.int64)
        kt = np.asarray(d['kintree_table'])
        self.rodzic = {int(kt[1, i]): int(kt[0, i]) for i in range(1, kt.shape[1])}

    @staticmethod
    def _rodrigues(r):
        th = np.linalg.norm(r, axis=1, keepdims=True) + 1e-12
        k = r / th
        K = np.zeros((len(r), 3, 3))
        K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
        K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
        th = th[:, :, None]
        return np.eye(3)[None] + np.sin(th) * K + (1 - np.cos(th)) * K @ K

    def __call__(self, betas, pose, trans=None, v_template=None):
        b = np.zeros(self.shapedirs.shape[-1])
        b[:min(10, len(betas))] = np.asarray(betas)[:10]
        v = (self.v_template if v_template is None else v_template) + self.shapedirs @ b
        J = self.J_regressor @ v
        R = self._rodrigues(np.asarray(pose, dtype=np.float64).reshape(24, 3))
        v = v + self.posedirs @ (R[1:] - np.eye(3)[None]).ravel()
        G = np.zeros((24, 4, 4))
        for i in range(24):
            T = np.eye(4)
            T[:3, :3] = R[i]
            T[:3, 3] = J[i] - (J[self.rodzic[i]] if i else 0)
            G[i] = T if i == 0 else G[self.rodzic[i]] @ T
        G2 = G.copy()
        G2[:, :3, 3] -= np.einsum('kij,kj->ki', G[:, :3, :3], J)
        T = np.einsum('vk,kij->vij', self.weights, G2)
        v = np.einsum('vij,vj->vi', T[:, :3, :3], v) + T[:, :3, 3]
        return v + (0 if trans is None else np.asarray(trans))
