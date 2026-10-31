# -*- coding: utf-8 -*-
"""Bezpieczne wczytanie plików .pkl z 3DPW (pickle z Pythona 2): przepuszcza tylko tablice numpy i typy
wbudowane, więc pobrany plik nie może niczego wykonać przy wczytaniu."""
import pickle

DOZWOLONE = {
    ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
    ('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'),
    ('__builtin__', 'set'), ('builtins', 'set'), ('copy_reg', '_reconstructor'), ('copyreg', '_reconstructor'),
    ('__builtin__', 'object'), ('builtins', 'object'),
}


class Bezpieczny(pickle.Unpickler):
    def find_class(self, modul, nazwa):
        if (modul, nazwa) in DOZWOLONE:
            return super().find_class(modul.replace('numpy.core', 'numpy.core'), nazwa)
        raise pickle.UnpicklingError('niedozwolona klasa w pliku: %s.%s' % (modul, nazwa))


def wczytaj(sciezka):
    with open(sciezka, 'rb') as f:
        return Bezpieczny(f, encoding='latin1').load()
