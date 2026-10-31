# -*- coding: utf-8 -*-
"""Gdzie leżą pobrane dane. Kolejność: zmienna KP_DANE, potem <repo>/dane, potem <repo>/../dane (układ autorki)."""
import os

KPSTUDY = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(KPSTUDY)


def _dane():
    e = os.environ.get('KP_DANE')
    if e:
        return os.path.abspath(e)
    for c in (os.path.join(REPO, 'dane'), os.path.join(REPO, '..', 'dane')):
        if os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.join(REPO, 'dane')


DANE = _dane()
