# -*- coding: utf-8 -*-
"""Pary do wskazania zdjęciem wzorcowym na zbiorze dane/kp_razem: punkt zaznaczony na INNYM zdjęciu -> ten sam punkt tutaj.

    python -P ref_pary.py        # zapisuje dane/kp_razem/ref.json, idzie na Heliosa razem ze zbiorem

Wzorzec to inny egzemplarz tego samego rodzaju, jeśli taki jest (ten sam punkt o tej samej nazwie w innej scenie:
inne biurko, inne auto, inny rower, inna osoba), do 3 scen. Obiekty BOP mają własne nazwy punktów, więc tam wzorcem jest ten
sam obiekt sfotografowany w innej scenie. Punkt bez wzorca nie ma odpowiedzi i obniża pokrycie.
ref.json: {scena: {punkt: [{"view": id, "u": u, "v": v}, ...]}}; to tylko piksele, prawda 3D zostaje na laptopie.
"""
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', 'dane', 'kp_razem'))
ILE = 3


def main():
    gt = json.load(open(os.path.join(ROOT, 'gt.json'), encoding='utf-8'))['scenes']
    views = json.load(open(os.path.join(ROOT, 'views.json')))
    rng = np.random.default_rng(0)
    # wzorce: wycinki tam, gdzie są (obiekt zajmuje kadr), inaczej pełne zdjęcia
    kand = collections.defaultdict(list)
    for v in views:
        z = v['zbior']
        if v['kind'] == 'pelny' or z == 'av2':
            continue
        rod = {'av2c': 'av2'}.get(z, z)
        kp = json.load(open(os.path.join(ROOT, v['id'], 'kp.json')))
        for n, k in kp.items():
            if k['visible']:
                kand[(rod, n)].append((v['scene'], v['id'], k['u'], k['v']))
    wyj, bez, razem = {}, 0, 0
    for s, g in gt.items():
        rod = {'av2c': 'av2'}.get(g['zbior'], g['zbior'])
        rdzen = s.split('_', 1)[1]
        wyj[s] = {}
        for n in g['points_world']:
            razem += 1
            bop = rod in ('hope', 'handal')
            if bop:
                c = [x for x in kand[(rod, n)] if x[0] == s]
            elif rod == '3dpw':
                sek = s.rsplit('_', 2)[0]
                c = [x for x in kand[(rod, n)] if x[0].rsplit('_', 2)[0] != sek]
            else:
                c = [x for x in kand[(rod, n)] if x[0].split('_', 1)[1] != rdzen]
            if not c:
                bez += 1
                continue
            sceny = collections.defaultdict(list)
            for x in c:
                sceny[x[0]].append(x)
            kl = sorted(sceny)
            wyb = []
            if bop:                                      # ten sam obiekt: po prostu kilka innych zdjęć, cel wyklucza wrapper
                for i in rng.permutation(len(c))[:ILE + 1]:
                    wyb.append(c[int(i)])
            else:
                for i in rng.permutation(len(kl))[:ILE]:
                    lst = sceny[kl[int(i)]]
                    wyb.append(lst[int(rng.integers(len(lst)))])
            wyj[s][n] = [dict(view=x[1], u=x[2], v=x[3]) for x in wyb]
    json.dump(wyj, open(os.path.join(ROOT, 'ref.json'), 'w'))
    print('punktów', razem, 'bez wzorca', bez)
    for r in ('arkit', 'hope', 'handal', 'av2', 'av2c', '3dpw'):
        p = [(s, n) for s, g in gt.items() if g['zbior'] == r for n in g['points_world']]
        print(r, len(p), 'z wzorcem', sum(n in wyj[s] for s, n in p))


if __name__ == '__main__':
    main()
