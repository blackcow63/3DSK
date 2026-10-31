# -*- coding: utf-8 -*-
"""Tabele sekcji Evaluation z wyników na zbiorze kp_razem: docs/paper/tab_eval.tex, tab_rozklad.tex i makra wyniki.tex.

    python report/tabela_eval.py --root ../../dane/kp_razem

Błąd podany jest w procentach odległości punktu od kamery, więc biurko i ulica dają się połączyć w jeden wynik. Wszystkie
źródła z pomiarem liczone razem, na tych samych zdjęciach: co dziewiąte zdjęcie typu frame (wspólna próbka).
Brak odpowiedzi to chybienie w progach, mediany z punktów z odpowiedzią. Rozkład błędu: prawdziwy piksel + głębia modelu oraz piksel modelu + prawdziwa głębia.
Przedziały ufności: bootstrap po scenach, 2000 losowań, ziarno 0.
"""
import argparse
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, '..', '..', 'docs', 'paper'))
MODELE = [
    ('robobrain25_8b', 'RoboBrain 2.5 8B-NV', 'A'), ('robobrain25_8b_mt', 'RoboBrain 2.5 8B-MT', 'A'), ('robobrain25_4b', 'RoboBrain 2.5 4B', 'A'),
    ('robobrain25_8b_q4', 'RoboBrain 2.5 8B-NV, 4-bit', 'A'), ('robobrain25_4b_q4', 'RoboBrain 2.5 4B, 4-bit', 'A'),
    ('qwen3vl8b_razem', 'Qwen3-VL-8B', 'B'), ('cosmos_r2_8b_razem', 'Cosmos-Reason2-8B', 'B'),
    ('rynnbrain11_9b', 'RynnBrain 1.1 9B$^a$', 'B'), ('rynnbrain11_2b', 'RynnBrain 1.1 2B$^a$', 'B'), ('mimo7b_razem', 'MiMo-Embodied-7B$^b$', 'B'),
    ('qwen3vl8b_3d', '8B, asked for $(x, y, d)$', 'control'), ('qwen3vl8b_box', '8B, box centre only', 'control'),
    ('qwen3vl32b_3d', '32B, asked for $(x, y, d)$', 'control'), ('qwen3vl32b_box', '32B, box centre only', 'control'),
]
MAKRA = (('robobrain25_8b', 'Rb'), ('robobrain25_8b_mt', 'RbMt'), ('robobrain25_4b', 'RbMaly'), ('robobrain25_8b_q4', 'RbKwant'), ('robobrain25_4b_q4', 'RbMalyKwant'),
         ('qwen3vl8b_razem', 'Qwen'), ('cosmos_r2_8b_razem', 'Cosmos'), ('rynnbrain11_2b', 'RynMaly'), ('rynnbrain11_9b', 'RynDuzy'), ('mimo7b_razem', 'Mimo'),
         ('qwen3vl8b_3d', 'QwenPyt'), ('qwen3vl8b_box', 'QwenBox'), ('qwen3vl32b_3d', 'QwenDuzyPyt'), ('qwen3vl32b_box', 'QwenDuzyBox'))
ZRODLA = ('arkit', 'av2', 'handal', 'hope')


class Dane:
    def __init__(self, root):
        self.root = root
        self.views = [v for v in json.load(open(os.path.join(root, 'views.json'))) if v['kind'] == 'frame' and v['id'].split('_')[0] in ZRODLA]
        self.gt = json.load(open(os.path.join(root, 'gt.json')))['scenes']
        self.zdj = {}

    def zdjecie(self, vid, scene):
        if vid not in self.zdj:
            c = json.load(open(os.path.join(self.root, vid, 'cam.json')))
            T, K = np.array(c['T_wc']), np.array(c['K'])
            kp = json.load(open(os.path.join(self.root, vid, 'kp.json')))
            P = {n: T[:3, :3].T @ (np.array(x) - T[:3, 3]) for n, x in self.gt[scene]['points_world'].items() if kp.get(n, {}).get('visible')}
            self.zdj[vid] = (K, P, kp)
        return self.zdj[vid]

    def pred(self, m, vid):
        p = os.path.join(self.root, 'preds', 'e2e', m, vid + '.json')
        return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None


def wyniki(ds, m, views):
    """Dla modelu: błędy w % odległości i w cm oraz liczba punktów."""
    w = dict(e=[], cm=[], n=0, zdjec=0)
    for v in views:
        d = ds.pred(m, v['id'])
        if d is None:
            continue
        w['zdjec'] += 1
        _, P, _ = ds.zdjecie(v['id'], v['scene'])
        w['n'] += len(P)
        for n, G in P.items():
            if n in d['points_cam']:
                X = np.array(d['points_cam'][n])
                w['cm'].append(np.linalg.norm(X - G) * 100)
                w['e'].append(np.linalg.norm(X - G) / np.linalg.norm(G) * 100)
    return w


def fm(x, f='%.0f'):
    return f % x if x is not None and np.isfinite(x) else '--'


def med(x):
    return float(np.median(x)) if len(x) else None


def proc(k, n):
    return '%.0f\\%%' % (100 * k / n) if n else '--'


def w_procent(w, prog):
    return proc(sum(e < prog for e in w['e']), w['n'])


def tab_eval(ds, views):
    wyj = [r'\begin{tabular}{@{}lrr@{}}', r'\toprule', r'Model & Median error & Points answered \\', r'\midrule']
    podsum = {}
    for typ, tytul in (('A', 'Type A: writes a point and its depth'), ('B', 'Type B: writes a pixel and a 3D box'), ('control', 'Qwen3-VL asked in other ways')):
        w_grupie = []
        for m, nazwa, t in MODELE:
            if t != typ:
                continue
            w = wyniki(ds, m, views)
            if w['zdjec']:
                podsum[m] = w
                w_grupie.append((med(w['e']), nazwa, w))
        if wyj[-1] != r'\midrule':
            wyj.append(r'\addlinespace')
        wyj.append(r'\multicolumn{3}{@{}l}{\emph{%s}} \\' % tytul)
        for e, nazwa, w in sorted(w_grupie, key=lambda x: x[0]):
            wyj.append(r'\quad %s & %s\%% & %s \\' % (nazwa, fm(e), proc(len(w['e']), w['n'])))
    wyj += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(wyj), podsum


def rozklad(ds, m, views):
    """Mediana błędu w %, gdy prawdziwa jest tylko jedna ze składowych: piksel albo głębia."""
    w = collections.defaultdict(list)
    for v in views:
        d = ds.pred(m, v['id'])
        if d is None:
            continue
        K, P, kp = ds.zdjecie(v['id'], v['scene'])
        for n, G in P.items():
            if n not in d['points_cam'] or n not in d['points_2d']:
                continue
            X = np.array(d['points_cam'][n])
            r_gt = np.array([(kp[n]['u'] - K[0, 2]) / K[0, 0], (kp[n]['v'] - K[1, 2]) / K[1, 1], 1.0])
            u, v_ = d['points_2d'][n][:2]
            r_m = np.array([(u - K[0, 2]) / K[0, 0], (v_ - K[1, 2]) / K[1, 1], 1.0])
            g = np.linalg.norm(G)
            w['all'].append(np.linalg.norm(X - G) / g * 100)
            w['depth'].append(np.linalg.norm(r_gt * X[2] - G) / g * 100)
            w['pixel'].append(np.linalg.norm(r_m * G[2] - G) / g * 100)
    return {k: float(np.median(x)) for k, x in w.items()}


def boot(ds, views, a, b):
    """Różnica median a - b w punktach procentowych, na punktach z odpowiedzią obu modeli, przedział 95% po scenach."""
    sc = collections.defaultdict(list)
    for v in views:
        da, db = ds.pred(a, v['id']), ds.pred(b, v['id'])
        if da is None or db is None:
            continue
        _, P, _ = ds.zdjecie(v['id'], v['scene'])
        for n, G in P.items():
            if n in da['points_cam'] and n in db['points_cam']:
                g = np.linalg.norm(G)
                sc[v['scene']].append((np.linalg.norm(np.array(da['points_cam'][n]) - G) / g * 100, np.linalg.norm(np.array(db['points_cam'][n]) - G) / g * 100))
    if len(sc) < 2:
        return None
    klucze = list(sc)
    rng = np.random.default_rng(0)

    def roznica(ks):
        x = np.array([p for k in ks for p in sc[k]])
        return np.median(x[:, 0]) - np.median(x[:, 1])
    bs = [roznica([klucze[i] for i in rng.integers(len(klucze), size=len(klucze))]) for _ in range(2000)]
    return roznica(klucze), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), sum(len(x) for x in sc.values()), len(klucze)


def opis_roznicy(x):
    if x is None:
        return 'not available'
    r, a, b, n, s = x
    pp = 'percentage point' if abs(round(r)) == 1 else 'percentage points'
    return '$%+.0f$ %s (95\\%% CI $%+.0f$ to $%+.0f$, %d points in %d scenes)' % (r, pp, a, b, n, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    args = ap.parse_args()
    ds = Dane(args.root)
    views = [v for v in json.load(open(os.path.join(args.root, 'views.json'))) if v['kind'] == 'frame'][::9]
    views = [v for v in views if v['id'].split('_')[0] in ZRODLA]
    t, podsum = tab_eval(ds, views)
    open(os.path.join(PAPER, 'tab_eval.tex'), 'w', encoding='utf-8').write(t + '\n')
    print(t)
    wyj = [r'\begin{tabular}{@{}lrrr@{}}', r'\toprule', r'Model & Whole answer & Only depth wrong & Only pixel wrong \\', r'\midrule']
    for m, n in (('robobrain25_8b', 'RoboBrain 2.5 8B-NV'), ('qwen3vl8b_razem', 'Qwen3-VL-8B'), ('cosmos_r2_8b_razem', 'Cosmos-Reason2-8B')):
        x = rozklad(ds, m, views)
        if x:
            wyj.append('%s & %s\\%% & %s\\%% & %s\\%% \\\\' % (n, fm(x['all']), fm(x['depth']), fm(x['pixel'])))
        print(n, x)
    wyj += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(PAPER, 'tab_rozklad.tex'), 'w', encoding='utf-8').write('\n'.join(wyj) + '\n')
    mak = ['% Wygenerowane przez kpstudy/report/tabela_eval.py. Nie edytować ręcznie.', '\\newcommand{\\nZdjecProbki}{%d}' % len(views)]
    for m, nm in (('robobrain25_8b', 'Rb'), ('qwen3vl8b_razem', 'Qwen')):
        x = rozklad(ds, m, views)
        mak.append('\\newcommand{\\nPrawdziwyPiksel%s}{%s\\%%}' % (nm, fm(x.get('depth'))))
        mak.append('\\newcommand{\\nPrawdziwaGlebia%s}{%s\\%%}' % (nm, fm(x.get('pixel'))))
    # porównania w tekście: wyłącznie zdjęcia, które mają oba główne modele
    wspolne = [v for v in ds.views if ds.pred('qwen3vl8b_razem', v['id']) and ds.pred('robobrain25_8b', v['id'])]
    for m, nm in MAKRA:
        w = podsum.get(m, dict(e=[], cm=[], n=0, zdjec=0))
        mak.append('\\newcommand{\\nBlad%s}{%s\\%%}' % (nm, fm(med(w['e']))))
        mak.append('\\newcommand{\\nOdp%s}{%s}' % (nm, proc(len(w['e']), w['n'])))
        mak.append('\\newcommand{\\nDziesiec%s}{%s}' % (nm, w_procent(w, 10)))
        c = wyniki(ds, m, wspolne)
        mak.append('\\newcommand{\\nCalyBlad%s}{%s\\%%}' % (nm, fm(med(c['e']))))
        mak.append('\\newcommand{\\nCalyPiec%s}{%s}' % (nm, w_procent(c, 5)))
        mak.append('\\newcommand{\\nCalyDziesiec%s}{%s}' % (nm, w_procent(c, 10)))
        mak.append('\\newcommand{\\nCalyOdp%s}{%s}' % (nm, proc(len(c['e']), c['n'])))
        mak.append('\\newcommand{\\nCalyZdjec%s}{%d}' % (nm, c['zdjec']))
        print(nm.ljust(14), 'próbka', fm(med(w['e'])), 'całość', fm(med(c['e'])), '<5%', w_procent(c, 5), '<10%', w_procent(c, 10), 'odp', proc(len(c['e']), c['n']), 'zdjęć', c['zdjec'], 'cm', fm(med(c['cm'])))
    ludzie = [v for v in json.load(open(os.path.join(args.root, 'views.json'))) if v['kind'] == 'crop' and v['id'].startswith('3dpw')
              and ds.pred('qwen3vl8b_razem', v['id']) and ds.pred('robobrain25_8b', v['id'])]
    mak.append('\\newcommand{\\nZdjecLudzi}{%d}' % len(ludzie))
    # MiMo-Embodied miesza skale głębi: ile głębi idzie w setki i tysiące i jak chybiają pozostałe odpowiedzi
    z, e = [], []
    for v in views:
        d = ds.pred('mimo7b_razem', v['id'])
        if d is None:
            continue
        _, P, _ = ds.zdjecie(v['id'], v['scene'])
        for n, G in P.items():
            if n in d['points_cam']:
                X = np.array(d['points_cam'][n])
                z.append(X[2])
                e.append(np.linalg.norm(X - G) / np.linalg.norm(G) * 100)
    z, e = np.array(z), np.array(e)
    mak.append('\\newcommand{\\nMimoSetki}{%s\\%%}' % fm(100 * np.mean(z >= 100)))
    mak.append('\\newcommand{\\nMimoReszta}{%s\\%%}' % fm(med(e[z < 100])))
    for m, nm in (('robobrain25_8b', 'Rb'), ('qwen3vl8b_razem', 'Qwen')):
        w = wyniki(ds, m, ludzie)
        print('ludzie', nm, fm(med(w['e'])), proc(len(w['e']), w['n']), w['zdjec'])
        mak.append('\\newcommand{\\nLudzie%s}{%s\\%%}' % (nm, fm(med(w['e']))))
        mak.append('\\newcommand{\\nLudzieOdp%s}{%s}' % (nm, proc(len(w['e']), w['n'])))
    x, y = boot(ds, views, 'qwen3vl8b_razem', 'robobrain25_8b'), boot(ds, wspolne, 'qwen3vl8b_razem', 'robobrain25_8b')
    print('próbka', x)
    print('całość', y)
    mak.append('\\newcommand{\\nRozQwen}{%s}' % opis_roznicy(x))
    mak.append('\\newcommand{\\nRozQwenCaly}{%s}' % opis_roznicy(y))
    mak.append('\\newcommand{\\nZdjecCaly}{%d}' % len(wspolne))
    open(os.path.join(PAPER, 'wyniki.tex'), 'w', encoding='utf-8').write('\n'.join(mak) + '\n')


if __name__ == '__main__':
    main()
