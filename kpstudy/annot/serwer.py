# -*- coding: utf-8 -*-
"""Lokalny serwer narzędzia do anotacji: strona, siatki ARKitScenes i zapis klikniętych punktów.

    python serwer.py            # potem w przeglądarce http://127.0.0.1:8765/

GET  /                 -> annotate.html (sceny ARKitScenes)
GET  /obiekty          -> annotate_obj.html (obiekty BOP: HOPE, HANDAL)
GET  /kontekst         -> annotate_ctx.html (ulice Argoverse 2: ludzie, auta, znaki...)
GET  /ludzie           -> annotate_ludzie.html (ciało SMPL, 3DPW)
GET  /bop/<zbior>/models/obj_XXXXXX.(ply|png) -> model i tekstura z dane/bop
GET  /<plik>           -> pliki z katalogu annot/ (meta.json, podglądy, slownik.json, sceny.json)
GET  /raw/<id>/mesh.ply-> siatka z dane/arkitscenes/raw (tylko mesh.ply)
POST /zapisz/<id>      -> zapisuje JSON z ciała żądania do annot/punkty/<id>.json
POST /zapisz_obj/<zbior>_<id> -> annot/punkty_obj/<zbior>_<id>.json
POST /zapisz_ctx/av2_<log>_<track> -> annot/punkty_ctx/av2_<log>_<track>.json
POST /zapisz_ludzie    -> annot/punkty_ludzie.json
Słucha tylko na 127.0.0.1.
"""
import json
import os
import sys
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import paths  # noqa: E402
RAW = os.path.join(paths.DANE, 'arkitscenes', 'raw')
BOP = os.path.join(paths.DANE, 'bop')
PUNKTY = os.path.join(HERE, 'punkty')
PUNKTY_OBJ = os.path.join(HERE, 'punkty_obj')
PUNKTY_CTX = os.path.join(HERE, 'punkty_ctx')


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=HERE, **k)

    def translate_path(self, path):
        m = re.match(r'^/raw/(\d+)/mesh\.ply$', path.split('?')[0])
        if m:
            return os.path.join(RAW, m.group(1), 'mesh.ply')
        m = re.match(r'^/bop/(hope|handal)/models/(obj_\d{6}\.(?:ply|png))$', path.split('?')[0])
        if m:
            return os.path.join(BOP, m.group(1), 'models', m.group(2))
        if path.split('?')[0] == '/obiekty':
            return os.path.join(HERE, 'annotate_obj.html')
        if path.split('?')[0] == '/kontekst':
            return os.path.join(HERE, 'annotate_ctx.html')
        if path.split('?')[0] == '/ludzie':
            return os.path.join(HERE, 'annotate_ludzie.html')
        if path.split('?')[0] in ('/', ''):
            return os.path.join(HERE, 'annotate.html')
        return super().translate_path(path)

    def do_POST(self):
        if self.path == '/zapisz_ludzie':
            dane = json.loads(self.rfile.read(int(self.headers['Content-Length'])).decode('utf-8'))
            json.dump(dane, open(os.path.join(HERE, 'punkty_ludzie.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
            return
        m = re.match(r'^/zapisz/(\d+)$', self.path)
        m2 = re.match(r'^/zapisz_obj/((?:hope|handal)_\d{6})$', self.path)
        m3 = re.match(r'^/zapisz_ctx/(av2_[0-9a-f]{8}_[0-9a-f]{8})$', self.path)
        if not (m or m2 or m3):
            self.send_error(404)
            return
        dane = json.loads(self.rfile.read(int(self.headers['Content-Length'])).decode('utf-8'))
        katalog, klucz = (PUNKTY, m.group(1)) if m else (PUNKTY_OBJ, m2.group(1)) if m2 else (PUNKTY_CTX, m3.group(1))
        os.makedirs(katalog, exist_ok=True)
        json.dump(dane, open(os.path.join(katalog, klucz + '.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


if __name__ == '__main__':
    print('http://127.0.0.1:8765/')
    ThreadingHTTPServer(('127.0.0.1', 8765), H).serve_forever()
