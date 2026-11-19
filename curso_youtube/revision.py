"""Genera un PDF de revisión: cada lámina (en su último paso) con la narración al lado.

Uso:  python -m curso_youtube.revision <trabajo> [salida.pdf]
Requiere las imágenes de `python -m curso_youtube.construir <trabajo>`.
"""

from __future__ import annotations

import html
import subprocess
import sys
from pathlib import Path

from PIL import Image

from . import deck
from .guion import _titulo
from .voz import CARACTERES_POR_SEGUNDO, a_voz

CSS = """
@page { size: A4 landscape; margin: 10mm; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Inter", sans-serif; color: #0F1E2D; font-size: 10.5pt; line-height: 1.38; }
h1 { font-family: "Inter Display", sans-serif; font-size: 22pt; margin-bottom: 4mm; }
h2 { font-family: "Inter Display", sans-serif; font-size: 15pt; margin: 2mm 0 3mm; padding-bottom: 1.5mm; border-bottom: 1.5pt solid #E3A72B; break-after: avoid; }
.nota { color: #4E5C6B; margin-bottom: 5mm; }
.l { display: grid; grid-template-columns: 118mm 1fr; gap: 6mm; padding: 3mm 0 4mm; border-bottom: .5pt solid #D3CBB9; break-inside: avoid; }
.l img { width: 118mm; border: .5pt solid #D3CBB9; border-radius: 1.5mm; }
.l .id { font-size: 8pt; font-weight: 700; letter-spacing: .08em; color: #A56B05; text-transform: uppercase; }
.l h3 { font-size: 11.5pt; margin: .5mm 0 1.5mm; }
.l p { margin-bottom: 1.6mm; }
.l p i { font-style: normal; color: #A56B05; font-weight: 700; margin-right: 1.5mm; }
.portada { height: 170mm; display: flex; flex-direction: column; justify-content: center; }
.portada table { border-collapse: collapse; margin-top: 6mm; font-size: 10.5pt; }
.portada td { padding: 1.2mm 6mm 1.2mm 0; border-bottom: .5pt solid #E4DDCF; }
"""


def construir(trabajo: Path, salida: Path) -> Path:
    laminas = deck.cargar()
    segs = deck.segmentos(laminas)
    mini = trabajo / "miniaturas"
    mini.mkdir(exist_ok=True)
    total = sum(g["caracteres"] for g in segs)
    minutos = (total / CARACTERES_POR_SEGUNDO + deck.PAUSA_ENTRE_SEGMENTOS * len(segs)) / 60
    filas = "".join(
        f"<tr><td>{html.escape(m['capitulo'])}</td><td>{sum(1 for s in laminas if s['modulo'] == i)} láminas</td>"
        f"<td>{sum(len(a_voz(deck._paso(p)[0])) for s in laminas if s['modulo'] == i for p in s['pasos']) / CARACTERES_POR_SEGUNDO / 60:.1f} min</td></tr>"
        for i, m in enumerate(deck.MODULOS))
    partes = [f'<div class="portada"><h1>Análisis de estados financieros con Claude</h1>'
              f'<p class="nota">Guion para revisión · {len(laminas)} láminas · {len(segs)} audios · ≈ {minutos:.0f} minutos de '
              f'narración. Cada párrafo numerado es un cambio en pantalla. «↳» indica que la frase continúa el párrafo anterior.</p>'
              f"<table>{filas}</table></div>"]
    modulo = -1
    for s in laminas:
        if s["modulo"] != modulo:
            modulo = s["modulo"]
            partes.append(f"<h2>{html.escape(deck.MODULOS[modulo]['capitulo'])}</h2>")
        origen = trabajo / "frames" / f"{s['id']}_p{len(s['pasos']) - 1:02d}.png"
        jpg = mini / f"{s['id']}.jpg"
        Image.open(origen).convert("RGB").resize((1120, 630), Image.LANCZOS).save(jpg, quality=82)
        parrafos = []
        for k, crudo in enumerate(s["pasos"], 1):
            texto, continua = deck._paso(crudo)
            parrafos.append(f"<p><i>{'↳' if continua else k}</i>{html.escape(texto)}</p>")
        partes.append(f'<div class="l"><img src="{jpg.resolve().as_uri()}"><div><div class="id">{s["id"]} · audio seg_{s["seg"]:03d}</div>'
                      f'<h3>{html.escape(_titulo(s))}</h3>{"".join(parrafos)}</div></div>')
    pagina = trabajo / "revision.html"
    pagina.write_text(f'<!doctype html><html lang="es"><head><meta charset="utf-8"><style>{CSS}</style></head>'
                      f'<body>{"".join(partes)}</body></html>', encoding="utf-8")
    script = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();"
              f"await p.goto('file://{pagina.resolve()}');await p.evaluate(()=>document.fonts.ready);"
              f"await p.pdf({{path:'{salida.resolve()}',format:'A4',landscape:true,printBackground:true,preferCSSPageSize:true}});"
              "await b.close();})();")
    subprocess.run(["node", "-e", script], check=True)
    return salida


if __name__ == "__main__":
    trabajo = Path(sys.argv[1])
    salida = Path(sys.argv[2]) if len(sys.argv) > 2 else trabajo / "guion_con_laminas.pdf"
    print("PDF de revisión:", construir(trabajo, salida))
