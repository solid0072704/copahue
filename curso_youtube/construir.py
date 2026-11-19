"""Genera el HTML de las láminas y lanza la captura a PNG (1920 x 1080) con Playwright.

Uso:
    python -m curso_youtube.construir [carpeta_de_trabajo] [prefijo_de_lámina ...]

Deja el HTML en <trabajo>/html y las imágenes en <trabajo>/frames (una por lámina y paso).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import deck
from .laminas import pagina

AQUI = Path(__file__).parent
TRABAJO_POR_DEFECTO = AQUI / "_trabajo"


def generar_html(trabajo: Path) -> list[dict]:
    destino = trabajo / "html"
    destino.mkdir(parents=True, exist_ok=True)
    laminas = deck.cargar()
    css = (AQUI / "estilo.css").resolve().as_uri()
    for s in laminas:
        ctx = dict(kicker=s["kicker"], modulo=s["modulo"], n_modulos=s["n_modulos"], avance=s["avance"])
        (destino / f"{s['id']}.html").write_text(pagina(s, ctx, css_href=css), encoding="utf-8")
    manifiesto = [dict(id=s["id"], pasos=len(s["pasos"])) for s in laminas]
    (destino / "laminas.json").write_text(json.dumps(manifiesto), encoding="utf-8")
    return laminas


def capturar(trabajo: Path, prefijos: list[str]) -> None:
    frames = trabajo / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    subprocess.run(["node", str(AQUI / "capturar.cjs"), str(trabajo / "html"), str(frames), *prefijos], check=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    trabajo = Path(args[0]) if args and not args[0].startswith("m") else TRABAJO_POR_DEFECTO
    prefijos = [a for a in args if a.startswith("m")]
    laminas = generar_html(trabajo)
    print(f"{len(laminas)} láminas en {trabajo / 'html'}")
    capturar(trabajo, prefijos)
