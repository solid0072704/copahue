"""Genera el HTML de las láminas y las captura a PNG (1920 x 1080) con Playwright.

Uso:
    python -m curso_youtube.construir <trabajo> [prefijo ...]                    # una imagen por paso (revisar diseño)
    python -m curso_youtube.construir <trabajo> --animar --audio <carpeta_audio>  # además, la animación cuadro a cuadro

Deja el HTML en <trabajo>/html y las imágenes en <trabajo>/frames. Con `--animar` cada paso tiene sus cuadros
(`<id>_pKK_fNNN.jpg`); con `--audio` se conoce cuánto dura cada paso en el video, de modo que la animación cabe
en su paso y las láminas con imagen de fondo se capturan completas (el fondo se mueve). `--hilos N` reparte la
captura en N navegadores y `--reanudar` no repite las láminas que ya están capturadas tal como quedarían ahora
(sirve para retomar una captura interrumpida).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from . import deck
from .laminas import _HEROES, pagina

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
    # `continua`: láminas con imagen de fondo en movimiento, que se capturan cuadro a cuadro de principio a fin
    manifiesto = [dict(id=s["id"], tipo=s["tipo"], pasos=len(s["pasos"]), continua=s["tipo"] in _HEROES) for s in laminas]
    (destino / "laminas.json").write_text(json.dumps(manifiesto), encoding="utf-8")
    return laminas


def duracion_de_pasos(trabajo: Path, carpeta_audio: Path, fin: float) -> Path:
    """Escribe <trabajo>/pasos.json: cuántos segundos dura en el video cada paso de cada lámina."""
    from . import ensamblar
    tl = ensamblar.linea_de_tiempo(trabajo, carpeta_audio, [], False, fin)
    ev = tl["eventos"]
    pasos = {}
    for i, e in enumerate(ev):
        hasta = ev[i + 1]["t"] if i + 1 < len(ev) else tl["total"]
        desde = 0.0 if i == 0 else e["t"]
        pasos[e["frame"][:-4]] = round(hasta - desde, 4)
    destino = trabajo / "pasos.json"
    destino.write_text(json.dumps(pasos), encoding="utf-8")
    return destino


def capturar(trabajo: Path, prefijos: list[str], animar: bool = False, pasos: Path | None = None,
             hilos: int | None = None, reanudar: bool = False) -> None:
    frames = trabajo / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    orden = ["node", str(AQUI / "capturar.cjs"), str(trabajo / "html"), str(frames)]
    if animar:
        orden.append("--animar")
    if pasos:
        orden += ["--tiempos", str(pasos)]
    if hilos:
        orden += ["--hilos", str(hilos)]
    if reanudar:
        orden.append("--reanudar")
    subprocess.run(orden + prefijos, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trabajo", nargs="?", type=Path, default=TRABAJO_POR_DEFECTO)
    ap.add_argument("prefijos", nargs="*", help="solo las láminas cuyo id empieza así (m05, m05_s03…)")
    ap.add_argument("--animar", action="store_true", help="captura también la animación de cada paso")
    ap.add_argument("--audio", type=Path, default=None, help="carpeta con los audios: fija cuánto dura cada paso")
    ap.add_argument("--fin", type=float, default=20.0, help="segundos de la última lámina al final del video")
    ap.add_argument("--hilos", type=int, default=None)
    ap.add_argument("--reanudar", action="store_true", help="no repite las láminas ya capturadas (retoma una captura)")
    a = ap.parse_args()
    laminas = generar_html(a.trabajo)
    print(f"{len(laminas)} láminas en {a.trabajo / 'html'}")
    pasos = duracion_de_pasos(a.trabajo, a.audio, a.fin) if a.audio else None
    capturar(a.trabajo, a.prefijos, a.animar, pasos, a.hilos, a.reanudar)
