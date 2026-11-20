"""Genera el HTML de las láminas y las captura con Playwright: imágenes fijas o el video de cada escena.

Uso:
    python -m curso_youtube.construir <trabajo> [prefijo ...]
        Una imagen por paso (1920 x 1080), para revisar el diseño. Avisa si algo se sale de la lámina.

    python -m curso_youtube.construir <trabajo> --continuo --audio <carpeta_audio> [--fin 20] [--reanudar]
        El video de cada lámina con movimiento continuo. Con el audio se sabe cuánto dura cada paso; cada lámina se
        reproduce de corrido y se codifica mientras se captura (queda en <trabajo>/escenas). `--reanudar` no repite
        las escenas ya hechas; `--solo seg_000 seg_016 …` limita el video a unos segmentos (una muestra).

Deja el HTML en <trabajo>/html y las imágenes en <trabajo>/frames. `--hilos N` reparte la captura en N navegadores.
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
    manifiesto = [dict(id=s["id"], tipo=s["tipo"], pasos=len(s["pasos"]), continua=s["tipo"] in _HEROES) for s in laminas]
    (destino / "laminas.json").write_text(json.dumps(manifiesto), encoding="utf-8")
    return laminas


def capturar(trabajo: Path, prefijos: list[str], hilos: int | None = None) -> None:
    """Imágenes fijas: una por paso."""
    frames = trabajo / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    orden = ["node", str(AQUI / "capturar.cjs"), str(trabajo / "html"), str(frames)]
    if hilos:
        orden += ["--hilos", str(hilos)]
    subprocess.run(orden + prefijos, check=True)


def capturar_escenas(trabajo: Path, carpeta_audio: Path, fin: float, solo: list[str], prefijos: list[str],
                     hilos: int | None = None, reanudar: bool = False) -> Path:
    """El video de cada lámina, con movimiento continuo y calzado con el audio. Devuelve la carpeta de escenas."""
    from . import ensamblar
    plan = ensamblar.plan_de_escenas(ensamblar.linea_de_tiempo(trabajo, carpeta_audio, solo, False, fin))
    escenas = trabajo / "escenas"
    escenas.mkdir(parents=True, exist_ok=True)
    (escenas / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    orden = ["node", str(AQUI / "capturar.cjs"), str(trabajo / "html"), str(escenas), "--continuo", str(escenas / "plan.json")]
    if hilos:
        orden += ["--hilos", str(hilos)]
    if reanudar:
        orden.append("--reanudar")
    subprocess.run(orden + prefijos, check=True)
    return escenas


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trabajo", nargs="?", type=Path, default=TRABAJO_POR_DEFECTO)
    ap.add_argument("prefijos", nargs="*", help="solo las láminas cuyo id empieza así (m05, m05_s03…)")
    ap.add_argument("--continuo", action="store_true", help="captura el video de cada lámina, con movimiento continuo")
    ap.add_argument("--audio", type=Path, default=None, help="carpeta con los audios: fija cuánto dura cada paso")
    ap.add_argument("--fin", type=float, default=20.0, help="segundos de la última lámina al final del video")
    ap.add_argument("--solo", nargs="*", default=[], help="solo estos segmentos de audio (para una muestra)")
    ap.add_argument("--hilos", type=int, default=None)
    ap.add_argument("--reanudar", action="store_true", help="no repite las escenas ya capturadas")
    a = ap.parse_args()
    laminas = generar_html(a.trabajo)
    print(f"{len(laminas)} láminas en {a.trabajo / 'html'}")
    if a.continuo:
        if not a.audio:
            ap.error("--continuo necesita --audio: el movimiento se reparte según lo que dura cada paso")
        capturar_escenas(a.trabajo, a.audio, a.fin, a.solo, a.prefijos, a.hilos, a.reanudar)
    else:
        capturar(a.trabajo, a.prefijos, a.hilos)
