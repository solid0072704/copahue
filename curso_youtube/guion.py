"""Exporta el guion del curso.

Uso:
    python -m curso_youtube.guion                 # escribe curso_youtube/GUION.md (para leer y corregir)
    python -m curso_youtube.guion --voz <carpeta> # un .txt por audio (texto exacto que se envía a la voz) + segmentos.json
    python -m curso_youtube.guion --voz <carpeta> --tono "[calm] [professional]"   # con indicación de tono
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import deck
from .voz import CARACTERES_POR_SEGUNDO

AQUI = Path(__file__).parent


def _titulo(s: dict) -> str:
    return s.get("titulo") or s.get("nombre") or {"gancho": "Gancho", "titulo_curso": "Título del curso",
                                                  "chips": "Lo único que necesitas"}.get(s["tipo"], s["tipo"])


def markdown() -> str:
    laminas = deck.cargar()
    segs = deck.segmentos(laminas)
    total = sum(g["caracteres"] for g in segs)
    minutos = (total / CARACTERES_POR_SEGUNDO + deck.PAUSA_ENTRE_SEGMENTOS * len(segs)) / 60
    L = ["# Guion · Análisis de estados financieros con Claude", "",
         f"{len(laminas)} láminas · {len(segs)} audios · {total:,} caracteres de narración · ≈ {minutos:.0f} minutos".replace(",", "."),
         "", "Cada párrafo es un cambio en pantalla. Las cifras se escriben como se ven; la voz las recibe en palabras.",
         "Para cambiar el texto, edita los archivos de `contenido/` y vuelve a generar este documento.", ""]
    modulo = -1
    for s in laminas:
        if s["modulo"] != modulo:
            modulo = s["modulo"]
            L += ["", f"## {deck.MODULOS[modulo]['capitulo']}", ""]
        L += [f"### {s['id']} · {_titulo(s)}", ""]
        for crudo in s["pasos"]:
            texto, continua = deck._paso(crudo)
            L.append(("↳ " if continua else "") + texto)
            L.append("")
    return "\n".join(L)


def prompts_markdown() -> str:
    """Los prompts del curso, listos para copiar (material descargable y comentario fijado)."""
    L = ["# Los prompts del curso · Análisis de estados financieros con Claude", "",
         "Copia cada prompt, reemplaza lo que está entre [corchetes] y úsalo en orden. "
         "Funcionan mejor dentro de un Proyecto de Claude que ya tenga el contexto de tu empresa.", ""]
    vistos = set()
    extras = []
    for s in deck.cargar():
        if s["tipo"] == "prompt" and s["numero"] not in vistos and "ejemplo" not in s["nombre"]:
            vistos.add(s["numero"])
            texto = "".join(seg if isinstance(seg, str) else seg[0] for seg in s["segs"]).strip()
            L += [f"## Prompt {s['numero']} · {s['nombre']}", "", "```", texto, "```", ""]
        elif s.get("previo") and s["previo"].get("pill", "PROMPT") == "PROMPT":
            extras.append(s["previo"]["t"])
    if extras:
        L += ["## Repreguntas usadas en el caso", ""] + [f"- {e}" for e in extras] + [""]
    L += ["## Cinco repreguntas que siempre sirven", "", "- ¿Por qué? Dame hipótesis y el dato para confirmarlas.",
          "- ¿Cómo lo calculaste? Muéstrame la fórmula y las cifras.", "- ¿Qué supusiste?",
          "- ¿Qué dato te falta?", "- ¿Qué pasaría si…?", ""]
    return "\n".join(L)


def exportar_voz(carpeta: Path, tono: str = "") -> list[dict]:
    """Un .txt por audio. `tono` antepone indicaciones de tono para modelos que las aceptan,
    por ejemplo «[calm] [professional]» en ElevenLabs v4 (la voz no las lee: solo cambian la entonación)."""
    carpeta.mkdir(parents=True, exist_ok=True)
    segs = deck.segmentos()
    manifiesto = []
    prefijo = tono.strip() + " " if tono.strip() else ""
    for g in segs:
        texto = prefijo + g["voz"]
        (carpeta / f"{g['id']}.txt").write_text(texto + "\n", encoding="utf-8")
        manifiesto.append(dict(id=g["id"], laminas=g["laminas"], caracteres=len(texto),
                               segundos_estimados=round(g["segundos_estimados"], 1)))
    (carpeta / "segmentos.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifiesto


if __name__ == "__main__":
    if "--voz" in sys.argv:
        destino = Path(sys.argv[sys.argv.index("--voz") + 1])
        tono = sys.argv[sys.argv.index("--tono") + 1] if "--tono" in sys.argv else ""
        m = exportar_voz(destino, tono)
        print(f"{len(m)} textos de voz en {destino} · {sum(x['caracteres'] for x in m)} caracteres · "
              f"el más largo: {max(x['caracteres'] for x in m)}")
    else:
        (AQUI / "GUION.md").write_text(markdown(), encoding="utf-8")
        (AQUI / "material").mkdir(exist_ok=True)
        (AQUI / "material" / "prompts.md").write_text(prompts_markdown(), encoding="utf-8")
        print("Guion escrito en", AQUI / "GUION.md", "y prompts en", AQUI / "material" / "prompts.md")
