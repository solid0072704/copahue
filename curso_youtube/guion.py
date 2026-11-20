"""Exporta el guion del curso.

Uso:
    python -m curso_youtube.guion                 # escribe curso_youtube/GUION.md (para leer y corregir)
    python -m curso_youtube.guion --pdf           # material/prompts.pdf: los prompts para compartir con la audiencia
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


_PROMPTS_CSS = """
@font-face{font-family:"Montserrat";src:url("__FUENTES__/montserrat.woff2") format("woff2");font-weight:100 900}
@font-face{font-family:"Source Sans 3";src:url("__FUENTES__/source-sans-3.woff2") format("woff2");font-weight:200 900}
@font-face{font-family:"Newsreader";font-style:italic;src:url("__FUENTES__/newsreader-italic.woff2") format("woff2");font-weight:200 800}
@font-face{font-family:"Geist Mono";src:url("__FUENTES__/geist-mono.woff2") format("woff2");font-weight:100 900}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Source Sans 3",sans-serif;color:#121E2E;font-size:10.6pt;line-height:1.45;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.portada{background:#101C2C;color:#F2EADB;border-radius:6px;padding:26px 30px 28px}
.k{font-family:"Geist Mono",monospace;font-size:8.4pt;font-weight:500;letter-spacing:.16em;text-transform:uppercase;color:#E3A62F}
.portada h1{font-family:"Montserrat",sans-serif;font-size:26pt;line-height:1.08;font-weight:700;letter-spacing:-.03em;margin-top:10px}
.portada h1 em{font-family:"Newsreader",serif;font-style:italic;font-weight:500;letter-spacing:-.02em;color:#E3A62F}
.portada p{color:#B4BEC6;margin-top:10px;font-size:11.4pt;max-width:150mm}
.como{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:16px 0 4px}
.como div{border-top:2.5px solid #E3A62F;padding-top:8px}
.como b{display:block;font-family:"Montserrat",sans-serif;font-size:10.6pt;font-weight:700;letter-spacing:-.01em}
.como span{color:#47535F;font-size:9.8pt;line-height:1.38;display:block;margin-top:2px}
.prompt{break-inside:avoid;margin-top:17px}
.prompt .k{color:#A86E05;font-size:7.8pt}
h2{font-family:"Montserrat",sans-serif;font-size:13pt;font-weight:700;letter-spacing:-.015em;margin:3px 0 7px;display:flex;align-items:center;gap:10px}
h2 .n{font-family:"Geist Mono",monospace;font-size:8.2pt;font-weight:600;letter-spacing:.1em;background:#101C2C;color:#F2EADB;padding:3px 8px;border-radius:4px;white-space:nowrap}
pre{font-family:"Geist Mono",monospace;font-size:8.7pt;line-height:1.52;background:#F7F2E7;border:1px solid #DED6C4;border-left:4px solid #E3A62F;border-radius:5px;padding:11px 14px;white-space:pre-wrap}
pre .var{color:#A86E05;font-weight:600}
pre .tit{font-weight:600;letter-spacing:.08em;color:#47535F}
.lista{break-inside:avoid;margin-top:20px}
.lista h3{font-family:"Montserrat",sans-serif;font-size:12pt;font-weight:700;letter-spacing:-.015em;margin-bottom:6px}
.lista li{list-style:none;position:relative;padding:5px 0 5px 18px;border-top:1px solid #DED6C4}
.lista li::before{content:"";position:absolute;left:2px;top:.78em;width:7px;height:2px;background:#E3A62F}
.cierre{break-inside:avoid;margin-top:22px;background:#101C2C;color:#F2EADB;border-radius:6px;padding:18px 24px 20px;display:grid;grid-template-columns:1fr auto;gap:26px;align-items:center}
.cierre b{display:block;font-family:"Montserrat",sans-serif;font-size:13.5pt;font-weight:700;letter-spacing:-.02em;margin-top:4px}
.cierre p{color:#B4BEC6;font-size:10.2pt;margin-top:3px}
.cierre .c{border-left:1.5px solid #405064;padding-left:22px;font-family:"Montserrat",sans-serif;font-size:10pt;font-weight:600;line-height:1.7;white-space:nowrap}
.nota{margin-top:12px;color:#86909B;font-size:8.6pt}
"""


def prompts_html() -> str:
    """Los prompts del curso como documento para imprimir o guardar en PDF (el material que se descarga)."""
    from .laminas import CORREO, CURSO, LINKEDIN_URL, MARCA, _prompt_texto, esc
    fuentes = (AQUI / "fuentes").resolve().as_uri()
    partes, vistos, extras = [], set(), []
    for s in deck.cargar():
        if s["tipo"] == "prompt" and s["numero"] not in vistos and "ejemplo" not in s["nombre"]:
            vistos.add(s["numero"])
            texto = "".join(seg if isinstance(seg, str) else seg[0] for seg in s["segs"]).strip()
            partes.append(f'<div class="prompt"><div class="k">{esc(s["kicker"])}</div>'
                          f'<h2><span class="n">PROMPT {s["numero"]}</span>{esc(s["nombre"])}</h2>'
                          f'<pre>{_prompt_texto(texto)}</pre></div>')
        elif s.get("previo") and s["previo"].get("pill", "PROMPT") == "PROMPT":
            extras.append(s["previo"]["t"])
    siempre = ["¿Por qué? Dame hipótesis y el dato para confirmarlas.", "¿Cómo lo calculaste? Muéstrame la fórmula y las cifras.",
               "¿Qué supusiste?", "¿Qué dato te falta?", "¿Qué pasaría si…?"]

    def lista(titulo: str, items: list[str]) -> str:
        return f'<div class="lista"><h3>{esc(titulo)}</h3><ul>{"".join(f"<li>{esc(x)}</li>" for x in items)}</ul></div>'

    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Los prompts del curso · {esc(CURSO)}</title>'
            f'<style>{_PROMPTS_CSS.replace("__FUENTES__", fuentes)}</style></head><body>'
            f'<div class="portada"><div class="k">Material del curso · {len(vistos)} prompts</div>'
            '<h1>Análisis de estados financieros con <em>Claude</em></h1>'
            '<p>Los prompts del método de 8 pasos, listos para copiar: de preparar los datos al informe para el directorio.</p></div>'
            '<div class="como"><div><b>1 · Copia el prompt</b><span>Úsalos en orden: cada uno se apoya en el anterior.</span></div>'
            '<div><b>2 · Reemplaza los [corchetes]</b><span>Con los datos de tu empresa y de tu período.</span></div>'
            '<div><b>3 · Revisa la respuesta</b><span>Pide siempre el cálculo y los supuestos antes de usarla.</span></div></div>'
            + "".join(partes)
            + (lista("Repreguntas usadas en el caso", extras) if extras else "")
            + lista("Cinco repreguntas que siempre sirven", siempre)
            + '<div class="cierre"><div><div class="k">Asesoría</div><b>¿Quieres implementar IA en tu área de finanzas?</b>'
              f'<p>Estoy disponible para asesorar a tu empresa. Escríbeme.</p></div><div class="c">{esc(MARCA)}<br>'
              f'{esc(LINKEDIN_URL.split("://www.")[-1])}<br>{esc(CORREO)}</div></div>'
            '<p class="nota">Funcionan mejor dentro de un Proyecto de Claude que ya tenga el contexto de tu empresa. Contenido educativo: '
            'verifica siempre los resultados antes de tomar decisiones.</p></body></html>')


def prompts_pdf(destino: Path) -> None:
    """Escribe los prompts en PDF (A4), con Chromium: es el archivo que se comparte con la audiencia."""
    import subprocess
    html = destino.with_suffix(".html")
    html.write_text(prompts_html(), encoding="utf-8")
    pie = ('<div style="font-family:sans-serif;font-size:7pt;color:#86909B;width:100%;padding:0 17mm;display:flex;'
           'justify-content:space-between"><span>Análisis de estados financieros con Claude · Los prompts del curso</span>'
           '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>')
    script = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();"
              f"await p.goto('file://{html.resolve()}');await p.evaluate(()=>document.fonts.ready);"
              f"await p.pdf({{path:'{destino.resolve()}',format:'A4',printBackground:true,displayHeaderFooter:true,"
              f"headerTemplate:'<span></span>',footerTemplate:{json.dumps(pie)},"
              "margin:{top:'15mm',bottom:'17mm',left:'17mm',right:'17mm'}});await b.close();})();")
    subprocess.run(["node", "-e", script], check=True)
    html.unlink()


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
    if "--pdf" in sys.argv:
        (AQUI / "material").mkdir(exist_ok=True)
        prompts_pdf(AQUI / "material" / "prompts.pdf")
        print("Prompts en PDF:", AQUI / "material" / "prompts.pdf")
    elif "--voz" in sys.argv:
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
