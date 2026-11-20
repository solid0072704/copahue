"""Genera el material para publicar en YouTube: título, descripción, capítulos, etiquetas y miniatura.

Uso:
    python -m curso_youtube.youtube [archivo_de_capitulos.txt]

Sin argumentos usa capítulos estimados a partir del largo del guion. Después del render final, pásale el
archivo `*_capitulos.txt` que deja `ensamblar` para tener los tiempos exactos.
Escribe `youtube/publicacion.md` y `youtube/miniatura.png` (1280 x 720).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import deck
from .ensamblar import ENTRADA, _marca
from .laminas import CORREO, LINKEDIN, LINKEDIN_URL, MARCA

AQUI = Path(__file__).parent
SALIDA = AQUI / "youtube"

TITULOS = [
    "Análisis de estados financieros con Claude (IA): curso completo en 1 hora",
    "Cómo analizar tus estados financieros con IA: método de 8 pasos con Claude",
    "Claude para finanzas: del balance a las decisiones, paso a paso",
]

ETIQUETAS = [
    "análisis de estados financieros", "estados financieros", "análisis financiero", "Claude", "Claude AI",
    "inteligencia artificial en finanzas", "IA para finanzas", "ratios financieros", "flujo de caja",
    "capital de trabajo", "análisis vertical y horizontal", "desviaciones presupuesto", "control de gestión",
    "informe al directorio", "prompts para finanzas", "CFO", "gerente de finanzas", "pymes", "contabilidad",
    "curso de finanzas", "artefactos de Claude", "conectores de Claude", "plugins de Claude", "Claude skills", MARCA,
]

APRENDIZAJES = [
    "Preparar tus estados financieros y darle contexto a Claude (y qué cuidar en confidencialidad).",
    "Verificar que el balance, el estado de resultados y el flujo de efectivo cuadren antes de analizar.",
    "Hacer análisis vertical y horizontal y encontrar qué crece más rápido que las ventas.",
    "Calcular e interpretar ratios de liquidez, endeudamiento, rentabilidad y eficiencia, con DuPont y ciclo de efectivo.",
    "Explicar por qué la utilidad no se transforma en caja, con un puente de flujo de efectivo.",
    "Analizar desviaciones contra el presupuesto, separando el efecto volumen del efecto margen.",
    "Detectar señales de alerta con un «abogado del diablo».",
    "Cuantificar palancas y escenarios, y redactar un informe de una página para el directorio.",
    "Conocer los artefactos, conectores, skills y plugins de Claude que ayudan en la gestión financiera.",
]


def capitulos_estimados() -> list[tuple[float, str]]:
    laminas = deck.cargar()
    segs = deck.segmentos(laminas)
    inicio_seg, cursor = {}, ENTRADA
    for g in segs:
        inicio_seg[g["n"]] = cursor
        cursor += g["segundos_estimados"] + deck.PAUSA_ENTRE_SEGMENTOS
    caps = []
    for s in laminas:
        if s["primera_del_modulo"]:
            caps.append((inicio_seg[s["seg"]] if caps else 0.0, deck.MODULOS[s["modulo"]]["capitulo"]))
    return caps


def publicacion(capitulos: list[str], estimados: bool) -> str:
    nota = ("_Tiempos estimados a partir del guion. Reemplázalos por el archivo de capítulos del render final._"
            if estimados else "_Tiempos exactos del render final._")
    descripcion = [
        "Aprende a analizar estados financieros con Claude, paso a paso: verificación, análisis vertical y "
        "horizontal, ratios, flujo de caja, desviaciones contra el presupuesto, señales de alerta e informe para "
        "el directorio. Con un caso práctico completo y 10 prompts listos para usar.",
        "",
        "📊 Archivo del caso (Excel) y los 10 prompts: [ENLACE A LOS MATERIALES]",
        f"💼 LinkedIn de {MARCA}: {LINKEDIN_URL}",
        "",
        "ASESORÍA",
        "¿Quieres implementar inteligencia artificial en el área de finanzas de tu empresa? Estoy disponible para "
        f"asesorarte. Escríbeme por LinkedIn ({LINKEDIN_URL}) o a {CORREO}.",
        "",
        "CAPÍTULOS",
        *capitulos,
        "",
        "LO QUE VAS A APRENDER",
        *[f"• {a}" for a in APRENDIZAJES],
        "",
        "SOBRE EL CASO",
        "Distribuidora Alerce es una empresa ficticia creada para este curso. En 2025 vendió 18 % más y tuvo la "
        "mayor utilidad de su historia, pero su caja bajó de 620 a 65 millones. El curso muestra, con Claude, por "
        "qué pasó y qué hacer.",
        "",
        "AVISO",
        "Contenido educativo; no constituye asesoría financiera, contable ni tributaria. Verifica siempre los "
        "resultados antes de tomar decisiones. Claude es un producto de Anthropic; este canal no tiene afiliación "
        "con Anthropic. La narración usa una versión sintetizada de la voz del autor.",
        "",
        "#finanzas #inteligenciaartificial #estadosfinancieros",
    ]
    L = ["# Publicación en YouTube", "",
         "## Título (elige uno)", "", *[f"{i}. {t}  ({len(t)} caracteres)" for i, t in enumerate(TITULOS, 1)], "",
         "## Descripción", "", nota, "", "```", *descripcion, "```", "",
         "## Etiquetas", "", ", ".join(ETIQUETAS), "",
         "## Comentario fijado (sugerido)", "",
         "```",
         "Los 10 prompts del curso, en orden: 1) contexto y reglas, 2) verificación, 3) primera lectura, "
         "4) análisis vertical y horizontal, 5) ratios, 6) flujo de caja, 7) desviaciones, 8) señales de alerta, "
         "9) palancas y escenarios, 10) informe para el directorio.",
         "Descárgalos junto con el archivo del caso: [ENLACE A LOS MATERIALES]",
         "¿Qué paso te gustaría ver con más profundidad? Te leo en los comentarios.",
         "```", "",
         "## Lista de verificación antes de publicar", "",
         "- [ ] Reemplazar el enlace entre [corchetes] (materiales), en la descripción y en el comentario fijado.",
         "- [ ] Subir `material/caso_distribuidora_alerce.xlsx` y `material/prompts.md` al lugar que enlazarás.",
         *(["- [ ] Pegar los capítulos definitivos (archivo `*_capitulos.txt` del render final)."] if estimados else []),
         "- [ ] Subir la miniatura `youtube/miniatura.png` (1280 x 720).",
         "- [ ] Subir los subtítulos `.srt` (idioma: español) o pegar el guion para que YouTube los sincronice.",
         "- [ ] Categoría: Educación. Idioma del video: español.",
         "- [ ] Revisar la declaración de «contenido alterado o sintético» de YouTube: la narración usa tu voz clonada.",
         "- [ ] Pantalla final en los últimos 20 segundos (suscripción y video recomendado) y tarjetas.",
         "- [ ] Publicar primero como «No listado», revisarlo completo y recién después hacerlo público.",
         ""]
    return "\n".join(L)


MINIATURA_HTML = """<!doctype html><html lang="es"><head><meta charset="utf-8"><style>
@font-face{font-family:"Montserrat";src:url("__FUENTES__/montserrat.woff2") format("woff2");font-weight:100 900}
@font-face{font-family:"Source Sans 3";src:url("__FUENTES__/source-sans-3.woff2") format("woff2");font-weight:200 900}
@font-face{font-family:"Newsreader";src:url("__FUENTES__/newsreader-italic.woff2") format("woff2");font-weight:200 800;font-style:italic}
@font-face{font-family:"Geist Mono";src:url("__FUENTES__/geist-mono.woff2") format("woff2");font-weight:100 900}
*{box-sizing:border-box;margin:0;padding:0}
body{width:1280px;height:720px;overflow:hidden;position:relative;font-family:"Montserrat",sans-serif;color:#F2EADB;
 background:radial-gradient(900px 600px at 0% 110%,rgba(63,98,148,.42),transparent 62%),#101C2C}
.foto{position:absolute;right:0;top:0;bottom:0;width:600px;background:url("__IMAGENES__/heroe_gestion.jpg") 58% 50%/cover;
 -webkit-mask-image:linear-gradient(90deg,transparent 0,rgba(0,0,0,.6) 30%,#000 62%);mask-image:linear-gradient(90deg,transparent 0,rgba(0,0,0,.6) 30%,#000 62%)}
.foto::after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(16,28,44,.1),rgba(16,28,44,0) 40%,rgba(16,28,44,.78))}
.iz{position:absolute;left:64px;top:60px;width:800px}
.sello{display:flex;align-items:center;gap:18px;font-family:"Geist Mono",monospace;font-size:27px;font-weight:600;letter-spacing:.16em;color:#E3A62F}
.sello::before{content:"";width:56px;height:3px;background:#E3A62F}
h1{font-size:112px;line-height:.97;font-weight:800;letter-spacing:-.045em;margin-top:28px}
h1 em{font-family:"Newsreader",serif;font-style:italic;font-weight:500;letter-spacing:-.02em;color:#E3A62F;font-size:1.12em;line-height:.8}
.sub{margin-top:30px;font-family:"Source Sans 3",sans-serif;font-size:42px;font-weight:600;color:#B4BEC6}
.marca{position:absolute;left:64px;bottom:44px;display:flex;align-items:center;gap:16px;font-family:"Geist Mono",monospace;font-size:24px;font-weight:600;letter-spacing:.2em;text-transform:uppercase;color:#F2EADB}
.marca svg{width:42px;height:42px;flex:none}
.dato{position:absolute;right:54px;bottom:46px;width:372px;background:#F2EADB;color:#101C2C;border-radius:8px;padding:24px 30px 26px;box-shadow:0 34px 70px -24px rgba(0,0,0,.75);border-top:6px solid #E3A62F}
.dato .t{font-family:"Geist Mono",monospace;font-size:21px;font-weight:600;letter-spacing:.16em;color:#77695A}
.dato .n{font-size:104px;font-weight:800;letter-spacing:-.05em;line-height:1;margin-top:6px;color:#D2402B}
.dato .d{font-family:"Source Sans 3",sans-serif;font-size:31px;font-weight:600;margin-top:6px}
</style></head><body>
<div class="foto"></div>
<div class="iz"><span class="sello">CURSO COMPLETO · 1 HORA</span>
<h1>Estados financieros con <em>Claude</em></h1>
<div class="sub">8 pasos · 10 prompts · 1 caso práctico</div></div>
<div class="marca">__LOGO____MARCA__</div>
<div class="dato"><div class="t">CAJA DEL AÑO</div><div class="n">−90 %</div><div class="d">con ventas +18 %</div></div>
</body></html>"""


def miniatura(destino: Path) -> None:
    aqui = Path(__file__).parent.resolve()
    html = destino.with_suffix(".html")
    html.write_text(MINIATURA_HTML.replace("__LOGO__", LINKEDIN).replace("__MARCA__", MARCA)
                    .replace("__FUENTES__", (aqui / "fuentes").as_uri()).replace("__IMAGENES__", (aqui / "imagenes").as_uri()),
                    encoding="utf-8")
    script = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();"
              "const c=await b.newContext({viewport:{width:1280,height:720},deviceScaleFactor:1});const p=await c.newPage();"
              f"await p.goto('file://{html.resolve()}');await p.evaluate(()=>document.fonts.ready);"
              f"await p.screenshot({{path:'{destino.resolve()}'}});await b.close();}})();")
    subprocess.run(["node", "-e", script], check=True)
    html.unlink()


if __name__ == "__main__":
    SALIDA.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        capitulos, estimados = Path(sys.argv[1]).read_text(encoding="utf-8").strip().splitlines(), False
    else:
        capitulos, estimados = [f"{_marca(t)} {n}" for t, n in capitulos_estimados()], True
    (SALIDA / "publicacion.md").write_text(publicacion(capitulos, estimados), encoding="utf-8")
    miniatura(SALIDA / "miniatura.png")
    print("Material de publicación en", SALIDA)
    print("\n".join(capitulos))
