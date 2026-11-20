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
from .laminas import CORREO, LINKEDIN, LINKEDIN_URL, MARCA, _ondas

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
    "capital de trabajo", "análisis vertical y horizontal", "control de gestión",
    "prompts para finanzas", "CFO", "gerente de finanzas", "pymes", "contabilidad",
    "curso de finanzas", "artefactos de Claude", "conectores de Claude", "plugins de Claude",
    "cómo descargar Claude", "planes de Claude", MARCA,
]



def largo_de_etiquetas(etiquetas: list[str]) -> int:
    """Caracteres que YouTube cuenta en las etiquetas: las que llevan espacios van entre comillas, y una coma entre cada una."""
    return sum(len(e) + (2 if " " in e else 0) for e in etiquetas) + max(0, len(etiquetas) - 1)


# Páginas oficiales que el capítulo de primeros pasos muestra en pantalla
DESCARGA_URL = "https://claude.com/download"
PRECIOS_URL = "https://claude.com/pricing"
# Carpeta pública («cualquier persona con el enlace puede ver») en el Google Drive del autor, con el Excel del
# caso y los prompts en PDF. Si se regeneran los archivos de `material/`, hay que volver a subirlos ahí.
MATERIALES_URL = "https://drive.google.com/drive/folders/1HBzyhRRWgxDncfWjbkE5c37ovR5C2kn-"
# El video en el canal del autor. Se subió como «No listado», con título, descripción, etiquetas, categoría
# Educación, miniatura y subtítulos en español; hacerlo público es decisión del autor, en YouTube Studio.
VIDEO_URL = "https://youtu.be/ioX_gRnrr60"
ESTUDIO_URL = "https://studio.youtube.com/video/ioX_gRnrr60/edit"

APRENDIZAJES = [
    "Descargar Claude (web, escritorio y teléfono) y elegir tu plan: Free, Pro, Max, Ultra, Team o Enterprise.",
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
        "📥 DESCARGA GRATIS LOS MATERIALES DEL CURSO",
        f"El archivo del caso (Excel) y los 10 prompts (PDF): {MATERIALES_URL}",
        "",
        "✉️ ¿TIENES DUDAS?",
        f"Escríbeme a {CORREO} o por LinkedIn: {LINKEDIN_URL}",
        "",
        "🤝 ASESORÍA",
        "¿Quieres implementar inteligencia artificial en el área de finanzas de tu empresa? Estoy disponible para "
        "asesorarte: escríbeme por cualquiera de esos dos medios.",
        "",
        "PARA EMPEZAR CON CLAUDE",
        f"Descarga oficial: {DESCARGA_URL}",
        f"Planes y precios vigentes: {PRECIOS_URL}",
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
        "con Anthropic. Los planes y precios de Claude que se muestran son de referencia (noviembre de 2026) y "
        "pueden cambiar: confírmalos en la página oficial. La narración usa una versión sintetizada de la voz del "
        "autor.",
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
         f"Descárgalos junto con el archivo del caso: {MATERIALES_URL}",
         f"¿Dudas? Escríbeme a {CORREO} o por LinkedIn: {LINKEDIN_URL}",
         "¿Qué paso te gustaría ver con más profundidad? Te leo en los comentarios.",
         "```", "",
         "## Estado", "",
         f"El video está subido al canal como «No listado»: {VIDEO_URL}",
         f"Para editarlo o publicarlo: {ESTUDIO_URL}", "",
         "Ya quedó puesto: título, descripción (con la descarga de los materiales y el contacto), etiquetas, categoría "
         "Educación, idioma español, miniatura y subtítulos en español.", "",
         "## Lo que queda por hacer, en YouTube Studio", "",
         "- [ ] Ver el video completo y abrir el enlace de los materiales en una ventana privada (deben verse el Excel "
         "del caso y los prompts en PDF).",
         "- [ ] Cambiar la visibilidad de «No listado» a «Público».",
         "- [ ] Publicar el comentario de arriba y fijarlo (YouTube solo deja fijar comentarios desde la página del video).",
         "- [ ] Pantalla final en los últimos 20 segundos (suscripción y video recomendado) y tarjetas.",
         "",
         "No hace falta declarar «contenido alterado o sintético»: clonar la propia voz para la narración está entre los "
         "casos que YouTube exime (https://support.google.com/youtube/answer/14328491). La descripción ya lo menciona.",
         *(["", "_Los capítulos de la descripción son estimados: reemplázalos por los del render final._"] if estimados else []),
         ""]
    return "\n".join(L)


MINIATURA_HTML = """<!doctype html><html lang="es"><head><meta charset="utf-8"><style>
@font-face{font-family:"Montserrat";src:url("__FUENTES__/montserrat.woff2") format("woff2");font-weight:100 900}
@font-face{font-family:"Source Sans 3";src:url("__FUENTES__/source-sans-3.woff2") format("woff2");font-weight:200 900}
@font-face{font-family:"Newsreader";src:url("__FUENTES__/newsreader-italic.woff2") format("woff2");font-weight:200 800;font-style:italic}
@font-face{font-family:"Geist Mono";src:url("__FUENTES__/geist-mono.woff2") format("woff2");font-weight:100 900}
*{box-sizing:border-box;margin:0;padding:0}
body{width:1280px;height:720px;overflow:hidden;position:relative;font-family:"Montserrat",sans-serif;color:#F2EADB;
 background:radial-gradient(900px 620px at -4% 112%,rgba(63,98,148,.46),transparent 62%),radial-gradient(700px 500px at 30% -30%,rgba(46,72,110,.5),transparent 64%),#101C2C}
.ondas{position:absolute;inset:0;width:1280px;height:720px;opacity:.26}
/* el autor, sobre un aro de luz que lo separa del fondo */
.aro{position:absolute;left:676px;top:74px;width:620px;height:620px;border-radius:50%;
 background:radial-gradient(circle at 46% 38%,#3A66A0 0%,#27456F 46%,#1A2E4B 100%);box-shadow:0 0 0 3px #E3A62F,0 0 110px 10px rgba(79,125,184,.32)}
.aro::after{content:"";position:absolute;inset:-34px;border-radius:50%;border:1.5px solid rgba(227,166,47,.45)}
.autor{position:absolute;left:704px;top:22px;height:838px;filter:drop-shadow(0 18px 36px rgba(4,8,14,.6))}
.velo{position:absolute;left:560px;right:0;bottom:0;height:300px;background:linear-gradient(180deg,rgba(16,28,44,0),rgba(16,28,44,.8) 52%,#101C2C 96%)}
.placa{position:absolute;right:44px;bottom:84px;display:flex;flex-direction:column;align-items:flex-end;gap:7px}
.placa .n{display:flex;align-items:center;gap:14px;font-family:"Geist Mono",monospace;font-size:25px;font-weight:600;letter-spacing:.2em;text-transform:uppercase;color:#F2EADB}
.placa .n svg{width:38px;height:38px;flex:none}
.placa .c{font-family:"Source Sans 3",sans-serif;font-size:22px;font-weight:600;color:#B4BEC6;border-top:2px solid #E3A62F;padding-top:7px}
.iz{position:absolute;left:60px;top:54px;width:660px}
.sello{display:flex;align-items:center;gap:18px;font-family:"Geist Mono",monospace;font-size:26px;font-weight:600;letter-spacing:.16em;color:#E3A62F}
.sello::before{content:"";width:54px;height:3px;background:#E3A62F}
h1{font-size:100px;line-height:.98;font-weight:800;letter-spacing:-.045em;margin-top:24px}
h1 em{font-family:"Newsreader",serif;font-style:italic;font-weight:500;letter-spacing:-.02em;color:#E3A62F;font-size:1.13em;line-height:.8}
.abajo{position:absolute;left:60px;bottom:42px;display:flex;align-items:stretch;gap:30px}
.dato{width:330px;background:#F2EADB;color:#101C2C;border-radius:8px;padding:16px 26px 18px;box-shadow:0 30px 60px -24px rgba(0,0,0,.75);border-top:6px solid #E3A62F}
.dato .t{font-family:"Geist Mono",monospace;font-size:18px;font-weight:600;letter-spacing:.16em;color:#77695A}
.dato .n{font-size:84px;font-weight:800;letter-spacing:-.05em;line-height:1;margin-top:2px;color:#D2402B;white-space:nowrap}
.dato .d{font-family:"Source Sans 3",sans-serif;font-size:27px;font-weight:600;margin-top:2px}
.lista{display:flex;flex-direction:column;justify-content:center;gap:8px;font-family:"Source Sans 3",sans-serif;font-size:31px;font-weight:600;color:#B4BEC6;white-space:nowrap}
.lista b{font-family:"Montserrat",sans-serif;font-weight:800;color:#F2EADB;letter-spacing:-.02em}
</style></head><body>
__ONDAS__
<div class="aro"></div>
<img class="autor" src="__IMAGENES__/autor.png" alt="">
<div class="velo"></div>
<div class="iz"><span class="sello">CURSO COMPLETO · 1 HORA</span>
<h1>Estados financieros con <em>Claude</em></h1></div>
<div class="abajo"><div class="dato"><div class="t">CAJA DEL AÑO</div><div class="n">−90 %</div><div class="d">con ventas +18 %</div></div>
<div class="lista"><span><b>8</b> pasos</span><span><b>10</b> prompts</span><span><b>1</b> caso práctico</span></div></div>
<div class="placa"><div class="n">__LOGO____MARCA__</div><div class="c">Contador auditor · Magíster en gestión financiera</div></div>
</body></html>"""


def miniatura(destino: Path) -> None:
    """Miniatura de 1280 x 720 con el retrato del autor (imagenes/autor.png)."""
    aqui = Path(__file__).parent.resolve()
    html = destino.with_suffix(".html")
    html.write_text(MINIATURA_HTML.replace("__LOGO__", LINKEDIN).replace("__MARCA__", MARCA)
                    .replace("__ONDAS__", _ondas(0).replace('viewBox="0 0 1920 1080"', 'viewBox="0 0 1920 1080" preserveAspectRatio="none"'))
                    .replace("__FUENTES__", (aqui / "fuentes").as_uri()).replace("__IMAGENES__", (aqui / "imagenes").as_uri()),
                    encoding="utf-8")
    script = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();"
              "const c=await b.newContext({viewport:{width:1280,height:720},deviceScaleFactor:1});const p=await c.newPage();"
              f"await p.goto('file://{html.resolve()}');await p.evaluate(()=>document.fonts.ready);"
              "await p.evaluate(()=>Promise.all([...document.images].map(i=>i.decode().catch(()=>{}))));"
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
