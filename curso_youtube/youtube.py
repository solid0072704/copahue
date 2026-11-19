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
from .laminas import LINKEDIN, MARCA

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
    "curso de finanzas", MARCA,
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
        "📥 Ebook: [ENLACE AL EBOOK]",
        "📊 Archivo del caso (Excel) y los 10 prompts: [ENLACE A LOS MATERIALES]",
        f"💼 LinkedIn de {MARCA}: [ENLACE A TU PERFIL DE LINKEDIN]",
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
         "- [ ] Reemplazar los tres enlaces entre [corchetes] (ebook, materiales y tu perfil de LinkedIn).",
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
*{box-sizing:border-box;margin:0;padding:0}
body{width:1280px;height:720px;overflow:hidden;font-family:"Inter Display","Inter",sans-serif;color:#fff;
 background:radial-gradient(900px 560px at 86% 8%,#1F4468 0%,rgba(31,68,104,0) 62%),radial-gradient(700px 480px at 0% 100%,rgba(227,167,43,.28) 0%,rgba(227,167,43,0) 62%),#0B1B2B}
.iz{position:absolute;left:64px;top:56px;width:760px}
.sello{display:inline-block;background:#E3A72B;color:#0B1B2B;font-weight:800;font-size:30px;letter-spacing:.1em;padding:10px 22px;border-radius:999px}
h1{font-size:104px;line-height:.98;font-weight:800;letter-spacing:-.04em;margin-top:30px}
h1 em{font-style:normal;color:#E3A72B}
.sub{margin-top:30px;font-family:"Inter",sans-serif;font-size:40px;font-weight:650;color:#D5DEE8}
.marca{position:absolute;left:64px;bottom:40px;display:flex;align-items:center;gap:16px;font-family:"Inter",sans-serif;font-size:24px;font-weight:800;letter-spacing:.22em;text-transform:uppercase;color:#C9D3DE}
.marca svg{width:40px;height:40px;flex:none}
.gr{position:absolute;right:56px;top:86px;width:400px;height:520px;background:linear-gradient(180deg,#17334F,#10263C);border:3px solid #2A4A6B;border-radius:34px;padding:34px;box-shadow:0 30px 70px rgba(0,0,0,.45)}
.gr .t{font-family:"Inter",sans-serif;font-size:25px;font-weight:700;letter-spacing:.12em;color:#8FB4DA;text-transform:uppercase}
.gr .n{font-size:108px;font-weight:800;letter-spacing:-.04em;line-height:1;margin-top:10px;color:#F0705C}
.gr .d{font-family:"Inter",sans-serif;font-size:30px;font-weight:650;color:#D5DEE8;margin-top:4px}
.gr svg{position:absolute;left:34px;right:34px;bottom:34px;width:332px;height:230px}
</style></head><body>
<div class="iz"><span class="sello">CURSO COMPLETO · 1 HORA</span>
<h1>Estados financieros con <em>Claude</em></h1>
<div class="sub">8 pasos · 10 prompts · 1 caso práctico</div></div>
<div class="marca">__LOGO____MARCA__</div>
<div class="gr"><div class="t">Caja</div><div class="n">−90 %</div><div class="d">con ventas +18 %</div>
<svg viewBox="0 0 332 230"><line x1="0" y1="60" x2="332" y2="60" stroke="#5B7895" stroke-width="3"/>
<rect x="6" y="14" width="46" height="46" rx="6" fill="#8FB4DA"/><rect x="62" y="0" width="46" height="14" rx="5" fill="#2FA36B"/>
<rect x="118" y="0" width="46" height="112" rx="6" fill="#E0563F"/><rect x="174" y="112" width="46" height="64" rx="6" fill="#E0563F"/>
<rect x="230" y="158" width="46" height="18" rx="5" fill="#2FA36B"/><rect x="286" y="60" width="46" height="98" rx="6" fill="#B33A28"/></svg></div>
</body></html>"""


def miniatura(destino: Path) -> None:
    html = destino.with_suffix(".html")
    html.write_text(MINIATURA_HTML.replace("__LOGO__", LINKEDIN).replace("__MARCA__", MARCA), encoding="utf-8")
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
