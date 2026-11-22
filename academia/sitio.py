"""Genera la plataforma web del programa: un sitio estático que sale del temario y de los bancos de preguntas.

    python -m academia.sitio <salida> [--curso curso.mp4]

Deja en `<salida>`:
    index.html          el sitio completo, para publicarlo en cualquier alojamiento estático
    vista_previa.html   la misma página sin su envoltorio y con los subtítulos dentro, para la vista previa privada
    autor.jpg, muestra.mp4, muestra.jpg, muestra.vtt

Con `--curso` (el video del curso gratuito, con su `.srt` y su `_capitulos.txt` al lado) se corta la lección de
muestra: el capítulo «Artefactos, conectores y plugins». Sin él, la plataforma sale sin lección de muestra.

Lo que todavía no hace el sitio, porque depende de servicios por contratar: cuentas de alumnos, cobro y entrega
protegida de los videos. El avance se guarda en el navegador (`Almacen`, en web/app.js).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

from . import evaluaciones, programa

AQUI = Path(__file__).parent
CURSO_GRATUITO = AQUI.parent / "curso_youtube"
FUENTES = ("https://fonts.googleapis.com/css2?family=Geist+Mono:wght@500&family=Montserrat:wght@600;700;800"
           "&family=Newsreader:ital,wght@1,500&family=Source+Sans+3:wght@400;600&display=swap")
CAPITULO_DE_MUESTRA = "Artefactos, conectores y plugins"
MUESTRA = dict(
    t="Artefactos, conectores y plugins",
    d="Tres herramientas de Claude para ir más lejos en la gestión financiera, presentadas en tres minutos.",
    puntos=["Artefactos: del análisis a un tablero o un simulador que se comparte con un enlace",
            "Conectores: Claude lee los datos donde ya están, con tus permisos",
            "Skills y plugins: flujos de trabajo ya armados, como el plugin de finanzas",
            "Cuándo conviene usar cada una"],
    nota="Esta lección viene del curso gratuito «Análisis de estados financieros con Claude». En el programa, cada una de "
         "estas herramientas tiene su propio módulo (5, 6 y 7), con demostraciones y práctica.",
)
BIO = ("Gerente de administración y finanzas y docente en transformación digital financiera e inteligencia artificial. "
       "Autor del curso gratuito «Análisis de estados financieros con Claude».")
PREGUNTAS = [
    ("¿Necesito saber programar?",
     "No. Cuando hace falta código, lo escribe y lo ejecuta Claude; tú aprendes a pedirlo y a revisar el resultado."),
    ("¿Qué plan de Claude necesito?",
     "Puedes empezar con el plan gratuito. Para trabajar con archivos grandes, proyectos y conectores conviene un plan de pago. "
     "Cada lección indica qué plan requiere lo que muestra."),
    ("Mi empresa usa otro CRM, o no usa Power BI. ¿Me sirve?",
     "Sí. Los módulos usan HubSpot y Power BI como ejemplo y muestran la alternativa con archivos exportados, que funciona con cualquier sistema."),
    ("¿Los datos del curso son reales?",
     "No. Distribuidora Alerce es una empresa ficticia creada para el curso, con estados financieros, libro mayor y CRM que cuadran entre sí."),
    ("¿Este programa es de Anthropic?",
     "No. Claude es un producto de Anthropic; este programa es independiente y no tiene afiliación con Anthropic."),
]
AVISO = ("Vista previa de la plataforma: el avance y las notas se guardan solo en este navegador, y las lecciones del "
         "programa están en producción.")
PIE = ["Claude es un producto de Anthropic. Este programa es independiente y no tiene afiliación con Anthropic.",
       "Distribuidora Alerce es una empresa ficticia creada para el curso.",
       f"© 2026 {programa.AUTOR['nombre']}"]


def datos(muestra: dict | None = None) -> dict:
    """Todo lo que lee la aplicación (web/app.js)."""
    return dict(
        nombre=programa.NOMBRE, bajada=programa.BAJADA, autor=dict(programa.AUTOR, bio=BIO, foto="autor.jpg"),
        etapas=programa.ETAPAS, bloques=[{k: b[k] for k in ("clave", "etapa", "t", "d")} for b in programa.BLOQUES],
        modulos=programa.modulos(), totales=programa.totales(), porEtapa={e: programa.totales(e) for e in programa.ETAPAS},
        logros=programa.LOGROS, preguntas=PREGUNTAS, evaluaciones=evaluaciones.para_la_web(), muestra=muestra)


def pagina(d: dict, completa: bool = True, aviso: bool = True) -> str:
    """La página: completa (con su envoltorio HTML) o solo el contenido, como la pide la vista previa."""
    marca = f'<a class="marca" href="#inicio"><i></i>{programa.NOMBRE}</a>'
    cuerpo = (
        f'<div class="barra"><div class="caja">{marca}<nav aria-label="Principal"><a href="#programa">Programa</a>'
        + ('<a href="#muestra">Lección de muestra</a>' if d["muestra"] else "") + '<a href="#curso">Mi curso</a></nav></div></div>'
        + (f'<div class="aviso-previa"><div class="caja">{AVISO}</div></div>' if aviso else "")
        + '<main id="vista"></main><footer class="pie"><div class="caja">' + "".join(f"<span>{x}</span>" for x in PIE) + "</div></footer>"
        + "<script>window.DATOS = " + json.dumps(d, ensure_ascii=False).replace("</", "<\\/") + ";</script>"
        + "<script>" + (AQUI / "web" / "app.js").read_text(encoding="utf-8") + "</script>")
    cabeza = (f"<title>{programa.NOMBRE}</title>\n<meta name=\"claude:viewport-fit-sides\" content=\"cover\">\n"
              f'<link rel="stylesheet" href="{FUENTES}">\n<style>\n' + (AQUI / "web" / "app.css").read_text(encoding="utf-8") + "</style>\n")
    if not completa:
        return cabeza + cuerpo
    return ('<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f'<meta name="description" content="{programa.BAJADA}">\n' + cabeza + "</head>\n<body>\n" + cuerpo + "\n</body>\n</html>\n")


def retrato(destino: Path) -> None:
    """El retrato del autor (recorte con fondo transparente) sobre azul noche, cuadrado."""
    foto = Image.open(CURSO_GRATUITO / "imagenes" / "autor.png").convert("RGBA")
    lado = foto.width
    fondo = Image.new("RGBA", (lado, lado), "#17273B")
    fondo.alpha_composite(foto.crop((0, 0, lado, lado)))
    fondo.convert("RGB").resize((400, 400), Image.LANCZOS).save(destino, quality=88)


def _segundos(marca: str) -> int:
    partes = [int(x) for x in marca.split(":")]
    return sum(x * 60 ** i for i, x in enumerate(reversed(partes)))


def cortar_muestra(curso: Path, salida: Path) -> dict:
    """Corta del curso gratuito el capítulo de muestra, con su imagen de portada y sus subtítulos."""
    capitulos = [linea.split(" ", 1) for linea in Path(str(curso.with_suffix("")) + "_capitulos.txt").read_text(encoding="utf-8").strip().splitlines()]
    i = next(k for k, (_, nombre) in enumerate(capitulos) if nombre.startswith(CAPITULO_DE_MUESTRA))
    a, b = _segundos(capitulos[i][0]), _segundos(capitulos[i + 1][0])
    ff = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error"]
    subprocess.run([*ff, "-ss", str(a), "-t", str(b - a), "-i", str(curso), "-vf", "scale=1280:720:flags=lanczos", "-c:v", "libx264", "-preset", "slow",
                    "-crf", "27", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-movflags", "+faststart", str(salida / "muestra.mp4")], check=True)
    subprocess.run([*ff, "-ss", str(a + 40), "-i", str(curso), "-frames:v", "1", "-vf", "scale=1280:720:flags=lanczos", "-q:v", "4", str(salida / "muestra.jpg")], check=True)
    # los subtítulos del tramo, con el reloj corrido al comienzo de la lección
    marca = lambda s: f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:06.3f}"  # noqa: E731
    vtt, subtitulos = ["WEBVTT", ""], []
    for bloque in curso.with_suffix(".srt").read_text(encoding="utf-8").strip().split("\n\n"):
        lineas = bloque.split("\n")
        t0, t1 = (sum(float(x) * 60 ** k for k, x in enumerate(reversed(re.split(":", t.replace(",", "."))))) for t in lineas[1].split(" --> "))
        if a <= t0 and t1 <= b:
            vtt += [f"{marca(t0 - a)} --> {marca(t1 - a)}", *lineas[2:], ""]
            subtitulos.append([round(t0 - a, 3), round(t1 - a, 3), "\n".join(lineas[2:])])
    (salida / "muestra.vtt").write_text("\n".join(vtt), encoding="utf-8")
    return dict(MUESTRA, video="muestra.mp4", poster="muestra.jpg", vtt="muestra.vtt", subtitulos=subtitulos, min=round((b - a) / 60))


def construir(salida: Path, curso: Path | None = None) -> dict:
    salida.mkdir(parents=True, exist_ok=True)
    retrato(salida / "autor.jpg")
    muestra = cortar_muestra(curso, salida) if curso else None
    # el sitio publicado lee los subtítulos de su archivo .vtt; la vista previa, que no sirve ese tipo de archivo, de los datos
    sin = lambda m, clave: {k: v for k, v in m.items() if k != clave} if m else None  # noqa: E731
    d = datos(sin(muestra, "subtitulos"))
    (salida / "index.html").write_text(pagina(d, completa=True), encoding="utf-8")
    (salida / "vista_previa.html").write_text(pagina(datos(sin(muestra, "vtt")), completa=False), encoding="utf-8")
    return d


if __name__ == "__main__":
    args = sys.argv[1:]
    curso = Path(args[args.index("--curso") + 1]) if "--curso" in args else None
    destino = [a for i, a in enumerate(args) if a != "--curso" and (i == 0 or args[i - 1] != "--curso")]
    if len(destino) != 1:
        sys.exit(__doc__)
    r = construir(Path(destino[0]), curso)
    t = r["totales"]
    print(f'Sitio en {destino[0]}: {t["modulos"]} módulos, {t["lecciones"]} lecciones, {len(r["evaluaciones"]["bancos"]) + 1} evaluaciones'
          + (f', lección de muestra de {r["muestra"]["min"]} min' if r["muestra"] else ", sin lección de muestra"))
