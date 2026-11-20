"""Arma un video corto de muestra del curso, para publicarlo en redes sociales.

Uso:
    python -m curso_youtube.muestra <curso.mp4> <trabajo> <cierre.mp3> <salida.mp4>

`<curso.mp4>` es el video final (con su `.srt` al lado). `<trabajo>` es la carpeta donde se capturaron sus escenas
(`construir --continuo`): de su plan salen los cuadros exactos de cada lámina. `<cierre.mp3>` es la frase hablada que
invita a ver el curso completo (el texto de `CIERRE["texto"]`, generado con la misma voz).

La muestra es vertical (1080 x 1350, la proporción que más espacio ocupa en el feed): arriba el título, al centro el
video del curso y abajo los subtítulos en grande, porque en las redes casi todos miran sin sonido. Usa fragmentos
reales del curso, cortados en cambios de paso (donde la voz hace una pausa); parte con la miniatura, que es el
cuadro que se ve antes de reproducir, y termina con una lámina que dice dónde está el curso completo.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

from . import deck, ensamblar, youtube
from .laminas import LINKEDIN, LINKEDIN_URL, MARCA, esc, pagina

AQUI = Path(__file__).parent
FPS = ensamblar.FPS
ANCHO, ALTO = 1080, 1350
VIDEO_Y, VIDEO_ALTO = 318, 608            # el video del curso, a todo el ancho (16:9)
ROTULO_Y, ROTULO_ALTO = 946, 284          # la franja de los subtítulos
PORTADA = 1.0                             # segundos con la miniatura al comienzo
FUNDE = 10                                # cuadros de fundido al dejar una lámina a medio camino
ESPERA_CIERRE, COLA_CIERRE = 0.6, 1.5     # silencio antes y después de la frase de cierre

# Fragmentos del curso, en orden. `pasos` corta la última lámina antes de ese paso; sin él va completa.
FRAGMENTOS = [
    dict(desde="m00_s01", hasta="m00_s02"),              # el gancho y la promesa
    dict(desde="m00_s06", hasta="m00_s07", pasos=3),     # el método de 8 pasos y el caso
    dict(desde="m04_s07", hasta="m04_s07", pasos=2),     # un prompt…
    dict(desde="m04_s08", hasta="m04_s08", pasos=1),     # …y lo que responde Claude
]
CIERRE = dict(sobre="Curso completo · gratis en YouTube",
              titulo_html="Análisis de estados financieros con <em>Claude</em>",
              sub="El enlace está en esta publicación",
              texto="¿Qué pasó con la caja? Mira el curso completo, gratis, en YouTube. El enlace está en esta publicación.")

_FUENTES = """
@font-face{font-family:"Montserrat";src:url("__F__/montserrat.woff2") format("woff2");font-weight:100 900}
@font-face{font-family:"Source Sans 3";src:url("__F__/source-sans-3.woff2") format("woff2");font-weight:200 900}
@font-face{font-family:"Newsreader";font-style:italic;src:url("__F__/newsreader-italic.woff2") format("woff2");font-weight:200 800}
@font-face{font-family:"Geist Mono";src:url("__F__/geist-mono.woff2") format("woff2");font-weight:100 900}
*{box-sizing:border-box;margin:0;padding:0}
"""
_MARCO = _FUENTES + """
body{width:1080px;height:1350px;overflow:hidden;position:relative;color:#F2EADB;
 background:radial-gradient(900px 700px at 108% 108%,rgba(63,98,148,.42),transparent 62%),radial-gradient(800px 560px at -10% -12%,rgba(46,72,110,.5),transparent 64%),#101C2C}
.k{position:absolute;left:64px;top:58px;display:flex;align-items:center;gap:18px;font-family:"Geist Mono",monospace;font-size:25px;font-weight:500;letter-spacing:.16em;text-transform:uppercase;color:#E3A62F}
.k::before{content:"";width:50px;height:3px;background:#E3A62F}
h1{position:absolute;left:64px;right:64px;top:106px;font-family:"Montserrat",sans-serif;font-size:66px;line-height:1.07;font-weight:700;letter-spacing:-.034em}
h1 em{font-family:"Newsreader",serif;font-style:italic;font-weight:500;letter-spacing:-.02em;color:#E3A62F}
.hueco{position:absolute;left:0;right:0;top:__VY__px;height:__VA__px;background:#0B1420;box-shadow:0 -3px 0 #E3A62F,0 30px 60px -30px rgba(0,0,0,.8)}
.quien{position:absolute;left:64px;right:64px;bottom:44px;display:flex;align-items:center;gap:16px;font-family:"Montserrat",sans-serif;font-size:29px;font-weight:600;letter-spacing:-.01em}
.quien svg{width:42px;height:42px;flex:none}
.quien span{margin-left:auto;font-family:"Geist Mono",monospace;font-size:21px;font-weight:500;letter-spacing:.04em;color:#B4BEC6}
"""
_ROTULO = _FUENTES + """
html,body{background:transparent}
body{width:1080px;height:__RA__px;display:flex;align-items:center;justify-content:center;padding:0 60px}
p{font-family:"Montserrat",sans-serif;font-size:49px;line-height:1.22;font-weight:600;letter-spacing:-.018em;color:#F2EADB;text-align:center;text-wrap:balance}
"""


def _node(script: str) -> None:
    subprocess.run(["node", "-e", script], check=True, cwd=AQUI)


def _marco(destino: Path) -> None:
    """El cuadro fijo de la muestra: título arriba, hueco para el video y firma abajo."""
    css = _MARCO.replace("__F__", (AQUI / "fuentes").resolve().as_uri()).replace("__VY__", str(VIDEO_Y)).replace("__VA__", str(VIDEO_ALTO))
    html = destino.with_suffix(".html")
    html.write_text(f'<!doctype html><html lang="es"><head><meta charset="utf-8"><style>{css}</style></head><body>'
                    f'<div class="k">{esc(CIERRE["sobre"])}</div><h1>{CIERRE["titulo_html"]}</h1><div class="hueco"></div>'
                    f'<div class="quien">{LINKEDIN}{esc(MARCA)}<span>{esc(LINKEDIN_URL.split("://www.")[-1])}</span></div>'
                    '</body></html>', encoding="utf-8")
    _node("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();"
          f"const c=await b.newContext({{viewport:{{width:{ANCHO},height:{ALTO}}},deviceScaleFactor:1}});const p=await c.newPage();"
          f"await p.goto('file://{html.resolve()}');await p.evaluate(()=>document.fonts.ready);"
          f"await p.screenshot({{path:'{destino.resolve()}'}});await b.close();}})();")


def _rotulos(textos: list[str], carpeta: Path) -> list[Path]:
    """Una imagen transparente por subtítulo (y una vacía, la número 0), con la tipografía del curso."""
    carpeta.mkdir(parents=True, exist_ok=True)
    css = _ROTULO.replace("__F__", (AQUI / "fuentes").resolve().as_uri()).replace("__RA__", str(ROTULO_ALTO))
    html = carpeta / "rotulo.html"
    html.write_text(f'<!doctype html><html lang="es"><head><meta charset="utf-8"><style>{css}</style></head><body><p></p></body></html>',
                    encoding="utf-8")
    (carpeta / "textos.json").write_text(json.dumps([""] + textos, ensure_ascii=False), encoding="utf-8")
    _node("const {chromium}=require('playwright');const fs=require('fs');(async()=>{const b=await chromium.launch();"
          f"const c=await b.newContext({{viewport:{{width:{ANCHO},height:{ROTULO_ALTO}}},deviceScaleFactor:1}});const p=await c.newPage();"
          f"await p.goto('file://{html.resolve()}');await p.evaluate(()=>document.fonts.ready);"
          f"const t=JSON.parse(fs.readFileSync('{(carpeta / 'textos.json').resolve()}','utf8'));"
          "for(let i=0;i<t.length;i++){const alto=await p.evaluate((x)=>{const e=document.querySelector('p');e.textContent=x;"
          "return e.getBoundingClientRect().height;},t[i]);"
          f"if(alto>{ROTULO_ALTO}-8)throw new Error('El subtítulo no cabe: '+t[i]);"
          f"await p.screenshot({{path:'{carpeta.resolve()}/r'+String(i).padStart(3,'0')+'.png',omitBackground:true}});}}"
          "await b.close();})();")
    return [carpeta / f"r{i:03d}.png" for i in range(len(textos) + 1)]


def _lamina_de_cierre(carpeta: Path, cuadros: int) -> Path:
    """La lámina final, con el mismo motor de movimiento del curso."""
    html = carpeta / "html"
    html.mkdir(parents=True, exist_ok=True)
    s = dict(id="cierre", tipo="titulo_curso", sobre=CIERRE["sobre"], titulo_html=CIERRE["titulo_html"], sub=CIERRE["sub"],
             quien=MARCA, pasos=[CIERRE["texto"]])
    (html / "cierre.html").write_text(pagina(s, dict(kicker="", modulo=0, n_modulos=1, avance=1),
                                             css_href=(AQUI / "estilo.css").resolve().as_uri()), encoding="utf-8")
    dur = cuadros / FPS
    plan = dict(fps=FPS, total=cuadros, codificacion=ensamblar.CODIFICACION_CONTINUA,
                escenas=[dict(id="cierre", tipo="titulo_curso", modulo=0, cuadro0=0, cuadros=cuadros, inicio=0.0,
                              duraciones=[round(dur, 4)], entra="", sale="", modIni=0.0, modFin=round(dur, 4))])
    (carpeta / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    subprocess.run(["node", str(AQUI / "capturar.cjs"), str(html), str(carpeta / "escena"), "--continuo", str(carpeta / "plan.json")],
                   check=True)
    return carpeta / "escena" / "cierre.mp4"


def tramo(escenas: dict[str, dict], f: dict) -> tuple[int, int, bool]:
    """(primer cuadro, cuadro siguiente al último, ¿termina a medio camino de una lámina?) de un fragmento."""
    a, ultima = escenas[f["desde"]]["cuadro0"], escenas[f["hasta"]]
    if f.get("pasos"):
        return a, ultima["cuadro0"] + math.ceil(sum(ultima["duraciones"][:f["pasos"]]) * FPS - 1e-6), True
    return a, ultima["cuadro0"] + ultima["cuadros"], False


def leer_srt(ruta: Path) -> list[tuple[float, float, str]]:
    def seg(t: str) -> float:
        h, m, s = t.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    out = []
    for bloque in ruta.read_text(encoding="utf-8").strip().split("\n\n"):
        lineas = bloque.split("\n")
        a, b = (seg(x) for x in lineas[1].split(" --> "))
        out.append((a, b, " ".join(lineas[2:])))
    return out


def rotulos_de(tramos: list[tuple[int, int, bool]], subtitulos: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Los subtítulos del curso que caen en los fragmentos, con el tiempo que les toca en la muestra."""
    out, cursor = [], PORTADA
    for a, b, _ in tramos:
        ta, tb = a / FPS, b / FPS
        for t0, t1, texto in subtitulos:
            ini, fin = max(t0, ta), min(t1, tb)
            if fin - ini >= 0.3:
                out.append((round(ini - ta + cursor, 3), round(fin - ta + cursor, 3), texto))
        cursor += tb - ta
    return out


def armar(curso: Path, trabajo: Path, cierre_audio: Path, salida: Path) -> dict:
    obra = salida.parent / (salida.stem + "_obra")
    obra.mkdir(parents=True, exist_ok=True)
    plan = json.loads((trabajo / "escenas" / "plan.json").read_text())
    escenas = {e["id"]: e for e in plan["escenas"]}
    laminas = {s["id"]: s for s in deck.cargar()}
    assert [laminas[f["desde"]]["tipo"] for f in FRAGMENTOS] == ["gancho", "metodo", "prompt", "respuesta"], \
        "las láminas de la muestra cambiaron de lugar: revisa FRAGMENTOS"
    tramos = [tramo(escenas, f) for f in FRAGMENTOS]

    # la frase de cierre y su lámina
    voz = obra / "cierre.wav"
    dur_voz = ensamblar.preparar_audio(cierre_audio, voz)
    n_cierre = math.ceil((ESPERA_CIERRE + dur_voz + COLA_CIERRE) * FPS)
    lamina_cierre = _lamina_de_cierre(obra, n_cierre)

    # subtítulos: los del curso más los de la frase de cierre
    rotulos = rotulos_de(tramos, leer_srt(curso.with_suffix(".srt")))
    t_cierre = PORTADA + sum(b - a for a, b, _ in tramos) / FPS + ESPERA_CIERRE
    trozos = ensamblar._trozos(CIERRE["texto"])
    inicios = ensamblar.tiempos_de_subtitulos(0.0, dur_voz, trozos, ensamblar.silencios(voz))
    for i, trozo in enumerate(trozos):
        fin = inicios[i + 1] if i + 1 < len(trozos) else dur_voz + 0.8
        rotulos.append((round(t_cierre + inicios[i], 3), round(t_cierre + fin, 3), trozo))
    total = PORTADA + (sum(b - a for a, b, _ in tramos) + n_cierre) / FPS
    imagenes = _rotulos([r[2] for r in rotulos], obra / "rotulos")
    lineas, cursor = ["ffconcat version 1.0"], 0.0
    for i, (t0, t1, _) in enumerate(rotulos, 1):
        if t0 > cursor + 1e-3:
            lineas += [f"file '{imagenes[0]}'", f"option framerate {FPS}", f"duration {t0 - cursor:.6f}"]
        lineas += [f"file '{imagenes[i]}'", f"option framerate {FPS}", f"duration {t1 - max(t0, cursor):.6f}"]
        cursor = t1
    lineas += [f"file '{imagenes[0]}'", f"option framerate {FPS}", f"duration {max(0.1, total - cursor):.6f}",
               f"file '{imagenes[0]}'", f"option framerate {FPS}"]
    (obra / "rotulos.ffconcat").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # el cuadro fijo y la portada (la miniatura del curso, a 1920 x 1080)
    _marco(obra / "marco.png")
    youtube.miniatura(obra / "portada.png", escala=1.5)

    # entradas de ffmpeg: portada, un corte del curso por fragmento, la lámina de cierre, el audio del curso y la voz
    ext = FUNDE / FPS
    entradas = ["-loop", "1", "-framerate", str(FPS), "-t", f"{PORTADA + ext:.4f}", "-i", str(obra / "portada.png")]
    for a, b, medias in tramos:
        entradas += ["-ss", f"{(a - 0.5) / FPS:.6f}", "-t", f"{(b - a + (FUNDE if medias else 0) + 2) / FPS:.6f}", "-i", str(curso)]
    n = len(tramos)
    entradas += ["-i", str(lamina_cierre), "-vn", "-i", str(curso), "-i", str(voz),
                 "-loop", "1", "-framerate", str(FPS), "-i", str(obra / "marco.png"),
                 "-f", "concat", "-safe", "0", "-i", str(obra / "rotulos.ffconcat")]
    i_cierre, i_audio, i_voz, i_marco, i_rot = n + 1, n + 2, n + 3, n + 4, n + 5

    comun = f"fps={FPS},format=yuv420p,settb=AVTB"
    f = [f"[0:v]scale=1920:1080:flags=lanczos:out_color_matrix=bt709:out_range=tv,{comun},setsar=1[v0]"]
    for k, (a, b, medias) in enumerate(tramos, 1):
        f.append(f"[{k}:v]setpts=PTS-STARTPTS,{comun},trim=end_frame={b - a + (FUNDE if medias else 0)},setpts=PTS-STARTPTS,setsar=1[v{k}]")
    f.append(f"[{i_cierre}:v]{comun},setpts=PTS-STARTPTS,setsar=1[v{n + 1}]")
    # se encadenan: fundido cuando lo anterior trae cuadros de sobra (portada y láminas cortadas a medio camino)
    actual, largo, funde_previo = "v0", PORTADA, True
    for k in range(1, n + 2):
        if funde_previo:
            f.append(f"[{actual}][v{k}]xfade=transition=fade:duration={ext:.4f}:offset={largo:.4f}[x{k}]")
        else:
            f.append(f"[{actual}][v{k}]concat=n=2:v=1:a=0,{comun}[x{k}]")
        actual = f"x{k}"
        if k <= n:
            a, b, funde_previo = tramos[k - 1]
            largo += (b - a) / FPS
    # el audio: silencio bajo la portada, cada fragmento con una entrada y salida de 30 ms, y la frase de cierre
    piezas = [f"anullsrc=r=48000:cl=mono,atrim=duration={PORTADA:.4f}[a0]"]
    for k, (a, b, _) in enumerate(tramos, 1):
        d = (b - a) / FPS
        piezas.append(f"[{i_audio}:a]atrim=start={a / FPS:.6f}:end={b / FPS:.6f},asetpts=PTS-STARTPTS,"
                      f"afade=t=in:d=0.03,afade=t=out:st={d - 0.03:.4f}:d=0.03[a{k}]")
    piezas.append(f"[{i_voz}:a]adelay={int(ESPERA_CIERRE * 1000)},apad,atrim=duration={n_cierre / FPS:.6f}[a{n + 1}]")
    f += piezas
    f.append("".join(f"[a{k}]" for k in range(n + 2)) + f"concat=n={n + 2}:v=0:a=1,aresample=48000[a]")
    # composición: el programa, reducido, sobre el cuadro fijo; encima, los subtítulos
    f.append(f"[{actual}]scale={ANCHO}:{VIDEO_ALTO}:flags=lanczos[prog]")
    f.append(f"[{i_marco}:v]scale={ANCHO}:{ALTO}:out_color_matrix=bt709:out_range=tv,{comun}[marco]")
    f.append(f"[{i_rot}:v]scale={ANCHO}:{ROTULO_ALTO}:out_color_matrix=bt709:out_range=tv,fps={FPS},format=yuva420p,settb=AVTB[rot]")
    f.append(f"[marco][prog]overlay=0:{VIDEO_Y}:shortest=1[base]")
    f.append(f"[base][rot]overlay=0:{ROTULO_Y}:eof_action=pass,format=yuv420p[v]")
    cuadros = round(total * FPS)
    ensamblar._ffmpeg(*entradas, "-filter_complex", ";".join(f), "-map", "[v]", "-map", "[a]", "-frames:v", str(cuadros),
                      "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", *ensamblar._COLOR,
                      "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-shortest", "-movflags", "+faststart", str(salida))
    info = dict(segundos=round(total, 2), cuadros=cuadros, rotulos=rotulos,
                fragmentos=[dict(desde=x["desde"], hasta=x["hasta"], cuadros=[a, b]) for x, (a, b, _) in zip(FRAGMENTOS, tramos)])
    (salida.parent / (salida.stem + "_tiempos.json")).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    r = armar(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    print(f"Muestra: {sys.argv[4]} · {int(r['segundos'] // 60)}:{r['segundos'] % 60:04.1f} · {len(r['rotulos'])} subtítulos")
