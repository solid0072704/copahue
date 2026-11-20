"""Arma el video vertical (9:16) que promociona el curso en TikTok. Sirve igual para Reels y Shorts.

Uso:
    python -m curso_youtube.tiktok --texto                      # el texto que se envía a la voz
    python -m curso_youtube.tiktok <voz.mp3> <trabajo> <salida.mp4> [marcas.json] [--guias]

No es un recorte del curso: es una pieza propia, de medio minuto, pensada para verse en el teléfono. Parte con el
caso (vender más, ganar más y quedarse sin caja), da la idea que lo explica, muestra qué trae el curso y termina
diciendo cómo encontrarlo en YouTube, porque en TikTok los enlaces del texto no se pueden abrir.

`GUION` es la única fuente: de ahí salen el texto de la voz, los subtítulos y los momentos en que entra cada cosa.
`<voz.mp3>` es ese texto narrado con la voz del curso (la generación usada está anotada en `tiktok.json`). Los
tiempos de cada frase salen de `marcas.json` ({"clave": segundos, …}, medidos sobre el audio); sin él se estiman
con las pausas de la voz. La página (tiktok.css + tiktok.js) se captura cuadro a cuadro con `capturar.cjs`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

from . import caso, deck, ensamblar, youtube
from .laminas import ICONOS, MARCA, _BOCETOS, esc, pagina
from .voz import a_voz

AQUI = Path(__file__).parent
FPS = ensamblar.FPS
ANCHO, ALTO = 1080, 1920
# Lo que la interfaz de TikTok tapa sobre el video: pestañas arriba, texto y música abajo, botones a la derecha.
# Fuera de estas franjas queda la zona segura, donde va todo lo que hay que leer (tiktok.css la respeta). TikTok
# no publica medidas y las guías que circulan no coinciden: estas son de las más prudentes.
ZONA = dict(arriba=210, abajo=440, derecha=150, botones_desde=800)
ENTRADA, COLA = 0.2, 1.7                              # segundos antes de la primera palabra y después de la última
PAUSAS = dict(pregunta=0.25, curso=0.2, busca=0.45)   # silencio que se agrega antes de esas frases (cambios de escena)
VOLUMEN = -14.0                                       # LUFS: en TikTok se escucha con el teléfono en la mano
VENTANA = (858, 597)                                  # la ventana donde se ven las láminas del curso (tiktok.css)
TITULO = "Análisis de estados financieros con Claude"

# Cada frase es un subtítulo y un momento del video: la clave nombra lo que entra en pantalla cuando empieza a
# sonar. Una frase que parte con «¶» abre párrafo (la voz hace una pausa). Entre asteriscos, lo que se destaca.
GUION = [
    ("ventas", "¶Una empresa vende *18 % más*"),
    ("utilidad", "y logra la *mayor utilidad* de su historia."),
    ("caja", "Y en diciembre, casi no puede *pagar los sueldos*."),
    ("pregunta", "¶¿Cómo es posible?"),
    ("idea", "Porque *utilidad* no es lo mismo que *caja*."),
    ("estados", "Y eso está en los estados financieros,"),
    ("mirar", "si sabes *dónde mirar*."),
    ("curso", "¶En mi *curso gratuito* de una hora"),
    ("ia", "lo analizo con inteligencia artificial,"),
    ("claude", "paso a paso, usando *Claude*."),
    ("metodo", "Te llevas un método de *8 pasos*,"),
    ("prompts", "los *prompts exactos* que uso"),
    ("excel", "y el archivo del caso en *Excel*,"),
    ("empresa", "para repetirlo con tu empresa."),
    ("busca", "¶Búscalo en *YouTube*:"),
    ("titulo", "Análisis de estados financieros con Claude."),
]
SIN_ROTULO = {"pregunta", "busca", "titulo"}        # lo que dice la voz ya está escrito en grande
# Las láminas del curso que se ven en la ventana, con otro ritmo: un prompt real y lo que responde Claude.
# `pesos` reparte el tiempo entre los pasos de la lámina; `solo` deja solo los primeros (del prompt se ve el primero:
# se escribe y se envía) y `velocidad` la acelera: el motor del curso solo escribe el prompt, en vez de pegarlo, si
# el paso dura lo suficiente, así que se captura más largo y se muestra más rápido. `desde` se salta el comienzo
# de la lámina: la respuesta llega con su tarjeta ya puesta, para reemplazar al prompt sin pasar por el fondo.
CINE = [
    dict(id="cine-a", lamina="m04_s07", tipo="prompt", selector=".prompt", pesos=[1, 1, 1, 1], solo=1, velocidad=1.7),
    dict(id="cine-b", lamina="m04_s08", tipo="respuesta", selector=".resp", pesos=[2.0, 1.4, 2.0, 1.6, 1.5], desde=0.5),
]
CLAVES = [k for k, _ in GUION]
# El texto de la publicación. En TikTok los enlaces del texto no se pueden abrir: por eso dice cómo buscar el curso.
PUBLICACION = [
    "¿Se puede vender más, ganar más… y quedarse sin caja? Sí: utilidad no es lo mismo que caja.",
    "",
    "En mi curso gratuito de 1 hora analizo este caso paso a paso con inteligencia artificial (Claude): un método de "
    "8 pasos, los prompts exactos que uso y el archivo del caso en Excel.",
    "",
    f"🔎 Búscalo en YouTube: {TITULO} (canal {MARCA}).",
    "",
    "El caso es de una empresa ficticia, creada para el curso.",
    "",
    "#finanzas #contabilidad #estadosfinancieros #flujodecaja #inteligenciaartificial #claude #excel #pymes",
]


def _limpio(frase: str) -> str:
    return frase.replace("¶", "").replace("*", "")


def texto() -> str:
    """El guion como se lee: un párrafo por línea."""
    parrafos: list[list[str]] = []
    for _, frase in GUION:
        if frase.startswith("¶"):
            parrafos.append([])
        parrafos[-1].append(_limpio(frase))
    return "\n\n".join(" ".join(p) for p in parrafos)


def texto_para_voz() -> str:
    """El guion como se envía a la voz: cifras en palabras y nombres escritos como suenan."""
    return "\n\n".join(a_voz(p) for p in texto().split("\n\n"))


def cifras() -> dict:
    """Las cifras del caso que el video muestra, tomadas de los estados financieros del curso."""
    e24, e25 = caso.eerr("2024"), caso.eerr("2025")
    b24, b25 = caso.balance("2024"), caso.balance("2025")
    return dict(ventas_pct=round((e25["ventas"] / e24["ventas"] - 1) * 100), ventas=e25["ventas"], utilidad=e25["utilidad"],
                utilidad_antes=e24["utilidad"], caja=b25["efectivo"], caja_antes=b24["efectivo"])


# --- Tiempos ---------------------------------------------------------------
def marcas_estimadas(duracion: float, pausas: list[tuple[float, float]]) -> dict[str, float]:
    """Segundo en que empieza cada frase, según el largo del texto y las pausas reales de la voz."""
    inicios = ensamblar.tiempos_de_subtitulos(0.0, duracion, [_limpio(f) for _, f in GUION], pausas)
    return {k: round(t, 3) for k, t in zip(CLAVES, inicios)} | {"fin": round(duracion, 3)}


def con_pausas(marcas: dict[str, float], pausas: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], dict[str, float]]:
    """Dónde se agrega silencio a la voz y cómo quedan las marcas después de agregarlo.

    Cada pausa de `PAUSAS` se inserta en medio del silencio real que precede a su frase (o justo antes de ella, si la
    voz no se detuvo). Devuelve los cortes [(segundo de la voz, segundos de silencio)] y las marcas ya corridas, con
    la entrada incluida.
    """
    cortes = []
    for clave, extra in PAUSAS.items():
        t = marcas[clave]
        previas = [(a, b) for a, b in pausas if abs(b - t) < 0.3 and a < t]
        cortes.append((round((previas[-1][0] + previas[-1][1]) / 2 if previas else max(0.0, t - 0.03), 4), extra))
    cortes.sort()
    corridas = {k: round(ENTRADA + t + sum(e for p, e in cortes if p <= t + 1e-6), 3) for k, t in marcas.items()}
    return cortes, corridas


def _audio(voz: Path, cortes: list[tuple[float, float]], destino: Path) -> float:
    """Escribe la voz con la entrada, las pausas agregadas y la cola. Devuelve la duración."""
    with wave.open(str(voz)) as w:
        tasa, datos = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    silencio = lambda s: np.zeros(int(round(s * tasa)), dtype=np.int16)  # noqa: E731
    piezas, previo = [silencio(ENTRADA)], 0
    for punto, extra in cortes:
        i = int(round(punto * tasa))
        piezas += [datos[previo:i], silencio(extra)]
        previo = i
    todo = np.concatenate(piezas + [datos[previo:], silencio(COLA)])
    with wave.open(str(destino), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(tasa)
        w.writeframes(todo.tobytes())
    return len(todo) / tasa


def rotulos(m: dict[str, float]) -> list[dict]:
    """Los subtítulos: cada frase desde que empieza a sonar hasta que empieza la siguiente."""
    out = []
    for i, (clave, frase) in enumerate(GUION):
        if clave in SIN_ROTULO:
            continue
        sigue = CLAVES[i + 1] if i + 1 < len(GUION) else "fin"
        fin = m[sigue] - (0.3 if sigue in SIN_ROTULO else 0.0)
        out.append(dict(t0=m[clave], t1=round(fin, 3), html=re.sub(r"\*(.+?)\*", r"<b>\1</b>", esc(frase.replace("¶", "")))))
    return out


def tiempos_del_cine(m: dict[str, float]) -> list[tuple[float, float]]:
    """(inicio, fin) de cada lámina del curso en la ventana: el prompt mientras se presenta el curso; la respuesta
    desde que la voz nombra a Claude hasta que cambia la escena."""
    return [(round(m["curso"] + 0.35, 3), round(m["claude"] + 0.25, 3)), (round(m["claude"] - 0.2, 3), round(m["busca"], 3))]


# --- La página -------------------------------------------------------------
def _ico(nombre: str) -> str:
    return f'<svg viewBox="0 0 24 24">{ICONOS[nombre]}</svg>'


def pagina_html(datos: dict) -> str:
    """La página con las cuatro escenas; `datos` es lo que lee tiktok.js (marcas, subtítulos, cine…)."""
    c = datos["cifras"]
    k = lambda t: f'<div class="k"><i></i><span>{esc(t)}</span></div>'  # noqa: E731
    tile = lambda ident, tono, eti, val, sub: (  # noqa: E731
        f'<div class="tile {tono}" id="{ident}"><i class="fl"></i><div class="eti">{esc(eti)}</div>'
        f'<div class="val">{val}</div><div class="sub">{esc(sub)}</div></div>')
    docs = "".join(f'<div class="doc" id="d-{b}"><div class="dt">{esc(t)}</div>{_BOCETOS[b]}</div>'
                   for t, b in (("Balance", "balance"), ("Estado de resultados", "resultados"), ("Flujo de efectivo", "flujo")))
    chips = "".join(f'<div class="chip" id="c-{ident}">{_ico(icono)}{esc(t)}</div>'
                    for ident, icono, t in (("metodo", "capas", "8 pasos"), ("prompts", "mensaje", "10 prompts"), ("excel", "tabla", "Caso en Excel")))
    cine = "".join(f'<img id="{x["id"]}" src="{x["dir"]}/0001.jpg" alt="">' for x in datos["cine"])
    guias = (f'<div class="guia" style="left:0;top:0;width:{ANCHO}px;height:{ZONA["arriba"]}px"></div>'
             f'<div class="guia" style="left:0;top:{ALTO - ZONA["abajo"]}px;width:{ANCHO}px;height:{ZONA["abajo"]}px"></div>'
             f'<div class="guia" style="left:{ANCHO - ZONA["derecha"]}px;top:{ZONA["botones_desde"]}px;width:{ZONA["derecha"]}px;'
             f'height:{ALTO - ZONA["abajo"] - ZONA["botones_desde"]}px"></div>')
    cuerpo = (
        '<canvas id="fondo" width="1080" height="1920"></canvas>'
        # 1 · el caso
        f'<section class="esc" id="caso">{k("Caso práctico del curso")}'
        '<h1>Vende más.<br>Gana más.<br>Y se queda <em>sin caja</em>.</h1><div class="tiles">'
        + tile("t-ventas", "pos", "Ventas 2025", '<span class="n">+0 %</span>', f'{caso.miles(c["ventas"])} millones de pesos')
        + tile("t-utilidad", "pos", "Utilidad neta", '<span class="n">0</span><small>millones</small>', "La más alta de su historia")
        + tile("t-caja", "neg", "Caja en diciembre",
               f'<span class="n">{c["caja_antes"]}</span><small>millones</small><div class="barra"><b></b></div>',
               f'Eran {c["caja_antes"]} a comienzos de año')
        + '</div></section>'
        # 2 · la idea
        f'<section class="esc" id="idea">{k("La clave")}<div class="preg" id="preg">¿Cómo es posible?</div>'
        f'<div class="lock"><div class="pal lu"><span>Utilidad</span><small>{c["utilidad"]} millones</small></div>'
        f'<div class="ne">≠</div><div class="pal lc"><span>Caja</span><small>{c["caja"]} millones</small></div></div>'
        f'<div class="docs">{docs}</div></section>'
        # 3 · el curso
        f'<section class="esc" id="curso">{k("Curso gratis · 1 hora")}<h2>Análisis de estados financieros con <em>Claude</em></h2>'
        f'<div class="vent" id="vent">{cine}</div><div class="chips">{chips}</div></section>'
        # 4 · dónde está
        f'<section class="esc" id="cta">{k("Gratis en YouTube")}<h2>Búscalo en <em>YouTube</em></h2>'
        f'<div class="busca" id="busca">{_ico("lupa")}<span id="q"></span><i class="cur"></i></div>'
        '<div class="video" id="video"><img src="miniatura.png" alt=""><div class="dur">1:04:37</div></div>'
        f'<div class="meta" id="meta"><b>{esc(MARCA)}</b><span>Curso completo · 1 hora · gratis</span></div></section>'
        '<div class="rot" id="rot"><p></p></div>' + guias)
    return ('<!doctype html><html lang="es"><head><meta charset="utf-8"><title>tiktok</title>'
            f'<link rel="stylesheet" href="{(AQUI / "tiktok.css").resolve().as_uri()}"></head><body>{cuerpo}'
            f'<script>window.DATOS = {json.dumps(datos, ensure_ascii=False)};</script>'
            f'<script src="{(AQUI / "tiktok.js").resolve().as_uri()}"></script></body></html>')


def _cine(carpeta: Path, x: dict, segundos: float) -> int:
    """Captura una lámina del curso, con sus pasos repartidos en `segundos`, recortada a la tarjeta. Devuelve los cuadros."""
    s = {t["id"]: t for t in deck.cargar()}[x["lamina"]]
    assert s["tipo"] == x["tipo"], f'la lámina {x["lamina"]} cambió de lugar: revisa CINE'
    assert len(s["pasos"]) == len(x["pesos"]), f'la lámina {x["lamina"]} cambió de pasos: revisa CINE'
    laminas = carpeta / "laminas"
    laminas.mkdir(parents=True, exist_ok=True)
    html = laminas / f'{s["id"]}.html'
    html.write_text(pagina(s, dict(kicker=s["kicker"], modulo=s["modulo"], n_modulos=s["n_modulos"], avance=s["avance"]),
                           css_href=(AQUI / "estilo.css").resolve().as_uri()), encoding="utf-8")
    velocidad, solo, desde = x.get("velocidad", 1), x.get("solo", len(x["pesos"])), x.get("desde", 0)
    duraciones = ([round((desde + segundos * velocidad) * p / sum(x["pesos"][:solo]), 4) for p in x["pesos"][:solo]]
                  + [1.0] * (len(x["pesos"]) - solo))
    info = dict(id=s["id"], tipo=s["tipo"], modulo=s["modulo"], cuadro0=0, cuadros=round(segundos * FPS), inicio=0.0, duraciones=duraciones,
                entra="", sale="", modIni=0.0, modFin=round(sum(duraciones), 4), fps=FPS, velocidad=velocidad, desde=desde, selector=x["selector"],
                margen=14, proporcion=VENTANA[0] / VENTANA[1])
    (laminas / f'{s["id"]}.json').write_text(json.dumps(info), encoding="utf-8")
    r = subprocess.run(["node", str(AQUI / "tiktok_cine.cjs"), str(html), str(carpeta / "html" / x["dir"]), str(laminas / f'{s["id"]}.json')],
                       check=True, capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])["n"]


def revisar(html: Path, instantes: list[float]) -> list[str]:
    """Lo que queda fuera de la zona segura en esos instantes del video (lo dice `revisar()`, de tiktok.js)."""
    script = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();"
              f"const c=await b.newContext({{viewport:{{width:{ANCHO},height:{ALTO}}},deviceScaleFactor:1}});const p=await c.newPage();"
              f"await p.goto('file://{html.resolve()}');await p.evaluate(()=>document.fonts.ready);await p.evaluate(()=>iniciarLamina());"
              f"const fuera=[];for(const t of {json.dumps(instantes)}){{await p.evaluate((s)=>cuadro(s),t);"
              "for(const x of await p.evaluate(()=>revisar()))fuera.push(t.toFixed(2)+' s · '+x);}"
              "console.log(JSON.stringify(fuera));await b.close();})();")
    r = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True, cwd=AQUI)
    return json.loads(r.stdout.strip().splitlines()[-1])


def armar(voz_mp3: Path, trabajo: Path, salida: Path, archivo_marcas: Path | None = None, guias: bool = False) -> dict:
    trabajo.mkdir(parents=True, exist_ok=True)
    html = trabajo / "html"
    html.mkdir(exist_ok=True)

    # la voz: nivelada, con sus marcas, y con las pausas que separan las escenas
    voz = trabajo / "voz.wav"
    dur_voz = ensamblar.preparar_audio(voz_mp3, voz)
    pausas = ensamblar.silencios(voz)
    medidas = json.loads(archivo_marcas.read_text()) if archivo_marcas else marcas_estimadas(dur_voz, pausas)
    medidas = medidas.get("marcas", medidas)
    medidas = {k: medidas[k] for k in CLAVES} | {"fin": min(medidas.get("fin", dur_voz), dur_voz)}
    assert all(medidas[a] < medidas[b] for a, b in zip(CLAVES, CLAVES[1:] + ["fin"])), "las marcas deben ir en orden"
    cortes, m = con_pausas(medidas, pausas)
    total = _audio(voz, cortes, trabajo / "audio.wav")
    cuadros = round(total * FPS)

    # las láminas del curso que se ven en la ventana
    cine = []
    for x, (t0, t1) in zip(CINE, tiempos_del_cine(m)):
        x = dict(x, dir=x["id"].replace("-", "_"))
        cine.append(dict(id=x["id"], dir=x["dir"], t0=t0, n=_cine(trabajo, x, t1 - t0)))

    # la página y su captura
    youtube.miniatura(html / "miniatura.png")
    datos = dict(fps=FPS, total=round(total, 3), marcas=m, rotulos=rotulos(m), cine=cine, titulo=TITULO, cifras=cifras(), guias=guias,
                 zona=ZONA, ancho=ANCHO, alto=ALTO)
    (html / "tiktok.html").write_text(pagina_html(datos), encoding="utf-8")
    inicios = [0.0] + [m[k] for k in CLAVES[1:]] + [total]
    filtro = ensamblar.FILTRO_COLOR.replace("1920:1080", f"{ANCHO}:{ALTO}")
    plan = dict(fps=FPS, total=cuadros, ancho=ANCHO, alto=ALTO,
                codificacion=["-vf", f"{filtro},noise=c0s=3:c0f=u", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-g", str(FPS * 2),
                              "-bf", "2", *ensamblar._COLOR],
                escenas=[dict(id="tiktok", tipo="tiktok", modulo=0, cuadro0=0, cuadros=cuadros, inicio=0.0,
                              duraciones=[round(b - a, 4) for a, b in zip(inicios, inicios[1:])], entra="", sale="", modIni=0.0,
                              modFin=round(total, 4))])
    (trabajo / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    subprocess.run(["node", str(AQUI / "capturar.cjs"), str(html), str(trabajo / "escena"), "--continuo", str(trabajo / "plan.json"),
                    "--hilos", "1"], check=True)

    # el video con su voz, al volumen de TikTok
    ganancia = VOLUMEN - ensamblar.VOLUMEN_OBJETIVO
    ensamblar._ffmpeg("-i", str(trabajo / "escena" / "tiktok.mp4"), "-i", str(trabajo / "audio.wav"), "-map", "0:v", "-map", "1:a",
                      "-c:v", "copy", "-af", f"volume={ganancia:.1f}dB,alimiter=limit=0.89:level=false", "-c:a", "aac", "-b:a", "160k",
                      "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(salida))
    info = dict(segundos=round(total, 2), cuadros=cuadros, marcas=m, cortes=cortes, rotulos=datos["rotulos"], cine=cine,
                fuera_de_zona=revisar(html / "tiktok.html", [round(t - 0.12, 3) for t in inicios[1:]]))
    (salida.parent / (salida.stem + "_tiempos.json")).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--guias"]
    if args == ["--texto"]:
        print(texto_para_voz())
    elif len(args) in (3, 4):
        r = armar(Path(args[0]), Path(args[1]), Path(args[2]), Path(args[3]) if len(args) == 4 else None, guias="--guias" in sys.argv)
        print(f"TikTok: {args[2]} · {r['segundos']:.1f} s · {len(r['rotulos'])} subtítulos")
        for aviso in r["fuera_de_zona"]:
            print("AVISO, fuera de la zona segura:", aviso)
    else:
        sys.exit(__doc__)
