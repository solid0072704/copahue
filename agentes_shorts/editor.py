"""Agente editor: elige los mejores momentos de un video y los convierte en Shorts verticales (9:16).

Cómo elige los momentos, en orden de preferencia:
  1. IA (Claude) leyendo la transcripción + las zonas «más repetidas» de YouTube.
  2. Mapa de «más repetido» (heatmap) que YouTube muestra en la barra de progreso.
  3. Energía del audio (los momentos más intensos suelen ser los más compartibles).
  4. Ventanas repartidas a lo largo del video.
"""

import logging
import os
import re
import shutil
import subprocess

log = logging.getLogger(__name__)

FUENTES_ROTULO = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


def fmt_tiempo(segundos):
    s = max(0, int(segundos))
    return f"{s // 60}:{s % 60:02d}" if s < 3600 else f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _fmt_srt(segundos):
    ms = int(round(max(0.0, segundos) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------- selección de momentos
def _sin_solape(candidatas, n, separacion):
    elegidas = []
    for ini, fin, puntaje in sorted(candidatas, key=lambda c: -c[2]):
        if all(fin + separacion <= e[0] or ini >= e[1] + separacion for e in elegidas):
            elegidas.append((ini, fin, puntaje))
        if len(elegidas) >= n:
            break
    return sorted(elegidas)


def ventanas_por_heatmap(heatmap, duracion_video, dur_corte, n=1, margen=0.03):
    """Ventanas de `dur_corte` segundos con mayor valor acumulado del mapa «más repetido»."""
    if not heatmap or duracion_video <= dur_corte:
        return []
    paso = 1.0
    inicio_min, fin_max = duracion_video * margen, duracion_video * (1 - margen)
    if fin_max - inicio_min < dur_corte:
        inicio_min, fin_max = 0.0, duracion_video
    valores = []
    t = inicio_min
    while t + dur_corte <= fin_max:
        valores.append((t, t + dur_corte, _valor_heatmap(heatmap, t, t + dur_corte)))
        t += paso
    maximo = max((v for _, _, v in valores), default=0) or 1
    return [(round(i, 2), round(f, 2), round(v / maximo, 3)) for i, f, v in _sin_solape(valores, n, dur_corte * 0.5)]


def _valor_heatmap(heatmap, inicio, fin):
    total = 0.0
    for tramo in heatmap:
        a, b = float(tramo.get("start_time", 0)), float(tramo.get("end_time", 0))
        solape = min(b, fin) - max(a, inicio)
        if solape > 0:
            total += solape * float(tramo.get("value", 0))
    return total


def energia_audio(archivo):
    """Nivel RMS (dB) por segundo usando ffmpeg; [] si falla."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", archivo, "-vn",
        "-af", "asetnsamples=n=48000,aresample=48000,astats=metadata=1:reset=1,ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=-",
        "-f", "null", "-",
    ]
    try:
        salida = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("No se pudo medir el audio: %s", e)
        return []
    niveles = []
    for m in re.finditer(r"RMS_level=(-?[\d.]+|-inf)", salida):
        niveles.append(-90.0 if m.group(1) == "-inf" else float(m.group(1)))
    return niveles


def ventanas_por_audio(niveles, duracion_video, dur_corte, n=1, margen=0.03):
    if len(niveles) < dur_corte + 2:
        return []
    inicio_min = int(duracion_video * margen)
    fin_max = min(len(niveles), int(duracion_video * (1 - margen)))
    candidatas = []
    for t in range(inicio_min, fin_max - int(dur_corte) + 1):
        tramo = niveles[t : t + int(dur_corte)]
        candidatas.append((float(t), float(t + dur_corte), sum(tramo) / len(tramo)))
    if not candidatas:
        return []
    minimo = min(c[2] for c in candidatas)
    maximo = max(c[2] for c in candidatas) - minimo or 1
    return [(i, f, round((v - minimo) / maximo, 3)) for i, f, v in _sin_solape(candidatas, n, dur_corte * 0.5)]


def ventanas_uniformes(duracion_video, dur_corte, n=1):
    if duracion_video <= dur_corte:
        return [(0.0, float(duracion_video), 0.0)]
    util = duracion_video * 0.9
    paso = util / (n + 1)
    ventanas = []
    for k in range(1, n + 1):
        ini = min(duracion_video * 0.05 + paso * k - dur_corte / 2, duracion_video - dur_corte)
        ventanas.append((round(max(0.0, ini), 2), round(max(0.0, ini) + dur_corte, 2), 0.0))
    return ventanas


def texto_ventana(subtitulos, inicio, fin):
    return " ".join(t for i, f, t in subtitulos if f > inicio and i < fin).strip()


def transcripcion_para_ia(subtitulos, bloque_s=10, max_caracteres=90_000):
    """Agrupa los subtítulos en líneas «[m:ss] texto» para que la IA elija momentos."""
    lineas, actual, inicio_bloque = [], [], None
    for ini, fin, txt in subtitulos:
        if inicio_bloque is None:
            inicio_bloque = ini
        actual.append(txt)
        if fin - inicio_bloque >= bloque_s:
            lineas.append(f"[{fmt_tiempo(inicio_bloque)}] {' '.join(actual)}")
            actual, inicio_bloque = [], None
    if actual:
        lineas.append(f"[{fmt_tiempo(inicio_bloque)}] {' '.join(actual)}")
    texto = "\n".join(lineas)
    return texto[:max_caracteres]


def _ajustar(inicio, fin, cfg, duracion_video):
    inicio = max(0.0, float(inicio))
    fin = min(float(duracion_video), float(fin))
    if fin - inicio > cfg.corte_max_s:
        fin = inicio + cfg.corte_max_s
    if fin - inicio < cfg.corte_min_s:
        fin = min(duracion_video, inicio + cfg.corte_min_s)
        inicio = max(0.0, fin - cfg.corte_min_s)
    return round(inicio, 2), round(fin, 2)


def elegir_cortes(cfg, video, material, asistente=None):
    """Decide los cortes de un video. Devuelve una lista de dicts listos para renderizar y publicar."""
    duracion = float(material.get("duracion_s") or video.get("duracion_s") or 0)
    n = max(1, cfg.cortes_por_video)
    subtitulos = material.get("subtitulos") or []
    pistas = ventanas_por_heatmap(material.get("heatmap"), duracion, cfg.corte_duracion_s, n=max(n, 3))

    cortes = []
    if asistente is not None and subtitulos:
        propuestos = asistente.elegir_cortes(
            video=video,
            transcripcion=transcripcion_para_ia(subtitulos),
            pistas=[(fmt_tiempo(i), fmt_tiempo(f)) for i, f, _ in pistas],
            n=n,
            duracion_video=duracion,
        )
        for p in propuestos or []:
            ini, fin = _ajustar(p["inicio_s"], p["fin_s"], cfg, duracion)
            cortes.append({"inicio_s": ini, "fin_s": fin, "metodo": "ia", "titulo": p.get("titulo", ""),
                           "descripcion": p.get("descripcion", ""), "etiquetas": p.get("etiquetas", [])})
        cortes = _quitar_solapes(cortes)[:n]

    if len(cortes) < n:
        faltan = n - len(cortes)
        ocupadas = [(c["inicio_s"], c["fin_s"]) for c in cortes]
        ventanas = [v for v in pistas if _libre(v, ocupadas)]
        metodo = "heatmap"
        if len(ventanas) < faltan and material.get("archivo") and os.path.exists(material["archivo"]):
            niveles = energia_audio(material["archivo"])
            ventanas += [v for v in ventanas_por_audio(niveles, duracion, cfg.corte_duracion_s, n=faltan * 2) if _libre(v, ocupadas + [(a, b) for a, b, _ in ventanas])]
            metodo = "audio" if not pistas else "heatmap+audio"
        if len(ventanas) < faltan:
            ventanas += [v for v in ventanas_uniformes(duracion, cfg.corte_duracion_s, n=faltan) if _libre(v, ocupadas + [(a, b) for a, b, _ in ventanas])]
            metodo = metodo if ventanas and metodo != "heatmap" else "uniforme"
        for ini, fin, _ in ventanas[:faltan]:
            ini, fin = _ajustar(ini, fin, cfg, duracion)
            cortes.append({"inicio_s": ini, "fin_s": fin, "metodo": metodo, "titulo": "", "descripcion": "", "etiquetas": []})

    cortes = sorted(_quitar_solapes(cortes), key=lambda c: c["inicio_s"])[:n]
    for indice, c in enumerate(cortes, 1):
        c["indice"] = indice
        c["video_id"] = video["video_id"]
        c["texto"] = texto_ventana(subtitulos, c["inicio_s"], c["fin_s"])
        if asistente is not None and not c["titulo"] and c["texto"]:
            datos = asistente.titular(video, c["texto"]) or {}
            c["titulo"], c["descripcion"], c["etiquetas"] = datos.get("titulo", ""), datos.get("descripcion", ""), datos.get("etiquetas", [])
    return cortes


def _libre(ventana, ocupadas):
    ini, fin = ventana[0], ventana[1]
    return all(fin <= a or ini >= b for a, b in ocupadas)


def _quitar_solapes(cortes):
    limpios = []
    for c in sorted(cortes, key=lambda c: c["inicio_s"]):
        if all(c["fin_s"] <= o["inicio_s"] or c["inicio_s"] >= o["fin_s"] for o in limpios):
            limpios.append(c)
    return limpios


# ---------------------------------------------------------------- render con ffmpeg
def escribir_srt(subtitulos, inicio, fin, ruta, max_caracteres=42):
    """Subtítulos del tramo [inicio, fin] con tiempos relativos al corte."""
    bloques, k = [], 1
    for ini, f, txt in subtitulos:
        if f <= inicio or ini >= fin:
            continue
        a, b = max(ini, inicio) - inicio, min(f, fin) - inicio
        if b - a < 0.15:
            continue
        for linea in _partir(txt, max_caracteres):
            bloques.append(f"{k}\n{_fmt_srt(a)} --> {_fmt_srt(b)}\n{linea}\n")
            k += 1
            break  # una línea por bloque: en pantalla vertical se lee mejor corto
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(bloques) + ("\n" if bloques else ""))
    return bool(bloques)


def _partir(texto, ancho):
    palabras, lineas, actual = texto.split(), [], ""
    for p in palabras:
        if len(actual) + len(p) + 1 > ancho and actual:
            lineas.append(actual)
            actual = p
        else:
            actual = f"{actual} {p}".strip()
    if actual:
        lineas.append(actual)
    return lineas or [texto[:ancho]]


def _fuente():
    for ruta in FUENTES_ROTULO:
        if os.path.exists(ruta):
            return ruta
    return None


def _escapar_ruta(ruta):
    # Dentro de un filtro de ffmpeg hay que escapar ':' y '\' (rutas de Windows incluidas).
    return ruta.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _filtro_video(cfg, srt=None, rotulo_txts=None, fuente=None):
    if cfg.estilo == "recorte":
        cadena = "[0:v]crop=ih*9/16:ih,scale=1080:1920[comp]"
    else:
        cadena = (
            "[0:v]split=2[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
            "boxblur=luma_radius=25:luma_power=2,eq=brightness=-0.08[bgb];"
            "[fg]scale=1080:1920:force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2[fgs];"
            "[bgb][fgs]overlay=(W-w)/2:(H-h)/2:shortest=1[comp]"
        )
    ultimo = "comp"
    for k, rotulo_txt in enumerate(rotulo_txts or []):
        if not fuente:
            break
        cadena += (
            f";[{ultimo}]drawtext=fontfile='{_escapar_ruta(fuente)}':textfile='{_escapar_ruta(rotulo_txt)}'"
            f":fontsize=58:fontcolor=white:borderw=4:bordercolor=black:x=(w-text_w)/2:y={190 + k * 72}[rot{k}]"
        )
        ultimo = f"rot{k}"
    if srt:
        # libass mide estos valores sobre una pantalla virtual de 288 px de alto (PlayResY), no sobre 1920.
        estilo = (
            "FontName=DejaVu Sans,FontSize=15,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            "BorderStyle=1,Outline=1.3,Shadow=0,Alignment=2,MarginV=52,MarginL=10,MarginR=10,WrapStyle=0"
        )
        cadena += f";[{ultimo}]subtitles=filename='{_escapar_ruta(srt)}':force_style='{estilo}'[sub]"
        ultimo = "sub"
    cadena += f";[{ultimo}]format=yuv420p[v]"
    return cadena


def renderizar(cfg, origen, inicio, fin, destino, rotulo=None, subtitulos=None):
    """Genera el Short. Si fallan los subtítulos o el rótulo, reintenta sin ellos antes de rendirse."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg no está instalado (https://ffmpeg.org/download.html)")
    carpeta = os.path.dirname(destino) or "."
    os.makedirs(carpeta, exist_ok=True)
    base = os.path.splitext(destino)[0]
    srt, rotulos = None, []
    if cfg.subtitulos and subtitulos:
        srt = base + ".srt"
        if not escribir_srt(subtitulos, inicio, fin, srt):
            os.remove(srt)
            srt = None
    if cfg.rotulo and rotulo:
        for k, linea in enumerate(_partir(rotulo, 26)[:3]):
            ruta = f"{base}.rotulo{k}.txt"
            with open(ruta, "w", encoding="utf-8") as fh:
                fh.write(linea)
            rotulos.append(ruta)

    intentos = [(srt, tuple(rotulos)), (None, tuple(rotulos)), (None, ())]
    ultimo_error = None
    for s, r in dict.fromkeys(intentos):
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{inicio:.3f}", "-t", f"{fin - inicio:.3f}", "-i", origen,
            "-filter_complex", _filtro_video(cfg, s, r, _fuente()),
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", cfg.preset_ffmpeg, "-crf", "22", "-profile:v", "high", "-level", "4.1",
            "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-ar", "48000",
            "-movflags", "+faststart", "-shortest", destino,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and os.path.exists(destino) and os.path.getsize(destino) > 0:
            return destino
        ultimo_error = proc.stderr.strip()[-800:]
        log.warning("ffmpeg falló (%s); reintentando con menos adornos", ultimo_error.splitlines()[-1] if ultimo_error else "?")
    raise RuntimeError(f"ffmpeg no pudo generar el corte: {ultimo_error}")


def ruta_corte(cfg, video_id, indice):
    return os.path.join(cfg.carpeta_trabajo, video_id, f"corte_{indice:02d}.mp4")
