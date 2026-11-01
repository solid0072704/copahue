"""Descarga del video original con yt-dlp, junto con su mapa de «más repetido» y los subtítulos."""

import glob
import json
import logging
import os

log = logging.getLogger(__name__)


def _opciones(cfg, carpeta, idiomas):
    opts = {
        "format": (
            f"bv*[height<={cfg.calidad_max}][ext=mp4]+ba[ext=m4a]/"
            f"bv*[height<={cfg.calidad_max}]+ba/b[height<={cfg.calidad_max}]/b"
        ),
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(carpeta, "%(id)s.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": list(dict.fromkeys(idiomas + ["es", "en"])),
        "subtitlesformat": "json3",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "fragment_retries": 3,
        "ignoreerrors": False,
    }
    if cfg.cookies and os.path.exists(cfg.cookies):
        opts["cookiefile"] = cfg.cookies
    return opts


def descargar(cfg, video_id, idiomas=None):
    """Devuelve {'archivo', 'heatmap', 'subtitulos', 'duracion_s', 'titulo'} o lanza una excepción."""
    import yt_dlp

    carpeta = os.path.join(cfg.carpeta_trabajo, video_id)
    os.makedirs(carpeta, exist_ok=True)
    idiomas = idiomas or cfg.idiomas
    with yt_dlp.YoutubeDL(_opciones(cfg, carpeta, idiomas)) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
    archivo = os.path.join(carpeta, f"{video_id}.mp4")
    if not os.path.exists(archivo):
        candidatos = [p for p in glob.glob(os.path.join(carpeta, f"{video_id}.*")) if not p.endswith((".json3", ".json", ".part"))]
        if not candidatos:
            raise RuntimeError("yt-dlp terminó pero no se encontró el archivo de video")
        archivo = candidatos[0]
    return {
        "archivo": archivo,
        "heatmap": info.get("heatmap") or [],
        "subtitulos": cargar_subtitulos(carpeta, video_id, idiomas),
        "duracion_s": float(info.get("duration") or 0),
        "titulo": info.get("title", ""),
        "idioma": (info.get("language") or "").lower(),
    }


def cargar_subtitulos(carpeta, video_id, idiomas):
    """Busca el archivo .json3 del mejor idioma disponible y lo convierte en segmentos."""
    for idioma in list(idiomas) + ["es", "en"]:
        for patron in (f"{video_id}.{idioma}.json3", f"{video_id}.{idioma}-*.json3", f"{video_id}.{idioma}*.json3"):
            for ruta in sorted(glob.glob(os.path.join(carpeta, patron))):
                try:
                    with open(ruta, encoding="utf-8") as f:
                        segmentos = parsear_json3(json.load(f))
                except (OSError, ValueError) as e:
                    log.warning("No se pudo leer %s: %s", ruta, e)
                    continue
                if segmentos:
                    return segmentos
    return []


def parsear_json3(datos):
    """Convierte el formato json3 de YouTube en [(inicio_s, fin_s, texto)], sin líneas vacías ni duplicadas."""
    segmentos = []
    for ev in datos.get("events", []):
        if "segs" not in ev or "tStartMs" not in ev:
            continue
        texto = "".join(s.get("utf8", "") for s in ev["segs"]).replace("\n", " ").strip()
        if not texto:
            continue
        inicio = ev["tStartMs"] / 1000.0
        duracion = ev.get("dDurationMs")
        fin = inicio + (duracion / 1000.0 if duracion else 2.0)
        if segmentos and segmentos[-1][2] == texto and abs(segmentos[-1][0] - inicio) < 0.05:
            continue
        segmentos.append((round(inicio, 3), round(fin, 3), texto))
    # Los subtítulos automáticos se solapan: recortamos cada uno al inicio del siguiente.
    for i in range(len(segmentos) - 1):
        ini, fin, txt = segmentos[i]
        if fin > segmentos[i + 1][0]:
            segmentos[i] = (ini, segmentos[i + 1][0], txt)
    return segmentos


def limpiar(cfg, video_id, conservar=()):
    """Borra el video original y los subtítulos; conserva los cortes indicados."""
    carpeta = os.path.join(cfg.carpeta_trabajo, video_id)
    if not os.path.isdir(carpeta):
        return
    for ruta in glob.glob(os.path.join(carpeta, "*")):
        if ruta in conservar or os.path.basename(ruta).startswith("corte_"):
            continue
        try:
            os.remove(ruta)
        except OSError:
            pass
