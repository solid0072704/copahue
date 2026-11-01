"""Agente publicador: sube los cortes a tu canal con título, descripción, atribución y etiquetas.

Respeta tres frenos: el máximo de subidas por día, la cuota de la API (1600 unidades por video)
y la privacidad configurada (por defecto `private`, para que revises antes de publicar).
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .db import ahora

log = logging.getLogger(__name__)

LICENCIAS_TEXTO = {
    "creativeCommon": "Licencia del video original: Creative Commons Atribución (CC BY). Fragmento con atribución al autor.",
    "": "",
}


@dataclass
class ResultadoPublicacion:
    subidos: list = field(default_factory=list)
    omitidos: list = field(default_factory=list)
    errores: list = field(default_factory=list)

    def texto(self):
        partes = [f"{len(self.subidos)} subidos"]
        if self.omitidos:
            partes.append(f"{len(self.omitidos)} en espera ({self.omitidos[0]})")
        if self.errores:
            partes.append(f"{len(self.errores)} errores")
        return ", ".join(partes)


def _acortar(texto, largo):
    texto = " ".join((texto or "").split())
    return texto if len(texto) <= largo else texto[: largo - 1].rstrip() + "…"


def armar_metadatos(cfg, corte, video):
    """Devuelve (titulo, descripcion, etiquetas) listos para YouTube."""
    titulo = (corte.get("titulo") or "").strip()
    if not titulo:
        base = _acortar(video.get("titulo") or "Momento destacado", 70)
        titulo = f"{base} · parte {corte.get('indice', 1)}" if corte.get("indice", 1) > 1 else base
    if "#shorts" not in titulo.lower():
        titulo = _acortar(titulo, 100 - len(" #Shorts")) + " #Shorts"
    titulo = titulo[:100]

    url = f"https://youtu.be/{video['video_id']}"
    lineas = []
    if corte.get("descripcion"):
        lineas += [corte["descripcion"].strip(), ""]
    lineas.append(f"Fragmento de «{video.get('titulo', '')}» del canal {video.get('canal', '')}: {url}")
    lineas.append(f"Minuto {_mmss(corte['inicio_s'])} a {_mmss(corte['fin_s'])} del video original.")
    licencia = LICENCIAS_TEXTO.get(video.get("licencia") or "", "Todos los derechos del contenido pertenecen a su autor original.")
    if licencia:
        lineas.append(licencia)
    if cfg.firma_descripcion:
        lineas += ["", cfg.firma_descripcion]
    etiquetas_corte = corte.get("etiquetas") or []
    etiquetas = list(dict.fromkeys([e.strip().lstrip("#") for e in etiquetas_corte + cfg.etiquetas_fijas if e.strip()]))
    hashtags = " ".join("#" + e.replace(" ", "") for e in etiquetas[:5])
    if hashtags:
        lineas += ["", hashtags]
    return titulo, "\n".join(lineas), etiquetas


def _mmss(segundos):
    s = int(segundos)
    return f"{s // 60}:{s % 60:02d}"


def _proxima_fecha_programada(db, cfg):
    """Escalona las publicaciones cada `programar_cada_horas` a partir de la última programada."""
    if cfg.programar_cada_horas <= 0:
        return None
    minimo = ahora() + timedelta(minutes=15)
    ultimo = db.ultimo_programado()
    if ultimo:
        try:
            siguiente = datetime.fromisoformat(ultimo.replace("Z", "+00:00")) + timedelta(hours=cfg.programar_cada_horas)
        except ValueError:
            siguiente = minimo
        minimo = max(minimo, siguiente)
    return minimo.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def publicar_pendientes(db, yt, cfg, simular=False, limite=None):
    resultado = ResultadoPublicacion()
    pendientes = db.cortes(estado="renderizado")
    if not pendientes:
        return resultado
    cupo = cfg.max_subidas_dia - db.subidos_hoy()
    if limite is not None:
        cupo = min(cupo, limite)
    for corte in pendientes:
        if cupo <= 0:
            resultado.omitidos.append(f"ya se alcanzó el máximo diario de {cfg.max_subidas_dia} subidas")
            break
        if not yt.cuota_disponible("videos.insert"):
            resultado.omitidos.append("sin cuota de la API de YouTube hasta mañana")
            break
        video = db.video(corte["video_id"]) or {"video_id": corte["video_id"], "titulo": corte.get("video_titulo", ""), "canal": corte.get("canal", "")}
        if not corte.get("archivo") or not os.path.exists(corte["archivo"]):
            db.marcar_corte_error(corte["id"], "el archivo del corte ya no existe")
            resultado.errores.append(f"{corte['video_id']}#{corte['indice']}: archivo perdido")
            _cerrar_video(db, corte["video_id"])
            continue
        titulo, descripcion, etiquetas = armar_metadatos(cfg, corte, video)
        programado = _proxima_fecha_programada(db, cfg)
        if simular:
            log.info("[simulación] Subiría %s como «%s» (%s)", corte["archivo"], titulo, cfg.privacidad)
            resultado.subidos.append({"corte_id": corte["id"], "youtube_id": "simulado", "titulo": titulo})
            cupo -= 1
            continue
        try:
            youtube_id = yt.subir(
                corte["archivo"], titulo, descripcion, etiquetas,
                categoria=video.get("categoria") or "22", privacidad=cfg.privacidad,
                publicar_en=programado, made_for_kids=cfg.made_for_kids,
            )
        except Exception as e:  # la API devuelve HttpError con detalle; lo guardamos y seguimos
            db.marcar_corte_error(corte["id"], e)
            db.anotar("publicador", f"Error al subir {corte['video_id']}#{corte['indice']}: {e}", "error")
            resultado.errores.append(f"{corte['video_id']}#{corte['indice']}: {e}")
            _cerrar_video(db, corte["video_id"])
            continue
        db.marcar_corte_subido(corte["id"], youtube_id, programado)
        db.anotar("publicador", f"Subido https://youtu.be/{youtube_id} «{titulo}» ({cfg.privacidad}{', programado ' + programado if programado else ''})")
        resultado.subidos.append({"corte_id": corte["id"], "youtube_id": youtube_id, "titulo": titulo})
        cupo -= 1
        _cerrar_video(db, corte["video_id"])
    return resultado


def _cerrar_video(db, video_id):
    """Cuando ya no quedan cortes pendientes, el video pasa a «publicado» (si subió alguno) o a «error»."""
    if db.cortes(estado="renderizado", video_id=video_id):
        return
    subidos = db.cortes(estado="subido", video_id=video_id)
    if subidos:
        db.cambiar_estado_video(video_id, "publicado", f"{len(subidos)} cortes subidos")
    else:
        db.cambiar_estado_video(video_id, "error", "ningún corte pudo subirse")
