"""Agente explorador: descubre los videos más virales y decide cuáles vale la pena recortar.

Fuentes (en orden de costo de cuota):
  · tendencias por región (1 unidad)      · canales con permiso (2 unidades)
  · búsquedas Creative Commons / temas (100 unidades cada una)
Score 0-100: velocidad de vistas, interacción, recencia y duración aprovechable.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .db import ahora

log = logging.getLogger(__name__)


@dataclass
class ResultadoExploracion:
    revisados: int = 0
    nuevos: int = 0
    seleccionados: list = field(default_factory=list)
    descartados: int = 0
    errores: list = field(default_factory=list)

    def texto(self):
        return (
            f"{self.revisados} videos revisados, {self.nuevos} nuevos, "
            f"{len(self.seleccionados)} seleccionados, {self.descartados} descartados"
            + (f", {len(self.errores)} errores" if self.errores else "")
        )


def _horas_desde(publicado_en, momento=None):
    if not publicado_en:
        return 24.0 * 30
    try:
        fecha = datetime.fromisoformat(publicado_en.replace("Z", "+00:00"))
    except ValueError:
        return 24.0 * 30
    if fecha.tzinfo is None:
        fecha = fecha.replace(tzinfo=timezone.utc)
    return max(1.0, ((momento or ahora()) - fecha).total_seconds() / 3600)


def puntuar(video, momento=None):
    """Devuelve (score 0-100, detalle [(factor, puntos), ...])."""
    detalle = []
    horas = _horas_desde(video.get("publicado_en"), momento)
    vistas = max(0, int(video.get("vistas") or 0))

    velocidad = vistas / horas  # vistas por hora desde la publicación
    pts = min(60, round(12 * math.log10(velocidad + 1)))
    detalle.append((f"{velocidad:,.0f} vistas/hora", pts))

    if vistas > 0:
        interaccion = (int(video.get("likes") or 0) + 3 * int(video.get("comentarios") or 0)) / vistas
        pts = min(25, round(interaccion * 500))
        detalle.append((f"interacción {interaccion:.1%}", pts))

    if horas <= 48:
        detalle.append(("publicado hace menos de 2 días", 10))
    elif horas <= 24 * 7:
        detalle.append(("publicado esta semana", 5))

    dur = int(video.get("duracion_s") or 0)
    if 180 <= dur <= 1500:
        detalle.append(("duración ideal para recortar (3-25 min)", 5))

    score = max(0, min(100, sum(p for _, p in detalle)))
    return score, detalle


def motivo_descarte(video, cfg):
    """None si el video sirve; si no, por qué se descarta."""
    if video.get("en_vivo"):
        return "transmisión en vivo"
    dur = int(video.get("duracion_s") or 0)
    if dur < cfg.duracion_min_s:
        return f"dura {dur}s (mínimo {cfg.duracion_min_s}s)"
    if dur > cfg.duracion_max_s:
        return f"dura {dur}s (máximo {cfg.duracion_max_s}s)"
    if str(video.get("categoria", "")) in cfg.categorias_excluidas:
        return f"categoría excluida ({video.get('categoria')})"
    permitido = video.get("canal_id") in cfg.canales or ("@" + (video.get("canal") or "")).lower() in [c.lower() for c in cfg.canales]
    if cfg.licencia == "creativeCommon" and not permitido and video.get("licencia") != "creativeCommon":
        return "sin licencia Creative Commons"
    if cfg.idiomas and video.get("idioma") and video["idioma"].split("-")[0] not in cfg.idiomas:
        return f"idioma {video['idioma']}"
    return None


def recolectar(yt, cfg):
    """Reúne candidatos de todas las fuentes configuradas respetando la cuota."""
    candidatos, errores = [], []

    def intentar(nombre, fn):
        try:
            candidatos.extend(fn())
        except Exception as e:  # una fuente caída no debe frenar a las demás
            log.warning("Fuente %s falló: %s", nombre, e)
            errores.append(f"{nombre}: {e}")

    for region in cfg.regiones:
        if yt.cuota_disponible("videos.list"):
            intentar(f"tendencias {region}", lambda r=region: yt.tendencias(r))
    for canal in cfg.canales:
        if yt.cuota_disponible("channels.list"):
            intentar(f"canal {canal}", lambda c=canal: yt.videos_de_canal(c))

    busquedas = []
    solo_cc = cfg.licencia == "creativeCommon"
    for tema in cfg.temas:
        for region in cfg.regiones[:1] or [None]:
            busquedas.append((f"tema «{tema}»", dict(consulta=tema, region=region, idioma=(cfg.idiomas or [None])[0], solo_cc=solo_cc)))
    if solo_cc and not cfg.temas:
        for region in cfg.regiones:
            busquedas.append((f"Creative Commons {region}", dict(region=region, idioma=(cfg.idiomas or [None])[0], solo_cc=True)))
    for nombre, params in busquedas[: cfg.busquedas_max]:
        if yt.cuota_disponible("search.list"):
            intentar(nombre, lambda p=params: yt.buscar(dias=cfg.publicado_hace_dias, **p))
        else:
            errores.append(f"{nombre}: sin cuota para buscar")

    unicos = {}
    for v in candidatos:
        if v.get("video_id") and v["video_id"] not in unicos:
            unicos[v["video_id"]] = v
    return list(unicos.values()), errores


def explorar(db, yt, cfg):
    resultado = ResultadoExploracion()
    candidatos, resultado.errores = recolectar(yt, cfg)
    momento = ahora()
    for video in candidatos:
        resultado.revisados += 1
        existente = db.video(video["video_id"])
        if existente and existente["estado"] not in ("descubierto",):
            db.guardar_video(video)  # refresca métricas, pero no vuelve a procesar
            continue
        if db.guardar_video(video):
            resultado.nuevos += 1
        score, detalle = puntuar(video, momento)
        db.puntuar_video(video["video_id"], score, detalle)
        motivo = motivo_descarte(video, cfg)
        if motivo:
            db.cambiar_estado_video(video["video_id"], "descartado", motivo)
            resultado.descartados += 1

    for video in db.candidatos(cfg.umbral_viral, cfg.videos_por_ciclo):
        db.cambiar_estado_video(video["video_id"], "seleccionado", f"score {video['score']}")
        resultado.seleccionados.append(video)
        db.anotar("explorador", f"Seleccionado «{video['titulo']}» ({video['canal']}) con score {video['score']}")
    if resultado.seleccionados or resultado.nuevos:
        db.anotar("explorador", resultado.texto())
    return resultado
