"""Ciclo completo del agente: explorar → cortar → subir. Cada etapa deja su rastro en la base."""

import logging
import os
from dataclasses import dataclass, field

from . import descarga, editor, explorador, publicador
from .llm import crear_asistente
from .youtube_api import YouTube

log = logging.getLogger(__name__)
ETAPAS = ("explorar", "cortar", "subir")


@dataclass
class Resultado:
    exploracion: explorador.ResultadoExploracion = None
    cortes: list = field(default_factory=list)
    publicacion: publicador.ResultadoPublicacion = None
    errores: list = field(default_factory=list)

    def texto(self):
        partes = []
        if self.exploracion:
            partes.append("Explorador: " + self.exploracion.texto())
        partes.append(f"Editor: {len(self.cortes)} cortes nuevos")
        if self.publicacion:
            partes.append("Publicador: " + self.publicacion.texto())
        if self.errores:
            partes.append(f"Errores: {len(self.errores)}")
        return " · ".join(partes)


def cortar_video(db, cfg, video, asistente=None, descargar_fn=None, renderizar_fn=None):
    """Descarga un video seleccionado, elige los momentos y renderiza los Shorts. Devuelve los cortes."""
    descargar_fn = descargar_fn or descarga.descargar
    renderizar_fn = renderizar_fn or editor.renderizar
    try:
        material = descargar_fn(cfg, video["video_id"], idiomas=[video["idioma"].split("-")[0]] + cfg.idiomas if video.get("idioma") else cfg.idiomas)
    except Exception as e:
        db.cambiar_estado_video(video["video_id"], "error", f"descarga: {str(e)[:300]}")
        db.anotar("editor", f"No se pudo descargar {video['video_id']}: {e}", "error")
        raise
    cortes = editor.elegir_cortes(cfg, video, material, asistente)
    hechos = []
    for corte in cortes:
        destino = editor.ruta_corte(cfg, video["video_id"], corte["indice"])
        try:
            corte["archivo"] = renderizar_fn(
                cfg, material["archivo"], corte["inicio_s"], corte["fin_s"], destino,
                rotulo=corte.get("titulo") or video.get("titulo"), subtitulos=material.get("subtitulos"),
            )
        except Exception as e:
            db.anotar("editor", f"Error al renderizar {video['video_id']}#{corte['indice']}: {e}", "error")
            log.warning("Render falló: %s", e)
            continue
        corte["id"] = db.guardar_corte(corte)
        hechos.append(corte)
        db.anotar(
            "editor",
            f"Corte {corte['indice']} de «{video['titulo']}»: {editor.fmt_tiempo(corte['inicio_s'])}-{editor.fmt_tiempo(corte['fin_s'])} "
            f"({corte['metodo']}) → {corte['archivo']}",
        )
    if hechos:
        db.cambiar_estado_video(video["video_id"], "cortado", f"{len(hechos)} cortes ({hechos[0]['metodo']})")
    else:
        db.cambiar_estado_video(video["video_id"], "error", "no se pudo generar ningún corte")
    if cfg.borrar_original:
        descarga.limpiar(cfg, video["video_id"], conservar=[c["archivo"] for c in hechos])
    return hechos


def cortar_pendientes(db, cfg, asistente=None, limite=None, **inyecciones):
    cortes, errores = [], []
    for video in db.videos(estado="seleccionado", limite=limite or cfg.videos_por_ciclo):
        try:
            cortes += cortar_video(db, cfg, video, asistente, **inyecciones)
        except Exception as e:
            errores.append(f"{video['video_id']}: {e}")
    return cortes, errores


def seleccionar_manual(db, yt, cfg, video_id):
    """Agrega un video elegido por ti (sin filtros de score) a la cola de cortes."""
    detalles = yt.detalles([video_id], fuente="manual")
    if not detalles:
        raise ValueError(f"YouTube no devolvió el video {video_id}")
    video = detalles[0]
    db.guardar_video(video)
    score, detalle = explorador.puntuar(video)
    db.puntuar_video(video_id, score, detalle)
    motivo = explorador.motivo_descarte(video, cfg)
    if motivo and "licencia" in motivo and cfg.licencia == "creativeCommon":
        log.warning("Ojo: %s. Lo agregas bajo tu responsabilidad.", motivo)
    db.cambiar_estado_video(video_id, "seleccionado", "elegido manualmente")
    db.anotar("explorador", f"Agregado manualmente «{video['titulo']}» ({video['canal']})")
    return db.video(video_id)


def ejecutar_ciclo(db, cfg, yt=None, etapas=ETAPAS, simular=False, usar_ia=True, **inyecciones):
    resultado = Resultado()
    yt = yt or YouTube(cfg, db)
    asistente = crear_asistente(cfg) if usar_ia else None
    os.makedirs(cfg.carpeta_trabajo, exist_ok=True)

    if "explorar" in etapas:
        try:
            resultado.exploracion = explorador.explorar(db, yt, cfg)
            resultado.errores += resultado.exploracion.errores
        except Exception as e:
            log.exception("El explorador falló")
            resultado.errores.append(f"explorador: {e}")
            db.anotar("explorador", f"Falló: {e}", "error")

    if "cortar" in etapas:
        cortes, errores = cortar_pendientes(db, cfg, asistente, **inyecciones)
        resultado.cortes, resultado.errores = cortes, resultado.errores + errores

    if "subir" in etapas:
        try:
            resultado.publicacion = publicador.publicar_pendientes(db, yt, cfg, simular=simular)
            resultado.errores += resultado.publicacion.errores
        except Exception as e:
            log.exception("El publicador falló")
            resultado.errores.append(f"publicador: {e}")
            db.anotar("publicador", f"Falló: {e}", "error")

    db.anotar("orquestador", resultado.texto())
    return resultado
