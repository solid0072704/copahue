"""Orquestador del embudo: lee HubSpot, diagnostica, destraba e informa en cada ciclo."""

import logging
from dataclasses import dataclass, field
from datetime import timedelta

from agentes_ebook.db import ahora_utc, iso

from . import desatascador, embudo, informe
from .api import ClienteHubSpot
from .db import MemoriaEmbudo
from .llm import crear_asistente

log = logging.getLogger(__name__)


@dataclass
class ResultadoCiclo:
    diagnostico: dict = None
    desatascador: object = None
    informe_ruta: str = ""
    correo: str = None
    markdown: str = ""
    errores: list = field(default_factory=list)

    def texto(self):
        if not self.diagnostico:
            return "sin diagnóstico · errores: " + "; ".join(self.errores)
        d, r = self.diagnostico, self.desatascador
        partes = [
            f"{d['total_abiertos']} negocios abiertos",
            f"{d['sin_proxima_total']} sin próxima actividad",
            f"{len(r.estancados)} estancados",
            f"{r.creadas} tareas creadas",
            f"{len(r.escalados)} por escalar",
        ]
        if self.informe_ruta:
            partes.append(f"informe: {self.informe_ruta}")
        if self.correo:
            partes.append(f"correo: {self.correo}")
        if self.errores:
            partes.append("errores: " + "; ".join(self.errores))
        return " · ".join(partes)


def abrir_db(cfg):
    return MemoriaEmbudo(cfg.ruta_db)


def _seguro(funcion, defecto, resultado, nombre):
    try:
        return funcion()
    except Exception as e:
        log.warning("No se pudo leer %s: %s", nombre, e)
        resultado.errores.append(f"{nombre}: {e}")
        return defecto


def ejecutar_ciclo(cfg, db, cliente=None, ahora=None, enviar=True, guardar=True, usar_ia=True):
    """Un ciclo completo: leer HubSpot → diagnosticar → detectar estancados → tareas → informe."""
    ahora = ahora or ahora_utc()
    r = ResultadoCiclo()
    cliente = cliente or ClienteHubSpot(cfg)
    try:
        etapas, pipeline_id, pipeline_nombre = cliente.etapas()
        abiertos = cliente.negocios_abiertos(pipeline_id, etapas)
        periodo = cliente.negocios_creados_desde(pipeline_id, etapas, ahora - timedelta(days=cfg.embudo_dias))
    except Exception as e:
        r.errores.append(f"Lectura de HubSpot: {e}")
        db.log("Orquestador", "Error leyendo HubSpot", str(e))
        return r
    propietarios = _seguro(cliente.propietarios, {}, r, "propietarios")
    portal = _seguro(cliente.portal_id, "", r, "id del portal") if hasattr(cliente, "portal_id") else ""

    estancados = desatascador.detectar(abiertos, etapas, cfg, ahora)
    anterior = db.foto_anterior(pipeline_id, cfg.comparar_dias, ahora)
    diag = embudo.diagnosticar(abiertos, periodo, etapas, cfg, ahora, pipeline_nombre, propietarios, anterior, estancados)
    resultado = desatascador.ejecutar(db, cfg, cliente, estancados, ahora)
    db.guardar_foto(diag, pipeline_id)
    db.log(embudo.AGENTE, "Diagnóstico", f"{diag['total_abiertos']} abiertos · {len(estancados)} estancados")

    lectura = None
    if usar_ia:
        asistente = crear_asistente(cfg)
        if asistente:
            lectura = asistente.lectura(diag)

    local = ahora.astimezone(cfg.tz)
    titulo, md, html, texto = informe.construir(diag, resultado, cfg, portal, propietarios, lectura, local)
    r.diagnostico, r.desatascador, r.markdown = diag, resultado, md
    if guardar:
        try:
            r.informe_ruta = informe.guardar(md, cfg, local, pipeline_nombre)
        except OSError as e:
            r.errores.append(f"No se pudo guardar el informe: {e}")
    if enviar:
        r.correo = informe.enviar(cfg, titulo, texto, html)
        if r.correo.startswith("error"):
            r.errores.append(f"Correo: {r.correo}")
    r.errores.extend(resultado.errores)
    db.set_meta("ultima_ejecucion", iso(ahora))
    db.log("Orquestador", "Ciclo completado", r.texto())
    return r


def preparar_demo(cfg, db, ahora=None):
    """Corre un ciclo con datos ficticios (sin HubSpot, sin correo, sin IA). Devuelve (resultado, cliente)."""
    from .demo import ClienteDemo

    ahora = ahora or ahora_utc()
    cliente = ClienteDemo(ahora)
    return ejecutar_ciclo(cfg, db, cliente, ahora, enviar=False, guardar=True, usar_ia=False), cliente
