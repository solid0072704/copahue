"""Orquestador del prospector: sincroniza con Gmail → califica → planifica → personaliza → avisa."""

import logging
from dataclasses import dataclass, field
from datetime import timedelta

from agentes_ebook.db import ahora_utc, iso
from agentes_ebook.gmail import ClienteGmail

from . import calificador, fuentes, notificador, secuencia, sincronizar
from .db import BaseProspector
from .llm import crear_asistente

log = logging.getLogger(__name__)


@dataclass
class ResultadoProspector:
    aceptaron: list = field(default_factory=list)
    respondieron: list = field(default_factory=list)
    conteo: dict = field(default_factory=dict)
    creadas: int = 0
    personalizados: int = 0
    cola: int = 0
    correo_cola: str = None
    correo_novedades: str = None
    errores: list = field(default_factory=list)

    def texto(self):
        c = self.conteo
        partes = [
            f"{sum(c.values())} prospectos (A={c.get('A', 0)}, B={c.get('B', 0)}, C={c.get('C', 0)})",
            f"{len(self.aceptaron)} aceptaron",
            f"{len(self.respondieron)} respondieron",
            f"{self.creadas} acciones nuevas",
            f"{self.cola} en la cola de hoy",
        ]
        if self.personalizados:
            partes.append(f"{self.personalizados} textos personalizados con IA")
        if self.correo_cola:
            partes.append(f"correo diario: {self.correo_cola}")
        if self.correo_novedades:
            partes.append(f"aviso de novedades: {self.correo_novedades}")
        if self.errores:
            partes.append("errores: " + "; ".join(self.errores))
        return " · ".join(partes)


def abrir_db(cfgp):
    return BaseProspector(cfgp.base.ruta_db)


def ejecutar_ciclo(db, cfgp, leer_gmail=True, notificar=True, usar_ia=True, forzar_cola=False, ahora=None):
    """Un ciclo completo. No abre LinkedIn ni envía nada allí: solo prepara tu cola y te avisa."""
    ahora = ahora or ahora_utc()
    r = ResultadoProspector()
    if leer_gmail and cfgp.base.gmail_configurado:
        try:
            with ClienteGmail(cfgp.base) as cliente:
                r.aceptaron, r.respondieron = sincronizar.sincronizar(db, cfgp, cliente, ahora)
        except Exception as e:
            r.errores.append(f"Sincronización con Gmail: {e}")
            db.log(sincronizar.AGENTE, "Error leyendo los avisos de LinkedIn", str(e))

    r.conteo = calificador.calificar(db, cfgp)
    r.creadas = len(secuencia.planificar(db, cfgp, ahora))
    if usar_ia:
        try:
            r.personalizados = secuencia.personalizar_cola(db, cfgp, crear_asistente(cfgp), ahora)
        except Exception as e:
            r.errores.append(f"IA: {e}")
    r.cola = len(secuencia.cola_de_hoy(db, cfgp, ahora))

    if notificar:
        r.correo_novedades = notificador.enviar_novedades(db, cfgp, r.aceptaron, r.respondieron)
        r.correo_cola = notificador.enviar_cola(db, cfgp, forzar=forzar_cola, ahora=ahora)
    db.set_meta("li_ultima_ejecucion", iso(ahora))
    db.log("Prospector", "Ciclo completado", r.texto())
    return r


def preparar_demo(db, cfgp):
    """Carga prospectos ficticios y simula una semana de trabajo para que veas el embudo en movimiento."""
    fuentes.cargar_demo(db)
    hace = lambda dias: ahora_utc() - timedelta(days=dias)  # noqa: E731
    ejecutar_ciclo(db, cfgp, leer_gmail=False, notificar=False, usar_ia=False, ahora=hace(7))
    demo = lambda a: a["fuente"] == "demo"  # noqa: E731
    invitaciones = [a for a in secuencia.pendientes(db, cfgp, hace(7))[1] if demo(a)]
    for a in invitaciones[:5]:
        secuencia.marcar_hecho(db, a["id"], hace(7))
    for a in invitaciones[:3]:
        secuencia.marcar_conectado(db, a["prospecto_id"], hace(6), origen="demo")
    secuencia.planificar(db, cfgp, hace(6))
    mensajes = [a for a in secuencia.pendientes(db, cfgp, hace(6))[0] if demo(a) and a["grado"] == 1]
    for a in mensajes[:3]:
        secuencia.marcar_hecho(db, a["id"], hace(5))
    if mensajes:
        secuencia.marcar_respondio(db, mensajes[0]["prospecto_id"], hace(3), origen="demo")
    return ejecutar_ciclo(db, cfgp, leer_gmail=False, notificar=False, usar_ia=False)
