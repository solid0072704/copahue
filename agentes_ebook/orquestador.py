"""Orquestador: ejecuta a los agentes en orden en cada ciclo."""

import logging
from dataclasses import dataclass, field

from . import analista, captura, notificador, reuniones, seguimiento
from .db import BaseDatos, ahora_utc, iso
from .gmail import ClienteGmail
from .llm import crear_asistente

log = logging.getLogger(__name__)


@dataclass
class ResultadoCiclo:
    nuevas_descargas: int = 0
    nuevos_leads: list = field(default_factory=list)
    respondieron: list = field(default_factory=list)
    calientes: list = field(default_factory=list)
    reuniones: list = field(default_factory=list)
    seguimientos: list = field(default_factory=list)
    alerta: str = None
    resumen: str = None
    errores: list = field(default_factory=list)

    def texto(self):
        partes = [
            f"{self.nuevas_descargas} descargas nuevas ({len(self.nuevos_leads)} leads nuevos)",
            f"{len(self.calientes)} leads calientes nuevos",
            f"{len(self.respondieron)} respuestas detectadas",
            f"{len(self.reuniones)} reuniones propuestas",
            f"{len(self.seguimientos)} correos de seguimiento preparados",
        ]
        if self.alerta:
            partes.append(f"alerta: {self.alerta}")
        if self.resumen:
            partes.append(f"resumen diario: {self.resumen}")
        if self.errores:
            partes.append("errores: " + "; ".join(self.errores))
        return " · ".join(partes)


def abrir_db(cfg):
    return BaseDatos(cfg.ruta_db)


def ejecutar_ciclo(db, cfg, leer_gmail=True, notificar=True, forzar_resumen=False, usar_ia=True):
    """Un ciclo completo: captura → sincroniza → analiza → reuniones → seguimiento → notifica."""
    r = ResultadoCiclo()
    asistente = crear_asistente(cfg) if usar_ia else None
    cliente = None
    if leer_gmail and cfg.gmail_configurado:
        try:
            cliente = ClienteGmail(cfg).__enter__()
        except Exception as e:
            r.errores.append(f"No se pudo conectar a Gmail: {e}")
            db.log("Captura", "Error de conexión", str(e))

    try:
        if cliente:
            try:
                r.nuevas_descargas, r.nuevos_leads = captura.desde_gmail(db, cfg, cliente)
            except Exception as e:
                r.errores.append(f"Lectura de descargas: {e}")
                db.log("Captura", "Error leyendo Gmail", str(e))
            try:
                r.respondieron = seguimiento.sincronizar(db, cfg, cliente)
            except Exception as e:
                r.errores.append(f"Sincronización: {e}")
                db.log("Seguimiento", "Error sincronizando", str(e))

        r.calientes = analista.analizar(db, cfg, asistente)
        reuniones.renovar_vencidas(db, cfg)
        r.reuniones = reuniones.proponer(db, cfg)
        r.seguimientos = seguimiento.ejecutar(db, cfg, cliente, asistente)
    finally:
        if cliente:
            cliente.__exit__(None, None, None)

    if notificar:
        r.alerta = notificador.enviar_alerta(db, cfg, r.respondieron)
        r.resumen = notificador.enviar_resumen(db, cfg, forzar=forzar_resumen)
    db.set_meta("ultima_ejecucion", iso(ahora_utc()))
    db.log("Orquestador", "Ciclo completado", r.texto())
    return r


def preparar_demo(db, cfg):
    """Carga leads ficticios con historial y deja a los agentes trabajar sobre ellos."""
    captura.cargar_demo(db)
    analista.analizar(db, cfg)
    captura.simular_historial_demo(db, cfg)
    return ejecutar_ciclo(db, cfg, leer_gmail=False, notificar=False, usar_ia=False)
