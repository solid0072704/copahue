"""Automatizaciones de Tu Cartera y su orquestación."""

import logging

from .. import calendario as cal
from .. import planes
from ..db import fila, filas, registrar_actividad
from .base import Contexto, retirar_obsoletas
from .cartera import Cartera
from .cobranza import Cobranza
from .control import Control
from .documentos import Documentos
from .informes import Informes
from .vencimientos import Vencimientos

log = logging.getLogger(__name__)

# El orden importa: el calendario crea las obligaciones que usan las demás.
AGENTES = [Vencimientos(), Documentos(), Control(), Cobranza(), Cartera(), Informes()]
POR_CLAVE = {a.clave: a for a in AGENTES}
MODOS = ("revisar", "automatico")


def asegurar_configuracion(conn, estudio_id):
    for agente in AGENTES:
        conn.execute("""
            INSERT OR IGNORE INTO automatizaciones (estudio_id, clave, activo, modo) VALUES (?, ?, 1, 'revisar')""",
            (estudio_id, agente.clave))


def ejecutar(conn, cfg, estudio_id, hoy, claves=None, llm=None, forzar=False):
    """Corre las automatizaciones activas (o las pedidas) y devuelve un resumen por cada una."""
    asegurar_configuracion(conn, estudio_id)
    estudio = fila(conn, "SELECT * FROM estudios WHERE id = ?", estudio_id)
    if not planes.suscripcion(estudio, hoy)["vigente"]:
        return []  # sin suscripción activa las automatizaciones quedan en pausa
    config = {a["clave"]: a for a in filas(conn, "SELECT * FROM automatizaciones WHERE estudio_id = ?", estudio_id)}
    resultados = []
    for agente in AGENTES:
        if claves and agente.clave not in claves:
            continue
        ajustes = config.get(agente.clave, {"activo": 1, "modo": "revisar"})
        if not ajustes["activo"] and not forzar:
            continue
        ctx = Contexto(conn=conn, cfg=cfg, estudio=estudio, hoy=hoy, llm=llm,
                       modo=ajustes["modo"], agente=agente.clave)
        try:
            agente.ejecutar(ctx)
            retirar_obsoletas(ctx)
            texto = agente.resumen(ctx)
        except Exception as e:  # una automatización con error no detiene a las demás
            log.exception("Falló la automatización %s", agente.clave)
            texto = f"Error: {e}"
        ahora = cal.ahora(cfg.tz)
        conn.execute("""
            UPDATE automatizaciones SET ultima_ejecucion = ?, ultimo_resultado = ? WHERE estudio_id = ? AND clave = ?""",
            (ahora, texto, estudio_id, agente.clave))
        if ctx.creadas or ctx.ejecutadas:
            partes = []
            if ctx.creadas:
                partes.append(f"dejó {ctx.creadas} propuesta{'s' if ctx.creadas != 1 else ''} para revisar")
            if ctx.ejecutadas:
                partes.append(f"ejecutó {ctx.ejecutadas} acci{'ones' if ctx.ejecutadas != 1 else 'ón'} en modo automático")
            frase = " y ".join(partes)
            registrar_actividad(conn, estudio_id, ahora, f"agente:{agente.clave}", frase[0].upper() + frase[1:])
        resultados.append({"clave": agente.clave, "nombre": agente.nombre, "creadas": ctx.creadas,
                           "ejecutadas": ctx.ejecutadas, "actualizadas": ctx.actualizadas, "texto": texto})
    return resultados
