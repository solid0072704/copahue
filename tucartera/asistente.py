"""Barra de instrucciones (⌘K): entiende pedidos breves del equipo y los convierte en trabajo.

Con Claude configurado interpreta lenguaje natural y usa herramientas. Sin Claude reconoce
pedidos frecuentes por palabras clave ("pide documentos", "revisa la cobranza"…).
"""

import unicodedata

from . import agentes, planes
from . import calendario as cal
from .db import filas

INTENCIONES = [
    (("document", "cartola", "antecedente", "pedir", "pide", "solicit", "faltan"), "documentos"),
    (("venc", "f29", "previred", "lre", "f22", "calendario", "declarac", "plazo"), "vencimientos"),
    (("cobr", "pago", "deuda", "honorario", "factura", "impag", "moros"), "cobranza"),
    (("cifra", "anomal", "variac", "control", "remanente", "compras", "ventas"), "control"),
    (("salud", "riesgo", "cartera", "oportunidad", "upsell"), "cartera"),
    (("informe", "resumen", "reporte", "cierre"), "informes"),
]
TODO = ("todo", "todas", "todos", "actualiza", "al dia", "ponte al dia")


def normalizar(texto):
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return " ".join(sin_tildes.lower().split())


def _resumen(resultados):
    if not resultados:
        return "Las automatizaciones están en pausa porque la suscripción no está activa. Revisa Plan y uso."
    nuevas = sum(r["creadas"] for r in resultados)
    ejecutadas = sum(r["ejecutadas"] for r in resultados)
    nombres = ", ".join(r["nombre"].lower() for r in resultados)
    if not nuevas and not ejecutadas:
        return f"Revisé {nombres}. No hay nada nuevo que proponer: todo lo pendiente ya está en Para revisar."
    partes = []
    if nuevas:
        partes.append(f"dejé {nuevas} propuesta{'s' if nuevas != 1 else ''} en Para revisar")
    if ejecutadas:
        partes.append(f"ejecuté {ejecutadas} en modo automático")
    return f"Revisé {nombres} y " + " y ".join(partes) + "."


def responder(conn, cfg, estudio, usuario, texto, hoy, claude=None):
    texto = (texto or "").strip()[:600]
    if not texto:
        return {"respuesta": "Escribe un pedido, por ejemplo: «pide los documentos que faltan».", "propuestas": []}

    def correr(clave):
        if clave not in agentes.POR_CLAVE:
            return "Esa automatización no existe."
        return _resumen(agentes.ejecutar(conn, cfg, estudio["id"], hoy, claves=[clave], forzar=True))

    aviso = ""
    if claude is not None:
        cupo = planes.cupo(conn, estudio, usuario["id"], hoy)
        if cupo["permitido"]:
            respuesta = claude.instruccion(
                conn, estudio, usuario, texto, hoy, correr,
                al_usar=lambda uso: planes.registrar_uso(conn, estudio["id"], usuario["id"], "instruccion", uso,
                                                         cal.ahora(cfg.tz)))
            if respuesta is not None:
                respuesta["modo"] = "asistido"
                return respuesta
        else:
            aviso = planes.mensaje_sin_cupo(cupo["motivo"])
    respuesta = _por_reglas(conn, cfg, estudio, texto, hoy)
    if aviso:
        respuesta["aviso"] = aviso
    return respuesta


def _por_reglas(conn, cfg, estudio, texto, hoy):
    norma = normalizar(texto)
    claves = [clave for palabras, clave in INTENCIONES if any(p in norma for p in palabras)]
    if not claves and any(p in norma.split() or p in norma for p in TODO):
        claves = [a.clave for a in agentes.AGENTES]
    if claves:
        resultados = agentes.ejecutar(conn, cfg, estudio["id"], hoy, claves=claves, forzar=True)
        return {"respuesta": _resumen(resultados), "propuestas": [], "modo": "reglas", "navegar": "#/revisar"}

    palabras = [p for p in norma.split() if len(p) > 2]
    for c in filas(conn, "SELECT id, razon_social, nombre_corto FROM clientes WHERE estudio_id = ?", estudio["id"]):
        nombre = normalizar(f"{c['razon_social']} {c['nombre_corto']}")
        if palabras and all(p in nombre for p in palabras):
            return {"respuesta": f"Abriendo {c['razon_social']}.", "propuestas": [], "modo": "reglas",
                    "navegar": f"#/clientes/{c['id']}"}
    return {
        "respuesta": ("No reconocí el pedido. Prueba con «pide los documentos que faltan», «revisa vencimientos», "
                      "«cobra los honorarios atrasados» o el nombre de un cliente."),
        "propuestas": [], "modo": "reglas",
    }
