"""Calendario tributario: genera las obligaciones de cada cliente y avisa las que están por vencer."""

from datetime import date, timedelta

from .. import calendario as cal
from ..db import filas, fila
from .base import Agente, proponer

ESTADOS = {"pendiente": "Sin comenzar", "preparacion": "En preparación", "revision": "En revisión"}


def generar_obligaciones(conn, estudio_id, cliente, hoy):
    """Crea las obligaciones del período cerrado, del mes en curso y la renta anual si corresponden."""
    servicios = cliente["servicios"]
    creadas = 0
    for periodo in (cal.periodo_cerrado(hoy), cal.periodo_de(hoy)):
        for tipo in cal.tipos_mensuales(servicios, cliente["trabajadores"]):
            creadas += _crear(conn, estudio_id, cliente, tipo, periodo)
    if "renta" in servicios:
        creadas += _crear(conn, estudio_id, cliente, "F22", f"AT{cal.anio_tributario_vigente(hoy)}")
    return creadas


def _crear(conn, estudio_id, cliente, tipo, periodo):
    monto = None
    if tipo == "F29":
        ind = fila(conn, "SELECT iva_debito, iva_credito FROM indicadores WHERE cliente_id = ? AND periodo = ?",
                   cliente["id"], periodo)
        if ind:
            monto = max(0, ind["iva_debito"] - ind["iva_credito"])
    cursor = conn.execute("""
        INSERT OR IGNORE INTO obligaciones (estudio_id, cliente_id, tipo, periodo, vence, estado, monto)
        VALUES (?, ?, ?, ?, ?, 'pendiente', ?)""",
        (estudio_id, cliente["id"], tipo, periodo, cal.vencimiento(tipo, periodo).isoformat(), monto))
    return cursor.rowcount


def cuando(hoy, vence):
    dias = (vence - hoy).days
    if dias < 0:
        return f"venció hace {-dias} día{'s' if dias != -1 else ''}"
    if dias == 0:
        return "vence hoy"
    if dias == 1:
        return "vence mañana"
    return f"vence el {cal.fecha_larga(vence)}"


class Vencimientos(Agente):
    clave = "vencimientos"
    nombre = "Calendario tributario"
    descripcion = "Genera F29, Previred, LRE y F22 de cada cliente y avisa lo que está por vencer."
    detalle = ("Cada día arma las obligaciones según los servicios contratados, corre las fechas al "
               "siguiente día hábil y propone asignar las que vencen en los próximos 3 días hábiles.")
    icono = "calendario"

    def ejecutar(self, ctx):
        for cliente in ctx.clientes():
            generar_obligaciones(ctx.conn, ctx.estudio_id, cliente, ctx.hoy)
        clientes = {c["id"]: c for c in ctx.clientes()}
        limite = cal.sumar_habiles(ctx.hoy, 3).isoformat()
        desde = (ctx.hoy - timedelta(days=45)).isoformat()
        for o in filas(ctx.conn, """
            SELECT * FROM obligaciones WHERE estudio_id = ? AND estado != 'presentada'
            AND vence BETWEEN ? AND ? ORDER BY vence""", ctx.estudio_id, desde, limite):
            cliente = clientes.get(o["cliente_id"])
            if not cliente:
                continue
            vence = date.fromisoformat(o["vence"])
            atrasada = vence < ctx.hoy
            if o["estado"] == "revision" and not atrasada:
                continue  # ya está en la revisión final
            tipo = cal.TIPOS[o["tipo"]]
            periodo = cal.nombre_periodo(o["periodo"])
            nombre_cliente = cliente["nombre_corto"] or cliente["razon_social"]
            motivos = [f"Estado: {ESTADOS.get(o['estado'], o['estado'])}"]
            if o["tipo"] in ("F29", "PREVIRED", "LRE"):
                faltan = filas(ctx.conn, """
                    SELECT tipo FROM documentos WHERE cliente_id = ? AND periodo = ? AND estado != 'recibido'""",
                    cliente["id"], o["periodo"])
                if faltan:
                    motivos.append(f"Faltan {len(faltan)} documento{'s' if len(faltan) > 1 else ''} del cliente")
            if o["monto"]:
                motivos.append(f"Monto estimado: ${o['monto']:,}".replace(",", "."))
            responsable = cliente.get("responsable_nombre")
            if responsable:
                motivos.append(f"Responsable: {responsable}")
            texto_cuando = cuando(ctx.hoy, vence)
            proponer(
                ctx,
                clave=f"venc:{o['id']}:{'vencida' if atrasada else 'proxima'}",
                tipo="tarea",
                cliente_id=cliente["id"],
                titulo=f"{tipo['nombre']} de {periodo} {texto_cuando}",
                resumen=f"{nombre_cliente} · {tipo['descripcion']}",
                motivos=motivos,
                prioridad=1 if atrasada or (vence - ctx.hoy).days <= 1 else 2,
                enfriamiento_dias=30,
                datos={
                    "tarea": {
                        "titulo": f"Presentar {tipo['nombre']} de {periodo} · {nombre_cliente}",
                        "detalle": f"{tipo['descripcion']} ({tipo['organismo']}). {texto_cuando.capitalize()}.",
                        "vence": o["vence"],
                        "responsable_id": cliente["responsable_id"],
                    },
                    "efectos": [{"tipo": "obligacion_estado", "id": o["id"], "estado": "preparacion"}]
                    if o["estado"] == "pendiente" else [],
                },
            )
