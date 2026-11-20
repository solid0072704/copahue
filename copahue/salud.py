"""Salud de cada cliente: qué tan al día está con el estudio (documentos, pagos, declaraciones)."""

from datetime import date

from . import calendario as cal
from .db import filas


def _clp(monto):
    return "$" + f"{int(monto):,}".replace(",", ".")


def evaluar_cartera(conn, estudio_id, hoy):
    """Devuelve {cliente_id: {"puntaje", "nivel", "motivos"}} para los clientes activos."""
    periodo = cal.periodo_cerrado(hoy)
    clientes = filas(conn, "SELECT id FROM clientes WHERE estudio_id = ? AND estado != 'pausado'", estudio_id)
    resultado = {c["id"]: {"puntaje": 100, "motivos": []} for c in clientes}

    def penalizar(cid, puntos, motivo):
        if cid in resultado:
            resultado[cid]["puntaje"] -= puntos
            resultado[cid]["motivos"].append(motivo)

    for o in filas(conn, """
        SELECT cliente_id, tipo, periodo, vence FROM obligaciones
        WHERE estudio_id = ? AND estado != 'presentada' AND vence < ?""", estudio_id, hoy.isoformat()):
        dias = (hoy - date.fromisoformat(o["vence"])).days
        nombre = cal.TIPOS[o["tipo"]]["nombre"]
        penalizar(o["cliente_id"], 35, f"{nombre} de {cal.nombre_periodo(o['periodo'])} vencido hace {dias} día{'s' if dias != 1 else ''}")

    for f in filas(conn, """
        SELECT cliente_id, COUNT(*) AS n, SUM(monto) AS total, MIN(vence) AS mas_antigua FROM facturas
        WHERE estudio_id = ? AND pagada_el IS NULL AND vence < ? GROUP BY cliente_id""", estudio_id, hoy.isoformat()):
        dias = (hoy - date.fromisoformat(f["mas_antigua"])).days
        puntos = 30 if dias > 45 else 15 if dias > 14 else 5
        texto = f"{f['n']} factura{'s' if f['n'] > 1 else ''} impaga{'s' if f['n'] > 1 else ''} ({_clp(f['total'])}, hasta {dias} días)"
        penalizar(f["cliente_id"], puntos, texto)

    vence_f29 = cal.vencimiento("F29", periodo)
    cerca = cal.habiles_hasta(hoy, vence_f29) <= 3
    for d in filas(conn, """
        SELECT cliente_id, COUNT(*) AS n, MAX(recordatorios) AS recordatorios FROM documentos
        WHERE estudio_id = ? AND periodo = ? AND estado != 'recibido' GROUP BY cliente_id""", estudio_id, periodo):
        if d["recordatorios"] >= 2:
            penalizar(d["cliente_id"], 20, f"No responde: {d['recordatorios']} recordatorios sin respuesta")
        elif cerca:
            penalizar(d["cliente_id"], 15, f"Faltan {d['n']} documento{'s' if d['n'] > 1 else ''} de {cal.nombre_periodo(periodo)}")
        else:
            penalizar(d["cliente_id"], 5, f"{d['n']} documento{'s' if d['n'] > 1 else ''} por recibir")

    for cid, datos in resultado.items():
        datos["puntaje"] = max(0, min(100, datos["puntaje"]))
        datos["nivel"] = "ok" if datos["puntaje"] >= 80 else "atencion" if datos["puntaje"] >= 55 else "riesgo"
    return resultado
