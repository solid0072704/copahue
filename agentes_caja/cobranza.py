"""Cobranza: morosidad por antigüedad, por sociedad y por compromiso, acción recomendada y efectividad histórica."""

from collections import defaultdict
from datetime import timedelta

from .fuentes import _sumar_mes

TRAMOS = [
    (1, 15, "1-15 días", "Recordatorio automático con link de pago (WhatsApp y correo).", "media"),
    (16, 45, "16-45 días", "Llamada de cobranza: acordar fecha de pago concreta y registrarla.", "alta"),
    (46, 90, "46-90 días", "Carta formal y propuesta de repactación; revisar las garantías del contrato.", "alta"),
    (91, 10**9, "más de 90 días", "Evaluar cobranza externa o judicial y provisionar como incobrable.", "crítica"),
]


def tramo(dias):
    """(nombre del tramo, acción recomendada, nivel) según los días de mora."""
    for inicio, fin, nombre, accion, nivel in TRAMOS:
        if inicio <= max(dias, 1) <= fin:
            return nombre, accion, nivel
    return TRAMOS[-1][2:]


def _clave(c):
    return (c["empresa"], c["compromiso"] or c["cliente"])


def _inicio_mes(d):
    return d.replace(day=1)


def _mes_anterior(d):
    return (_inicio_mes(d) - timedelta(days=1)).replace(day=1)


def efectividad_mensual(cuotas, hoy, meses=3):
    """Por cada uno de los últimos meses cerrados: lo que vencía, cuánto de eso se pagó y lo recaudado en el mes."""
    filas, inicio = [], _mes_anterior(hoy)
    for _ in range(meses):
        fin = _sumar_mes(inicio, 1) - timedelta(days=1)
        vencidas = [c for c in cuotas if c["vence"] and inicio <= c["vence"] <= fin and c["estado"] != "anulada"]
        vencido = sum(c["monto"] for c in vencidas)
        cobrado = sum(c["monto"] for c in vencidas if c["estado"] == "pagada")
        recaudado = sum(
            c["monto"]
            for c in cuotas
            if c["estado"] == "pagada" and (c["pagado_en"] or c["vence"]) and inicio <= (c["pagado_en"] or c["vence"]) <= fin
        )
        filas.append(
            {
                "mes": f"{inicio.year}-{inicio.month:02d}",
                "vencido": round(vencido),
                "cobrado": round(cobrado),
                "puntualidad": round(100 * cobrado / vencido, 1) if vencido else None,
                "recaudado": round(recaudado),
                "cuotas_vencidas": len(vencidas),
            }
        )
        inicio = _mes_anterior(inicio)
    return list(reversed(filas))


def tasa_cobro_historica(efectividad, hay_pagadas):
    """Promedio de puntualidad de los meses con vencimientos, o None si no hay historial de pagos."""
    if not hay_pagadas:
        return None
    valores = [f["puntualidad"] for f in efectividad if f["puntualidad"] is not None and f["vencido"] > 0]
    return round(sum(valores) / len(valores) / 100, 3) if valores else None


def _resumen(lista):
    return {"monto": round(sum(c["monto"] for c in lista)), "cuotas": len(lista), "compromisos": len({_clave(c) for c in lista})}


def analizar(cuotas, hoy, top=25):
    morosas = [c for c in cuotas if c["estado"] == "moroso"]
    por_vencer = [c for c in cuotas if c["estado"] == "por_vencer"]
    pagadas = [c for c in cuotas if c["estado"] == "pagada"]
    total_moroso = sum(c["monto"] for c in morosas)
    total_por_vencer = sum(c["monto"] for c in por_vencer)
    total_pendiente = total_moroso + total_por_vencer
    pct_mora = round(100 * total_moroso / total_pendiente, 1) if total_pendiente else 0.0

    tramos = []
    for inicio, fin, nombre, accion, nivel in TRAMOS:
        grupo = [c for c in morosas if inicio <= max(c["dias_mora"], 1) <= fin]
        monto = sum(c["monto"] for c in grupo)
        tramos.append(
            {
                "tramo": nombre,
                "cuotas": len(grupo),
                "monto": round(monto),
                "pct": round(100 * monto / total_moroso, 1) if total_moroso else 0.0,
                "accion": accion,
                "nivel": nivel,
            }
        )

    en_30, hace_30 = hoy + timedelta(days=30), hoy - timedelta(days=30)
    por_empresa = []
    for empresa in sorted({c["empresa"] for c in cuotas}):
        cs = [c for c in cuotas if c["empresa"] == empresa]
        m = sum(c["monto"] for c in cs if c["estado"] == "moroso")
        pv = sum(c["monto"] for c in cs if c["estado"] == "por_vencer")
        por_empresa.append(
            {
                "empresa": empresa or "Sin sociedad",
                "pendiente": round(m + pv),
                "moroso": round(m),
                "pct_mora": round(100 * m / (m + pv), 1) if (m + pv) else 0.0,
                "cuotas_morosas": sum(1 for c in cs if c["estado"] == "moroso"),
                "compromisos_morosos": len({_clave(c) for c in cs if c["estado"] == "moroso"}),
                "vence_30d": round(sum(c["monto"] for c in cs if c["estado"] == "por_vencer" and c["vence"] and c["vence"] <= en_30)),
                "recaudado_30d": round(
                    sum(c["monto"] for c in cs if c["estado"] == "pagada" and (c["pagado_en"] or c["vence"]) and hace_30 <= (c["pagado_en"] or c["vence"]) <= hoy)
                ),
            }
        )
    por_empresa.sort(key=lambda x: -x["moroso"])

    grupos = defaultdict(list)
    for c in morosas:
        grupos[_clave(c)].append(c)
    prioritarios = []
    for (empresa, compromiso), cs in grupos.items():
        monto = sum(c["monto"] for c in cs)
        dias = max(c["dias_mora"] for c in cs)
        nombre_tramo, accion, nivel = tramo(dias)
        vencen = sum(1 for c in por_vencer if _clave(c) == (empresa, compromiso) and c["vence"] and c["vence"] <= en_30)
        prioritarios.append(
            {
                "empresa": empresa,
                "compromiso": compromiso,
                "cliente": cs[0]["cliente"],
                "proyecto": cs[0]["proyecto"],
                "monto": round(monto),
                "cuotas": len(cs),
                "dias_mora": dias,
                "tramo": nombre_tramo,
                "accion": accion,
                "nivel": nivel,
                "vencen_30d": vencen,
                "prioridad": round(monto * (0.5 + min(dias, 120) / 60)),
            }
        )
    prioritarios.sort(key=lambda x: (-x["prioridad"], -x["dias_mora"]))
    monto_top10 = sum(p["monto"] for p in prioritarios[:10])

    en_7 = hoy + timedelta(days=7)
    claves_morosas = set(grupos)
    proximas7 = [c for c in por_vencer if c["vence"] and c["vence"] <= en_7]
    proximas30 = [c for c in por_vencer if c["vence"] and c["vence"] <= en_30]
    efectividad = efectividad_mensual(cuotas, hoy)
    return {
        "total_pendiente": round(total_pendiente),
        "total_moroso": round(total_moroso),
        "total_por_vencer": round(total_por_vencer),
        "pct_mora": pct_mora,
        "cuotas_morosas": len(morosas),
        "compromisos_morosos": len(prioritarios),
        "compromisos_activos": len({_clave(c) for c in morosas + por_vencer}),
        "en_cobro": _resumen([c for c in cuotas if c["estado"] == "en_cobro"]),
        "repactadas": _resumen([c for c in cuotas if c["estado"] == "repactada"]),
        "tramos": tramos,
        "por_empresa": por_empresa,
        "prioritarios": prioritarios[:top],
        "total_prioritarios": len(prioritarios),
        "monto_top10": round(monto_top10),
        "concentracion_top10": round(100 * monto_top10 / total_moroso, 1) if total_moroso else 0.0,
        "vence_7d": {
            **_resumen(proximas7),
            "con_mora_monto": round(sum(c["monto"] for c in proximas7 if _clave(c) in claves_morosas)),
            "con_mora_cuotas": sum(1 for c in proximas7 if _clave(c) in claves_morosas),
        },
        "vence_30d": _resumen(proximas30),
        "efectividad": efectividad,
        "tasa_cobro_historica": tasa_cobro_historica(efectividad, bool(pagadas)),
    }
