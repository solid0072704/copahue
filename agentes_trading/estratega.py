"""Agente Estratega: mira tu cartera y el mercado y propone qué comprar o vender.

Estrategia (simple, conocida y explicable):
1. Cartera objetivo según tu perfil de riesgo (ETFs diversificados de bajo costo).
2. Filtro de tendencia: si un activo cae bajo su media de 200 días, su peso se mueve a bonos
   de corto plazo (SHY) hasta que recupere la tendencia. Reduce caídas fuertes.
3. Rebalanceo: solo propone órdenes cuando un activo se desvía más del umbral de su objetivo,
   o cuando hay efectivo nuevo (un depósito) para invertir.
4. Stop de pérdida: si una posición cae más del X% bajo su costo, propone venderla.

El Estratega solo PROPONE. Nada se ejecuta sin tu aprobación.
"""

import math
import secrets

from .config import ACTIVO_REFUGIO
from .mercado import indicadores

AGENTE = "Estratega"


def _redondear_abajo(x, decimales=4):
    f = 10**decimales
    return math.floor(x * f) / f


def valorizar(cuenta, mercado):
    posiciones = {}
    for simbolo, p in cuenta["posiciones"].items():
        precio = mercado.precio(simbolo)
        valor = p["cantidad"] * precio
        costo = p["costo_promedio"]
        posiciones[simbolo] = {
            "simbolo": simbolo,
            "cantidad": p["cantidad"],
            "costo_promedio": costo,
            "precio": precio,
            "valor": valor,
            "ganancia": valor - p["cantidad"] * costo,
            "ganancia_pct": (precio / costo - 1) * 100 if costo else 0.0,
        }
    invertido = sum(p["valor"] for p in posiciones.values())
    total = cuenta["efectivo"] + invertido
    for p in posiciones.values():
        p["peso"] = p["valor"] / total * 100 if total else 0.0
    return {"efectivo": cuenta["efectivo"], "invertido": invertido, "total": total, "posiciones": posiciones}


def pesos_tacticos(cfg, inds):
    """Aplica el filtro de tendencia a la cartera objetivo. Devuelve (pesos, notas por símbolo)."""
    pesos = dict(cfg.pesos_objetivo)
    notas = {}
    if not cfg.filtro_tendencia:
        return pesos, notas
    for simbolo in list(pesos):
        ind = inds.get(simbolo)
        if simbolo == ACTIVO_REFUGIO or not ind or ind["sobre_media"] is not False:
            continue
        notas[simbolo] = (
            f"{simbolo} está bajo su media de {cfg.media_tendencia_dias} días "
            f"(US$ {ind['precio']:,.2f} vs US$ {ind['media']:,.2f}): su peso pasa a {ACTIVO_REFUGIO} hasta que recupere la tendencia."
        )
        pesos[ACTIVO_REFUGIO] = pesos.get(ACTIVO_REFUGIO, 0) + pesos[simbolo]
        pesos[simbolo] = 0.0
    if ACTIVO_REFUGIO in pesos:
        pesos[ACTIVO_REFUGIO] = min(pesos[ACTIVO_REFUGIO], cfg.max_posicion_pct)
    return pesos, notas


def _texto_tendencia(ind, cfg):
    if not ind or ind["sobre_media"] is None:
        return ""
    partes = []
    if ind["sobre_media"]:
        partes.append(f"Tendencia alcista: precio sobre su media de {cfg.media_tendencia_dias} días.")
    if ind.get("ret_12m") is not None:
        partes.append(f"Rentabilidad 12 meses: {ind['ret_12m']:+.1f}%.")
    return " ".join(partes)


def stops(valor, cfg):
    """Posiciones que cayeron más del stop de pérdida bajo su costo promedio."""
    return [
        p for s, p in valor["posiciones"].items() if s in cfg.universo and p["ganancia_pct"] <= -cfg.stop_perdida_pct
    ]


def orden_stop(p, cfg):
    return {
        "tipo": "stop",
        "simbolo": p["simbolo"],
        "lado": "venta",
        "cantidad": p["cantidad"],
        "precio_ref": p["precio"],
        "monto": p["valor"],
        "motivo": (
            f"Stop de pérdida: {p['simbolo']} cae {p['ganancia_pct']:.1f}% bajo tu costo promedio "
            f"(US$ {p['costo_promedio']:,.2f}); tu límite es {cfg.stop_perdida_pct:g}%. Vender protege el resto de tu capital."
        ),
    }


def nuevo_plan_id():
    return secrets.token_hex(4)


def generar_plan(cfg, mercado, valor, sin_compras=()):
    """Calcula las órdenes para llevar la cartera a su objetivo. Devuelve (órdenes, resumen).

    Las ventas van primero porque liberan el efectivo que usan las compras.
    sin_compras: símbolos que no se recompran por ahora (p. ej. vendidos hace poco por stop de pérdida).
    """
    total = valor["total"]
    if total < cfg.min_orden_usd:
        return [], "No hay fondos suficientes para invertir. Deposita dinero para que el agente empiece."

    inds = {s: indicadores(mercado, s, cfg.media_tendencia_dias) for s in cfg.universo}
    pesos, notas = pesos_tacticos(cfg, inds)
    invertible = total * (1 - cfg.reserva_efectivo_pct / 100)
    umbral = max(cfg.min_orden_usd, total * cfg.umbral_rebalanceo_pct / 100)
    efectivo_ocioso = valor["efectivo"] - total * cfg.reserva_efectivo_pct / 100
    hay_aporte = efectivo_ocioso >= umbral
    en_stop = {p["simbolo"] for p in stops(valor, cfg)}

    ventas, compras = [], []
    for simbolo in cfg.universo:
        pos = valor["posiciones"].get(simbolo)
        if simbolo in en_stop:
            ventas.append(orden_stop(pos, cfg))
            continue
        ind = inds[simbolo]
        precio = ind["precio"]
        actual = pos["valor"] if pos else 0.0
        objetivo = pesos.get(simbolo, 0.0) / 100 * invertible
        dif = objetivo - actual
        peso_actual, peso_obj = actual / total * 100, objetivo / total * 100
        salida_total = objetivo == 0 and actual >= cfg.min_orden_usd
        if dif < 0 and (abs(dif) >= umbral or salida_total):
            cantidad = pos["cantidad"] if salida_total else min(_redondear_abajo(-dif / precio), pos["cantidad"])
            tipo = "tendencia" if simbolo in notas else "rebalanceo"
            motivo = notas.get(simbolo) or (
                f"Rebalanceo: {simbolo} subió a {peso_actual:.1f}% de tu cartera y su objetivo es {peso_obj:.1f}%. "
                "Vender una parte toma ganancias y mantiene tu nivel de riesgo."
            )
            ventas.append({"tipo": tipo, "simbolo": simbolo, "lado": "venta", "cantidad": cantidad, "precio_ref": precio, "motivo": motivo})
        elif dif > 0 and simbolo not in sin_compras and (dif >= umbral or (hay_aporte and dif >= cfg.min_orden_usd)):
            refugio = (
                f" Incluye el peso de {', '.join(notas)} mientras estén bajo su tendencia."
                if simbolo == ACTIVO_REFUGIO and notas
                else ""
            )
            compras.append({
                "tipo": "aporte" if hay_aporte else "rebalanceo",
                "simbolo": simbolo,
                "lado": "compra",
                "cantidad": _redondear_abajo(dif / precio),
                "precio_ref": precio,
                "motivo": (
                    f"{'Invertir tu efectivo disponible' if hay_aporte else 'Rebalanceo'}: {simbolo} pesa "
                    f"{peso_actual:.1f}% y su objetivo es {peso_obj:.1f}%.{refugio} " + _texto_tendencia(ind, cfg)
                ).strip(),
            })

    # Las compras no pueden gastar más que el efectivo disponible (más lo que liberen las ventas).
    comision = 1 + cfg.comision_simulada_pct / 100
    disponible = efectivo_ocioso + sum(v["cantidad"] * v["precio_ref"] for v in ventas) / comision
    necesario = sum(c["cantidad"] * c["precio_ref"] * comision for c in compras)
    if necesario > disponible > 0:
        factor = disponible / necesario
        for c in compras:
            c["cantidad"] = _redondear_abajo(c["cantidad"] * factor)
    compras = [c for c in compras if c["cantidad"] * c["precio_ref"] >= cfg.min_orden_usd] if disponible > 0 else []

    ordenes = ventas + compras
    for o in ordenes:
        o["monto"] = round(o["cantidad"] * o["precio_ref"], 2)
    if not ordenes:
        return [], "Tu cartera está en línea con su objetivo. No hace falta operar."
    resumen = f"{len(ventas)} ventas y {len(compras)} compras para llevar tu cartera ({cfg.perfil}) a su objetivo."
    if notas:
        resumen += " Filtro de tendencia activo en " + ", ".join(notas) + "."
    return ordenes, resumen
