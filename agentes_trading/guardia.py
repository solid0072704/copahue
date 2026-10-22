"""Agente Guardia de riesgo: revisa cada orden antes de mostrártela y otra vez antes de ejecutarla.

Sus reglas no las puede saltar nadie: ni el Estratega, ni la IA, ni una aprobación tuya
por Telegram. Si una orden no las cumple, queda bloqueada y te explica por qué.
"""

AGENTE = "Guardia"


def perdida_desde_maximo_pct(db, valor_total, aportes):
    """Cuánto ha retrocedido tu ganancia desde su mejor momento, como % de lo que depositaste."""
    if aportes <= 0:
        return 0.0
    mejor = max(db.mejor_ganancia(), 0.0)
    return max(0.0, (mejor - (valor_total - aportes)) / aportes * 100)


def freno_activo(db, cfg, valor):
    return perdida_desde_maximo_pct(db, valor["total"], db.total_aportes()) >= cfg.freno_perdida_pct


def evaluar(prop, valor, cfg, db, efectivo_extra=0.0, ordenes_hoy=None, precio_actual=None):
    """Devuelve la lista de motivos por los que la orden NO puede ejecutarse (vacía = aprobada por riesgo).

    valor: resultado de estratega.valorizar() con la cuenta actual del broker.
    efectivo_extra: efectivo que liberarán las ventas del mismo plan (solo al planificar).
    ordenes_hoy: órdenes ya ejecutadas hoy (solo al ejecutar).
    precio_actual: precio justo antes de ejecutar, para detectar movimientos bruscos.
    """
    motivos = []
    simbolo, lado = prop["simbolo"], prop["lado"]
    cantidad = prop["cantidad"]
    precio = precio_actual or prop["precio_ref"]
    monto = cantidad * precio
    total = valor["total"]
    pos = valor["posiciones"].get(simbolo)

    if simbolo not in cfg.universo:
        motivos.append(f"{simbolo} no está en tu lista de activos permitidos ({', '.join(cfg.universo)}).")
    if lado not in ("compra", "venta"):
        motivos.append(f"Tipo de orden desconocido: {lado}.")
    if cantidad <= 0 or precio <= 0:
        motivos.append("La cantidad y el precio deben ser mayores a cero.")

    if precio_actual and prop["precio_ref"]:
        desvio = abs(precio_actual / prop["precio_ref"] - 1) * 100
        if desvio > cfg.desvio_precio_max_pct:
            motivos.append(
                f"El precio se movió {desvio:.1f}% desde que aprobaste (US$ {prop['precio_ref']:,.2f} → "
                f"US$ {precio_actual:,.2f}); el máximo es {cfg.desvio_precio_max_pct:g}%."
            )

    if ordenes_hoy is not None and ordenes_hoy >= cfg.max_ordenes_dia:
        motivos.append(f"Ya se ejecutaron {ordenes_hoy} órdenes hoy (máximo {cfg.max_ordenes_dia}).")

    if lado == "venta":
        tengo = pos["cantidad"] if pos else 0.0
        if cantidad > tengo + 1e-6:
            motivos.append(f"Solo tienes {tengo:.4f} {simbolo}; no se permite vender en corto.")
        cierra_posicion = pos and cantidad >= tengo - 1e-6
        if monto < cfg.min_orden_usd and not cierra_posicion:
            motivos.append(f"La orden (US$ {monto:,.2f}) es menor al mínimo de US$ {cfg.min_orden_usd:,.2f}.")
        return motivos

    # --- compras ---
    if monto < cfg.min_orden_usd:
        motivos.append(f"La orden (US$ {monto:,.2f}) es menor al mínimo de US$ {cfg.min_orden_usd:,.2f}.")
    reserva = total * cfg.reserva_efectivo_pct / 100
    disponible = valor["efectivo"] + efectivo_extra - reserva
    costo = monto * (1 + cfg.comision_simulada_pct / 100)
    if costo > disponible + 0.01:
        motivos.append(
            f"No alcanza el efectivo: la compra cuesta US$ {costo:,.2f} y hay US$ {max(disponible, 0):,.2f} "
            f"disponibles (se guarda una reserva de {cfg.reserva_efectivo_pct:g}%). Nunca se usa apalancamiento."
        )
    if total > 0:
        peso_final = ((pos["valor"] if pos else 0.0) + monto) / total * 100
        if peso_final > cfg.max_posicion_pct + 0.5:
            motivos.append(f"{simbolo} quedaría con {peso_final:.1f}% de tu cartera (máximo {cfg.max_posicion_pct:g}%).")
    if freno_activo(db, cfg, valor):
        motivos.append(
            f"Freno de pérdidas activo: la cartera retrocedió {cfg.freno_perdida_pct:g}% o más de lo depositado "
            "desde su máximo. Las compras se pausan hasta que lo revises."
        )
    return motivos
