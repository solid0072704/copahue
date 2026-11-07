"""Proyección de un plan de aportes mensuales hacia una meta de renta (aritmética, sin promesas)."""

import math


def tasa_mensual(cagr):
    return (1 + cagr) ** (1 / 12) - 1


def patrimonio_futuro(aporte_mensual, cagr, meses, capital_inicial=0.0):
    """Valor de una cartera con aportes mensuales constantes tras `meses`, a una rentabilidad anual `cagr`."""
    r = tasa_mensual(cagr)
    crecido = capital_inicial * (1 + r) ** meses
    if abs(r) < 1e-12:
        return crecido + aporte_mensual * meses
    return crecido + aporte_mensual * ((1 + r) ** meses - 1) / r


def capital_para_renta(renta_mensual, tasa_retiro=0.04):
    """Capital necesario para retirar `renta_mensual` cada mes sin agotarlo (regla del 4 %)."""
    return renta_mensual * 12 / tasa_retiro if tasa_retiro > 0 else math.inf


def meses_hasta(objetivo, aporte_mensual, cagr, capital_inicial=0.0, maximo=1200):
    """Meses hasta alcanzar `objetivo`; None si no se alcanza en `maximo` meses."""
    if objetivo <= capital_inicial:
        return 0
    if aporte_mensual <= 0 and cagr <= 0:
        return None
    for m in range(1, maximo + 1):
        if patrimonio_futuro(aporte_mensual, cagr, m, capital_inicial) >= objetivo:
            return m
    return None


def escenarios(aporte_mensual, renta_meta, tasa_retiro=0.04, cagrs=(0.0, 0.05, 0.10, 0.15), horizontes=(1, 5, 10, 20), capital_inicial=0.0):
    """Tabla de escenarios: para cada rentabilidad anual, patrimonio y renta posible a distintos horizontes."""
    capital_meta = capital_para_renta(renta_meta, tasa_retiro)
    filas = []
    for cagr in cagrs:
        fila = {"cagr": cagr, "horizontes": {}}
        for años in horizontes:
            v = patrimonio_futuro(aporte_mensual, cagr, años * 12, capital_inicial)
            fila["horizontes"][años] = {"aportado": capital_inicial + aporte_mensual * años * 12, "patrimonio": v, "renta_mensual": v * tasa_retiro / 12}
        m = meses_hasta(capital_meta, aporte_mensual, cagr, capital_inicial)
        fila["años_para_meta"] = m / 12 if m is not None else None
        filas.append(fila)
    return {"capital_meta": capital_meta, "filas": filas}


def aporte_necesario(renta_meta, cagr, años, tasa_retiro=0.04, capital_inicial=0.0):
    """Aporte mensual que haría falta para lograr la renta meta en `años` a la rentabilidad `cagr`."""
    objetivo = capital_para_renta(renta_meta, tasa_retiro)
    r = tasa_mensual(cagr)
    n = años * 12
    restante = objetivo - capital_inicial * (1 + r) ** n
    if restante <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return restante / n
    return restante * r / ((1 + r) ** n - 1)


def texto_escenarios(aporte_mensual, renta_meta, tasa_retiro=0.04, cagr_historico=None, capital_inicial=0.0, moneda="CLP"):
    cagrs = [0.0, 0.05, 0.10, 0.15]
    if cagr_historico is not None and all(abs(cagr_historico - c) > 0.005 for c in cagrs):
        cagrs.append(cagr_historico)
    cagrs.sort()
    e = escenarios(aporte_mensual, renta_meta, tasa_retiro, cagrs, capital_inicial=capital_inicial)
    f = lambda v: f"{v:,.0f}".replace(",", ".")
    lineas = [
        f"Plan: {f(aporte_mensual)} {moneda} al mes · meta: renta de {f(renta_meta)} {moneda} al mes",
        f"Capital necesario para esa renta retirando {tasa_retiro:.0%} al año: {f(e['capital_meta'])} {moneda}",
        "",
        f"{'Rent. anual':>12} {'1 año':>14} {'5 años':>14} {'10 años':>14} {'20 años':>14}   Años hasta la meta",
    ]
    for fila in e["filas"]:
        celdas = " ".join(f"{f(fila['horizontes'][a]['patrimonio']):>14}" for a in (1, 5, 10, 20))
        años = f"{fila['años_para_meta']:.0f}" if fila["años_para_meta"] is not None else "más de 100"
        etiqueta = f"{fila['cagr']:.1%}" + (" (hist.)" if cagr_historico is not None and abs(fila["cagr"] - cagr_historico) < 1e-9 else "")
        lineas.append(f"{etiqueta:>12} {celdas}   {años}")
    lineas.append("")
    lineas.append("Aporte mensual que haría falta para lograr la meta:")
    for años in (5, 10, 20):
        lineas.append("  · en %2d años: " % años + " · ".join(f"{f(aporte_necesario(renta_meta, c, años, tasa_retiro, capital_inicial))} al {c:.0%}" for c in (0.05, 0.10, 0.15)))
    return "\n".join(lineas)
