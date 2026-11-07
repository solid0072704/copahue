"""Motor diario: ejecuta las órdenes pendientes en la apertura y decide las nuevas al cierre.

Es el mismo código para el backtest y para la operación diaria, así lo que ves en las
pruebas históricas es exactamente lo que hará el agente en vivo.
"""

import pandas as pd

from .estrategia import caida_excesiva, revisar_stops, señal_objetivo


def _fila(tabla, fecha):
    """Precios de una fecha como {ticker: valor} sin NaN."""
    if fecha not in tabla.index:
        return {}
    fila = tabla.loc[fecha]
    return {t: float(v) for t, v in fila.items() if pd.notna(v)}


def ejecutar_pendiente(cartera, aperturas, fecha, cfg):
    """Ejecuta a precio de apertura las órdenes decididas en la rueda anterior."""
    ordenes = cartera.pendiente
    if not ordenes:
        return []
    precios = _fila(aperturas, fecha)
    hechas = []
    for t in ordenes.get("vender", []):
        if t in precios and t in cartera.posiciones:
            op = cartera.vender_todo(t, precios[t], fecha, cfg.comision_pct, ordenes.get("motivo_venta", "venta"))
            if op:
                hechas.append(op)
    objetivo = ordenes.get("objetivo")
    if objetivo is not None:
        patrimonio = cartera.patrimonio(precios)
        # primero se vende lo que sobra, luego se compra lo que falta
        for t in list(cartera.posiciones):
            peso_obj = objetivo.get(t, 0.0)
            if t not in precios:
                continue
            valor = cartera.posiciones[t]["cantidad"] * precios[t]
            exceso = valor - peso_obj * patrimonio
            if peso_obj == 0:
                op = cartera.vender_todo(t, precios[t], fecha, cfg.comision_pct, "rebalanceo: sale de la cartera")
            elif exceso > max(25.0, 0.01 * patrimonio):
                op = cartera.vender(t, exceso / precios[t], precios[t], fecha, cfg.comision_pct, "rebalanceo: reduce peso")
            else:
                op = None
            if op:
                hechas.append(op)
        for t, peso_obj in objetivo.items():
            if t not in precios:
                continue
            valor = cartera.posiciones.get(t, {"cantidad": 0})["cantidad"] * precios[t]
            falta = peso_obj * patrimonio - valor
            if falta > max(25.0, 0.01 * patrimonio):
                op = cartera.comprar(t, falta, precios[t], fecha, cfg.comision_pct, "rebalanceo: entra o sube peso")
                if op:
                    hechas.append(op)
    cartera.pendiente = {}
    return hechas


def decidir(cartera, cierres, fecha, cfg):
    """Al cierre de `fecha`: revisa stops, caída de cartera y rebalanceo. Deja las órdenes en `pendiente`."""
    precios = _fila(cierres, fecha)
    patrimonio = cartera.anotar_curva(fecha, precios)
    cartera.ultima_fecha = str(fecha)[:10]
    pendiente = {}
    motivos = []

    vender = revisar_stops(cartera.posiciones, precios, cfg)
    if vender:
        pendiente["vender"] = vender
        pendiente["motivo_venta"] = f"stop-loss de {cfg.stop_loss_pct:.0%}"
        motivos.append(f"Stop-loss en {', '.join(vender)}.")

    toca_rebalancear = True
    if cartera.ultimo_rebalanceo:
        dias = len(cierres.loc[cartera.ultimo_rebalanceo:fecha]) - 1
        toca_rebalancear = dias >= cfg.rebalanceo_dias

    if cartera.posiciones and caida_excesiva(patrimonio, cartera.maximo, cfg):
        pendiente["vender"] = sorted(set(pendiente.get("vender", [])) | set(cartera.posiciones))
        pendiente["motivo_venta"] = f"la cartera cayó {cfg.max_caida_cartera_pct:.0%} desde su máximo"
        cartera.refugio_hasta = _fecha_futura(cierres, fecha, cfg.rebalanceo_dias)
        cartera.maximo = patrimonio   # el máximo se reinicia para no quedar en efectivo para siempre
        cartera.ultimo_rebalanceo = str(fecha)[:10]
        motivos.append(f"Caída de {cfg.max_caida_cartera_pct:.0%} desde el máximo: todo a efectivo hasta {cartera.refugio_hasta}.")
    elif toca_rebalancear and (not cartera.refugio_hasta or str(fecha)[:10] >= cartera.refugio_hasta):
        señal = señal_objetivo(cierres, cfg, fecha)
        cartera.refugio_hasta = ""
        cartera.ultimo_rebalanceo = str(fecha)[:10]
        cartera.ultima_señal = {
            "fecha": str(señal.fecha)[:10],
            "objetivo": señal.objetivo,
            "ranking": [{"ticker": t, "momentum": round(m, 4), "tendencia": e} for t, m, e in señal.ranking],
            "motivo": señal.motivo,
        }
        objetivo = {t: p for t, p in señal.objetivo.items() if t not in pendiente.get("vender", [])}
        actual = cartera.pesos(precios)
        if _cambia(actual, objetivo):
            pendiente["objetivo"] = objetivo
            motivos.append(señal.motivo)
        else:
            motivos.append("Rebalanceo revisado: la cartera ya está en su objetivo.")

    if pendiente:
        pendiente["fecha"] = str(fecha)[:10]
        pendiente["motivo"] = " ".join(motivos)
    cartera.pendiente = pendiente
    return pendiente


def _cambia(actual, objetivo, tolerancia=0.02):
    tickers = set(actual) | set(objetivo)
    return any(abs(actual.get(t, 0.0) - objetivo.get(t, 0.0)) > tolerancia for t in tickers)


def _fecha_futura(cierres, fecha, ruedas):
    posteriores = cierres.index[cierres.index > fecha]
    if len(posteriores) >= ruedas:
        return str(posteriores[ruedas - 1])[:10]
    return str(fecha + pd.Timedelta(days=int(ruedas * 1.5)))[:10]


def paso(cartera, aperturas, cierres, fecha, cfg):
    """Una rueda completa: ejecuta lo pendiente en la apertura y decide al cierre."""
    hechas = ejecutar_pendiente(cartera, aperturas, fecha, cfg)
    pendiente = decidir(cartera, cierres, fecha, cfg)
    return hechas, pendiente
