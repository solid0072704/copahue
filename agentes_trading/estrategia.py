"""Estrategia: momentum relativo con filtro de tendencia y control de riesgo.

Reglas (en español sencillo):
  1. Cada ~mes se ordenan los activos por su rentabilidad de los últimos 6 meses.
  2. Solo se compran los mejores N que además estén por encima de su media de 200 ruedas
     y tengan rentabilidad positiva. Si ninguno cumple, la cartera queda en efectivo.
  3. El peso de cada activo es inverso a su volatilidad (los más tranquilos pesan más).
  4. Todos los días se revisan los stops: una posición que cae X% desde su compra se vende,
     y si la cartera cae Y% desde su máximo, se vende todo hasta el próximo rebalanceo.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Señal:
    fecha: pd.Timestamp
    objetivo: dict                     # {ticker: peso objetivo (0-1)}
    ranking: list = field(default_factory=list)   # [(ticker, momentum, en_tendencia)]
    motivo: str = ""


def momentum(cierres, ventana):
    return cierres.iloc[-1] / cierres.iloc[-1 - ventana] - 1


def en_tendencia(cierres, ventana):
    media = cierres.iloc[-ventana:].mean()
    return cierres.iloc[-1] > media


def volatilidad(cierres, ventana):
    return cierres.pct_change().iloc[-ventana:].std()


def señal_objetivo(cierres, cfg, fecha=None):
    """Calcula la cartera objetivo con los datos disponibles hasta `fecha` (incluida)."""
    if fecha is not None:
        cierres = cierres.loc[:fecha]
    necesarios = max(cfg.ventana_momentum, cfg.ventana_tendencia) + 1
    fecha = cierres.index[-1]
    if len(cierres) < necesarios:
        return Señal(fecha, {}, [], f"Faltan datos: se necesitan {necesarios} ruedas y hay {len(cierres)}.")

    ranking = []
    for t in cierres.columns:
        serie = cierres[t].dropna()
        if len(serie) < necesarios or serie.index[-1] != fecha:
            continue
        ranking.append((t, float(momentum(serie, cfg.ventana_momentum)), bool(en_tendencia(serie, cfg.ventana_tendencia))))
    ranking.sort(key=lambda x: x[1], reverse=True)

    elegidos = [t for t, mom, tend in ranking if mom > 0 and tend][: cfg.max_posiciones]
    if not elegidos:
        return Señal(fecha, {}, ranking, "Ningún activo cumple momentum positivo y tendencia alcista: cartera en efectivo.")

    inv_vol = {}
    for t in elegidos:
        vol = volatilidad(cierres[t].dropna(), cfg.ventana_volatilidad)
        inv_vol[t] = 1.0 / vol if vol and not np.isnan(vol) and vol > 0 else 1.0
    total = sum(inv_vol.values())
    invertible = 1.0 - cfg.reserva_efectivo_pct
    objetivo = {t: round(invertible * v / total, 4) for t, v in inv_vol.items()}
    return Señal(fecha, objetivo, ranking, f"Se eligen {', '.join(elegidos)} (mejor momentum y en tendencia).")


def revisar_stops(posiciones, precios_hoy, cfg):
    """Devuelve los tickers que deben venderse por stop-loss individual."""
    vender = []
    for t, pos in posiciones.items():
        precio = precios_hoy.get(t)
        if precio is None or np.isnan(precio) or not pos.get("precio_promedio"):
            continue
        if precio <= pos["precio_promedio"] * (1 - cfg.stop_loss_pct):
            vender.append(t)
    return vender


def caida_excesiva(patrimonio, maximo, cfg):
    return maximo > 0 and patrimonio <= maximo * (1 - cfg.max_caida_cartera_pct)
