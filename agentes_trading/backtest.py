"""Prueba histórica de la estrategia y métricas de riesgo/rentabilidad."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cartera import Cartera
from .datos import tabla
from .motor import paso

RUEDAS_AÑO = 252


@dataclass
class ResultadoBacktest:
    curva: pd.Series
    referencia: pd.Series
    operaciones: list
    metricas: dict
    metricas_referencia: dict
    probabilidad_meta: float
    meses_para_meta: float | None
    aportado_total: float = 0.0
    patrimonio_final: float = 0.0
    cartera: Cartera = field(repr=False, default=None)

    def resumen(self):
        m, r = self.metricas, self.metricas_referencia
        return (
            f"Estrategia: {m['rentabilidad_total']:+.1%} total, {m['cagr']:+.1%} anual, caída máxima {m['caida_maxima']:.1%}, "
            f"Sharpe {m['sharpe']:.2f}, {m['operaciones']} operaciones · Referencia: {r['cagr']:+.1%} anual, caída máxima {r['caida_maxima']:.1%}."
        )


def metricas(curva):
    """Rentabilidad, CAGR, volatilidad, Sharpe, caída máxima y % de meses positivos de una curva de patrimonio."""
    curva = curva.dropna()
    if len(curva) < 2 or curva.iloc[0] <= 0:
        return {k: 0.0 for k in ("rentabilidad_total", "cagr", "volatilidad", "sharpe", "caida_maxima", "meses_positivos", "operaciones")}
    retornos = curva.pct_change().dropna()
    años = len(curva) / RUEDAS_AÑO
    total = curva.iloc[-1] / curva.iloc[0] - 1
    cagr = (1 + total) ** (1 / años) - 1 if años > 0 else 0.0
    vol = retornos.std() * np.sqrt(RUEDAS_AÑO)
    sharpe = (retornos.mean() * RUEDAS_AÑO) / vol if vol > 0 else 0.0
    caida = (curva / curva.cummax() - 1).min()
    mensual = curva.resample("ME").last().pct_change().dropna()
    return {
        "rentabilidad_total": float(total),
        "cagr": float(cagr),
        "volatilidad": float(vol),
        "sharpe": float(sharpe),
        "caida_maxima": float(caida),
        "meses_positivos": float((mensual > 0).mean()) if len(mensual) else 0.0,
        "años": float(años),
    }


def probabilidad_de_meta(curva, capital, meta, ruedas=RUEDAS_AÑO):
    """Fracción de ventanas históricas de `ruedas` días en que `capital` habría ganado al menos `meta`."""
    curva = curva.dropna()
    if len(curva) <= ruedas:
        return 0.0
    ratio = curva.values[ruedas:] / curva.values[:-ruedas]
    return float(np.mean(capital * (ratio - 1) >= meta))


def meses_estimados(cagr, capital, meta):
    """Meses que tardaría en alcanzarse la meta si se repitiera el CAGR histórico (None si no crece)."""
    if cagr <= 0 or capital <= 0 or meta <= 0:
        return None
    años = np.log(1 + meta / capital) / np.log(1 + cagr)
    return float(años * 12)


def curva_sin_aportes(puntos, capital_inicial):
    """Curva ponderada por tiempo: quita el efecto de los depósitos para medir solo la estrategia."""
    indice, valores = [], []
    anterior, nivel = None, float(capital_inicial) if capital_inicial > 0 else 100.0
    for p in puntos:
        patrimonio, aporte = float(p["patrimonio"]), float(p.get("aporte", 0.0))
        if anterior is not None and anterior > 0:
            nivel *= (patrimonio - aporte) / anterior
        indice.append(pd.Timestamp(p["fecha"]))
        valores.append(nivel)
        anterior = patrimonio
    return pd.Series(valores, index=indice, name="estrategia")


def ejecutar_backtest(precios, cfg, desde=None, hasta=None, aporte_usd=0.0):
    """Simula la estrategia rueda a rueda con el mismo motor que usa el agente en vivo.

    `aporte_usd` suma ese importe el primer día hábil de cada mes, como hará el agente.
    Las métricas se calculan sobre la curva sin aportes (rentabilidad ponderada por tiempo).
    """
    cierres, aperturas = tabla(precios, "Close"), tabla(precios, "Open")
    if desde:
        cierres, aperturas = cierres.loc[desde:], aperturas.loc[desde:]
    if hasta:
        cierres, aperturas = cierres.loc[:hasta], aperturas.loc[:hasta]
    cartera = Cartera(modo="backtest", capital_inicial=cfg.capital_inicial, efectivo=cfg.capital_inicial, maximo=cfg.capital_inicial)
    inicio = max(cfg.ventana_momentum, cfg.ventana_tendencia) + 1
    fechas = cierres.index[inicio:]
    for fecha in fechas:
        paso(cartera, aperturas, cierres, fecha, cfg, aporte_usd=aporte_usd)
    curva = curva_sin_aportes(cartera.curva, cfg.capital_inicial)
    ref = cierres[cfg.referencia].loc[fechas[0]:] if cfg.referencia in cierres else cierres.iloc[:, 0].loc[fechas[0]:]
    referencia = (ref / ref.iloc[0] * float(curva.iloc[0])).rename("referencia")
    m = metricas(curva)
    m["operaciones"] = len(cartera.operaciones)
    r = metricas(referencia)
    r["operaciones"] = 1
    return ResultadoBacktest(
        curva=curva,
        referencia=referencia,
        operaciones=cartera.operaciones,
        metricas=m,
        metricas_referencia=r,
        probabilidad_meta=probabilidad_de_meta(curva, cfg.capital_inicial, cfg.meta_usd),
        meses_para_meta=meses_estimados(m["cagr"], cfg.capital_inicial, cfg.meta_usd),
        aportado_total=cartera.capital_aportado,
        patrimonio_final=cartera.curva[-1]["patrimonio"] if cartera.curva else cfg.capital_inicial,
        cartera=cartera,
    )
