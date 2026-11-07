"""Precios diarios: descarga (Stooq, con respaldo en Yahoo), caché local y datos sintéticos."""

import io
import json
import logging
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("trading.datos")
COLUMNAS = ["Open", "High", "Low", "Close", "Volume"]
USER_AGENT = "Mozilla/5.0 (compatible; copahue-trading/1.0)"


def _normalizar(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for col in COLUMNAS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[COLUMNAS].astype(float)
    df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].ffill()
    return df.dropna(subset=["Close"])


def descargar_stooq(ticker, timeout=30):
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    if "Date" not in r.text[:100]:
        raise ValueError(f"Stooq no devolvió datos para {ticker}")
    df = pd.read_csv(io.StringIO(r.text), index_col="Date")
    return _normalizar(df)


def descargar_yahoo(ticker, años=10, timeout=30):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range={años}y&interval=1d"
    r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    resultado = r.json()["chart"]["result"][0]
    quote = resultado["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"],
            "Volume": quote["volume"],
        },
        index=pd.to_datetime(resultado["timestamp"], unit="s").normalize(),
    )
    return _normalizar(df)


def _ruta_cache(ruta_precios, ticker):
    return os.path.join(ruta_precios, f"{ticker.upper()}.csv")


def leer_cache(ruta_precios, ticker):
    ruta = _ruta_cache(ruta_precios, ticker)
    if not os.path.exists(ruta):
        return None
    try:
        return _normalizar(pd.read_csv(ruta, index_col="Date"))
    except Exception as e:  # archivo corrupto: se vuelve a descargar
        log.warning("Caché ilegible para %s (%s)", ticker, e)
        return None


def guardar_cache(ruta_precios, ticker, df):
    os.makedirs(ruta_precios, exist_ok=True)
    df.to_csv(_ruta_cache(ruta_precios, ticker))


def cache_actualizada(df, hoy=None):
    """La caché sirve si tiene datos del último día hábil (viernes cuenta para el fin de semana)."""
    if df is None or df.empty:
        return False
    hoy = hoy or date.today()
    ultimo_habil = hoy
    while ultimo_habil.weekday() >= 5:
        ultimo_habil -= timedelta(days=1)
    return df.index[-1].date() >= ultimo_habil - timedelta(days=1)


def obtener_precios(ticker, ruta_precios, años=10, forzar=False, hoy=None):
    """Devuelve el histórico diario de un ticker, usando la caché cuando está al día."""
    cache = leer_cache(ruta_precios, ticker)
    if not forzar and cache_actualizada(cache, hoy):
        return cache
    errores = []
    for fuente in (descargar_stooq, lambda t: descargar_yahoo(t, años)):
        try:
            df = fuente(ticker)
            if len(df) < 50:
                raise ValueError("histórico demasiado corto")
            if cache is not None:
                df = _normalizar(pd.concat([cache, df]))
            guardar_cache(ruta_precios, ticker, df)
            return df
        except Exception as e:
            errores.append(str(e))
    if cache is not None:
        log.warning("Sin conexión para %s, se usa la caché (%s)", ticker, "; ".join(errores))
        return cache
    raise RuntimeError(f"No se pudieron obtener precios de {ticker}: {'; '.join(errores)}")


def cargar_universo(tickers, ruta_precios, años=10, forzar=False, hoy=None):
    """Devuelve {ticker: DataFrame} y la lista de tickers que fallaron."""
    precios, fallidos = {}, []
    for t in tickers:
        try:
            precios[t] = obtener_precios(t, ruta_precios, años, forzar, hoy)
        except Exception as e:
            log.error("%s", e)
            fallidos.append(t)
    return precios, fallidos


def tabla(precios, columna="Close"):
    """Une un diccionario {ticker: DataFrame} en una tabla fechas × tickers."""
    if not precios:
        return pd.DataFrame()
    df = pd.concat({t: d[columna] for t, d in precios.items()}, axis=1).sort_index()
    return df.ffill().dropna(how="all")


def precios_sinteticos(tickers, dias=1500, semilla=7, inicio="2019-01-01"):
    """Series ficticias con tendencia y volatilidad distintas por activo (para demo y pruebas)."""
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range(inicio, periods=dias)
    precios = {}
    for i, t in enumerate(tickers):
        deriva = rng.uniform(-0.0002, 0.0007)
        vol = rng.uniform(0.006, 0.02)
        # régimen: bloques de 120 ruedas con tendencia alcista o bajista
        regimen = np.repeat(rng.choice([-1.5, 1.0, 1.5], size=dias // 120 + 1), 120)[:dias]
        retornos = deriva * regimen + vol * rng.standard_normal(dias)
        close = 100 * (1 + i * 0.3) * np.exp(np.cumsum(retornos))
        open_ = close * (1 + 0.002 * rng.standard_normal(dias))
        high = np.maximum(open_, close) * (1 + abs(0.003 * rng.standard_normal(dias)))
        low = np.minimum(open_, close) * (1 - abs(0.003 * rng.standard_normal(dias)))
        volumen = rng.integers(1_000_000, 5_000_000, dias).astype(float)
        precios[t] = _normalizar(
            pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volumen}, index=fechas)
        )
    return precios


def descargar_usdclp(timeout=20):
    """Dólar observado del Banco Central de Chile vía mindicador.cl (sin clave)."""
    r = requests.get("https://mindicador.cl/api/dolar", timeout=timeout, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    serie = r.json()["serie"]
    return float(serie[0]["valor"]), serie[0]["fecha"][:10]


def tipo_cambio_usdclp(ruta_archivo, defecto=950.0, hoy=None):
    """Devuelve (USD/CLP, fecha). Usa la caché del día; si no hay red, la última guardada o el valor por defecto."""
    hoy = str(hoy or date.today())[:10]
    cache = None
    if os.path.exists(ruta_archivo):
        try:
            with open(ruta_archivo, encoding="utf-8") as f:
                cache = json.load(f)
            if cache.get("consultado") == hoy:
                return float(cache["valor"]), cache["fecha"]
        except Exception:
            cache = None
    try:
        valor, fecha = descargar_usdclp()
        os.makedirs(os.path.dirname(ruta_archivo) or ".", exist_ok=True)
        with open(ruta_archivo, "w", encoding="utf-8") as f:
            json.dump({"valor": valor, "fecha": fecha, "consultado": hoy}, f)
        return valor, fecha
    except Exception as e:
        log.warning("Sin tipo de cambio en línea (%s)", e)
        if cache:
            return float(cache["valor"]), cache["fecha"]
        return float(defecto), "valor por defecto"
