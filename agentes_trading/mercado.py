"""Datos de mercado: precios históricos y actuales de los ETFs del universo.

- MercadoYahoo: precios reales (diarios, con cierre ajustado) desde Yahoo Finance, sin API key.
- MercadoSintetico: precios inventados y reproducibles, para la demo y las pruebas sin internet.
"""

import math
import random
from datetime import date, timedelta

import requests


class ErrorMercado(Exception):
    pass


class MercadoYahoo:
    URL = "https://query1.finance.yahoo.com/v8/finance/chart/{simbolo}"

    def __init__(self, sesion=None):
        self.sesion = sesion or requests.Session()
        self.sesion.headers.setdefault("User-Agent", "Mozilla/5.0 (copahue-trading)")
        self._cache = {}

    def _datos(self, simbolo):
        if simbolo not in self._cache:
            try:
                r = self.sesion.get(
                    self.URL.format(simbolo=simbolo), params={"range": "2y", "interval": "1d"}, timeout=20
                )
                r.raise_for_status()
                res = r.json()["chart"]["result"][0]
                ind = res.get("indicators", {})
                cierres = (ind.get("adjclose") or [{}])[0].get("adjclose") or ind["quote"][0]["close"]
            except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as e:
                raise ErrorMercado(f"No se pudo leer el precio de {simbolo}: {e}") from e
            serie = [
                (date.fromtimestamp(t).isoformat(), float(c))
                for t, c in zip(res.get("timestamp", []), cierres)
                if c is not None
            ]
            precio = res.get("meta", {}).get("regularMarketPrice") or (serie[-1][1] if serie else None)
            if not precio:
                raise ErrorMercado(f"Yahoo no devolvió precio para {simbolo}.")
            self._cache[simbolo] = (serie, float(precio))
        return self._cache[simbolo]

    def historial(self, simbolo):
        """Lista de (fecha ISO, cierre ajustado), del más antiguo al más reciente."""
        return self._datos(simbolo)[0]

    def precio(self, simbolo):
        return self._datos(simbolo)[1]

    def refrescar(self):
        self._cache.clear()


class MercadoSintetico:
    """Caminata aleatoria reproducible. Puedes fijar tendencias o precios para simular escenarios."""

    def __init__(self, semilla=7, dias=320, tendencias=None):
        self.semilla = semilla
        self.dias = dias
        self.tendencias = tendencias or {}
        self._fijos = {}
        self._cache = {}

    def historial(self, simbolo):
        if simbolo not in self._cache:
            rnd = random.Random(f"{self.semilla}-{simbolo}")
            deriva = self.tendencias.get(simbolo, 0.0003)
            precio = 50 + rnd.random() * 250
            hoy = date.today()
            serie = []
            for i in range(self.dias):
                precio *= math.exp(deriva + rnd.gauss(0, 0.009))
                serie.append(((hoy - timedelta(days=self.dias - 1 - i)).isoformat(), round(precio, 2)))
            self._cache[simbolo] = serie
        serie = self._cache[simbolo]
        if simbolo in self._fijos:
            return serie[:-1] + [(serie[-1][0], self._fijos[simbolo])]
        return serie

    def precio(self, simbolo):
        return self.historial(simbolo)[-1][1]

    def fijar_precio(self, simbolo, precio):
        self._fijos[simbolo] = float(precio)

    def refrescar(self):
        pass


def crear_mercado(cfg):
    return MercadoSintetico() if cfg.mercado == "sintetico" else MercadoYahoo()


# ---------------- indicadores ----------------
def media_movil(cierres, n):
    return sum(cierres[-n:]) / n if len(cierres) >= n else None


def retorno_pct(cierres, n):
    if len(cierres) <= n or not cierres[-n - 1]:
        return None
    return (cierres[-1] / cierres[-n - 1] - 1) * 100


def volatilidad_anual_pct(cierres, n=60):
    if len(cierres) <= n:
        return None
    rets = [math.log(cierres[i] / cierres[i - 1]) for i in range(len(cierres) - n, len(cierres))]
    prom = sum(rets) / len(rets)
    var = sum((r - prom) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252) * 100


def indicadores(mercado, simbolo, dias_media=200):
    cierres = [c for _, c in mercado.historial(simbolo)]
    precio = mercado.precio(simbolo)
    media = media_movil(cierres, dias_media)
    return {
        "simbolo": simbolo,
        "precio": precio,
        "media": media,
        "sobre_media": None if media is None else precio >= media,
        "ret_3m": retorno_pct(cierres, 63),
        "ret_12m": retorno_pct(cierres, 252),
        "volatilidad": volatilidad_anual_pct(cierres),
    }
