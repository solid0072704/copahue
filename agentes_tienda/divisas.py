"""Tipo de cambio USD → CLP (dólar observado del Banco Central vía mindicador.cl)."""

import logging

import requests

from .almacen import ahora, dias_desde

log = logging.getLogger(__name__)


def tipo_cambio(almacen, cfg, usar_red=True):
    """Devuelve el dólar del día; si no hay red, el último conocido o TIPO_CAMBIO."""
    estado = almacen.estado
    if cfg.moneda != "CLP":
        return 1.0
    guardado = estado.get("tipo_cambio")
    if guardado and dias_desde(estado.get("tipo_cambio_fecha")) < 0.5:
        return guardado
    if usar_red:
        try:
            r = requests.get("https://mindicador.cl/api/dolar", timeout=15)
            r.raise_for_status()
            valor = float(r.json()["serie"][0]["valor"])
            if 300 < valor < 3000:
                estado["tipo_cambio"] = round(valor, 2)
                estado["tipo_cambio_fecha"] = ahora()
                return estado["tipo_cambio"]
        except Exception as e:
            log.warning("No se pudo leer el dólar observado: %s", e)
    return guardado or cfg.tipo_cambio_defecto
