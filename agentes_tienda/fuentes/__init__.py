"""Fuentes de productos. Cada fuente entrega productos 'normalizados' con los mismos campos."""

from .aliexpress import ClienteAliExpress
from .cj import ClienteCJ


def clientes_activos(cfg):
    clientes = {}
    if cfg.aliexpress_configurado:
        clientes["aliexpress"] = ClienteAliExpress(cfg)
    if cfg.cj_configurado:
        clientes["cj"] = ClienteCJ(cfg)
    return clientes


__all__ = ["ClienteAliExpress", "ClienteCJ", "clientes_activos"]
