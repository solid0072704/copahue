"""Las pasarelas de pago. Todas hacen lo mismo con nombres distintos:

    iniciar(orden, request)   -> la dirección a la que se envía al comprador para pagar
    confirmar(request)        -> procesa el aviso que la pasarela manda al servidor y devuelve la orden (o None)
    de_retorno(request)       -> la orden de un comprador que vuelve desde la pasarela (o None)

Qué pasarelas están activas lo decide la configuración (`PASARELAS`): cada una se enciende con sus credenciales, y la
de prueba solo en desarrollo. La orden nunca se da por pagada por lo que diga el navegador del comprador: se paga
cuando la pasarela lo confirma al servidor y el servidor lo comprueba con ella.
"""

from __future__ import annotations

from django.conf import settings

from . import flow, mercadopago, prueba

TODAS = {"flow": flow.Flow, "mercadopago": mercadopago.MercadoPago, "prueba": prueba.Prueba}


def activas() -> list:
    return [TODAS[clave]() for clave in settings.PASARELAS if clave in TODAS]


def pasarela(clave: str):
    return TODAS[clave]() if clave in settings.PASARELAS and clave in TODAS else None
