"""Flow (flow.cl): tarjetas, Webpay y transferencias en Chile.

Cómo funciona su API: se crea el pago (`payment/create`) y Flow devuelve una dirección y un token; el comprador paga
ahí. Flow avisa al servidor (`urlConfirmation`) enviando solo el token, y el estado real se pregunta con
`payment/getStatus`: 1 pendiente, 2 pagada, 3 rechazada, 4 anulada. Cada llamada va firmada: los parámetros ordenados
por nombre, pegados nombre y valor, con HMAC-SHA256 y la clave secreta del comercio.

Escrito según la documentación de Flow; hay que probarlo en su ambiente sandbox (FLOW_URL) antes de cobrar de verdad.
"""

from __future__ import annotations

import hashlib
import hmac

import requests
from django.conf import settings
from django.urls import reverse

from ..models import Orden

ESTADOS = {1: "pendiente", 2: "pagada", 3: "rechazada", 4: "anulada"}


def firmar(parametros: dict, secreto: str) -> str:
    texto = "".join(f"{k}{parametros[k]}" for k in sorted(parametros))
    return hmac.new(secreto.encode(), texto.encode(), hashlib.sha256).hexdigest()


class Flow:
    clave = "flow"
    nombre = "Flow"
    descripcion = "Tarjetas de crédito y débito, Webpay y transferencia."

    def _llamar(self, metodo: str, servicio: str, parametros: dict) -> dict:
        datos = {**parametros, "apiKey": settings.FLOW_API_KEY}
        datos["s"] = firmar(datos, settings.FLOW_SECRET_KEY)
        url = f"{settings.FLOW_URL}/{servicio}"
        r = requests.post(url, data=datos, timeout=20) if metodo == "POST" else requests.get(url, params=datos, timeout=20)
        r.raise_for_status()
        return r.json()

    def iniciar(self, orden: Orden, request) -> str:
        r = self._llamar("POST", "payment/create", dict(
            commerceOrder=orden.codigo, subject=orden.curso.titulo[:140], currency=orden.moneda, amount=orden.monto, email=orden.usuario.email,
            urlConfirmation=settings.SITIO_URL + reverse("pago_confirmacion", args=[self.clave]),
            urlReturn=settings.SITIO_URL + reverse("pago_retorno", args=[self.clave])))
        orden.referencia = r["token"]
        orden.save(update_fields=["referencia"])
        return f'{r["url"]}?token={r["token"]}'

    def _actualizar(self, token: str) -> Orden | None:
        """Pregunta a Flow por el pago de ese token y deja la orden como corresponde."""
        if not token:
            return None
        estado = self._llamar("GET", "payment/getStatus", dict(token=token))
        orden = Orden.objects.filter(codigo=estado.get("commerceOrder", ""), pasarela=self.clave).first()
        if not orden:
            return None
        resultado = ESTADOS.get(estado.get("status"))
        if resultado == "pagada" and int(estado.get("amount", 0)) == orden.monto:
            orden.pagar(referencia=str(estado.get("flowOrder", token)), detalle=dict(flow=estado))
        elif resultado in ("rechazada", "anulada"):
            orden.cerrar(resultado, dict(flow=estado))
        return orden

    def confirmar(self, request) -> Orden | None:
        return self._actualizar(request.POST.get("token", ""))

    def de_retorno(self, request) -> Orden | None:
        return self._actualizar(request.POST.get("token") or request.GET.get("token", ""))
