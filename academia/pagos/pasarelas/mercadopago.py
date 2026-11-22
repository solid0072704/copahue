"""Mercado Pago (Checkout Pro): tarjetas, cuotas y saldo en cuenta.

Cómo funciona su API: se crea una «preferencia» con el curso y el número de la orden (`external_reference`), y Mercado
Pago devuelve la dirección de pago (`init_point`). Cuando el pago cambia de estado avisa al servidor
(`notification_url`) con el identificador del pago; el estado real se consulta en `/v1/payments/<id>`. Si se configura
la clave secreta de los avisos, además se valida su firma (`x-signature`).

Escrito según la documentación de Mercado Pago; hay que probarlo con sus credenciales de prueba antes de cobrar de verdad.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import requests
from django.conf import settings
from django.urls import reverse

from .. import precios
from ..models import Orden

API = "https://api.mercadopago.com"
ESTADOS = {"approved": "pagada", "rejected": "rechazada", "cancelled": "anulada"}


def firma_valida(request, secreto: str) -> bool:
    """La firma del aviso: HMAC-SHA256 de «id:<data.id>;request-id:<x-request-id>;ts:<ts>;» con la clave secreta."""
    partes = dict(p.strip().split("=", 1) for p in request.headers.get("x-signature", "").split(",") if "=" in p)
    id_del_aviso = request.GET.get("data.id") or request.GET.get("id", "")
    texto = f'id:{id_del_aviso};request-id:{request.headers.get("x-request-id", "")};ts:{partes.get("ts", "")};'
    esperada = hmac.new(secreto.encode(), texto.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(esperada, partes.get("v1", ""))


class MercadoPago:
    clave = "mercadopago"
    nombre = "Mercado Pago"
    descripcion = "Tarjetas de crédito y débito, cuotas y saldo de Mercado Pago."

    def _cabeceras(self) -> dict:
        return {"Authorization": f"Bearer {settings.MERCADOPAGO_TOKEN}", "Content-Type": "application/json"}

    def iniciar(self, orden: Orden, request) -> str:
        retorno = settings.SITIO_URL + reverse("pago_retorno", args=[self.clave]) + f"?orden={orden.codigo}"
        r = requests.post(f"{API}/checkout/preferences", headers=self._cabeceras(), timeout=20, json=dict(
            items=[dict(title=orden.curso.titulo, quantity=1, unit_price=precios.para_pasarela(orden.monto), currency_id=orden.moneda)],
            payer=dict(email=orden.usuario.email), external_reference=orden.codigo,
            back_urls=dict(success=retorno, pending=retorno, failure=retorno), auto_return="approved",
            notification_url=settings.SITIO_URL + reverse("pago_confirmacion", args=[self.clave])))
        r.raise_for_status()
        preferencia = r.json()
        orden.detalle = {**orden.detalle, "preferencia": preferencia.get("id", "")}
        orden.save(update_fields=["detalle"])
        return preferencia["init_point"]

    def _actualizar(self, id_del_pago: str) -> Orden | None:
        if not str(id_del_pago).isdigit():
            return None
        r = requests.get(f"{API}/v1/payments/{id_del_pago}", headers=self._cabeceras(), timeout=20)
        r.raise_for_status()
        pago = r.json()
        orden = Orden.objects.filter(codigo=pago.get("external_reference") or "", pasarela=self.clave).first()
        if not orden:
            return None
        resultado = ESTADOS.get(pago.get("status"))
        resumen = dict(mercadopago={k: pago.get(k) for k in ("id", "status", "status_detail", "transaction_amount", "currency_id", "date_approved")})
        if resultado == "pagada" and precios.mismo_monto(pago.get("transaction_amount"), orden.monto) and pago.get("currency_id") == orden.moneda:
            orden.pagar(referencia=str(pago["id"]), detalle=resumen)
        elif resultado in ("rechazada", "anulada"):
            orden.cerrar(resultado, resumen)
        return orden

    def confirmar(self, request) -> Orden | None:
        if settings.MERCADOPAGO_SECRETO and not firma_valida(request, settings.MERCADOPAGO_SECRETO):
            return None
        try:
            cuerpo = json.loads(request.body or b"{}")
        except ValueError:
            cuerpo = {}
        cuerpo = cuerpo if isinstance(cuerpo, dict) else {}
        tipo = request.GET.get("type") or request.GET.get("topic") or cuerpo.get("type", "")
        if tipo and tipo != "payment":
            return None                 # avisos de otras cosas (órdenes comerciales, contracargos): no cambian una orden
        return self._actualizar(request.GET.get("data.id") or request.GET.get("id") or str((cuerpo.get("data") or {}).get("id", "")))

    def de_retorno(self, request) -> Orden | None:
        """El comprador vuelve con el identificador del pago en la dirección; si viene, se consulta de inmediato."""
        if request.GET.get("payment_id", "").isdigit():
            orden = self._actualizar(request.GET["payment_id"])
            if orden:
                return orden
        return Orden.objects.filter(codigo=request.GET.get("orden", ""), pasarela=self.clave).first()
