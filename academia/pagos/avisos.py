"""Los correos que acompañan una compra."""

from __future__ import annotations

import logging

from django.conf import settings
from django.core.mail import send_mail
from django.urls import reverse

registro = logging.getLogger(__name__)


def compra_confirmada(orden) -> None:
    """Le confirma al alumno que el pago llegó y le deja el enlace al curso. Si el correo falla, la compra sigue en pie."""
    enlace = settings.SITIO_URL + reverse("aprender", args=[orden.curso.slug])
    nombre = orden.usuario.first_name or "Hola"
    try:
        send_mail(
            f"Tu inscripción en «{orden.curso.titulo}»",
            f"{nombre}:\n\nRecibimos tu pago de {orden.moneda} {orden.monto:,}".replace(",", ".") + f" (orden {orden.codigo}).\n"
            f"Ya puedes entrar al curso:\n{enlace}\n\n{settings.SITIO_NOMBRE}",
            settings.DEFAULT_FROM_EMAIL, [orden.usuario.email])
    except Exception:          # noqa: BLE001 — cualquier falla del servicio de correo
        registro.exception("No se pudo enviar la confirmación de la orden %s", orden.codigo)
