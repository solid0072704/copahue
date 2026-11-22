"""Pasarela de prueba: una página propia con dos botones, aprobar o rechazar. Sirve para recorrer la compra completa en
desarrollo y en las pruebas automáticas, sin dinero ni cuentas en ningún servicio. No se activa sola en el servidor."""

from __future__ import annotations

from django.urls import reverse

from ..models import Orden


class Prueba:
    clave = "prueba"
    nombre = "Pago de prueba"
    descripcion = "Simula el pago, sin dinero. Solo para probar la academia."

    def iniciar(self, orden: Orden, request) -> str:
        return reverse("pago_de_prueba", args=[orden.codigo])

    def confirmar(self, request) -> Orden | None:
        orden = Orden.objects.filter(codigo=request.POST.get("orden", ""), pasarela=self.clave).first()
        if orden and request.POST.get("resultado") == "aprobar":
            orden.pagar(referencia="prueba")
        elif orden:
            orden.cerrar("rechazada")
        return orden

    def de_retorno(self, request) -> Orden | None:
        return Orden.objects.filter(codigo=request.GET.get("orden", ""), pasarela=self.clave).first()
