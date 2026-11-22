"""Las órdenes de compra. Una orden nace pendiente cuando el alumno decide comprar; la pasarela avisa cuando se pagó,
y recién entonces se crea la inscripción."""

from __future__ import annotations

import secrets

from django.conf import settings
from django.db import models, transaction
from django.utils import timezone

from academia.cursos.models import Curso, Inscripcion


def _codigo() -> str:
    return secrets.token_hex(10)          # 20 caracteres: sirve como número de orden ante cualquier pasarela


class Orden(models.Model):
    ESTADOS = [("pendiente", "Pendiente"), ("pagada", "Pagada"), ("rechazada", "Rechazada"), ("anulada", "Anulada")]
    codigo = models.CharField("código", max_length=24, unique=True, default=_codigo, editable=False)
    usuario = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="ordenes")
    curso = models.ForeignKey(Curso, on_delete=models.PROTECT, related_name="ordenes")
    monto = models.PositiveIntegerField()
    moneda = models.CharField(max_length=3)
    estado = models.CharField(max_length=20, choices=ESTADOS, default="pendiente")
    pasarela = models.CharField(max_length=20)
    referencia = models.CharField(max_length=200, blank=True, help_text="El identificador del pago en la pasarela")
    detalle = models.JSONField(default=dict, blank=True)
    creada = models.DateTimeField(auto_now_add=True)
    pagada_en = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-creada"]
        verbose_name = "orden"
        verbose_name_plural = "órdenes"

    def __str__(self) -> str:
        return f"{self.codigo} · {self.curso} · {self.get_estado_display()}"

    @transaction.atomic
    def pagar(self, referencia: str = "", detalle: dict | None = None) -> bool:
        """Marca la orden como pagada e inscribe al alumno. Devuelve False si ya lo estaba: el aviso de la pasarela
        puede llegar más de una vez y no debe tener efecto la segunda."""
        orden = Orden.objects.select_for_update().get(pk=self.pk)
        if orden.estado == "pagada":
            return False
        orden.estado, orden.pagada_en = "pagada", timezone.now()
        orden.referencia = referencia or orden.referencia
        orden.detalle = {**orden.detalle, **(detalle or {})}
        orden.save()
        Inscripcion.objects.get_or_create(usuario=orden.usuario, curso=orden.curso, defaults=dict(origen="compra"))
        self.refresh_from_db()
        from .avisos import compra_confirmada          # aquí, y no arriba: avisos usa este módulo
        transaction.on_commit(lambda: compra_confirmada(orden))
        return True

    def cerrar(self, estado: str, detalle: dict | None = None) -> None:
        """Deja la orden rechazada o anulada. Una orden pagada no se toca."""
        if self.estado == "pendiente" and estado in ("rechazada", "anulada"):
            self.estado, self.detalle = estado, {**self.detalle, **(detalle or {})}
            self.save(update_fields=["estado", "detalle"])
