"""Las monedas en que se muestran los precios y las órdenes de compra. Una orden nace pendiente cuando el alumno
decide comprar; la pasarela avisa cuando se pagó, y recién entonces se crea la inscripción."""

from __future__ import annotations

import secrets

from django.conf import settings
from django.db import models, transaction
from django.utils import timezone

from academia.cursos.models import Curso, Inscripcion


class Moneda(models.Model):
    """Una moneda en la que se muestran precios (o se cobra) y cuántas unidades de ella vale un dólar.

    Los cursos tienen su precio en dólares; el tipo de cambio de cada moneda se trae una vez al día (`cambio.py`) o
    se escribe a mano aquí. Marcado como fijo, la actualización automática no lo toca."""
    codigo = models.CharField("código", max_length=3, unique=True, help_text="El código ISO de la moneda: CLP, MXN, EUR…")
    nombre = models.CharField(max_length=60, help_text="En plural y con minúscula: «pesos chilenos»")
    simbolo = models.CharField("símbolo", max_length=6, default="$", help_text="Si es solo «$», el precio lleva además el código: $100.000 CLP")
    decimales = models.PositiveSmallIntegerField(default=2, help_text="Decimales con que se cobra en esta moneda. El peso chileno, 0.")
    por_dolar = models.DecimalField("unidades por dólar", max_digits=18, decimal_places=6, null=True, blank=True,
                                    help_text="Cuántas unidades de esta moneda vale un dólar. En blanco, los precios no se muestran en ella.")
    fija = models.BooleanField("tipo de cambio fijo", default=False, help_text="Marcado, la actualización automática de cada día no cambia este valor")
    actualizada = models.DateTimeField("valor de la fuente del", null=True, blank=True, editable=False,
                                       help_text="La fecha que informó la fuente. Vacío si el valor se escribió a mano.")
    observacion = models.CharField("observación", max_length=200, blank=True, editable=False)

    class Meta:
        ordering = ["codigo"]

    def __str__(self) -> str:
        return f"{self.codigo} · {self.nombre}"

    @property
    def automatica(self) -> bool:
        """El tipo de cambio vino de la fuente pública, que pide ser nombrada donde se usa."""
        return not self.fija and self.actualizada is not None


def _codigo() -> str:
    return secrets.token_hex(10)          # 20 caracteres: sirve como número de orden ante cualquier pasarela


class Orden(models.Model):
    ESTADOS = [("pendiente", "Pendiente"), ("pagada", "Pagada"), ("rechazada", "Rechazada"), ("anulada", "Anulada")]
    codigo = models.CharField("código", max_length=24, unique=True, default=_codigo, editable=False)
    usuario = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="ordenes")
    curso = models.ForeignKey(Curso, on_delete=models.PROTECT, related_name="ordenes")
    monto = models.DecimalField(max_digits=12, decimal_places=2, help_text="Lo que se cobra, en la moneda de la orden")
    moneda = models.CharField(max_length=3)
    precio_usd = models.DecimalField("precio en dólares", max_digits=8, decimal_places=2, default=0, help_text="El precio del curso al comprar")
    cambio = models.DecimalField("tipo de cambio", max_digits=18, decimal_places=6, default=0, help_text="Unidades de la moneda de la orden por dólar con que se calculó el monto")
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

    @property
    def cobrado(self) -> str:
        """El monto como se le muestra a una persona: «$100.394 CLP»."""
        from . import precios                          # aquí, y no arriba: precios usa este módulo
        return precios.formato(self.monto, precios.moneda(self.moneda))

    @property
    def en_dolares(self) -> str:
        """El precio en dólares con que se hizo la compra («US$ 99»); vacío en las órdenes anteriores a los precios en dólares."""
        from . import precios
        return precios.formato(self.precio_usd, precios.moneda("USD")) if self.precio_usd and self.moneda != "USD" else ""

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
