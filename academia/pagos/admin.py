from django.contrib import admin, messages

from academia.cursos.admin import ConComaDecimal

from . import cambio
from .models import Moneda, Orden


@admin.register(Moneda)
class MonedaAdmin(ConComaDecimal, admin.ModelAdmin):
    """Las monedas en que se muestran los precios. El tipo de cambio llega solo una vez al día; aquí se puede escribir
    a mano («1014,08») y dejarlo fijo, o pedir la actualización ahora."""
    list_display = ("codigo", "nombre", "simbolo", "por_dolar", "fija", "actualizada", "observacion")
    list_editable = ("por_dolar", "fija")
    readonly_fields = ("actualizada", "observacion")
    actions = ["actualizar_ahora"]

    def save_model(self, request, obj, form, change) -> None:
        if "por_dolar" in form.changed_data:            # escrito a mano: ya no es un valor de la fuente, y la observación se da por vista
            obj.actualizada, obj.observacion = None, ""
        super().save_model(request, obj, form, change)

    @admin.action(description="Traer ahora los tipos de cambio del día (no cambia los fijos)")
    def actualizar_ahora(self, request, monedas) -> None:
        try:
            r = cambio.actualizar()
        except Exception as e:          # noqa: BLE001 — cualquier falla de la fuente se le muestra a quien apretó el botón
            self.message_user(request, f"No se pudieron traer los tipos de cambio: {e}", messages.ERROR)
            return
        self.message_user(request, f"{len(r['actualizadas'])} monedas actualizadas.", messages.SUCCESS)
        if r["rechazadas"]:
            self.message_user(request, f"Sin aplicar por estar muy lejos del valor guardado: {', '.join(r['rechazadas'])}. Revisa su observación.", messages.WARNING)


@admin.register(Orden)
class OrdenAdmin(admin.ModelAdmin):
    list_display = ("codigo", "usuario", "curso", "monto", "moneda", "precio_usd", "estado", "pasarela", "creada", "pagada_en")
    list_filter = ("estado", "pasarela", "curso")
    search_fields = ("codigo", "referencia", "usuario__email")
    readonly_fields = ("codigo", "usuario", "curso", "monto", "moneda", "precio_usd", "cambio", "pasarela", "referencia", "detalle", "creada", "pagada_en")
    actions = ["dar_por_pagadas"]

    def has_add_permission(self, request) -> bool:
        return False            # las órdenes nacen de una compra; una cortesía se da como inscripción

    @admin.action(description="Dar por pagadas (pago recibido por otra vía) e inscribir")
    def dar_por_pagadas(self, request, ordenes) -> None:
        n = sum(o.pagar(referencia="manual", detalle=dict(manual=request.user.email)) for o in ordenes)
        self.message_user(request, f"{n} órdenes pagadas e inscritas.", messages.SUCCESS)
