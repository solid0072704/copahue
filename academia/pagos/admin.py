from django.contrib import admin, messages

from .models import Orden


@admin.register(Orden)
class OrdenAdmin(admin.ModelAdmin):
    list_display = ("codigo", "usuario", "curso", "monto", "moneda", "estado", "pasarela", "creada", "pagada_en")
    list_filter = ("estado", "pasarela", "curso")
    search_fields = ("codigo", "referencia", "usuario__email")
    readonly_fields = ("codigo", "usuario", "curso", "monto", "moneda", "pasarela", "referencia", "detalle", "creada", "pagada_en")
    actions = ["dar_por_pagadas"]

    def has_add_permission(self, request) -> bool:
        return False            # las órdenes nacen de una compra; una cortesía se da como inscripción

    @admin.action(description="Dar por pagadas (pago recibido por otra vía) e inscribir")
    def dar_por_pagadas(self, request, ordenes) -> None:
        n = sum(o.pagar(referencia="manual", detalle=dict(manual=request.user.email)) for o in ordenes)
        self.message_user(request, f"{n} órdenes pagadas e inscritas.", messages.SUCCESS)
