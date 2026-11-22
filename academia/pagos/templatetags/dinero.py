"""Lo que las páginas necesitan para escribir precios: el de un curso para quien lo mira, y la nota que los explica."""

from django import template
from django.conf import settings

from academia.cursos.models import Perfil

from .. import cambio, precios

register = template.Library()


@register.simple_tag(takes_context=True)
def precio(context, curso) -> precios.Precio:
    """`{% precio curso as p %}`: el precio en dólares (`p.usd`) y en la moneda del país de quien mira (`p.en_su_moneda`)."""
    request = context.get("request")
    return precios.precio(curso, getattr(request, "user", None), request)


@register.inclusion_tag("pagos/_nota_de_precios.html", takes_context=True)
def nota_de_precios(context) -> dict:
    """La línea que acompaña a los precios: de dónde sale el valor en la moneda del alumno o cómo llegar a verlo."""
    request = context["request"]
    usuario = request.user
    local = precios.moneda_del_alumno(usuario, request)
    return dict(
        anonimo=not usuario.is_authenticated, sin_pais=usuario.is_authenticated and not Perfil.pais_de(usuario),
        moneda=local, referencia=local is not None and local.codigo != settings.MONEDA, de_cobro=precios.moneda(settings.MONEDA, request),
        fuente=cambio.ATRIBUCION if local is not None and local.automatica else None, vuelta=request.get_full_path())
