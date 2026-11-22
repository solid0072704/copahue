from django import template

register = template.Library()


@register.filter
def miles(n) -> str:
    """50000 -> «50.000»: el separador de miles que se usa en Chile, sin depender del idioma del navegador."""
    try:
        return f"{int(n):,}".replace(",", ".")
    except (TypeError, ValueError):
        return str(n)
