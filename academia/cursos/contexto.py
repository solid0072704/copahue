from django.conf import settings


def sitio(request) -> dict:
    """Lo que todas las páginas necesitan saber del sitio."""
    return dict(SITIO_NOMBRE=settings.SITIO_NOMBRE, SITIO_BAJADA=settings.SITIO_BAJADA, CON_GOOGLE=settings.CON_GOOGLE)
