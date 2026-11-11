"""Indicadores en línea: valor de la UF del día.

Fuente: mindicador.cl, un servicio gratuito y sin clave que publica los indicadores del
Banco Central de Chile. Si no hay conexión o el servicio no responde, la plataforma sigue
funcionando con el último valor guardado.
"""

from datetime import date, datetime, timedelta, timezone

from . import cobranza

URL_UF = "https://mindicador.cl/api/uf"
FUENTE = "mindicador.cl (datos del Banco Central de Chile)"
META_FECHA = "uf_fecha"
META_FUENTE = "uf_fuente"
META_INTENTO = "uf_intento"
META_ERROR = "uf_error"
# La UF se mueve unos pocos pesos al día: un valor fuera de este rango es un error de la fuente.
UF_MINIMA, UF_MAXIMA = 20_000, 200_000
REINTENTAR_CADA = timedelta(hours=3)


class ErrorIndicador(Exception):
    pass


def valor_del_dia(datos, hoy):
    """Extrae (valor, fecha) de la respuesta de mindicador.cl: el valor de hoy o, si aún no está, el más reciente."""
    serie = datos.get("serie") if isinstance(datos, dict) else None
    if not isinstance(serie, list) or not serie:
        raise ErrorIndicador("La respuesta no trae la serie de la UF.")
    candidatos = []
    for punto in serie:
        try:
            fecha = date.fromisoformat(str(punto["fecha"])[:10])
            valor = float(punto["valor"])
        except (KeyError, TypeError, ValueError):
            continue
        if fecha <= hoy and UF_MINIMA <= valor <= UF_MAXIMA:
            candidatos.append((fecha, valor))
    if not candidatos:
        raise ErrorIndicador("La respuesta no trae un valor de UF utilizable.")
    fecha, valor = max(candidatos)
    if (hoy - fecha).days > 10:
        raise ErrorIndicador(f"El último valor publicado es del {fecha:%d/%m/%Y}: demasiado antiguo.")
    return valor, fecha


def consultar_uf(hoy, timeout=5, get=None):
    """Consulta la UF en línea. Devuelve (valor, fecha) o lanza ErrorIndicador."""
    if get is None:
        import requests

        get = requests.get
    try:
        respuesta = get(URL_UF, timeout=timeout, headers={"Accept": "application/json"})
        respuesta.raise_for_status()
        datos = respuesta.json()
    except ErrorIndicador:
        raise
    except Exception as error:  # sin conexión, tiempo agotado, respuesta que no es JSON…
        raise ErrorIndicador(f"No se pudo consultar la UF en línea ({type(error).__name__}).") from error
    return valor_del_dia(datos, hoy)


def estado_uf(db, cfg):
    """Lo que la plataforma sabe hoy de la UF: valor, de qué fecha es y de dónde salió."""
    return {
        "valor": cobranza.valor_uf(db, cfg),
        "fecha": db.get_meta(META_FECHA),
        "fuente": db.get_meta(META_FUENTE) or ("configuración inicial" if cfg.valor_uf else ""),
        "error": db.get_meta(META_ERROR) or "",
    }


def guardar_uf_manual(db, valor, hoy):
    db.set_meta(cobranza.META_UF, str(float(valor)))
    db.set_meta(META_FECHA, hoy.isoformat())
    db.set_meta(META_FUENTE, "ingresada a mano")
    db.set_meta(META_ERROR, "")


def actualizar_uf(db, cfg, hoy, forzar=False, get=None, ahora=None):
    """Actualiza la UF guardada si corresponde. Nunca lanza: devuelve 'actualizada', 'vigente', 'en_espera' o 'error'.

    Sin `forzar`, no consulta si ya tiene el valor de hoy ni si el último intento fallido fue hace poco
    (para que la app no se quede esperando en cada clic cuando no hay internet).
    """
    ahora = ahora or datetime.now(timezone.utc)
    if not forzar:
        if db.get_meta(META_FECHA) == hoy.isoformat():
            return "vigente"
        intento = db.get_meta(META_INTENTO)
        try:
            if intento and ahora - datetime.fromisoformat(intento) < REINTENTAR_CADA:
                return "en_espera"
        except ValueError:
            pass
    db.set_meta(META_INTENTO, ahora.isoformat())
    try:
        valor, fecha = consultar_uf(hoy, get=get)
    except ErrorIndicador as error:
        db.set_meta(META_ERROR, str(error))
        return "error"
    db.set_meta(cobranza.META_UF, str(valor))
    db.set_meta(META_FECHA, fecha.isoformat())
    db.set_meta(META_FUENTE, FUENTE)
    db.set_meta(META_ERROR, "")
    return "actualizada"
