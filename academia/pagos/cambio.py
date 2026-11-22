"""Los tipos de cambio: cuántas unidades de cada moneda vale un dólar.

Vienen de un servicio público, la dirección abierta de ExchangeRate-API: no pide clave, publica valores una vez al día
y exige nombrar la fuente («Rates By Exchange Rate API») en las páginas donde se usan. Hay dos maneras de mantenerlos
al día, y conviven:

- sola: la primera página que necesita precios después de un día sin datos nuevos los va a buscar (`al_dia`);
- a pedido: `manage.py actualizar_monedas`, a mano o como tarea programada, y el botón del panel.

Un tipo de cambio marcado como fijo en el panel no se toca. Y un valor que se aleja mucho del guardado no se aplica
solo: queda anotado en la moneda para que una persona lo mire, porque con ese número se calcula lo que se cobra.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from datetime import timezone as utc
from decimal import Decimal, InvalidOperation

import requests
from django.conf import settings
from django.utils import timezone

from .models import Moneda

registro = logging.getLogger(__name__)
VIGENCIA = timedelta(hours=25)          # la fuente publica cada 24 horas y unos minutos: se le pregunta cuando ya lo hizo
ESPERA = 3600                           # segundos entre dos intentos de un mismo proceso: la fuente pide no pasar de uno por hora
SALTO = Decimal("0.30")                 # una diferencia mayor que esta con el valor guardado no se aplica sola
TOPE = Decimal(10) ** 11                # ninguna moneda vale tanto por dólar: un número así es un error de la fuente
ATRIBUCION = ("Rates By Exchange Rate API", "https://www.exchangerate-api.com")       # el texto y el enlace que pide la fuente
_ultimo_intento: float | None = None


def actualizar() -> dict[str, list[str]]:
    """Trae los tipos de cambio del día y los guarda en las monedas que no están fijas.

    Devuelve los códigos de las monedas `actualizadas`, las `rechazadas` (el valor nuevo está demasiado lejos del
    guardado) y las que quedaron `sin_dato` (la fuente no las trae). Si la fuente no responde, o responde otra cosa,
    lanza la excepción y no cambia nada."""
    respuesta = requests.get(settings.CAMBIO_URL, timeout=5)
    respuesta.raise_for_status()
    datos = respuesta.json()
    tasas = datos.get("rates") if isinstance(datos, dict) else None
    if not isinstance(tasas, dict) or datos.get("result") != "success" or datos.get("base_code") != "USD":
        raise ValueError("La fuente de tipos de cambio respondió algo inesperado")
    try:                                # la fecha de los valores es la que dice la fuente, no la de la consulta
        fecha = min(datetime.fromtimestamp(int(datos["time_last_update_unix"]), utc.utc), timezone.now())
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        fecha = timezone.now()

    resultado: dict[str, list[str]] = dict(actualizadas=[], rechazadas=[], sin_dato=[])
    for m in Moneda.objects.filter(fija=False):
        try:
            nuevo = Decimal(str(tasas[m.codigo])).quantize(Decimal("0.000001"))
        except (KeyError, InvalidOperation):
            nuevo = None
        if nuevo is None or not nuevo.is_finite() or not 0 < nuevo < TOPE:
            resultado["sin_dato"].append(m.codigo)
        elif m.por_dolar and abs(nuevo - m.por_dolar) / m.por_dolar > SALTO:
            m.observacion = (f"El {fecha:%d-%m-%Y} la fuente informó {nuevo.normalize():f} por dólar, muy lejos del valor guardado: "
                             "no se aplicó. Si es correcto, escríbelo a mano.")[:200]
            m.save(update_fields=["observacion"])
            resultado["rechazadas"].append(m.codigo)
            registro.error("Tipo de cambio de %s no aplicado: la fuente informó %s y el valor guardado es %s", m.codigo, nuevo, m.por_dolar)
        else:
            m.por_dolar, m.actualizada, m.observacion = nuevo, fecha, ""
            m.save(update_fields=["por_dolar", "actualizada", "observacion"])
            resultado["actualizadas"].append(m.codigo)
    return resultado


def al_dia(monedas) -> bool:
    """Si hace más de un día que no llegan tipos de cambio, los va a buscar. Devuelve si hubo valores nuevos.

    Lo llama cualquier página que muestra precios, así que nunca la interrumpe: si la fuente falla se sigue con los
    valores guardados, y cada proceso vuelve a intentarlo a lo más una vez por hora."""
    global _ultimo_intento
    if not settings.CAMBIO_AUTOMATICO:
        return False
    libres = [m for m in monedas if not m.fija]
    if not libres:
        return False
    ultima = max((m.actualizada for m in libres if m.actualizada), default=None)
    if ultima and timezone.now() - ultima < VIGENCIA:
        return False
    ahora = time.monotonic()
    if _ultimo_intento is not None and ahora - _ultimo_intento < ESPERA:
        return False
    _ultimo_intento = ahora
    try:
        return bool(actualizar()["actualizadas"])
    except (requests.RequestException, ValueError) as e:        # la fuente no respondió, o respondió otra cosa
        registro.warning("No se pudieron actualizar los tipos de cambio; se sigue con los guardados (%s)", e)
    except Exception:           # noqa: BLE001 — pase lo que pase aquí, la página que muestra precios no se cae por esto
        registro.exception("Falló la actualización de los tipos de cambio; se sigue con los guardados")
    return False
