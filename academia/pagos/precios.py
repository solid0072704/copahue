"""Los precios: cada curso tiene el suyo en dólares, y de ahí sale lo que ve cada alumno y lo que se cobra.

- Todos ven el precio en dólares. Quien tiene cuenta lo ve además en la moneda de su país —el que indicó al
  registrarse—, al tipo de cambio guardado (`cambio.py`).
- El cobro se hace en la moneda de las pasarelas (`ACADEMIA_MONEDA`; por defecto, pesos chilenos): el precio en
  dólares por el tipo de cambio de esa moneda. Para quien vive en otro país, el precio en su moneda es una
  referencia: la conversión final la hace su banco o su tarjeta.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from django.conf import settings

from academia.cursos import paises
from academia.cursos.models import Perfil

from . import cambio
from .models import Moneda

ESPACIO = " "          # espacio que no corta la línea: el símbolo, el número y el código van siempre juntos


# --- Monedas ----------------------------------------------------------------
def monedas(request=None) -> dict[str, Moneda]:
    """Todas las monedas, por código. Se leen una vez por página, y de paso se revisa que los tipos de cambio estén al día."""
    if request is not None and hasattr(request, "_monedas"):
        return request._monedas
    tabla = {m.codigo: m for m in Moneda.objects.all()}
    if cambio.al_dia(tabla.values()):
        tabla = {m.codigo: m for m in Moneda.objects.all()}
    tabla.setdefault("USD", Moneda(codigo="USD", nombre="dólares", simbolo="US$", decimales=2, por_dolar=Decimal(1), fija=True))
    if request is not None:
        request._monedas = tabla
    return tabla


def moneda(codigo: str, request=None) -> Moneda:
    """La moneda de ese código. Si ya no está en el panel (una orden antigua), una genérica que solo sirve para escribir el monto."""
    return monedas(request).get(codigo) or Moneda(codigo=codigo, nombre=codigo, simbolo="$", decimales=2)


def _redondear(monto: Decimal, decimales: int) -> Decimal:
    return monto.quantize(Decimal(1).scaleb(-decimales), rounding=ROUND_HALF_UP)


# --- Cómo se escribe un monto -----------------------------------------------
def numero(monto: Decimal, decimales: int) -> str:
    """1234567.5 con 2 decimales → «1.234.567,50»: miles con punto y decimales con coma, sin depender del navegador."""
    entero, _, fraccion = f"{_redondear(monto, decimales):,.{decimales}f}".partition(".")
    return entero.replace(",", ".") + (f",{fraccion}" if fraccion else "")


def formato(monto, en: Moneda, aproximado: bool = False, decimales: int | None = None) -> str:
    """El monto como lo lee una persona: «US$ 99», «$100.394 CLP», «S/ 341».

    Los centavos se muestran solo si los hay y, en un valor aproximado, solo en montos chicos. Una moneda cuyo
    símbolo es el signo peso a secas lleva además su código, porque ese signo lo comparten muchos países.
    `decimales` reemplaza a los de la moneda: sirve para escribir un tipo de cambio («$1.014,08 CLP»)."""
    monto, decimales = Decimal(str(monto)), en.decimales if decimales is None else decimales
    exacto = _redondear(monto, decimales)
    sin_centavos = exacto == exacto.to_integral_value() or (aproximado and abs(exacto) >= 100)
    texto = numero(monto, 0 if sin_centavos else decimales)
    if en.simbolo.strip() == "$":
        return f"${texto}{ESPACIO}{en.codigo}"
    return f"{en.simbolo.strip()}{ESPACIO}{texto}"


# --- Lo que se muestra ------------------------------------------------------
@dataclass(frozen=True)
class Precio:
    """El precio de un curso para quien mira la página."""
    gratuito: bool
    usd: str = ""                       # «US$ 99»
    local: str = ""                     # «$100.394 CLP»: en la moneda del país del alumno; vacío si es el dólar o no se conoce su cambio
    aproximado: bool = False            # esa moneda no es la del cobro: el valor es una referencia
    moneda: Moneda | None = None        # la moneda del país del alumno

    @property
    def en_su_moneda(self) -> str:
        """«$100.394 CLP» o, si es una referencia, «≈ $1.893 MXN»."""
        return f"≈{ESPACIO}{self.local}" if self.aproximado and self.local else self.local

    def __str__(self) -> str:
        """En una línea: «US$ 99 · $100.394 CLP», «US$ 99 · ≈ $1.893 MXN», «US$ 99» o «Gratis»."""
        return "Gratis" if self.gratuito else " · ".join(x for x in (self.usd, self.en_su_moneda) if x)


def moneda_del_alumno(usuario, request=None) -> Moneda | None:
    """La moneda del país que el alumno indicó, si no es el dólar y se conoce su tipo de cambio."""
    local = monedas(request).get(paises.moneda_de(Perfil.pais_de(usuario)))
    return local if local is not None and local.codigo != "USD" and local.por_dolar else None


def precio(curso, usuario=None, request=None) -> Precio:
    usd = Decimal(str(curso.precio_usd))
    if not usd:
        return Precio(gratuito=True)
    en_dolares = formato(usd, moneda("USD", request))
    local = moneda_del_alumno(usuario, request)
    if local is None:
        return Precio(False, en_dolares)
    aproximado = local.codigo != settings.MONEDA
    return Precio(False, en_dolares, formato(usd * local.por_dolar, local, aproximado), aproximado, local)


# --- Lo que se cobra --------------------------------------------------------
@dataclass(frozen=True)
class Cobro:
    """Lo que se le pide a la pasarela por un curso."""
    monto: Decimal                      # en la moneda de las pasarelas, con sus decimales
    moneda: Moneda
    cambio: Decimal                     # unidades de esa moneda por dólar con que se calculó

    def __str__(self) -> str:
        return formato(self.monto, self.moneda)

    @property
    def tipo_de_cambio(self) -> str:
        """«US$ 1 = $1.014,08 CLP»; vacío si se cobra en dólares."""
        return "" if self.moneda.codigo == "USD" else f"US${ESPACIO}1 = {formato(self.cambio, self.moneda, decimales=2)}"


def cobro(curso, request=None) -> Cobro | None:
    """Lo que se cobra por el curso, en la moneda de las pasarelas. None si no se conoce su tipo de cambio."""
    en = monedas(request).get(settings.MONEDA)
    if en is None or not en.por_dolar:
        return None
    return Cobro(_redondear(Decimal(str(curso.precio_usd)) * en.por_dolar, en.decimales), en, en.por_dolar)


def para_pasarela(monto: Decimal) -> int | float:
    """El monto como número para enviarlo a una pasarela: entero si no tiene centavos (un peso chileno no los tiene)."""
    return int(monto) if monto == monto.to_integral_value() else float(monto)


def mismo_monto(informado, monto: Decimal) -> bool:
    """Si lo que la pasarela dice haber cobrado es exactamente el monto de la orden."""
    try:
        return _redondear(Decimal(str(informado)), 2) == _redondear(Decimal(monto), 2)
    except (InvalidOperation, TypeError, ValueError):
        return False
