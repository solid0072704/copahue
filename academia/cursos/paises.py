"""Los países que se ofrecen al crear la cuenta y la moneda en que cada uno ve los precios, además del dólar.

Donde el dólar es la moneda de uso (Ecuador, El Salvador, Panamá, Puerto Rico, Estados Unidos) o los precios se
acostumbran a mirar en dólares (Cuba, Venezuela), el alumno ve solo el precio en dólares. Para agregar un país basta
una línea aquí; si su moneda es nueva, se crea además en el panel (Ventas → Monedas).
"""

from __future__ import annotations

PAISES = [      # (código ISO del país, nombre, código ISO de su moneda)
    ("AR", "Argentina", "ARS"), ("BO", "Bolivia", "BOB"), ("BR", "Brasil", "BRL"), ("CA", "Canadá", "CAD"),
    ("CL", "Chile", "CLP"), ("CO", "Colombia", "COP"), ("CR", "Costa Rica", "CRC"), ("CU", "Cuba", "USD"),
    ("EC", "Ecuador", "USD"), ("SV", "El Salvador", "USD"), ("ES", "España", "EUR"), ("US", "Estados Unidos", "USD"),
    ("GT", "Guatemala", "GTQ"), ("HN", "Honduras", "HNL"), ("MX", "México", "MXN"), ("NI", "Nicaragua", "NIO"),
    ("PA", "Panamá", "USD"), ("PY", "Paraguay", "PYG"), ("PE", "Perú", "PEN"), ("PR", "Puerto Rico", "USD"),
    ("DO", "República Dominicana", "DOP"), ("UY", "Uruguay", "UYU"), ("VE", "Venezuela", "USD"),
]
OTRO = "XX"     # «Otro país»: ve los precios en dólares
OPCIONES = [(codigo, nombre) for codigo, nombre, _ in PAISES] + [(OTRO, "Otro país")]
NOMBRES = dict(OPCIONES)
MONEDAS = {codigo: moneda for codigo, _, moneda in PAISES}


def moneda_de(pais: str) -> str:
    """El código de la moneda en que ve los precios quien vive en ese país; el dólar si no se sabe."""
    return MONEDAS.get(pais, "USD")
