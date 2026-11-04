"""Lectura de formatos ("1 L", "6 x 125 g", "12 un") y precios por unidad comparables."""

import re
import unicodedata

# unidad escrita -> (unidad base, factor)
UNIDADES = {
    "kg": ("g", 1000), "kgs": ("g", 1000), "kilo": ("g", 1000), "kilos": ("g", 1000),
    "g": ("g", 1), "gr": ("g", 1), "grs": ("g", 1), "gramos": ("g", 1),
    "l": ("ml", 1000), "lt": ("ml", 1000), "lts": ("ml", 1000), "litro": ("ml", 1000), "litros": ("ml", 1000),
    "ml": ("ml", 1), "cc": ("ml", 1), "cm3": ("ml", 1),
    "un": ("un", 1), "und": ("un", 1), "unid": ("un", 1), "unidad": ("un", 1), "unidades": ("un", 1),
    "u": ("un", 1), "rollos": ("un", 1), "rollo": ("un", 1), "huevos": ("un", 1), "capsulas": ("un", 1),
    "bolsas": ("un", 1), "panales": ("un", 1),
}
NOMBRE_BASE = {"g": "kg", "ml": "L", "un": "un"}
FACTOR_MOSTRAR = {"g": 1000, "ml": 1000, "un": 1}

_NUM = r"\d+(?:[.,]\d+)?"
_PATRON = re.compile(
    rf"(?:(\d+)\s*[x×]\s*)?({_NUM})\s*({'|'.join(sorted(UNIDADES, key=len, reverse=True))})\b",
    re.IGNORECASE,
)
_PACK = re.compile(r"\bpack\s*(?:de\s*)?(\d+)\b|\b(\d+)\s*[x×]\s*$", re.IGNORECASE)


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", str(texto or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", texto.lower()).strip()


def _numero(texto, unidad):
    texto = texto.strip()
    # "1.000 g" en Chile es mil gramos
    if re.fullmatch(r"\d{1,3}\.\d{3}", texto) and UNIDADES[unidad][0] in ("g", "ml"):
        return float(texto.replace(".", ""))
    return float(texto.replace(",", "."))


def parsear_formato(texto):
    """Devuelve (cantidad_en_unidad_base, unidad_base) o (None, None).

    >>> parsear_formato("Yogurt Batido 6 x 125 g")
    (750.0, 'g')
    """
    limpio = normalizar(texto)
    encontrados = []
    for m in _PATRON.finditer(limpio):
        multiplicador = int(m.group(1)) if m.group(1) else 1
        unidad = m.group(3).lower()
        base, factor = UNIDADES[unidad]
        try:
            cantidad = _numero(m.group(2), unidad) * factor * multiplicador
        except ValueError:
            continue
        if cantidad > 0:
            encontrados.append((cantidad, base, m.group(1) is not None))
    if not encontrados:
        return None, None
    # Preferimos peso/volumen sobre conteo ("4 rollos 50 m" -> rollos; "6 un 1 L" -> 6 L)
    con_medida = [e for e in encontrados if e[1] in ("g", "ml")]
    cantidad, base, tiene_x = (con_medida or encontrados)[0]
    if base != "un" and not tiene_x:
        pack = _PACK.search(limpio)
        if pack:
            cantidad *= int(pack.group(1) or pack.group(2))
        elif any(e[1] == "un" for e in encontrados) and con_medida:
            unidades = next(e[0] for e in encontrados if e[1] == "un")
            if 1 < unidades <= 48:
                cantidad *= unidades
    return round(cantidad, 3), base


def precio_por_unidad(precio, cantidad, unidad):
    """Precio por kg, L o unidad (para mostrar)."""
    if not precio or not cantidad or not unidad:
        return None
    return precio / cantidad * FACTOR_MOSTRAR[unidad]


def texto_formato(cantidad, unidad):
    if not cantidad or not unidad:
        return ""
    if unidad in ("g", "ml") and cantidad >= 1000:
        valor = cantidad / 1000
        return f"{valor:g} {'kg' if unidad == 'g' else 'L'}"
    return f"{cantidad:g} {unidad}"


def pesos(valor):
    if valor is None:
        return "—"
    return "$" + f"{round(valor):,}".replace(",", ".")
