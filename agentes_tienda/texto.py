"""Utilidades de texto: slugs, limpieza de títulos de proveedor, precios y formatos."""

import math
import re
import unicodedata


def sin_acentos(texto):
    texto = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in texto if not unicodedata.combining(c))


def normalizar(texto):
    return re.sub(r"\s+", " ", sin_acentos(texto).lower()).strip()


_COLGANTES = {"de", "del", "la", "el", "los", "las", "para", "con", "sin", "y", "o", "en", "por", "a", "al", "e", "un", "una"}


def quitar_colgantes(texto, sep=" "):
    """Quita preposiciones o conjunciones que quedan sueltas al final tras cortar: 'contador digital de' -> 'contador digital'."""
    partes = texto.split(sep)
    while len(partes) > 1 and normalizar(partes[-1]).strip(",;:-") in _COLGANTES:
        partes.pop()
    return sep.join(partes)


def slugify(texto, maximo=70):
    s = re.sub(r"[^a-z0-9]+", "-", normalizar(texto)).strip("-")
    if len(s) > maximo:
        s = quitar_colgantes(s[:maximo].rsplit("-", 1)[0], "-")
    return s or "producto"


def truncar(texto, maximo, sufijo="…"):
    """Corta en el último espacio antes de `maximo` caracteres."""
    texto = re.sub(r"\s+", " ", texto or "").strip()
    if len(texto) <= maximo:
        return texto
    corte = quitar_colgantes(texto[: maximo - len(sufijo)].rsplit(" ", 1)[0]).rstrip(",;:-–—|·")
    return corte + sufijo


def capitalizar(texto):
    texto = (texto or "").strip()
    return texto[:1].upper() + texto[1:] if texto else texto


def contar_palabras(texto):
    return len(re.findall(r"\w+", texto or ""))


# --- Limpieza de títulos de AliExpress/CJ (vienen cargados de palabras de relleno) ---
_RELLENO = [
    r"\b20[12]\d\b", r"\bnew\b", r"\bnuevo(s|a|as)?\b", r"\bhot( sale)?\b", r"\bventa caliente\b",
    r"\bfree shipping\b", r"\benv[ií]o gratis\b", r"\boriginal(es)?\b", r"\bhigh quality\b",
    r"\balta calidad\b", r"\bdropshipping\b", r"\bdrop shipping\b", r"\bwholesale\b", r"\bal por mayor\b",
    r"\bfactory\b", r"\bbest seller\b", r"\bm[aá]s vendido(s)?\b", r"\bpromo(ci[oó]n)?\b", r"\boferta\b",
    r"\b\d+\s?(pcs|pzs|unids?)\b", r"\bcreativ[oa]s?\b",
]
_RELLENO_RE = re.compile("|".join(_RELLENO), re.IGNORECASE)


def limpiar_titulo(titulo, maximo=56):
    """'2024 Nuevo Masajeador eléctrico de cuello, almohada Shiatsu, envío gratis' -> 'Masajeador eléctrico de cuello'."""
    t = re.sub(r"[【\[\(].*?[】\]\)]", " ", titulo or "")
    t = _RELLENO_RE.sub(" ", t)
    t = re.sub(r"[|/+•·]+", ",", t)
    t = re.sub(r"\s+", " ", t).strip(" ,.-–")
    partes = [p.strip(" -–") for p in t.split(",") if p.strip(" -–")]
    if not partes:
        return capitalizar(truncar(titulo or "Producto", maximo, ""))
    nombre = partes[0]
    # Si la primera parte es muy corta, sumamos la siguiente ("Masajeador" + "cervical eléctrico").
    if len(nombre) < 18 and len(partes) > 1:
        nombre = f"{nombre} {partes[1]}"
    # Quita palabras repetidas consecutivas ("de de", "Kit Kit").
    nombre = re.sub(r"\b(\w+)( \1\b)+", r"\1", nombre, flags=re.IGNORECASE)
    return capitalizar(truncar(nombre, maximo, ""))


def caracteristicas_de_titulo(titulo, maximo=6):
    """Las frases separadas por coma del título original suelen ser características."""
    t = re.sub(r"[【\[\(].*?[】\]\)]", " ", titulo or "")
    t = _RELLENO_RE.sub(" ", t)
    partes = [re.sub(r"\s+", " ", p).strip(" -–.") for p in re.split(r"[,|/+•·]", t)]
    vistas, salida = set(), []
    for p in partes[1:]:
        clave = normalizar(p)
        if len(p) >= 4 and clave not in vistas:
            vistas.add(clave)
            salida.append(capitalizar(p))
        if len(salida) >= maximo:
            break
    return salida


# --- Números y precios ---
def a_float(valor, defecto=None):
    """'12.50', 'US $12,50', '96.5%', '1.23-4.56' (toma el menor) -> float."""
    if valor is None or valor == "":
        return defecto
    if isinstance(valor, (int, float)):
        return float(valor)
    texto = str(valor).replace("%", "").strip()
    numeros = re.findall(r"\d+(?:[.,]\d+)?", texto)
    if not numeros:
        return defecto
    try:
        return min(float(n.replace(",", ".")) for n in numeros)
    except ValueError:
        return defecto


def redondear_precio(valor, terminacion=990):
    """Lleva un precio en CLP a la terminación psicológica más cercana hacia arriba: 23.417 -> 23.990."""
    if valor <= 0:
        return 0
    if not terminacion:
        return int(math.ceil(valor))
    paso = 10 ** len(str(terminacion))  # 990 -> 1000
    base = math.floor(valor / paso) * paso + terminacion
    return int(base if base >= valor else base + paso)


def miles(n):
    """1234567 -> '1.234.567' (formato chileno)."""
    try:
        return f"{int(round(n)):,}".replace(",", ".")
    except (TypeError, ValueError):
        return ""


def precio_texto(n, moneda="CLP"):
    if not n:
        return ""
    if moneda == "CLP":
        return f"${miles(n)}"
    return f"{moneda} {n:,.2f}"


def cantidad_corta(n):
    """1500 -> '1,5 mil', 12000 -> '12 mil'."""
    if not n:
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} M".replace(".0 ", " ").replace(".", ",")
    if n >= 10_000:
        return f"{n // 1000} mil"
    if n >= 1000:
        return f"{n / 1000:.1f} mil".replace(".0 ", " ").replace(".", ",")
    return str(int(n))
