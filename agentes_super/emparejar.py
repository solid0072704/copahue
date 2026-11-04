"""Decide qué productos de un supermercado corresponden a lo que quieres comprar."""

import re

from .unidades import normalizar

PALABRAS_VACIAS = {"de", "del", "la", "el", "los", "las", "en", "y", "para", "a", "al"}
# Si el nombre empieza con una de estas palabras (y no la pediste) es otro producto:
# «Dulce de Leche», «Galletas de Avena», «Jugo de Naranja», «Salsa de Tomate»…
DERIVADOS = {
    "dulce", "manjar", "galleta", "galletas", "helado", "sabor", "caramelo", "salsa", "sopa", "jugo", "nectar",
    "barra", "barras", "alfajor", "bombon", "bombones", "queque", "torta", "cereal", "flan", "postre", "crema",
}


def _raiz(palabra):
    if len(palabra) > 4 and palabra.endswith("es"):
        return palabra[:-2]
    if len(palabra) > 3 and palabra.endswith("s"):
        return palabra[:-1]
    return palabra


def tokens(texto):
    return [t for t in re.findall(r"[a-z0-9%]+", normalizar(texto)) if t not in PALABRAS_VACIAS]


def _contiene(palabras, token):
    r = _raiz(token)
    return any(p == token or _raiz(p) == r or (len(token) >= 4 and p.startswith(token)) for p in palabras)


def coincide(consulta, nombre):
    """Todas las palabras de la consulta están en el nombre y alguna aparece al inicio.

    Así «leche» encuentra «Leche Entera 1 L» pero no «Dulce de Leche».
    """
    buscadas = tokens(consulta)
    palabras = tokens(nombre)
    if not buscadas or not palabras:
        return False
    if not all(_contiene(palabras, t) for t in buscadas):
        return False
    if palabras[0] in DERIVADOS and not _contiene(buscadas, palabras[0]):
        return False
    return any(_contiene(palabras[:3], t) for t in buscadas)


def lista_texto(texto):
    return [x.strip() for x in str(texto or "").split(",") if x.strip()]


def cumple_filtros(item, producto):
    """Filtros del ítem: marcas aceptadas y palabras a excluir."""
    nombre = normalizar(f"{producto.nombre} {producto.marca}")
    for palabra in lista_texto(item.get("excluir")):
        if normalizar(palabra) in nombre:
            return False
    marcas = [normalizar(m) for m in lista_texto(item.get("marcas"))]
    if marcas and not any(m in nombre for m in marcas):
        return False
    return producto.disponible


def candidatos(item, productos):
    return [p for p in productos if coincide(item["consulta"], p.nombre) and cumple_filtros(item, p)]
