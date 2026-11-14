"""Investigación de palabras clave con el autocompletado de Google (lo que la gente en Chile escribe de verdad)."""

import json
import logging
import re
import time

import requests

from .almacen import ahora, dias_desde
from .texto import normalizar

log = logging.getLogger(__name__)

URL = "https://suggestqueries.google.com/complete/search"
VACIAS = {
    "de", "del", "la", "el", "los", "las", "para", "con", "sin", "y", "o", "en", "por", "a", "un", "una",
    "tu", "su", "al", "e", "x", "mini", "set", "kit", "pro", "max", "plus",
}
MODIFICADORES = ("{b}", "comprar {b}", "{b} precio", "{b} chile", "mejor {b}")


class Investigador:
    """Consulta el autocompletado con caché de 30 días y un límite de consultas por ciclo."""

    def __init__(self, almacen, cfg, usar_red=True, max_consultas=80):
        self.almacen = almacen
        self.cfg = cfg
        self.activo = usar_red and cfg.sugerencias_google
        self.restantes = max_consultas
        self.sesion = requests.Session()
        self.sesion.headers["User-Agent"] = "Mozilla/5.0 (compatible; tienda-seo-bot/1.0)"
        self._fallos = 0

    def sugerencias(self, consulta):
        clave = normalizar(consulta)
        guardado = self.almacen.keywords.get(clave)
        if guardado and dias_desde(guardado.get("fecha")) < 30:
            return guardado.get("s", [])
        if not self.activo or self.restantes <= 0 or self._fallos >= 3:
            return guardado.get("s", []) if guardado else []
        self.restantes -= 1
        try:
            idioma = self.cfg.idioma.split("-")[0]
            r = self.sesion.get(
                URL,
                params={"client": "firefox", "hl": idioma, "gl": self.cfg.pais.lower(), "ie": "utf-8", "oe": "utf-8", "q": consulta},
                timeout=10,
            )
            r.raise_for_status()
            datos = json.loads(r.content.decode("utf-8", errors="replace"))
            lista = [s for s in datos[1] if isinstance(s, str)][:10]
            time.sleep(0.4)
        except Exception as e:
            self._fallos += 1
            log.info("Autocompletado no disponible (%s): %s", consulta, e)
            return guardado.get("s", []) if guardado else []
        self.almacen.keywords[clave] = {"fecha": ahora(), "s": lista}
        return lista


def termino_base(nombre, palabras=3):
    """'Masajeador cervical eléctrico para cuello' -> 'masajeador cervical electrico'."""
    tokens = [t for t in re.findall(r"[a-záéíóúñü]+", (nombre or "").lower()) if t not in VACIAS and len(t) > 2]
    return " ".join(tokens[:palabras])


def termino_natural(nombre, palabras=3):
    """'Pistola de masaje muscular profunda' -> 'pistola de masaje muscular' (se lee como lo escribe una persona)."""
    salida, contenido = [], 0
    for t in re.findall(r"[\wáéíóúñü]+", (nombre or "").lower()):
        if t.isdigit():
            break
        salida.append(t)
        if t not in VACIAS:
            contenido += 1
        if contenido >= palabras:
            break
    while salida and salida[-1] in VACIAS:
        salida.pop()
    return " ".join(salida)


def investigar(investigador, nombre):
    """Devuelve {'principal': str, 'secundarias': [...], 'interes': int}."""
    base3 = termino_base(nombre, 3)
    base2 = termino_base(nombre, 2) or base3
    if not base2:
        return {"principal": termino_natural(nombre) or normalizar(nombre), "secundarias": [], "interes": 0}
    encontradas = []
    for base in dict.fromkeys([base2, base3]):
        for plantilla in MODIFICADORES:
            encontradas += investigador.sugerencias(plantilla.format(b=base))
            if plantilla == "{b}" and not encontradas and base == base2:
                break  # nadie busca esto: no gastamos más consultas
    tokens_base = set(normalizar(base2).split())
    vistas, relevantes = set(), []
    for s in encontradas:
        n = normalizar(s)
        if n not in vistas and tokens_base & set(n.split()):
            vistas.add(n)
            relevantes.append(s.strip())
    # La principal: la búsqueda más corta que contiene el término base (suele ser la de más volumen).
    contienen = [s for s in relevantes if tokens_base <= set(normalizar(s).split())]
    principal = min(contienen, key=len) if contienen else (termino_natural(nombre) or base2)
    secundarias = [s for s in relevantes if normalizar(s) != normalizar(principal)][:10]
    return {"principal": principal, "secundarias": secundarias, "interes": len(relevantes)}
