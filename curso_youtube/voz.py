"""Convierte el texto del guion (como se lee en pantalla) en el texto que se envía a la voz.

El guion se escribe con cifras («11.800 millones», «18 %», «1,65 veces») porque así se
revisa y se subtitula mejor. Para la narración, las cifras se pasan a palabras y algunas
palabras se reescriben como suenan, de modo que cualquier modelo de voz las lea bien.
"""

from __future__ import annotations

import re

# Palabras que la voz en español podría leer mal: se envían escritas como suenan.
PRONUNCIACION = [
    (r"\bclaude\.ai\b", "clod punto a i"),
    (r"\bClaude\b", "Clod"),
    (r"\bEBITDA\b", "ébitda"),
    (r"\bebooks?\b", "íbuk"),
    (r"\bPDF\b", "pe de efe"),
    (r"\bCSV\b", "ce ese ve"),
    (r"\bDuPont\b", "Dupont"),
    (r"\bcontroller\b", "contróler"),
    (r"\bLinkedIn\b", "Línkedin"),
    (r"\bplugins\b", "pláguins"),
    (r"\bplugin\b", "pláguin"),
    (r"\bskills\b", "eskíls"),
    (r"\bskill\b", "eskil"),
    # nombres de los planes y de los equipos, que una voz en español tiende a leer letra por letra
    (r"\bFree\b", "Fri"),
    (r"\bTeam\b", "Tim"),
    (r"\bEnterprise\b", "Énterprais"),
    (r"\biPhone\b", "aifon"),
]

_UNIDADES = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
             "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho",
             "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro", "veinticinco",
             "veintiséis", "veintisiete", "veintiocho", "veintinueve"]
_DECENAS = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta", 8: "ochenta", 9: "noventa"}
_CENTENAS = {1: "ciento", 2: "doscientos", 3: "trescientos", 4: "cuatrocientos", 5: "quinientos",
             6: "seiscientos", 7: "setecientos", 8: "ochocientos", 9: "novecientos"}

# Sustantivos femeninos que pueden seguir a una cifra («21 semanas», «200 cuotas»).
_FEMENINOS = {"veces", "vez", "semanas", "semana", "horas", "hora", "líneas", "línea", "cosas", "partes",
              "páginas", "página", "cifras", "cuotas", "personas", "empresas", "palancas", "preguntas",
              "reglas", "cadenas", "señales", "alertas", "causas", "acciones", "decisiones", "formas",
              "razones", "lecturas", "columnas", "filas", "cuentas", "familias", "etapas"}
# Palabras que no son sustantivos: ante ellas la cifra se lee completa («de 1 a 1,19» -> «de uno a…»).
_NO_SUSTANTIVOS = {"a", "al", "de", "del", "y", "o", "u", "e", "en", "por", "con", "sin", "que", "es", "son",
                   "fue", "era", "eran", "se", "lo", "la", "las", "los", "el", "un", "una", "para", "como",
                   "más", "menos", "contra", "sobre", "entre", "hasta", "desde", "pero", "porque", "cuando",
                   "si", "no", "ni", "ya", "hay", "da", "dan", "queda", "quedan", "quedaban", "arriba", "abajo"}


def _menor_mil(n: int, femenino: bool = False, apocope: bool = False) -> str:
    """0 a 999 en palabras."""
    if n == 100:
        return "cien"
    partes = []
    c, r = divmod(n, 100)
    if c:
        palabra = _CENTENAS[c]
        if femenino and c > 1:
            palabra = palabra[:-2] + "as"
        partes.append(palabra)
    if r or not c:
        if r < 30:
            palabra = _UNIDADES[r]
        else:
            d, u = divmod(r, 10)
            palabra = _DECENAS[d] + (" y " + _UNIDADES[u] if u else "")
        if palabra.endswith("uno"):
            if femenino:
                palabra = palabra[:-3] + "una"
            elif apocope:
                palabra = palabra[:-3] + ("ún" if palabra.startswith("veinti") else "un")
        partes.append(palabra)
    return " ".join(partes)


def entero_a_palabras(n: int, femenino: bool = False, apocope: bool = False) -> str:
    """Entero no negativo (hasta 999.999.999.999) en palabras."""
    if n < 0:
        return "menos " + entero_a_palabras(-n, femenino, apocope)
    if n < 1000:
        return _menor_mil(n, femenino, apocope)
    partes = []
    miles_de_millones, resto = divmod(n, 1_000_000_000)
    millones, resto = divmod(resto, 1_000_000)
    miles, unidades = divmod(resto, 1000)
    if miles_de_millones:
        partes.append(("" if miles_de_millones == 1 else _menor_mil(miles_de_millones, apocope=True) + " ") + "mil")
    if millones or miles_de_millones:
        if millones == 1 and not miles_de_millones:
            partes.append("un millón")
        else:
            partes.append((_menor_mil(millones, apocope=True) + " " if millones else "") + "millones")
    if miles:
        partes.append(("" if miles == 1 else _menor_mil(miles, femenino, apocope=True) + " ") + "mil")
    if unidades:
        partes.append(_menor_mil(unidades, femenino, apocope))
    return " ".join(partes)


def _decimales_a_palabras(dec: str) -> str:
    if dec.startswith("0"):
        return " ".join(_UNIDADES[int(d)] for d in dec)
    return entero_a_palabras(int(dec))


_NUMERO = re.compile(r"(?<![\w,.])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(?![\d])")


def numeros_a_palabras(texto: str) -> str:
    def reemplazo(m: re.Match) -> str:
        entero = int(m.group(1).replace(".", ""))
        dec = m.group(2)
        resto = texto[m.end():]
        siguiente = re.match(r"\s*([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+|%)", resto)
        palabra = siguiente.group(1).lower() if siguiente else ""
        if dec is not None:
            palabras = f"{entero_a_palabras(entero)} coma {_decimales_a_palabras(dec)}"
        else:
            femenino = palabra in _FEMENINOS
            es_sustantivo = bool(palabra) and palabra != "%" and palabra not in _NO_SUSTANTIVOS
            palabras = entero_a_palabras(entero, femenino=femenino, apocope=es_sustantivo and not femenino)
        antes = texto[:m.start()]
        if not antes.strip() or re.search(r"[.?!]\s+$", antes):  # la cifra abre la frase
            palabras = palabras[0].upper() + palabras[1:]
        return palabras

    return _NUMERO.sub(reemplazo, texto)


def a_voz(texto: str) -> str:
    """Texto listo para la voz: pronunciaciones, cifras en palabras y signos que no se leen."""
    t = texto
    for patron, reemplazo in PRONUNCIACION:
        t = re.sub(patron, reemplazo, t)
    t = numeros_a_palabras(t)
    t = re.sub(r"\s*%", " por ciento", t)
    t = t.replace("«", "").replace("»", "")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


CARACTERES_POR_SEGUNDO = 17.1  # medido con la narración completa de la voz clonada (60.885 caracteres en 59,2 min)


def duracion_estimada(texto: str) -> float:
    """Segundos aproximados de narración para un texto del guion."""
    return len(a_voz(texto)) / CARACTERES_POR_SEGUNDO
