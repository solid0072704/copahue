"""Fechas, días hábiles y calendario tributario chileno.

Las reglas siguen los plazos habituales (F29 al día 20 con declaración electrónica,
Previred al 13, Libro de Remuneraciones Electrónico al 15, F22 al 30 de abril) y se
corren al siguiente día hábil. Verifica cada año el calendario oficial del SII y
agrega feriados nuevos en FERIADOS.
"""

from datetime import date, datetime, timedelta

MESES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

# Feriados nacionales de Chile (irrenunciables y legales) 2025-2027.
FERIADOS = {
    date(2025, 1, 1), date(2025, 4, 18), date(2025, 4, 19), date(2025, 5, 1), date(2025, 5, 21),
    date(2025, 6, 20), date(2025, 6, 29), date(2025, 7, 16), date(2025, 8, 15), date(2025, 9, 18),
    date(2025, 9, 19), date(2025, 10, 12), date(2025, 10, 31), date(2025, 11, 1), date(2025, 12, 8),
    date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 4, 3), date(2026, 4, 4), date(2026, 5, 1), date(2026, 5, 21),
    date(2026, 6, 21), date(2026, 6, 29), date(2026, 7, 16), date(2026, 8, 15), date(2026, 9, 18),
    date(2026, 9, 19), date(2026, 10, 12), date(2026, 10, 31), date(2026, 11, 1), date(2026, 12, 8),
    date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 3, 26), date(2027, 3, 27), date(2027, 5, 1), date(2027, 5, 21),
    date(2027, 6, 21), date(2027, 6, 28), date(2027, 7, 16), date(2027, 8, 15), date(2027, 9, 17),
    date(2027, 9, 18), date(2027, 9, 19), date(2027, 10, 11), date(2027, 10, 31), date(2027, 11, 1),
    date(2027, 12, 8), date(2027, 12, 25),
}

# Obligaciones que genera Copahue. `servicio` indica qué servicio contratado la activa.
TIPOS = {
    "F29": {
        "nombre": "F29",
        "descripcion": "Declaración mensual de IVA y PPM",
        "organismo": "SII",
        "servicio": "iva",
    },
    "PREVIRED": {
        "nombre": "Previred",
        "descripcion": "Pago de cotizaciones previsionales",
        "organismo": "Previred",
        "servicio": "remuneraciones",
    },
    "LRE": {
        "nombre": "LRE",
        "descripcion": "Libro de Remuneraciones Electrónico",
        "organismo": "Dirección del Trabajo",
        "servicio": "remuneraciones",
    },
    "F22": {
        "nombre": "F22",
        "descripcion": "Declaración anual de impuesto a la renta",
        "organismo": "SII",
        "servicio": "renta",
    },
}

REGIMENES = {
    "14D3": "Pro Pyme General (14 D N°3)",
    "14D8": "Pro Pyme Transparente (14 D N°8)",
    "14A": "Régimen General (14 A)",
    "PN": "Persona natural",
}

SERVICIOS = {
    "contabilidad": "Contabilidad",
    "iva": "IVA mensual",
    "remuneraciones": "Remuneraciones",
    "renta": "Operación renta",
}

# Documentos que se piden al cliente cada mes, según los servicios contratados.
DOCUMENTOS_POR_SERVICIO = {
    "contabilidad": ["Cartolas bancarias", "Comprobantes de gastos y caja chica"],
    "remuneraciones": ["Novedades del mes (altas, bajas, horas extra)", "Licencias médicas y permisos"],
}


def es_habil(d):
    return d.weekday() < 5 and d not in FERIADOS


def habil_desde(d):
    """El mismo día si es hábil; si no, el siguiente hábil."""
    while not es_habil(d):
        d += timedelta(days=1)
    return d


def sumar_habiles(d, n):
    paso = 1 if n >= 0 else -1
    restantes = abs(n)
    while restantes:
        d += timedelta(days=paso)
        if es_habil(d):
            restantes -= 1
    return d


def habiles_hasta(desde, hasta):
    """Días hábiles desde `desde` (excluido) hasta `hasta` (incluido). Negativo si ya pasó."""
    if hasta == desde:
        return 0
    if hasta < desde:
        return -habiles_hasta(hasta, desde)
    n, d = 0, desde
    while d < hasta:
        d += timedelta(days=1)
        if es_habil(d):
            n += 1
    return n


def periodo_de(d):
    return f"{d.year:04d}-{d.month:02d}"


def mover_periodo(periodo, meses):
    anio, mes = int(periodo[:4]), int(periodo[5:7])
    total = anio * 12 + (mes - 1) + meses
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def periodo_cerrado(hoy):
    """Último mes completo: el que se declara este mes."""
    return mover_periodo(periodo_de(hoy), -1)


def nombre_periodo(periodo, con_anio=False):
    if periodo.startswith("AT"):
        return f"año tributario {periodo[2:]}"
    nombre = MESES[int(periodo[5:7]) - 1]
    return f"{nombre} {periodo[:4]}" if con_anio else nombre


def fecha_larga(d, con_dia=True):
    if isinstance(d, str):
        d = date.fromisoformat(d[:10])
    texto = f"{d.day} de {MESES[d.month - 1]}"
    return f"{DIAS[d.weekday()]} {texto}" if con_dia else texto


def vencimiento(tipo, periodo):
    """Fecha de vencimiento de una obligación para un período."""
    if tipo == "F22":
        anio = int(periodo[2:])
        return habil_desde(date(anio, 4, 30))
    siguiente = mover_periodo(periodo, 1)
    anio, mes = int(siguiente[:4]), int(siguiente[5:7])
    dia = {"F29": 20, "PREVIRED": 13, "LRE": 15}[tipo]
    return habil_desde(date(anio, mes, dia))


def tipos_mensuales(servicios, trabajadores):
    tipos = []
    if "iva" in servicios:
        tipos.append("F29")
    if "remuneraciones" in servicios and trabajadores > 0:
        tipos += ["PREVIRED", "LRE"]
    return tipos


def documentos_requeridos(servicios, trabajadores):
    docs = list(DOCUMENTOS_POR_SERVICIO["contabilidad"]) if "contabilidad" in servicios else []
    if "remuneraciones" in servicios and trabajadores > 0:
        docs += DOCUMENTOS_POR_SERVICIO["remuneraciones"]
    return docs


def anio_tributario_vigente(hoy):
    """Año tributario cuyo F22 es el próximo a presentar."""
    return hoy.year if hoy <= date(hoy.year, 4, 30) else hoy.year + 1


def ahora(tz):
    return datetime.now(tz).isoformat(timespec="seconds")
