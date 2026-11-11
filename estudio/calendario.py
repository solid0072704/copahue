"""Calendario de vencimientos: qué le corresponde a cada cliente y cuándo vence.

Las fechas del catálogo son REFERENCIALES. El SII, la Dirección del Trabajo y Previred
publican cada año sus calendarios (y las declaraciones juradas cambian de fecha con
frecuencia): revísalas en ⚙️ Configuración y ajusta el día que corresponda.

Cuando un vencimiento cae en sábado, domingo o feriado se corre al día hábil siguiente.
Solo se consideran los feriados que este módulo conoce con certeza; ante la duda la fecha
calculada queda ANTES de la real, nunca después.
"""

import calendar
from dataclasses import dataclass, replace
from datetime import date, timedelta

from .db import ESTADOS_CERRADOS

SERVICIOS = [
    "Contabilidad mensual",
    "IVA mensual (F29)",
    "Remuneraciones",
    "Renta anual (F22)",
    "Declaraciones juradas",
    "Patente municipal",
    "Asesoría tributaria",
]


@dataclass(frozen=True)
class TipoObligacion:
    codigo: str
    nombre: str
    frecuencia: str  # "mensual" o "anual"
    dia: int
    servicio: str
    mes: int = 0  # solo anuales
    desfase: int = 1  # anuales: años después del año comercial; mensuales: siempre el mes siguiente
    ayuda: str = ""


CATALOGO = {
    t.codigo: t
    for t in (
        TipoObligacion(
            "F29", "F29 · IVA, PPM y retenciones", "mensual", 20, "IVA mensual (F29)",
            ayuda="Día 20 para quienes emiten documentos electrónicos y declaran por internet; día 12 para el resto.",
        ),
        TipoObligacion(
            "PREVIRED", "Cotizaciones previsionales (Previred)", "mensual", 13, "Remuneraciones",
            ayuda="Día 13 con pago electrónico; día 10 si se paga con planilla presencial.",
        ),
        TipoObligacion(
            "LRE", "Libro de Remuneraciones Electrónico (DT)", "mensual", 15, "Remuneraciones",
            ayuda="Se informa a la Dirección del Trabajo dentro de los primeros 15 días del mes siguiente.",
        ),
        TipoObligacion(
            "DJ1887", "DJ 1887 · Sueldos", "anual", 20, "Declaraciones juradas", mes=3,
            ayuda="Fecha referencial: el SII la fija cada año por resolución.",
        ),
        TipoObligacion(
            "DJ1879", "DJ 1879 · Honorarios", "anual", 25, "Declaraciones juradas", mes=3,
            ayuda="Fecha referencial: el SII la fija cada año por resolución.",
        ),
        TipoObligacion(
            "DJRENTA", "DJ de rentas del régimen (1947 / 1948 / 1943)", "anual", 30, "Declaraciones juradas", mes=3,
            ayuda="Depende del régimen tributario del cliente. Fecha referencial: confirma la de cada formulario.",
        ),
        TipoObligacion(
            "F22", "F22 · Declaración anual de renta", "anual", 30, "Renta anual (F22)", mes=4,
            ayuda="Operación Renta: abril del año siguiente al año comercial.",
        ),
        TipoObligacion(
            "PATENTE1", "Patente municipal · cuota de julio", "anual", 31, "Patente municipal", mes=7, desfase=0,
            ayuda="Primera cuota del período julio–junio.",
        ),
        TipoObligacion(
            "PATENTE2", "Patente municipal · cuota de enero", "anual", 31, "Patente municipal", mes=1, desfase=1,
            ayuda="Segunda cuota del período julio–junio.",
        ),
    )
}

META_AJUSTES = "catalogo_ajustes"


# ---------------- feriados y días hábiles ----------------
def _pascua(anio):
    """Domingo de Pascua (algoritmo gregoriano anónimo)."""
    a, b, c = anio % 19, anio // 100, anio % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes, dia = divmod(h + l - 7 * m + 114, 31)
    return date(anio, mes, dia + 1)


def _al_lunes(d):
    """Ley 19.668: martes a jueves se traslada al lunes de esa semana; viernes, al lunes siguiente."""
    dia = d.weekday()
    if dia in (1, 2, 3):
        return d - timedelta(days=dia)
    if dia == 4:
        return d + timedelta(days=3)
    return d


def feriados(anio):
    """Feriados nacionales de fecha conocida. No incluye el Día de los Pueblos Indígenas (solsticio),
    elecciones ni feriados regionales: agrégalos en ESTUDIO_FERIADOS_EXTRA."""
    pascua = _pascua(anio)
    iglesias = date(anio, 10, 31)
    if iglesias.weekday() == 1:  # martes → viernes anterior
        iglesias -= timedelta(days=4)
    elif iglesias.weekday() == 2:  # miércoles → viernes siguiente
        iglesias += timedelta(days=2)
    return {
        date(anio, 1, 1),
        pascua - timedelta(days=2),  # Viernes Santo
        pascua - timedelta(days=1),  # Sábado Santo
        date(anio, 5, 1),
        date(anio, 5, 21),
        _al_lunes(date(anio, 6, 29)),  # San Pedro y San Pablo
        date(anio, 7, 16),
        date(anio, 8, 15),
        date(anio, 9, 18),
        date(anio, 9, 19),
        _al_lunes(date(anio, 10, 12)),  # Encuentro de Dos Mundos
        iglesias,
        date(anio, 11, 1),
        date(anio, 12, 8),
        date(anio, 12, 25),
    }


def es_habil(d, feriados_extra=()):
    return d.weekday() < 5 and d not in feriados(d.year) and d not in set(feriados_extra)


def siguiente_habil(d, feriados_extra=()):
    while not es_habil(d, feriados_extra):
        d += timedelta(days=1)
    return d


# ---------------- períodos ----------------
def periodo_de(d):
    return f"{d.year:04d}-{d.month:02d}"


def mes_siguiente(anio, mes, cuantos=1):
    indice = anio * 12 + (mes - 1) + cuantos
    return indice // 12, indice % 12 + 1


def periodo_anterior(d):
    """Período mensual que se declara durante el mes de la fecha dada (el mes anterior)."""
    anio, mes = mes_siguiente(d.year, d.month, -1)
    return f"{anio:04d}-{mes:02d}"


def partes_periodo(periodo):
    """'2026-10' → (2026, 10) · '2026' → (2026, None). Lanza ValueError si no es un período."""
    texto = str(periodo).strip()
    if len(texto) == 4 and texto.isdigit():
        return int(texto), None
    anio, _, mes = texto.partition("-")
    if len(anio) == 4 and anio.isdigit() and mes.isdigit() and 1 <= int(mes) <= 12:
        return int(anio), int(mes)
    raise ValueError(f"Período no válido: «{periodo}» (usa AAAA-MM o AAAA).")


MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def nombre_periodo(periodo):
    try:
        anio, mes = partes_periodo(periodo)
    except ValueError:
        return str(periodo or "")
    return f"{MESES[mes - 1]} {anio}" if mes else f"año {anio}"


# ---------------- catálogo ----------------
def catalogo(ajustes=None):
    """Catálogo con los días/meses que el estudio haya ajustado en Configuración."""
    resultado = {}
    for codigo, tipo in CATALOGO.items():
        a = (ajustes or {}).get(codigo) or {}
        try:
            dia = min(31, max(1, int(a.get("dia", tipo.dia))))
            mes = min(12, max(1, int(a.get("mes", tipo.mes)))) if tipo.frecuencia == "anual" else 0
        except (TypeError, ValueError):
            dia, mes = tipo.dia, tipo.mes
        resultado[codigo] = replace(tipo, dia=dia, mes=mes)
    return resultado


def vencimiento(tipo, periodo, feriados_extra=()):
    """Fecha de vencimiento de un tipo de obligación para un período ('AAAA-MM' o 'AAAA')."""
    anio, mes = partes_periodo(periodo)
    if tipo.frecuencia == "mensual":
        if mes is None:
            raise ValueError(f"{tipo.codigo} es mensual: el período debe ser AAAA-MM.")
        anio, mes = mes_siguiente(anio, mes)
    else:
        if mes is not None:
            raise ValueError(f"{tipo.codigo} es anual: el período debe ser AAAA.")
        anio, mes = anio + tipo.desfase, tipo.mes
    dia = min(tipo.dia, calendar.monthrange(anio, mes)[1])
    return siguiente_habil(date(anio, mes, dia), feriados_extra)


def tipos_para(cliente, tipos=None):
    """Obligaciones que le corresponden a un cliente según los servicios contratados."""
    servicios = set(cliente.get("servicios") or [])
    return [t for t in (tipos or CATALOGO).values() if t.servicio in servicios]


def generar(db, cfg, periodo):
    """Crea los vencimientos del período para todos los clientes activos. No duplica los existentes.

    Con 'AAAA-MM' genera las obligaciones mensuales; con 'AAAA' (año comercial), las anuales.
    Devuelve cuántas obligaciones nuevas se crearon.
    """
    _, mes = partes_periodo(periodo)
    frecuencia = "mensual" if mes else "anual"
    tipos = catalogo(db.get_json(META_AJUSTES, {}))
    creadas = 0
    for cliente in db.clientes(estados=["Activo"]):
        for tipo in tipos_para(cliente, tipos):
            if tipo.frecuencia != frecuencia:
                continue
            nueva = db.crear_obligacion(
                cliente["id"], tipo.codigo, tipo.nombre, vencimiento(tipo, periodo, cfg.feriados_extra),
                periodo=periodo, responsable=cliente["responsable"],
            )
            creadas += bool(nueva)
    return creadas


# ---------------- semáforo ----------------
SEMAFOROS = ["Atrasada", "Vence hoy", "Próxima", "Al día", "Cerrada"]
ICONO_SEMAFORO = {"Atrasada": "🔴", "Vence hoy": "🟠", "Próxima": "🟡", "Al día": "🟢", "Cerrada": "✅"}


def semaforo(obligacion, hoy, dias_alerta=7):
    if obligacion["estado"] in ESTADOS_CERRADOS:
        return "Cerrada"
    dias = (date.fromisoformat(obligacion["vence"]) - hoy).days
    if dias < 0:
        return "Atrasada"
    if dias == 0:
        return "Vence hoy"
    return "Próxima" if dias <= dias_alerta else "Al día"


def texto_plazo(vence, hoy):
    dias = (date.fromisoformat(vence) - hoy).days
    if dias == 0:
        return "hoy"
    if dias == 1:
        return "mañana"
    if dias == -1:
        return "ayer"
    return f"en {dias} días" if dias > 0 else f"hace {-dias} días"
