"""Calendario del ayuno: qué aviso toca y cuándo.

Todas las horas se calculan en tu zona horaria (con cambios de horario incluidos)
y se comparan en UTC, para que un cambio de hora no adelante ni atrase los avisos.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

UTC = timezone.utc


@dataclass(frozen=True)
class Aviso:
    tipo: str
    dia: date  # día del plan: la fecha en que se abre la ventana de comida
    cuando: datetime  # UTC
    caduca: datetime  # UTC: después de esto el aviso ya no tiene sentido
    ayuno_desde: datetime  # UTC: el ayuno al que se refiere el aviso
    ayuno_hasta: datetime
    ventana_desde: datetime  # UTC: la ventana de comida del día
    ventana_hasta: datetime

    @property
    def clave(self):
        return f"{self.dia.isoformat()}:{self.tipo}"


def _local(dia, hora, cfg):
    return datetime.combine(dia, hora, tzinfo=cfg.tz).astimezone(UTC)


def ventana(dia, cfg):
    """(abre, cierra) en UTC de la ventana de comida que se abre el `dia`."""
    abre = _local(dia, cfg.ventana_inicio, cfg)
    cierre_dia = dia if cfg.ventana_fin > cfg.ventana_inicio else dia + timedelta(days=1)
    return abre, _local(cierre_dia, cfg.ventana_fin, cfg)


def _avisos_del_dia(dia, cfg):
    """Avisos del día `dia` (sin caducidad todavía)."""
    abre, cierra = ventana(dia, cfg)
    _, cierra_anterior = ventana(dia - timedelta(days=1), cfg)
    abre_siguiente, _ = ventana(dia + timedelta(days=1), cfg)
    previo = timedelta(minutes=cfg.aviso_previo_min)
    candidatos = []

    # El ayuno que termina hoy existe si hoy es un día de ayuno.
    if dia.weekday() in cfg.dias:
        ayuno = (cierra_anterior, abre)
        manana = _local(dia, cfg.hora_manana, cfg)
        if cierra_anterior < manana < abre - previo:  # solo si cae dentro del ayuno
            candidatos.append(("manana", manana, ayuno))
        if cfg.aviso_previo_min:
            candidatos.append(("apertura_previa", abre - previo, ayuno))
        candidatos.append(("apertura", abre, ayuno))

    # Al cerrar la ventana empieza el ayuno que termina mañana (si mañana es día de ayuno).
    if (dia + timedelta(days=1)).weekday() in cfg.dias:
        ayuno = (cierra, abre_siguiente)
        if cfg.aviso_previo_min:
            candidatos.append(("cierre_previo", cierra - previo, ayuno))
        candidatos.append(("inicio_ayuno", cierra, ayuno))

    return [(tipo, cuando, ayuno, (abre, cierra)) for tipo, cuando, ayuno in candidatos if tipo in cfg.avisos]


def calendario(cfg, desde, hasta):
    """Avisos con `cuando` entre `desde` y `hasta` (UTC), ordenados.

    Cada aviso caduca cuando llega el siguiente o pasada la tolerancia: así nunca
    llega un "en 30 minutos abres tu ventana" cuando la ventana ya está abierta.
    """
    tz = cfg.tz
    primer_dia = desde.astimezone(tz).date() - timedelta(days=2)
    ultimo_dia = hasta.astimezone(tz).date() + timedelta(days=2)
    crudos = []
    dia = primer_dia
    while dia <= ultimo_dia:
        crudos.extend((dia, *datos) for datos in _avisos_del_dia(dia, cfg))
        dia += timedelta(days=1)
    crudos.sort(key=lambda x: x[2])

    tolerancia = timedelta(minutes=cfg.tolerancia_min)
    avisos = []
    for i, (dia, tipo, cuando, ayuno, vent) in enumerate(crudos):
        caduca = cuando + tolerancia
        if i + 1 < len(crudos):
            caduca = min(caduca, crudos[i + 1][2])
        if desde <= cuando <= hasta:
            avisos.append(Aviso(tipo, dia, cuando, caduca, ayuno[0], ayuno[1], vent[0], vent[1]))
    return avisos


def pendientes(cfg, ahora, ya_enviado=lambda clave: False):
    """Avisos que ya debieron enviarse, siguen vigentes y no se han enviado."""
    margen = timedelta(minutes=cfg.tolerancia_min)
    return [
        a for a in calendario(cfg, ahora - margen, ahora)
        if a.cuando <= ahora < a.caduca and not ya_enviado(a.clave)
    ]


def proximo(cfg, ahora):
    """El siguiente aviso después de `ahora` (o None si no hay en una semana)."""
    futuros = calendario(cfg, ahora + timedelta(microseconds=1), ahora + timedelta(days=8))
    return futuros[0] if futuros else None


def ahora_utc():
    return datetime.now(UTC)
