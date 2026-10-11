"""Agente de Reuniones: propone horarios a los leads listos y genera invitaciones."""

from datetime import datetime, time, timedelta, timezone
from urllib.parse import urlencode

import pandas as pd

from .analista import nombre_de_pila
from .db import ahora_utc, desde_iso, iso

AGENTE = "Reuniones"


def proximos_horarios(cfg, ocupados=(), cantidad=3, desde=None, rotacion=0):
    """Próximos horarios hábiles (lun-vie) en la zona configurada, evitando los ya agendados.

    `rotacion` cambia la hora de partida para que no todos los leads reciban los mismos horarios.
    """
    tz = cfg.tz
    ahora = (desde or ahora_utc()).astimezone(tz)
    horas = []
    for h in cfg.reunion_horas:
        try:
            hh, mm = h.split(":")
            horas.append(time(int(hh), int(mm)))
        except ValueError:
            continue
    horas = sorted(horas) or [time(10, 0)]
    ocupados = set(ocupados)
    elegidos, dia = [], ahora.date() + timedelta(days=1)
    # Un horario por día para dar opciones en días distintos.
    while len(elegidos) < cantidad and dia < ahora.date() + timedelta(days=30):
        if dia.weekday() < 5:
            k = (len(elegidos) + rotacion) % len(horas)
            for hora in horas[k:] + horas[:k]:
                inicio = datetime.combine(dia, hora, tzinfo=tz)
                if iso(inicio) not in ocupados:
                    elegidos.append(inicio)
                    break
        dia += timedelta(days=1)
    return elegidos


def _titulo(lead, cfg):
    quien = lead.get("nombre") or lead["email"]
    return f"{cfg.oferta_reunion.capitalize()} · {quien} / {cfg.empresa_nombre}"


def _descripcion(lead, cfg):
    lineas = [
        f"Reunión con {lead.get('nombre', '')} ({lead.get('cargo', '')}, {lead.get('empresa', '')}).",
        f"Origen: descarga de {cfg.ebook_nombre}.",
    ]
    if lead.get("motivo"):
        lineas.append(f"Motivo de descarga: {lead['motivo']}")
    if lead.get("angulo"):
        lineas.append(f"Enfoque sugerido: {lead['angulo']}")
    return "\n".join(lineas)


def link_google_calendar(lead, cfg, inicio):
    """Enlace que abre Google Calendar con el evento prellenado e invitando al lead."""
    fin = inicio + timedelta(minutes=cfg.reunion_duracion_min)
    formato = "%Y%m%dT%H%M%SZ"
    parametros = {
        "action": "TEMPLATE",
        "text": _titulo(lead, cfg),
        "dates": f"{inicio.astimezone(timezone.utc):{formato}}/{fin.astimezone(timezone.utc):{formato}}",
        "details": _descripcion(lead, cfg),
        "add": lead["email"],
        "ctz": cfg.zona_horaria,
    }
    return "https://calendar.google.com/calendar/render?" + urlencode(parametros)


def archivo_ics(lead, cfg, inicio):
    """Contenido de un archivo .ics (sirve para Outlook, Apple Calendar y Google)."""
    fin = inicio + timedelta(minutes=cfg.reunion_duracion_min)
    utc = timezone.utc
    f = "%Y%m%dT%H%M%SZ"

    def esc(texto):
        return texto.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")

    lineas = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Tablero Ebook//Agentes//ES",
        "METHOD:REQUEST",
        "BEGIN:VEVENT",
        f"UID:reunion-{lead['id']}-{inicio.astimezone(utc):{f}}@tablero-ebook",
        f"DTSTAMP:{ahora_utc():{f}}",
        f"DTSTART:{inicio.astimezone(utc):{f}}",
        f"DTEND:{fin.astimezone(utc):{f}}",
        f"SUMMARY:{esc(_titulo(lead, cfg))}",
        f"DESCRIPTION:{esc(_descripcion(lead, cfg))}",
        f"ORGANIZER;CN={esc(cfg.firma)}:mailto:{cfg.gmail_usuario or 'agenda@example.com'}",
        f"ATTENDEE;CN={esc(lead.get('nombre') or lead['email'])};RSVP=TRUE:mailto:{lead['email']}",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    return "\r\n".join(lineas) + "\r\n"


def es_candidato(lead):
    if lead["etapa"] in ("Reunión agendada", "Cliente", "Descartado", "Reunión propuesta"):
        return False
    return lead["etapa"] == "Respondió" or lead["segmento"] == "Caliente"


def proponer(db, cfg):
    """Crea propuestas de reunión para los leads calientes o que respondieron."""
    nuevas = []
    ocupados = db.horarios_ocupados()
    for lead in db.leads():
        if not es_candidato(lead) or db.reunion_de(lead["id"]):
            continue
        horarios = proximos_horarios(cfg, ocupados, rotacion=lead["id"])
        propuestas = [iso(h) for h in horarios]
        db.crear_reunion(lead["id"], propuestas)
        db.actualizar_lead(lead["id"], etapa="Reunión propuesta")
        db.log(
            AGENTE,
            "Reunión propuesta",
            f"{nombre_de_pila(lead['nombre'])} <{lead['email']}>: "
            + ", ".join(h.strftime("%d/%m %H:%M") for h in horarios),
            lead["id"],
        )
        nuevas.append(lead["id"])
    return nuevas


def renovar_vencidas(db, cfg):
    """Si todos los horarios propuestos ya pasaron, ofrece horarios nuevos."""
    ahora = ahora_utc()
    ocupados = db.horarios_ocupados()
    for r in db.reuniones():
        if r["estado"] != "propuesta" or not r["propuestas"]:
            continue
        if all(desde_iso(p) < ahora for p in r["propuestas"]):
            nuevos = [iso(h) for h in proximos_horarios(cfg, ocupados, rotacion=r["lead_id"])]
            db.actualizar_reunion(r["lead_id"], propuestas=nuevos)
            db.log(AGENTE, "Horarios renovados", f"{r['email']}", r["lead_id"])


def horarios_de(db, lead_id, cfg):
    r = db.reunion_de(lead_id)
    if not r or r["estado"] != "propuesta":
        return []
    ahora = ahora_utc()
    return [desde_iso(p).astimezone(cfg.tz) for p in r["propuestas"] if desde_iso(p) > ahora]


def agendar(db, lead_id, inicio):
    db.actualizar_reunion(lead_id, estado="agendada", fecha_agendada=iso(inicio))
    db.actualizar_lead(lead_id, etapa="Reunión agendada")
    db.log(AGENTE, "Reunión agendada", inicio.strftime("%d/%m/%Y %H:%M"), lead_id)


def exportar_crm(leads):
    """Contactos listos para importar en HubSpot / Pipedrive / Zoho (CSV)."""
    etapa_a_estado = {
        "Nuevo": "NEW",
        "En seguimiento": "ATTEMPTED_TO_CONTACT",
        "Respondió": "CONNECTED",
        "Reunión propuesta": "CONNECTED",
        "Reunión agendada": "OPEN_DEAL",
        "Cliente": "OPEN_DEAL",
        "Descartado": "UNQUALIFIED",
    }
    filas = []
    for l in leads:
        partes = (l.get("nombre") or "").split()
        filas.append(
            {
                "First Name": partes[0] if partes else "",
                "Last Name": " ".join(partes[1:]),
                "Email": l["email"],
                "Phone Number": l.get("telefono", ""),
                "Company Name": l.get("empresa", ""),
                "Job Title": l.get("cargo", ""),
                "Country/Region": l.get("pais", ""),
                "Lead Status": etapa_a_estado.get(l.get("etapa"), "NEW"),
                "Lifecycle Stage": "customer" if l.get("etapa") == "Cliente" else (
                    "salesqualifiedlead" if l.get("segmento") == "Caliente" else "lead"
                ),
                "Lead Score": l.get("score", 0),
                "Segmento": l.get("segmento", ""),
                "Etapa": l.get("etapa", ""),
                "Motivo de descarga": l.get("motivo", ""),
                "Primera descarga": l.get("primera_descarga", ""),
                "Notas": l.get("notas", ""),
            }
        )
    return pd.DataFrame(filas)
