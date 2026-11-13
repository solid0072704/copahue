"""Sincronización con Gmail: detecta quién aceptó tu invitación y quién te respondió.

LinkedIn te avisa por correo («X aceptó tu invitación», «X te envió un mensaje»). Este agente lee
solo el remitente, el asunto y la fecha de esos avisos en tu propio Gmail —nunca entra a LinkedIn—
y actualiza el embudo. Es una detección de mejor esfuerzo: los asuntos de LinkedIn cambian con el
tiempo y el idioma, así que siempre puedes marcar «Aceptó» o «Respondió» a mano en el tablero.
"""

import re
from datetime import timedelta

from agentes_ebook.db import ahora_utc, desde_iso, iso
from agentes_ebook.gmail import normalizar

from . import secuencia

AGENTE = "Sincronización"
REMITENTE = "linkedin.com"

# Se aplican sobre el asunto normalizado (minúsculas, sin tildes).
PATRONES_ACEPTO = [
    r"^(?P<n>.+?) (?:ha aceptado|acepto) tu (?:invitacion|solicitud)",
    r"^(?P<n>.+?) accepted your invitation",
    r"^(?P<n>.+?) is now (?:a|your) connection",
    r"^(?P<n>.+?) (?:ya )?es (?:ahora )?tu contacto",
    r"nuevo contacto,? (?P<n>.+?)$",
    r"new connection,? (?P<n>.+?)$",
]
PATRONES_MENSAJE = [
    r"^(?P<n>.+?) te (?:ha enviado|envio|mando) (?:un|\d+) (?:nuevos? )?mensajes?",
    r"^(?P<n>.+?) (?:te ha escrito|te escribio|acaba de enviarte un mensaje|te respondio|respondio a tu mensaje)",
    r"^(?P<n>.+?) (?:just )?(?:sent you (?:a|\d+) (?:new )?messages?|messaged you|replied to your message)",
    r"^(?:nuevo mensaje de|tienes un (?:nuevo mensaje|mensaje nuevo) de|new message from|you have a new message from) (?P<n>.+?)$",
]
# «María Pérez via LinkedIn» como nombre del remitente en los avisos de mensajes.
PATRON_REMITENTE = r"^(?P<n>.+?) (?:via|a traves de|en|por|on) linkedin$"
BUZON_MENSAJES = r"messag|hit-reply|inmail"


def _tokens(nombre):
    return [t for t in re.split(r"[^a-z0-9]+", normalizar(nombre)) if len(t) > 1]


def _mismo_nombre(candidato, prospecto):
    """Mismo primer nombre y al menos dos palabras en común, con un nombre contenido en el otro."""
    if len(candidato) < 2 or len(prospecto) < 2 or candidato[0] != prospecto[0]:
        return False
    a, b = set(candidato), set(prospecto)
    return len(a & b) >= 2 and (a <= b or b <= a)


def clasificar(encabezado):
    """Devuelve ('acepto' | 'mensaje', nombre) o (None, '') para un aviso de LinkedIn."""
    asunto = normalizar(encabezado.get("asunto"))
    for patron in PATRONES_ACEPTO:
        m = re.search(patron, asunto)
        if m:
            return "acepto", m.group("n")
    for patron in PATRONES_MENSAJE:
        m = re.search(patron, asunto)
        if m:
            return "mensaje", m.group("n")
    buzon = (encabezado.get("de_correo") or "").split("@")[0]
    m = re.match(PATRON_REMITENTE, normalizar(encabezado.get("de_nombre")))
    if m and re.search(BUZON_MENSAJES, buzon):
        return "mensaje", m.group("n")
    return None, ""


def detectar(encabezados, prospectos):
    """Cruza los avisos con tus prospectos. Devuelve ({id: fecha} de aceptaciones, {id: fecha} de respuestas)."""
    nombres = [(p, _tokens(p["nombre"])) for p in prospectos]
    aceptaron, respondieron = {}, {}
    for e in sorted(encabezados, key=lambda x: x["fecha"]):
        tipo, nombre = clasificar(e)
        if not tipo:
            continue
        candidato = _tokens(nombre)
        coincidencias = [p for p, tokens in nombres if _mismo_nombre(candidato, tokens)]
        if len(coincidencias) != 1:
            continue  # nadie o más de uno con ese nombre: mejor no adivinar
        p = coincidencias[0]
        if tipo == "acepto":
            if p["etapa"] in ("Por invitar", "Invitación enviada"):
                aceptaron[p["id"]] = e["fecha"]
        else:
            # Solo cuenta como respuesta si tú ya le escribiste o lo invitaste antes de ese aviso.
            referencia = desde_iso(p["ultimo_mensaje_en"] or p["invitado_en"])
            if (
                referencia
                and e["fecha"] >= referencia
                and p["etapa"] in ("Invitación enviada", "Conectado", "En conversación", "Sin respuesta")
            ):
                respondieron[p["id"]] = e["fecha"]
    return aceptaron, respondieron


def sincronizar(db, cfgp, cliente, ahora=None):
    """Lee los avisos de LinkedIn en Gmail y actualiza el embudo. Devuelve (ids que aceptaron, ids que respondieron)."""
    ahora = ahora or ahora_utc()
    ultima = desde_iso(db.get_meta("li_ultima_sync"))
    desde = max(ahora - timedelta(days=30), (ultima - timedelta(days=2)) if ultima else ahora - timedelta(days=14))
    activos = [p for p in db.prospectos() if p["etapa"] not in ("Respondió", "Reunión agendada", "Cliente", "Descartado")]
    if not activos:
        return [], []
    encabezados = cliente.encabezados_de(REMITENTE, desde)
    aceptaron, respondieron = detectar(encabezados, activos)
    for pid, fecha in aceptaron.items():
        if pid not in respondieron:
            secuencia.marcar_conectado(db, pid, ahora=fecha, origen="aviso de LinkedIn en Gmail")
    for pid, fecha in respondieron.items():
        secuencia.marcar_respondio(db, pid, ahora=fecha, origen="aviso de LinkedIn en Gmail")
    db.set_meta("li_ultima_sync", iso(ahora))
    return [p for p in aceptaron if p not in respondieron], list(respondieron)
