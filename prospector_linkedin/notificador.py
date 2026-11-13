"""Avisos por correo del prospector: tu cola diaria de LinkedIn y las novedades (aceptaron / respondieron)."""

from html import escape

from agentes_ebook.db import ahora_utc
from agentes_ebook.gmail import enviar_correo
from agentes_ebook.notificador import COLOR_BORDE, COLOR_FONDO, COLOR_SECUNDARIO, COLOR_TEXTO, _boton

from . import secuencia
from .fuentes import link_perfil
from .mensajes import nombre_tipo, redactar

AGENTE = "Avisos LinkedIn"


def _envoltorio(titulo, contenido, cfgp):
    return f"""<!doctype html><html><body style="margin:0;background:{COLOR_FONDO};">
<div style="max-width:640px;margin:0 auto;padding:24px 16px;font-family:system-ui,-apple-system,'Segoe UI',Arial,sans-serif;color:{COLOR_TEXTO};">
<p style="margin:0 0 4px;font-size:12px;color:{COLOR_SECUNDARIO};letter-spacing:.04em;text-transform:uppercase;">Prospector de LinkedIn · {escape(cfgp.base.empresa_nombre)}</p>
<h1 style="margin:0 0 16px;font-size:22px;line-height:1.3;">{escape(titulo)}</h1>
{contenido}
<p style="margin:24px 0 0;font-size:12px;color:{COLOR_SECUNDARIO};">El prospector prepara a quién escribir y qué decir. Nada se envía en LinkedIn de forma automática: tú haces clic y luego lo marcas como enviado en el tablero.</p>
</div></body></html>"""


def _tarjeta(p, subtitulo, texto, cfgp):
    linea = " · ".join(x for x in (p.get("cargo"), p.get("empresa")) if x)
    mensaje = (
        f'<div style="margin:10px 0 6px;padding:10px 12px;background:{COLOR_FONDO};border-radius:6px;'
        f'font-size:14px;line-height:1.45;white-space:pre-wrap;">{escape(texto)}</div>'
        if texto
        else ""
    )
    return (
        f'<div style="margin:0 0 12px;padding:14px 16px;background:#ffffff;border:1px solid {COLOR_BORDE};border-radius:8px;">'
        f'<div style="font-size:12px;color:{COLOR_SECUNDARIO};">{escape(subtitulo)}</div>'
        f'<div style="font-size:16px;font-weight:600;margin-top:2px;">{escape(p.get("nombre", ""))}</div>'
        f'<div style="font-size:13px;color:{COLOR_SECUNDARIO};">{escape(linea)}</div>'
        f"{mensaje}{_boton('Abrir en LinkedIn', link_perfil(p), principal=True)}</div>"
    )


def construir_cola(db, cfgp, cola):
    mensajes = [a for a in cola if a["tipo"] != "invitacion"]
    invitaciones = [a for a in cola if a["tipo"] == "invitacion"]
    titulo = f"Tu LinkedIn de hoy: {len(mensajes)} mensajes y {len(invitaciones)} invitaciones"
    html, texto = [], [titulo, ""]
    html.append(
        f'<p style="margin:0 0 16px;font-size:14px;">Son unos {max(5, 1 + len(cola))} minutos. Abre cada perfil, '
        "pega el texto y marca lo enviado en el tablero para que la secuencia avance.</p>"
        + _boton("Abrir el tablero", cfgp.tablero_url)
        + '<div style="height:16px;"></div>'
    )
    for a in cola:
        subtitulo = f"{nombre_tipo(a['tipo'])} · prioridad {a['prioridad']} ({a['score']})"
        html.append(_tarjeta(a, subtitulo, a["texto"], cfgp))
        texto += [f"— {a['nombre']} ({a.get('cargo', '')}, {a.get('empresa', '')}) · {nombre_tipo(a['tipo'])}",
                  link_perfil(a), a["texto"], ""]
    return titulo, _envoltorio(titulo, "".join(html), cfgp), "\n".join(texto)


def toca_cola(db, cfgp, ahora=None):
    local = (ahora or ahora_utc()).astimezone(cfgp.tz)
    return local.hour >= cfgp.base.resumen_hora and db.get_meta("li_ultima_cola") != local.date().isoformat()


def enviar_cola(db, cfgp, forzar=False, ahora=None):
    """Una vez al día: el correo con lo que te toca hacer hoy en LinkedIn."""
    ahora = ahora or ahora_utc()
    if not forzar and not toca_cola(db, cfgp, ahora):
        return None
    cola = [a for a in secuencia.cola_de_hoy(db, cfgp, ahora) if a["fuente"] != "demo"]
    if not cola:
        return None
    if not (cfgp.base.gmail_configurado and cfgp.base.notificar_a):
        return "sin_configurar"
    titulo, html, texto = construir_cola(db, cfgp, cola)
    try:
        enviar_correo(cfgp.base, cfgp.base.notificar_a, f"💼 {titulo}", texto, html)
    except Exception as e:
        db.log(AGENTE, "Error enviando la cola diaria", str(e))
        return "error"
    db.set_meta("li_ultima_cola", ahora.astimezone(cfgp.tz).date().isoformat())
    db.log(AGENTE, "Cola diaria enviada", f"{len(cola)} acciones a {cfgp.base.notificar_a}")
    return "enviado"


def construir_novedades(db, cfgp, aceptaron, respondieron):
    titulo = f"LinkedIn: {len(respondieron)} respuestas y {len(aceptaron)} invitaciones aceptadas"
    html, texto = [], [titulo, ""]
    for pid in respondieron:
        p = db.prospecto(pid)
        sugerido = redactar("reunion", p, cfgp)
        html.append(_tarjeta(p, "💬 Te respondió · léelo en LinkedIn y, si calza, propón la reunión:", sugerido, cfgp))
        texto += [f"Respondió: {p['nombre']} · {link_perfil(p)}", sugerido, ""]
    for pid in aceptaron:
        p = db.prospecto(pid)
        html.append(_tarjeta(p, "🤝 Aceptó tu invitación · su primer mensaje ya está en tu cola", "", cfgp))
        texto += [f"Aceptó: {p['nombre']} · {link_perfil(p)}", ""]
    html.append(_boton("Abrir el tablero", cfgp.tablero_url))
    return titulo, _envoltorio(titulo, "".join(html), cfgp), "\n".join(texto)


def enviar_novedades(db, cfgp, aceptaron, respondieron):
    """Aviso inmediato cuando alguien acepta tu invitación o te responde."""
    reales = lambda ids: [i for i in ids if (db.prospecto(i) or {}).get("fuente") != "demo"]  # noqa: E731
    aceptaron, respondieron = reales(aceptaron), reales(respondieron)
    if not (aceptaron or respondieron):
        return None
    if not (cfgp.base.gmail_configurado and cfgp.base.notificar_a):
        return "sin_configurar"
    titulo, html, texto = construir_novedades(db, cfgp, aceptaron, respondieron)
    try:
        enviar_correo(cfgp.base, cfgp.base.notificar_a, f"🔔 {titulo}", texto, html)
    except Exception as e:
        db.log(AGENTE, "Error enviando novedades", str(e))
        return "error"
    db.log(AGENTE, "Novedades enviadas", f"{len(respondieron)} respuestas, {len(aceptaron)} aceptaciones")
    return "enviado"
