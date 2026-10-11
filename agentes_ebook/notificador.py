"""Agente Notificador: te avisa por correo de leads calientes, respuestas y el resumen diario."""

from datetime import timedelta
from html import escape
from urllib.parse import quote

from . import reuniones
from .analista import insights
from .db import ahora_utc, desde_iso
from .gmail import enviar_correo
from .plantillas import fecha_larga, link_mailto, link_whatsapp, redactar

AGENTE = "Notificador"

COLOR_TEXTO = "#0b0b0b"
COLOR_SECUNDARIO = "#52514e"
COLOR_ACENTO = "#2a78d6"
COLOR_BORDE = "#e1e0d9"
COLOR_FONDO = "#f9f9f7"
ICONO_SEGMENTO = {"Caliente": "🔥", "Tibio": "🌤️", "Frío": "❄️"}


def _boton(texto, url, principal=False):
    if not url:
        return ""
    estilo = (
        f"background:{COLOR_ACENTO};color:#ffffff;border:1px solid {COLOR_ACENTO};"
        if principal
        else f"background:#ffffff;color:{COLOR_ACENTO};border:1px solid {COLOR_ACENTO};"
    )
    return (
        f'<a href="{escape(url)}" style="{estilo}display:inline-block;padding:7px 12px;margin:4px 6px 0 0;'
        f'border-radius:6px;font-size:13px;text-decoration:none;">{escape(texto)}</a>'
    )


def _envoltorio(titulo, contenido, cfg):
    return f"""<!doctype html><html><body style="margin:0;background:{COLOR_FONDO};">
<div style="max-width:640px;margin:0 auto;padding:24px 16px;font-family:system-ui,-apple-system,'Segoe UI',Arial,sans-serif;color:{COLOR_TEXTO};">
<p style="margin:0 0 4px;font-size:12px;color:{COLOR_SECUNDARIO};letter-spacing:.04em;text-transform:uppercase;">Agentes del ebook · {escape(cfg.empresa_nombre)}</p>
<h1 style="margin:0 0 16px;font-size:22px;line-height:1.3;">{escape(titulo)}</h1>
{contenido}
<p style="margin:24px 0 0;font-size:12px;color:{COLOR_SECUNDARIO};">Este correo lo generaron tus agentes automáticamente. Ningún correo a clientes se envía sin tu revisión salvo que actives el modo automático.</p>
</div></body></html>"""


def _tarjeta_lead(lead, cfg, db, seguimiento=None):
    icono = ICONO_SEGMENTO.get(lead.get("segmento"), "")
    linea = " · ".join(x for x in (lead.get("cargo"), lead.get("empresa"), lead.get("pais")) if x)
    if seguimiento:
        asunto, cuerpo = seguimiento["asunto"], seguimiento["cuerpo"]
    else:
        horarios = reuniones.horarios_de(db, lead["id"], cfg) if db else []
        asunto, cuerpo = redactar("bienvenida_caliente", lead, cfg, horarios)
    botones = _boton("✉️ Responder con borrador", link_mailto(lead["email"], asunto, cuerpo), principal=True)
    botones += _boton("💬 WhatsApp", link_whatsapp(lead, cfg))
    if lead.get("respondio_en"):
        botones += _boton("📨 Ver su respuesta", f"https://mail.google.com/mail/u/0/#search/from%3A{quote(lead['email'])}")
    for h in (reuniones.horarios_de(db, lead["id"], cfg) if db else [])[:3]:
        botones += _boton(f"📅 {h:%d/%m %H:%M}", reuniones.link_google_calendar(lead, cfg, h))
    lectura = escape(lead.get("lectura") or "")
    angulo = escape(lead.get("angulo") or "")
    motivo = escape(lead.get("motivo") or "—")
    return f"""<div style="background:#ffffff;border:1px solid {COLOR_BORDE};border-radius:10px;padding:14px 16px;margin:0 0 12px;">
<p style="margin:0;font-size:16px;font-weight:600;">{icono} {escape(lead.get('nombre') or lead['email'])} <span style="font-weight:400;color:{COLOR_SECUNDARIO};font-size:13px;">· score {lead.get('score', 0)}</span></p>
<p style="margin:2px 0 8px;font-size:13px;color:{COLOR_SECUNDARIO};">{escape(linea)} · {escape(lead['email'])} {('· ' + escape(lead['telefono'])) if lead.get('telefono') else ''}</p>
<p style="margin:0 0 4px;font-size:14px;"><b>Motivo:</b> {motivo}</p>
<p style="margin:0 0 4px;font-size:14px;">{lectura}</p>
<p style="margin:0 0 6px;font-size:14px;"><b>Cómo abordarlo:</b> {angulo}</p>
{botones}
</div>"""


def _seguimiento_abierto(db, lead_id):
    abiertos = db.seguimientos(lead_id=lead_id, estados=("pendiente", "borrador"))
    return abiertos[0] if abiertos else None


def construir_alerta(db, cfg, leads, respondieron_ids):
    respondieron = [l for l in leads if l["id"] in respondieron_ids]
    calientes = [l for l in leads if l["id"] not in respondieron_ids]
    partes = []
    if respondieron:
        partes.append(f'<h2 style="font-size:17px;margin:8px 0;">📨 Te respondieron ({len(respondieron)})</h2>')
        partes += [_tarjeta_lead(l, cfg, db) for l in respondieron]
    if calientes:
        partes.append(f'<h2 style="font-size:17px;margin:8px 0;">🔥 Leads calientes nuevos ({len(calientes)})</h2>')
        partes += [_tarjeta_lead(l, cfg, db, _seguimiento_abierto(db, l["id"])) for l in calientes]
    total = len(respondieron) + len(calientes)
    titulo = f"{total} oportunidad{'es' if total != 1 else ''} para agendar reunión"
    texto = "\n".join(
        f"- {l.get('nombre')} <{l['email']}> · {l.get('cargo')} · {l.get('empresa')} · score {l.get('score')}"
        for l in respondieron + calientes
    )
    return titulo, _envoltorio(titulo, "".join(partes), cfg), texto


def enviar_alerta(db, cfg, respondieron_ids=()):
    """Avisa de leads de los segmentos configurados que aún no te notificamos y de respuestas nuevas."""
    reales = [l for l in db.leads() if l.get("fuente") != "demo"]
    pendientes = [
        l for l in reales
        if (l["segmento"] in cfg.notificar_segmentos and not l["notificado"]) or l["id"] in respondieron_ids
    ]
    if not pendientes:
        return None
    if not (cfg.gmail_configurado and cfg.notificar_a):
        return "sin_configurar"
    titulo, html, texto = construir_alerta(db, cfg, pendientes, set(respondieron_ids))
    try:
        enviar_correo(cfg, cfg.notificar_a, f"🔥 {titulo}", texto, html, cc=cfg.notificar_cc)
    except Exception as e:
        db.log(AGENTE, "Error enviando alerta", str(e))
        return "error"
    for l in pendientes:
        db.actualizar_lead(l["id"], notificado=1)
    db.log(AGENTE, "Alerta enviada", f"{len(pendientes)} leads a {cfg.notificar_a}")
    return "enviado"


def _kpi(valor, etiqueta):
    return (
        f'<td style="padding:10px 12px;background:#ffffff;border:1px solid {COLOR_BORDE};border-radius:8px;">'
        f'<div style="font-size:22px;font-weight:600;">{valor}</div>'
        f'<div style="font-size:12px;color:{COLOR_SECUNDARIO};">{escape(etiqueta)}</div></td>'
    )


def construir_resumen(db, cfg, leads=None, ahora=None):
    ahora = ahora or ahora_utc()
    leads = leads if leads is not None else db.leads()
    ids = {l["id"] for l in leads}
    descargas = [d for d in db.descargas() if d["lead_id"] in ids]
    d24 = [d for d in descargas if ahora - desde_iso(d["fecha"]) <= timedelta(hours=24)]
    d7 = [d for d in descargas if ahora - desde_iso(d["fecha"]) <= timedelta(days=7)]
    abiertos = [s for s in db.seguimientos(estados=("pendiente", "borrador")) if s["lead_id"] in ids]
    reus = [r for r in db.reuniones() if r["lead_id"] in ids]
    calientes = [l for l in leads if l["segmento"] == "Caliente"]

    kpis = (
        '<table role="presentation" style="width:100%;border-collapse:separate;border-spacing:6px;margin:0 -6px 8px;"><tr>'
        + _kpi(len(d24), "descargas 24 h")
        + _kpi(len(d7), "descargas 7 días")
        + _kpi(len(calientes), "leads calientes")
        + _kpi(len(abiertos), "correos por aprobar")
        + _kpi(sum(1 for r in reus if r["estado"] == "agendada"), "reuniones agendadas")
        + "</tr></table>"
    )
    hallazgos = "".join(f"<li style='margin:0 0 4px;'>{escape(h)}</li>" for h in insights(leads, descargas, ahora, cfg.tz))
    contenido = [kpis, f"<h2 style='font-size:17px;margin:16px 0 6px;'>🧠 Lo que ve el analista</h2><ul style='padding-left:18px;margin:0;font-size:14px;'>{hallazgos}</ul>"]

    nuevos_ids = {d["lead_id"] for d in d24}
    nuevos = sorted((l for l in leads if l["id"] in nuevos_ids), key=lambda l: -l["score"])[:15]
    if nuevos:
        filas = "".join(
            f"<tr><td style='padding:6px 4px;border-bottom:1px solid {COLOR_BORDE};'>{ICONO_SEGMENTO.get(l['segmento'], '')} {escape(l['nombre'] or l['email'])}</td>"
            f"<td style='padding:6px 4px;border-bottom:1px solid {COLOR_BORDE};color:{COLOR_SECUNDARIO};'>{escape(l['cargo'] or '')} · {escape(l['empresa'] or '')}</td>"
            f"<td style='padding:6px 4px;border-bottom:1px solid {COLOR_BORDE};text-align:right;'>{l['score']}</td></tr>"
            for l in nuevos
        )
        contenido.append(
            f"<h2 style='font-size:17px;margin:16px 0 6px;'>📥 Descargas de las últimas 24 h</h2>"
            f"<table style='width:100%;border-collapse:collapse;font-size:14px;'>{filas}</table>"
        )
    if abiertos:
        items = "".join(
            f"<li style='margin:0 0 6px;'><b>{escape(s['nombre'] or s['email'])}</b> · {escape(s['nombre_paso'])} "
            f"({'borrador en Gmail' if s['estado'] == 'borrador' else 'por aprobar'}) "
            f"<a href='{escape(link_mailto(s['email'], s['asunto'], s['cuerpo']))}' style='color:{COLOR_ACENTO};'>abrir correo</a></li>"
            for s in abiertos[:10]
        )
        contenido.append(f"<h2 style='font-size:17px;margin:16px 0 6px;'>✉️ Seguimientos listos para enviar</h2><ul style='padding-left:18px;margin:0;font-size:14px;'>{items}</ul>")
    proximas = sorted(
        (r for r in reus if r["estado"] == "agendada" and r["fecha_agendada"] and desde_iso(r["fecha_agendada"]) > ahora),
        key=lambda r: r["fecha_agendada"],
    )
    if proximas:
        items = "".join(
            f"<li>{escape(fecha_larga(desde_iso(r['fecha_agendada']).astimezone(cfg.tz)))} · {escape(r['nombre'] or r['email'])} ({escape(r['empresa'] or '')})</li>"
            for r in proximas
        )
        contenido.append(f"<h2 style='font-size:17px;margin:16px 0 6px;'>🤝 Próximas reuniones</h2><ul style='padding-left:18px;margin:0;font-size:14px;'>{items}</ul>")
    titulo = f"Resumen del ebook · {ahora.astimezone(cfg.tz):%d/%m/%Y}"
    texto = f"{titulo}\nDescargas 24 h: {len(d24)} · 7 días: {len(d7)} · Calientes: {len(calientes)} · Por aprobar: {len(abiertos)}"
    return titulo, _envoltorio(titulo, "".join(contenido), cfg), texto


def toca_resumen(db, cfg, ahora=None):
    local = (ahora or ahora_utc()).astimezone(cfg.tz)
    return local.hour >= cfg.resumen_hora and db.get_meta("ultimo_resumen") != local.date().isoformat()


def enviar_resumen(db, cfg, forzar=False, ahora=None):
    ahora = ahora or ahora_utc()
    if not forzar and not toca_resumen(db, cfg, ahora):
        return None
    if not (cfg.gmail_configurado and cfg.notificar_a):
        return "sin_configurar"
    reales = [l for l in db.leads() if l.get("fuente") != "demo"]
    titulo, html, texto = construir_resumen(db, cfg, reales, ahora)
    try:
        enviar_correo(cfg, cfg.notificar_a, f"📊 {titulo}", texto, html, cc=cfg.notificar_cc)
    except Exception as e:
        db.log(AGENTE, "Error enviando resumen", str(e))
        return "error"
    db.set_meta("ultimo_resumen", ahora.astimezone(cfg.tz).date().isoformat())
    db.log(AGENTE, "Resumen diario enviado", f"a {cfg.notificar_a}")
    return "enviado"
