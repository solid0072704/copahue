"""Agente Notificador: te consulta cada orden por Telegram (con botones Ejecutar / Rechazar)
y te avisa por correo. También lee tus respuestas de Telegram.

Seguridad: solo se aceptan respuestas que vengan de TU chat (TELEGRAM_CHAT_ID).
"""

import logging
import smtplib
from email.message import EmailMessage
from html import escape

import requests

from .db import desde_iso
from .llm import ETIQUETA_OPINION

log = logging.getLogger(__name__)
AGENTE = "Notificador"
LADO = {"compra": "🟢 COMPRAR", "venta": "🔴 VENDER"}


class ErrorTelegram(Exception):
    pass


class Telegram:
    def __init__(self, cfg, sesion=None):
        self.chat_id = str(cfg.telegram_chat_id)
        self.url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/"
        self.sesion = sesion or requests.Session()

    def _api(self, metodo, **datos):
        try:
            r = self.sesion.post(self.url + metodo, json=datos, timeout=40)
            cuerpo = r.json()
        except (requests.RequestException, ValueError) as e:
            raise ErrorTelegram(f"Sin conexión con Telegram: {e}") from e
        if not cuerpo.get("ok"):
            raise ErrorTelegram(f"Telegram respondió: {cuerpo.get('description', r.status_code)}")
        return cuerpo["result"]

    def enviar(self, texto, botones=None):
        datos = {"chat_id": self.chat_id, "text": texto, "parse_mode": "HTML", "disable_web_page_preview": True}
        if botones:
            datos["reply_markup"] = {"inline_keyboard": botones}
        return self._api("sendMessage", **datos)["message_id"]

    def editar(self, message_id, texto):
        try:
            self._api("editMessageText", chat_id=self.chat_id, message_id=message_id, text=texto, parse_mode="HTML")
        except ErrorTelegram as e:
            log.info("No se pudo editar el mensaje %s: %s", message_id, e)

    def responder_boton(self, callback_id, texto):
        try:
            self._api("answerCallbackQuery", callback_query_id=callback_id, text=texto)
        except ErrorTelegram:
            pass

    def actualizaciones(self, offset):
        return self._api("getUpdates", offset=offset, timeout=0, allowed_updates=["callback_query", "message"])

    def es_mi_chat(self, chat_id, usuario_id):
        return str(chat_id) == self.chat_id and str(usuario_id) == self.chat_id


def crear_telegram(cfg):
    return Telegram(cfg) if cfg.telegram_configurado else None


# ---------------- textos ----------------
def _usd(x):
    return f"US$ {x:,.2f}"


def _hora_local(texto_iso, cfg):
    dt = desde_iso(texto_iso)
    return dt.astimezone(cfg.tz).strftime("%d/%m %H:%M") if dt else ""


def texto_propuesta(prop, cfg, decision=None):
    lineas = [
        f"<b>{LADO[prop['lado']]} {escape(prop['simbolo'])}</b> · propuesta #{prop['id']}",
        f"Cantidad: {prop['cantidad']:.4f} a ~{_usd(prop['precio_ref'])} = <b>{_usd(prop['monto'])}</b>",
        "",
        f"<b>Por qué:</b> {escape(prop['motivo'])}",
    ]
    if prop.get("opinion_ia"):
        lineas.append(
            f"<b>Segunda opinión IA ({ETIQUETA_OPINION.get(prop['opinion_ia'], '')}):</b> {escape(prop.get('explicacion_ia') or '')}"
        )
    if decision:
        lineas += ["", decision]
    else:
        modo = "💵 DINERO REAL" if cfg.dinero_real else "🧪 práctica (dinero ficticio)"
        lineas += ["", f"Vence: {_hora_local(prop['vence_en'], cfg)} · {modo}"]
    return "\n".join(lineas)


def botones_propuesta(prop):
    return [[
        {"text": "✅ Ejecutar", "callback_data": f"si:{prop['id']}"},
        {"text": "❌ Rechazar", "callback_data": f"no:{prop['id']}"},
    ]]


def texto_cartera(valor, aportes, cfg):
    ganancia = valor["total"] - aportes
    lineas = [
        "<b>📊 Tu cartera</b>",
        f"Valor: <b>{_usd(valor['total'])}</b> · Depositado: {_usd(aportes)}",
        f"Resultado: {_usd(ganancia)} ({(ganancia / aportes * 100) if aportes else 0:+.1f}%)",
        f"Efectivo: {_usd(valor['efectivo'])}",
    ]
    for p in sorted(valor["posiciones"].values(), key=lambda p: -p["valor"]):
        lineas.append(f"• {escape(p['simbolo'])}: {_usd(p['valor'])} ({p['peso']:.1f}%) {p['ganancia_pct']:+.1f}%")
    return "\n".join(lineas)


# ---------------- envío ----------------
def enviar_correo(cfg, asunto, texto):
    if not cfg.correo_configurado:
        return "sin configurar"
    msg = EmailMessage()
    msg["From"] = cfg.gmail_usuario
    msg["To"] = cfg.notificar_a
    msg["Subject"] = asunto
    msg.set_content(texto)
    with smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_puerto, timeout=30) as smtp:
        smtp.login(cfg.gmail_usuario, cfg.gmail_app_password)
        smtp.send_message(msg)
    return "ok"


def avisar(db, cfg, tg, texto, asunto=None):
    """Mensaje informativo por Telegram y, si se indica asunto, también por correo."""
    if tg:
        try:
            tg.enviar(texto)
        except ErrorTelegram as e:
            db.log(AGENTE, "Error de Telegram", str(e))
    if asunto:
        try:
            enviar_correo(cfg, asunto, texto.replace("<b>", "").replace("</b>", ""))
        except Exception as e:
            db.log(AGENTE, "Error de correo", str(e))


def consultar(db, cfg, tg, props, resumen=""):
    """Te pide aprobación de cada propuesta. Devuelve cuántas se enviaron por Telegram."""
    enviadas = 0
    if tg:
        try:
            if resumen:
                tg.enviar(f"🤖 <b>Tu agente tiene {len(props)} propuesta(s)</b>\n{escape(resumen)}")
            for prop in props:
                msg_id = tg.enviar(texto_propuesta(prop, cfg), botones_propuesta(prop))
                db.actualizar_propuesta(prop["id"], telegram_msg_id=msg_id)
                enviadas += 1
        except ErrorTelegram as e:
            db.log(AGENTE, "Error de Telegram", str(e))
    if cfg.correo_configurado:
        cuerpo = [resumen, ""] + [
            f"#{p['id']} {p['lado'].upper()} {p['simbolo']} {p['cantidad']:.4f} (~{_usd(p['monto'])})\n  {p['motivo']}"
            for p in props
        ]
        cuerpo += ["", "Apruébalas o recházalas en Telegram o en tu tablero" + (f": {cfg.tablero_url}" if cfg.tablero_url else ".")]
        try:
            enviar_correo(cfg, f"🤖 {len(props)} propuesta(s) de inversión esperan tu decisión", "\n".join(cuerpo))
        except Exception as e:
            db.log(AGENTE, "Error de correo", str(e))
    db.log(AGENTE, "Propuestas consultadas", f"{len(props)} propuestas · {enviadas} por Telegram")
    return enviadas


def leer_respuestas(db, tg):
    """Lee los botones y comandos que tocaste en Telegram.

    Devuelve (decisiones, comandos): decisiones = [(prop_id, 'si'|'no', callback_id)], comandos = ['/cartera', ...].
    Ignora (y registra) cualquier mensaje que no venga de tu chat.
    """
    offset = int(db.get_meta("telegram_offset", 0) or 0)
    decisiones, comandos = [], []
    for u in tg.actualizaciones(offset):
        offset = max(offset, u["update_id"] + 1)
        cq = u.get("callback_query")
        msg = u.get("message")
        if cq:
            chat = (cq.get("message") or {}).get("chat", {}).get("id")
            if not tg.es_mi_chat(chat, cq.get("from", {}).get("id")):
                db.log(AGENTE, "Respuesta ignorada", f"Botón tocado desde un chat no autorizado ({chat}).")
                continue
            accion, _, prop_id = (cq.get("data") or "").partition(":")
            if accion in ("si", "no") and prop_id.isdigit():
                decisiones.append((int(prop_id), accion, cq["id"]))
        elif msg and (msg.get("text") or "").startswith("/"):
            if tg.es_mi_chat(msg.get("chat", {}).get("id"), msg.get("from", {}).get("id")):
                comandos.append(msg["text"].split()[0].split("@")[0].lower())
    db.set_meta("telegram_offset", offset)
    return decisiones, comandos
