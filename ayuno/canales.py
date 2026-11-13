"""Canales para escribirte al celular: Telegram, WhatsApp (CallMeBot) y ntfy.

Solo usa la biblioteca estándar. Los errores nunca muestran tokens ni teléfonos.
"""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

TIMEOUT = 20


class ErrorCanal(RuntimeError):
    pass


def _http(url, datos=None, cabeceras=None):
    pedido = urllib.request.Request(url, data=datos, headers=cabeceras or {}, method="POST" if datos else "GET")
    try:
        with urllib.request.urlopen(pedido, timeout=TIMEOUT) as respuesta:
            return respuesta.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        cuerpo = ""
        try:
            cuerpo = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        raise ErrorCanal(f"HTTP {e.code} {cuerpo}".strip()) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        motivo = getattr(e, "reason", e)
        raise ErrorCanal(f"sin conexión ({motivo})") from None


class Telegram:
    nombre = "telegram"

    def __init__(self, token, chat_id):
        self.token, self.chat_id = token, chat_id

    def enviar(self, texto):
        datos = json.dumps({"chat_id": self.chat_id, "text": texto}).encode("utf-8")
        respuesta = _http(
            f"https://api.telegram.org/bot{self.token}/sendMessage",
            datos,
            {"Content-Type": "application/json"},
        )
        if not json.loads(respuesta).get("ok"):
            raise ErrorCanal("Telegram no aceptó el mensaje")


class WhatsAppCallMeBot:
    nombre = "whatsapp"

    def __init__(self, telefono, apikey):
        self.telefono, self.apikey = telefono, apikey

    def enviar(self, texto):
        consulta = urllib.parse.urlencode({"phone": self.telefono, "text": texto, "apikey": self.apikey})
        respuesta = _http(f"https://api.callmebot.com/whatsapp.php?{consulta}")
        # CallMeBot responde 200 incluso con algunos errores: revisamos el texto.
        bajo = respuesta.lower()
        if ("error" in bajo or "invalid" in bajo) and "queued" not in bajo:
            raise ErrorCanal("CallMeBot rechazó el mensaje (revisa tu teléfono y APIKEY)")


class Ntfy:
    nombre = "ntfy"

    def __init__(self, servidor, tema):
        self.servidor, self.tema = servidor, tema

    def enviar(self, texto):
        _http(
            f"{self.servidor}/{urllib.parse.quote(self.tema)}",
            texto.encode("utf-8"),
            {"Title": "Ayuno intermitente", "Tags": "fork_and_knife"},
        )


def crear_canales(cfg):
    canales = []
    if cfg.telegram_token and cfg.telegram_chat_id:
        canales.append(Telegram(cfg.telegram_token, cfg.telegram_chat_id))
    if cfg.whatsapp_telefono and cfg.callmebot_apikey:
        canales.append(WhatsAppCallMeBot(cfg.whatsapp_telefono, cfg.callmebot_apikey))
    if cfg.ntfy_tema:
        canales.append(Ntfy(cfg.ntfy_servidor, cfg.ntfy_tema))
    return canales


def enviar_a_todos(canales, texto):
    """Envía a cada canal. Devuelve los nombres de los canales que funcionaron."""
    ok = []
    for canal in canales:
        try:
            canal.enviar(texto)
            ok.append(canal.nombre)
        except ErrorCanal as e:
            log.warning("No pude enviar por %s: %s", canal.nombre, e)
    return ok


def chats_de_telegram(token):
    """Chats que le escribieron a tu bot (para encontrar tu TELEGRAM_CHAT_ID)."""
    respuesta = json.loads(_http(f"https://api.telegram.org/bot{token}/getUpdates"))
    chats = {}
    for actualizacion in respuesta.get("result", []):
        mensaje = actualizacion.get("message") or actualizacion.get("edited_message") or {}
        chat = mensaje.get("chat")
        if chat:
            nombre = chat.get("first_name") or chat.get("title") or chat.get("username") or ""
            chats[chat["id"]] = nombre
    return chats
