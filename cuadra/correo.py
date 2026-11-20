"""Envío de correos a clientes.

Sin SMTP configurado, o si el destinatario es de un dominio de ejemplo (.example,
usado en la demostración), el correo queda registrado como "simulado" y no sale.
"""

import logging
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr, make_msgid

log = logging.getLogger(__name__)

DOMINIOS_SIMULADOS = (".example", ".test", ".invalid", ".localhost")


def es_simulado(direccion):
    dominio = direccion.rsplit("@", 1)[-1].lower().strip()
    return not dominio or dominio.endswith(DOMINIOS_SIMULADOS) or dominio in ("example.com", "example.org")


def enviar(cfg, para, asunto, cuerpo, remitente_nombre="", responder_a=""):
    """Devuelve (estado, detalle) con estado en: enviado, simulado, error."""
    if es_simulado(para):
        return "simulado", "Destinatario de demostración: el correo no se envió."
    if not cfg.smtp_configurado:
        return "simulado", "SMTP sin configurar: el correo quedó registrado pero no se envió."
    msg = EmailMessage()
    remitente = cfg.smtp_remitente or cfg.smtp_usuario
    msg["From"] = formataddr((remitente_nombre, remitente)) if remitente_nombre else remitente
    msg["To"] = para
    msg["Subject"] = asunto
    msg["Message-ID"] = make_msgid(domain=remitente.rsplit("@", 1)[-1])
    if responder_a:
        msg["Reply-To"] = responder_a
    msg.set_content(cuerpo)
    try:
        contexto = ssl.create_default_context()
        if cfg.smtp_puerto == 465:
            with smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_puerto, context=contexto, timeout=30) as smtp:
                smtp.login(cfg.smtp_usuario, cfg.smtp_clave)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(cfg.smtp_host, cfg.smtp_puerto, timeout=30) as smtp:
                smtp.starttls(context=contexto)
                smtp.login(cfg.smtp_usuario, cfg.smtp_clave)
                smtp.send_message(msg)
    except (smtplib.SMTPException, OSError) as e:
        log.warning("No se pudo enviar el correo a %s: %s", para, e)
        return "error", f"No se pudo enviar: {e}"
    return "enviado", ""
