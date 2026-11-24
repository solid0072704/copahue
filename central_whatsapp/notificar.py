"""Aviso opcional por correo al vendedor (la tarea de HubSpot ya genera su propia notificación)."""

import logging
import smtplib
import ssl
from email.message import EmailMessage

log = logging.getLogger(__name__)


class Notificador:
    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def activo(self):
        return self.cfg.correo_configurado

    def enviar(self, para, asunto, texto):
        """Envía un correo de texto. Nunca interrumpe el ruteo si falla."""
        if not self.activo or not para:
            return False
        msg = EmailMessage()
        msg["From"] = self.cfg.smtp_remitente or self.cfg.smtp_usuario
        msg["To"] = para
        msg["Subject"] = asunto
        msg.set_content(texto)
        try:
            if self.cfg.smtp_puerto == 465:
                with smtplib.SMTP_SSL(self.cfg.smtp_host, 465, context=ssl.create_default_context(), timeout=20) as s:
                    s.login(self.cfg.smtp_usuario, self.cfg.smtp_password)
                    s.send_message(msg)
            else:
                with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_puerto, timeout=20) as s:
                    s.starttls(context=ssl.create_default_context())
                    s.login(self.cfg.smtp_usuario, self.cfg.smtp_password)
                    s.send_message(msg)
            return True
        except (smtplib.SMTPException, OSError) as e:
            log.warning("No se pudo enviar el aviso por correo: %s", type(e).__name__)
            return False
