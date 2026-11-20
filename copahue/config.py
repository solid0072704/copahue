"""Configuración de Copahue desde variables de entorno (o un archivo .env en desarrollo)."""

import os
from dataclasses import dataclass
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv es opcional
    pass


def _env(nombre, defecto=""):
    valor = os.environ.get(nombre)
    return valor.strip() if valor and valor.strip() else defecto


def _env_int(nombre, defecto):
    try:
        return int(_env(nombre, str(defecto)))
    except ValueError:
        return defecto


def _env_bool(nombre, defecto=False):
    valor = _env(nombre)
    if not valor:
        return defecto
    return valor.lower() in ("1", "true", "si", "sí", "yes", "on")


@dataclass
class Config:
    db: str = "data/copahue.db"
    zona_horaria: str = "America/Santiago"
    # Fecha fija (AAAA-MM-DD) para demos y pruebas; vacío = hoy real.
    hoy_fijo: str = ""

    # Sesiones
    sesion_dias: int = 14
    cookie_segura: bool = False

    # Correo saliente (SMTP). Sin SMTP los correos quedan registrados, no se envían.
    smtp_host: str = ""
    smtp_puerto: int = 465
    smtp_usuario: str = ""
    smtp_clave: str = ""
    smtp_remitente: str = ""

    # Redacción asistida y barra de instrucciones (opcional)
    anthropic_api_key: str = ""
    modelo: str = "claude-opus-5-5"

    # Cada cuántos minutos el servidor corre las automatizaciones de los estudios (0 = nunca)
    intervalo_min: int = 30
    # Crea el estudio de demostración al iniciar
    demo: bool = True

    @property
    def tz(self):
        return ZoneInfo(self.zona_horaria)

    @property
    def smtp_configurado(self):
        return bool(self.smtp_host and self.smtp_usuario and self.smtp_clave)

    @property
    def ia_configurada(self):
        return bool(self.anthropic_api_key)


def cargar():
    return Config(
        db=_env("COPAHUE_DB", "data/copahue.db"),
        zona_horaria=_env("ZONA_HORARIA", "America/Santiago"),
        hoy_fijo=_env("COPAHUE_HOY"),
        sesion_dias=_env_int("COPAHUE_SESION_DIAS", 14),
        cookie_segura=_env_bool("COPAHUE_COOKIE_SEGURA", False),
        smtp_host=_env("SMTP_HOST"),
        smtp_puerto=_env_int("SMTP_PUERTO", 465),
        smtp_usuario=_env("SMTP_USUARIO"),
        smtp_clave=_env("SMTP_CLAVE"),
        smtp_remitente=_env("SMTP_REMITENTE"),
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        modelo=_env("COPAHUE_MODELO", "claude-opus-5-5"),
        intervalo_min=_env_int("COPAHUE_INTERVALO_MIN", 30),
        demo=_env_bool("COPAHUE_DEMO", True),
    )
