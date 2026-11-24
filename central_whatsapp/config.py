"""Configuración técnica de la central de WhatsApp.

Todo se lee desde variables de entorno (o un archivo .env en desarrollo).
Las reglas de negocio (qué vendedor atiende qué proyecto, horarios, SLA) van en
el archivo de ruteo (ver ruteo.ejemplo.yaml), no aquí.
"""

import os
from dataclasses import dataclass

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
    valor = _env(nombre, "")
    if not valor:
        return defecto
    return valor.lower() in ("1", "true", "si", "sí", "yes", "on")


# Valor que se escribe en la propiedad "Fuente" de los negocios creados por la central.
FUENTE_CENTRAL = "WhatsApp Central"

# Cuándo cambiar el propietario del contacto al vendedor asignado:
#   siempre  -> el contacto pasa al último vendedor asignado
#   si_libre -> solo si no tiene propietario, si el propietario está inactivo o si no tiene
#               negocios abiertos con ese contacto (recomendado)
#   nunca    -> solo se asigna el negocio y la tarea
POLITICAS_PROPIETARIO = ("siempre", "si_libre", "nunca")


@dataclass
class ConfigCentral:
    # --- HubSpot (app privada: Configuración > Integraciones > Apps privadas) ---
    hubspot_token: str = ""
    hubspot_portal_id: str = ""

    # --- Seguridad del webhook (el workflow de HubSpot envía este secreto en un header) ---
    webhook_secreto: str = ""
    webhook_header: str = "X-Central-Token"

    # --- Reglas de ruteo ---
    ruteo_archivo: str = "central_whatsapp/ruteo.yaml"
    ruteo_yaml: str = ""  # contenido YAML completo (alternativa al archivo, ideal para un Secret)

    # --- Comportamiento ---
    fuente: str = FUENTE_CENTRAL
    propietario_contacto: str = "si_libre"
    ventana_duplicado_min: int = 30  # webhooks repetidos dentro de esta ventana no crean otro negocio
    simular: bool = False  # True = lee HubSpot pero no escribe nada (para probar en producción)

    # --- Consulta periódica (reemplaza al webhook): cada cuántos segundos buscar pendientes ---
    consulta_cada_seg: int = 60

    # --- SLA: reasignar leads no atendidos ---
    sla_en_servidor: bool = True  # el servicio web revisa el SLA en segundo plano
    sla_cada_min: int = 5

    # --- Aviso opcional por correo al vendedor (además de la tarea de HubSpot) ---
    smtp_host: str = ""
    smtp_puerto: int = 465
    smtp_usuario: str = ""
    smtp_password: str = ""
    smtp_remitente: str = ""

    @classmethod
    def desde_entorno(cls):
        politica = _env("CENTRAL_PROPIETARIO_CONTACTO", "si_libre").lower()
        if politica not in POLITICAS_PROPIETARIO:
            politica = "si_libre"
        usuario_smtp = _env("CENTRAL_SMTP_USER")
        return cls(
            hubspot_token=_env("HUBSPOT_TOKEN"),
            hubspot_portal_id=_env("HUBSPOT_PORTAL_ID"),
            webhook_secreto=_env("CENTRAL_WEBHOOK_SECRET"),
            webhook_header=_env("CENTRAL_WEBHOOK_HEADER", "X-Central-Token"),
            ruteo_archivo=_env("CENTRAL_RUTEO_ARCHIVO", "central_whatsapp/ruteo.yaml"),
            ruteo_yaml=os.environ.get("CENTRAL_RUTEO_YAML", "").strip(),
            fuente=_env("CENTRAL_FUENTE", FUENTE_CENTRAL),
            propietario_contacto=politica,
            ventana_duplicado_min=_env_int("CENTRAL_VENTANA_DUPLICADO_MIN", 30),
            simular=_env_bool("CENTRAL_SIMULAR", False),
            consulta_cada_seg=max(15, _env_int("CENTRAL_CONSULTA_CADA_SEG", 60)),
            sla_en_servidor=_env_bool("CENTRAL_SLA_EN_SERVIDOR", True),
            sla_cada_min=max(1, _env_int("CENTRAL_SLA_CADA_MIN", 5)),
            smtp_host=_env("CENTRAL_SMTP_HOST"),
            smtp_puerto=_env_int("CENTRAL_SMTP_PORT", 465),
            smtp_usuario=usuario_smtp,
            smtp_password=_env("CENTRAL_SMTP_PASSWORD").replace(" ", ""),
            smtp_remitente=_env("CENTRAL_SMTP_FROM", usuario_smtp),
        )

    @property
    def hubspot_configurado(self):
        return bool(self.hubspot_token)

    @property
    def correo_configurado(self):
        return bool(self.smtp_host and self.smtp_usuario and self.smtp_password)
