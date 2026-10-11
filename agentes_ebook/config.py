"""Configuración del sistema de agentes del ebook.

Todo se lee desde variables de entorno (o un archivo .env en desarrollo, o los
Secrets de Streamlit Cloud). Nunca escribas contraseñas en el código.
"""

import os
from dataclasses import dataclass, field
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
    valor = _env(nombre, "")
    if not valor:
        return defecto
    return valor.lower() in ("1", "true", "si", "sí", "yes", "on")


def _env_lista(nombre, defecto=""):
    return [x.strip() for x in _env(nombre, defecto).split(",") if x.strip()]


MODOS_SEGUIMIENTO = ("borradores", "automatico", "manual")


@dataclass
class Config:
    # --- Cuenta de Gmail que recibe las notificaciones de descarga ---
    gmail_usuario: str = ""
    gmail_app_password: str = ""
    imap_host: str = "imap.gmail.com"
    smtp_host: str = "smtp.gmail.com"
    smtp_puerto: int = 465

    # --- Cómo reconocer una notificación de descarga ---
    descargas_remitente: str = "submissions@formsubmit.co"
    descargas_asunto: str = "descarga del ebook"
    descargas_dias: int = 120
    excluir_correos: list = field(default_factory=list)

    # --- A quién avisan los agentes ---
    notificar_a: str = ""
    notificar_cc: list = field(default_factory=list)
    notificar_segmentos: list = field(default_factory=lambda: ["Caliente"])
    resumen_hora: int = 8

    # --- Tu marca y tu oferta ---
    empresa_nombre: str = "Hyperion Advisory"
    sitio_web: str = "https://www.hyperionadvisory.cl"
    firma: str = "Equipo Hyperion Advisory"
    ebook_nombre: str = "nuestro ebook"
    ebook_tema: str = "análisis financiero"
    oferta_reunion: str = "diagnóstico financiero sin costo de 30 minutos"
    link_agenda: str = ""
    whatsapp_negocio: str = ""

    # --- Reuniones ---
    zona_horaria: str = "America/Santiago"
    reunion_duracion_min: int = 30
    reunion_horas: list = field(default_factory=lambda: ["10:00", "12:30", "16:00"])

    # --- Seguimiento ---
    seguimiento_modo: str = "borradores"
    seguimiento_max_dia: int = 25
    mercado_objetivo: list = field(default_factory=lambda: ["Chile"])
    umbral_caliente: int = 70
    umbral_tibio: int = 40

    # --- IA (opcional) ---
    anthropic_api_key: str = ""
    llm_modelo: str = "claude-opus-5-5"
    llm_max_leads_por_ciclo: int = 15

    # --- Almacenamiento ---
    ruta_db: str = "data/tablero_ebook.db"

    @classmethod
    def desde_entorno(cls):
        modo = _env("SEGUIMIENTO_MODO", "borradores").lower()
        if modo not in MODOS_SEGUIMIENTO:
            modo = "borradores"
        usuario = _env("GMAIL_USER")
        notificar = _env("NOTIFY_EMAIL", usuario)
        excluir = {c.lower() for c in _env_lista("EXCLUIR_CORREOS")}
        excluir.update(c.lower() for c in (usuario, notificar) if c)
        return cls(
            gmail_usuario=usuario,
            gmail_app_password=_env("GMAIL_APP_PASSWORD").replace(" ", ""),
            imap_host=_env("IMAP_HOST", "imap.gmail.com"),
            smtp_host=_env("SMTP_HOST", "smtp.gmail.com"),
            smtp_puerto=_env_int("SMTP_PORT", 465),
            descargas_remitente=_env("DESCARGAS_REMITENTE", "submissions@formsubmit.co"),
            descargas_asunto=_env("DESCARGAS_ASUNTO", "descarga del ebook"),
            descargas_dias=_env_int("DESCARGAS_DIAS", 120),
            excluir_correos=sorted(excluir),
            notificar_a=notificar,
            notificar_cc=_env_lista("NOTIFY_CC"),
            notificar_segmentos=_env_lista("NOTIFICAR_SEGMENTOS", "Caliente"),
            resumen_hora=_env_int("RESUMEN_HORA", 8),
            empresa_nombre=_env("EMPRESA_NOMBRE", "Hyperion Advisory"),
            sitio_web=_env("SITIO_WEB", "https://www.hyperionadvisory.cl"),
            firma=_env("FIRMA", "Equipo Hyperion Advisory"),
            ebook_nombre=_env("EBOOK_NOMBRE", "nuestro ebook"),
            ebook_tema=_env("EBOOK_TEMA", "análisis financiero"),
            oferta_reunion=_env("OFERTA_REUNION", "diagnóstico financiero sin costo de 30 minutos"),
            link_agenda=_env("LINK_AGENDA"),
            whatsapp_negocio=_env("WHATSAPP_NEGOCIO"),
            zona_horaria=_env("ZONA_HORARIA", "America/Santiago"),
            reunion_duracion_min=_env_int("REUNION_DURACION_MIN", 30),
            reunion_horas=_env_lista("REUNION_HORAS", "10:00,12:30,16:00"),
            seguimiento_modo=modo,
            seguimiento_max_dia=_env_int("SEGUIMIENTO_MAX_DIA", 25),
            mercado_objetivo=_env_lista("MERCADO_OBJETIVO", "Chile"),
            umbral_caliente=_env_int("UMBRAL_CALIENTE", 70),
            umbral_tibio=_env_int("UMBRAL_TIBIO", 40),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            llm_modelo=_env("LLM_MODELO", "claude-opus-5-5"),
            llm_max_leads_por_ciclo=_env_int("LLM_MAX_LEADS_POR_CICLO", 15),
            ruta_db=_env("TABLERO_DB", "data/tablero_ebook.db"),
        )

    # --- Estado de conexiones ---
    @property
    def gmail_configurado(self):
        return bool(self.gmail_usuario and self.gmail_app_password)

    @property
    def ia_configurada(self):
        return bool(self.anthropic_api_key)

    @property
    def tz(self):
        return ZoneInfo(self.zona_horaria)


def aplicar_secrets_streamlit():
    """Copia los Secrets de Streamlit al entorno (para Streamlit Cloud)."""
    try:
        import streamlit as st

        for clave, valor in st.secrets.items():
            if isinstance(valor, (str, int, float, bool)) and clave not in os.environ:
                os.environ[clave] = str(valor)
    except Exception:
        # Sin secrets.toml: se usa solo el entorno / .env
        pass
