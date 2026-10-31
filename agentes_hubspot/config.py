"""Configuración de los agentes del embudo de HubSpot.

Todo se lee desde variables de entorno (o un archivo .env en desarrollo). Nunca escribas
el token en el código.
"""

from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from agentes_ebook.config import _env, _env_int, _env_lista

MODOS = ("lectura", "tareas")

PROPIEDADES_EXTRA_DEFECTO = (
    "razon_perdido,closed_lost_reason,fecha_acordada_para_proximo_contacto,proximo_paso_acordado_con_el_cliente"
)


def _env_mapa_dias(nombre):
    """'Sin Contacto=2,Promesa Firmada=30' -> {'sin contacto': 2, 'promesa firmada': 30}."""
    mapa = {}
    for par in _env_lista(nombre):
        clave, separador, valor = par.partition("=")
        if separador and valor.strip().isdigit():
            mapa[clave.strip().lower()] = int(valor)
    return mapa


@dataclass
class ConfigHubSpot:
    # --- Conexión con HubSpot (token de app privada) ---
    token: str = ""
    pipeline: str = "default"
    modo: str = "lectura"  # lectura = solo informa · tareas = además crea tareas en HubSpot
    api_base: str = "https://api.hubapi.com"
    portal_id: str = ""
    propiedades_extra: list = field(default_factory=lambda: PROPIEDADES_EXTRA_DEFECTO.split(","))
    propiedades_razon_perdida: list = field(default_factory=lambda: ["razon_perdido", "closed_lost_reason"])

    # --- Analista de embudo ---
    embudo_dias: int = 30  # ventana para conversión y razones de pérdida
    sin_contacto_dias: int = 14  # desde cuántos días un negocio cuenta como "sin contacto"
    comparar_dias: int = 7  # con qué foto anterior se compara el diagnóstico

    # --- Desatascador: días en la etapa a partir de los cuales un negocio está estancado ---
    dias_primer_contacto: int = 2
    dias_prospeccion: int = 10
    dias_visita: int = 3
    dias_negociacion: int = 14
    dias_administrativo: int = 45
    dias_por_etapa: dict = field(default_factory=dict)  # umbrales por nombre o id de etapa
    repetir_tarea_dias: int = 7  # no crear otra tarea para el mismo negocio antes de estos días
    escalar_dias: int = 7  # si sigue estancado tantos días después de la tarea, se escala
    max_tareas_por_ciclo: int = 40
    tarea_vence_dias: int = 1
    supervisor_id: str = ""  # propietario de HubSpot que recibe las escalaciones
    max_estancados_informe: int = 40

    # --- Avisos por correo (misma cuenta Gmail de los agentes del ebook) ---
    gmail_usuario: str = ""
    gmail_app_password: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_puerto: int = 465
    notificar_a: str = ""
    notificar_cc: list = field(default_factory=list)
    empresa_nombre: str = "Mi empresa"
    firma: str = "Agentes del embudo"
    zona_horaria: str = "America/Santiago"

    # --- IA opcional (Claude) ---
    anthropic_api_key: str = ""
    llm_modelo: str = "claude-opus-5-5"

    # --- Memoria e informes ---
    ruta_db: str = "data/embudo_hubspot.db"
    carpeta_informes: str = "data/informes"

    @classmethod
    def desde_entorno(cls):
        modo = _env("HUBSPOT_MODO", "lectura").lower()
        if modo not in MODOS:
            modo = "lectura"
        usuario = _env("GMAIL_USER")
        return cls(
            token=_env("HUBSPOT_TOKEN"),
            pipeline=_env("HUBSPOT_PIPELINE", "default"),
            modo=modo,
            api_base=_env("HUBSPOT_API_BASE", "https://api.hubapi.com").rstrip("/"),
            portal_id=_env("HUBSPOT_PORTAL_ID"),
            propiedades_extra=_env_lista("HUBSPOT_PROPIEDADES_EXTRA", PROPIEDADES_EXTRA_DEFECTO),
            propiedades_razon_perdida=_env_lista("HUBSPOT_RAZON_PERDIDA", "razon_perdido,closed_lost_reason"),
            embudo_dias=_env_int("EMBUDO_DIAS", 30),
            sin_contacto_dias=_env_int("EMBUDO_SIN_CONTACTO_DIAS", 14),
            comparar_dias=_env_int("EMBUDO_COMPARAR_DIAS", 7),
            dias_primer_contacto=_env_int("DESATASCAR_DIAS_PRIMER_CONTACTO", 2),
            dias_prospeccion=_env_int("DESATASCAR_DIAS_PROSPECCION", 10),
            dias_visita=_env_int("DESATASCAR_DIAS_VISITA", 3),
            dias_negociacion=_env_int("DESATASCAR_DIAS_NEGOCIACION", 14),
            dias_administrativo=_env_int("DESATASCAR_DIAS_ADMINISTRATIVO", 45),
            dias_por_etapa=_env_mapa_dias("DESATASCAR_DIAS_POR_ETAPA"),
            repetir_tarea_dias=_env_int("DESATASCAR_REPETIR_DIAS", 7),
            escalar_dias=_env_int("DESATASCAR_ESCALAR_DIAS", 7),
            max_tareas_por_ciclo=_env_int("DESATASCAR_MAX_TAREAS", 40),
            tarea_vence_dias=_env_int("DESATASCAR_TAREA_VENCE_DIAS", 1),
            supervisor_id=_env("HUBSPOT_SUPERVISOR_ID"),
            max_estancados_informe=_env_int("INFORME_MAX_ESTANCADOS", 40),
            gmail_usuario=usuario,
            gmail_app_password=_env("GMAIL_APP_PASSWORD").replace(" ", ""),
            smtp_host=_env("SMTP_HOST", "smtp.gmail.com"),
            smtp_puerto=_env_int("SMTP_PORT", 465),
            notificar_a=_env("EMBUDO_NOTIFY_EMAIL", _env("NOTIFY_EMAIL", usuario)),
            notificar_cc=_env_lista("EMBUDO_NOTIFY_CC"),
            empresa_nombre=_env("EMPRESA_NOMBRE", "Mi empresa"),
            firma=_env("FIRMA", "Agentes del embudo"),
            zona_horaria=_env("ZONA_HORARIA", "America/Santiago"),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            llm_modelo=_env("LLM_MODELO", "claude-opus-5-5"),
            ruta_db=_env("EMBUDO_DB", "data/embudo_hubspot.db"),
            carpeta_informes=_env("EMBUDO_INFORMES", "data/informes"),
        )

    @property
    def hubspot_configurado(self):
        return bool(self.token)

    @property
    def gmail_configurado(self):
        return bool(self.gmail_usuario and self.gmail_app_password)

    @property
    def ia_configurada(self):
        return bool(self.anthropic_api_key)

    @property
    def escribe_en_hubspot(self):
        return self.modo == "tareas"

    @property
    def tz(self):
        return ZoneInfo(self.zona_horaria)
