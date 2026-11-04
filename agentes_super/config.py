"""Configuración del agente de compras de supermercado.

Todo se lee desde variables de entorno (o un archivo .env, o los Secrets de Streamlit).
"""

from dataclasses import dataclass
from zoneinfo import ZoneInfo

from agentes_ebook.config import _env, _env_int, aplicar_secrets_streamlit  # noqa: F401 - se reexporta


def _env_float(nombre, defecto):
    try:
        return float(_env(nombre, str(defecto)).replace(",", "."))
    except ValueError:
        return defecto


@dataclass
class ConfigSuper:
    # --- Correo para avisos (opcional; mismo Gmail que el tablero del ebook) ---
    gmail_usuario: str = ""
    gmail_app_password: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_puerto: int = 465
    notificar_a: str = ""
    firma: str = "Agente de compras"
    empresa_nombre: str = "Agente de compras"

    # --- Optimización ---
    max_tiendas: int = 2  # en cuántos supermercados como máximo se divide una compra
    dias_anticipacion: int = 1  # cuántos días antes de cada compra recurrente se cotiza y avisa
    umbral_oferta: float = 0.15  # 15 % bajo la mediana de 30 días = oferta para abastecerse
    resultados_por_busqueda: int = 24
    pausa_entre_consultas: float = 0.4  # segundos, para no saturar a los supermercados

    # --- IA opcional ---
    anthropic_api_key: str = ""
    llm_modelo: str = "claude-opus-5-5"
    llm_max_consultas_por_ciclo: int = 40

    zona_horaria: str = "America/Santiago"
    ruta_db: str = "data/supermercado.db"

    @classmethod
    def desde_entorno(cls):
        usuario = _env("GMAIL_USER")
        return cls(
            gmail_usuario=usuario,
            gmail_app_password=_env("GMAIL_APP_PASSWORD").replace(" ", ""),
            smtp_host=_env("SMTP_HOST", "smtp.gmail.com"),
            smtp_puerto=_env_int("SMTP_PORT", 465),
            notificar_a=_env("SUPER_NOTIFY_EMAIL", _env("NOTIFY_EMAIL", usuario)),
            firma=_env("SUPER_FIRMA", "Agente de compras"),
            empresa_nombre=_env("SUPER_FIRMA", "Agente de compras"),
            max_tiendas=max(1, _env_int("SUPER_MAX_TIENDAS", 2)),
            dias_anticipacion=max(0, _env_int("SUPER_DIAS_ANTICIPACION", 1)),
            umbral_oferta=_env_float("SUPER_UMBRAL_OFERTA", 0.15),
            resultados_por_busqueda=_env_int("SUPER_RESULTADOS", 24),
            pausa_entre_consultas=_env_float("SUPER_PAUSA", 0.4),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            llm_modelo=_env("LLM_MODELO", "claude-opus-5-5"),
            llm_max_consultas_por_ciclo=_env_int("SUPER_LLM_MAX", 40),
            zona_horaria=_env("ZONA_HORARIA", "America/Santiago"),
            ruta_db=_env("SUPER_DB", "data/supermercado.db"),
        )

    @property
    def gmail_configurado(self):
        return bool(self.gmail_usuario and self.gmail_app_password)

    @property
    def ia_configurada(self):
        return bool(self.anthropic_api_key)

    @property
    def tz(self):
        return ZoneInfo(self.zona_horaria)
