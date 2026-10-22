"""Configuración del sistema de trading agéntico.

Todo se lee desde variables de entorno (o un archivo .env en desarrollo, o los
Secrets de Streamlit Cloud). Nunca escribas claves de tu broker en el código.
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


def _env_num(nombre, defecto):
    try:
        return float(_env(nombre, str(defecto)).replace(",", "."))
    except ValueError:
        return defecto


def _env_bool(nombre, defecto=False):
    valor = _env(nombre, "")
    if not valor:
        return defecto
    return valor.lower() in ("1", "true", "si", "sí", "yes", "on")


# Carteras modelo de ETFs listados en EE.UU. (pesos en %). Son diversificadas y de bajo costo:
# el agente no busca "adivinar" el mercado, sino mantener la cartera de tu perfil y protegerla.
PERFILES = {
    "conservador": {"SHY": 40, "AGG": 25, "VTI": 20, "VEA": 10, "GLD": 5},
    "moderado": {"VTI": 40, "VEA": 15, "VWO": 10, "AGG": 25, "GLD": 10},
    "agresivo": {"VTI": 45, "QQQ": 25, "VEA": 15, "VWO": 10, "GLD": 5},
}
# Donde se refugia el peso de un activo que pierde su tendencia (bonos del Tesoro de corto plazo).
ACTIVO_REFUGIO = "SHY"
BROKERS = ("simulado", "alpaca")
MERCADOS = ("yahoo", "sintetico")


def parsear_cartera(texto):
    """'VTI:60, AGG:40' → {'VTI': 60.0, 'AGG': 40.0}"""
    cartera = {}
    for parte in texto.split(","):
        if ":" not in parte:
            continue
        simbolo, peso = parte.split(":", 1)
        try:
            cartera[simbolo.strip().upper()] = float(peso)
        except ValueError:
            continue
    return {s: p for s, p in cartera.items() if s and p > 0}


@dataclass
class Config:
    # --- Broker ---
    broker: str = "simulado"
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True
    # Para operar con dinero real hay que desactivar el modo paper Y escribir esta confirmación.
    trading_real_confirmado: bool = False
    comision_simulada_pct: float = 0.1

    # --- Datos de mercado ---
    mercado: str = "yahoo"

    # --- Estrategia ---
    perfil: str = "moderado"
    cartera_objetivo: dict = field(default_factory=dict)
    filtro_tendencia: bool = True
    media_tendencia_dias: int = 200
    umbral_rebalanceo_pct: float = 5.0
    plan_cada_horas: float = 24

    # --- Límites de riesgo (la Guardia los aplica a cada orden, incluso si tú la apruebas) ---
    reserva_efectivo_pct: float = 3.0
    max_posicion_pct: float = 50.0
    min_orden_usd: float = 10.0
    max_ordenes_dia: int = 12
    stop_perdida_pct: float = 15.0
    freno_perdida_pct: float = 20.0
    desvio_precio_max_pct: float = 2.0
    vigencia_propuesta_horas: float = 24

    # --- Cómo te consulta el agente ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    gmail_usuario: str = ""
    gmail_app_password: str = ""
    notificar_a: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_puerto: int = 465
    tablero_url: str = ""

    # --- IA (opcional) ---
    anthropic_api_key: str = ""
    llm_modelo: str = "claude-opus-5-5"

    # --- General ---
    zona_horaria: str = "America/Santiago"
    ruta_db: str = "data/trading.db"

    @classmethod
    def desde_entorno(cls):
        broker = _env("BROKER", "simulado").lower()
        perfil = _env("PERFIL_RIESGO", "moderado").lower()
        mercado = _env("MERCADO_DATOS", "yahoo").lower()
        usuario = _env("GMAIL_USER")
        return cls(
            broker=broker if broker in BROKERS else "simulado",
            alpaca_api_key=_env("ALPACA_API_KEY"),
            alpaca_secret_key=_env("ALPACA_SECRET_KEY"),
            alpaca_paper=_env_bool("ALPACA_PAPER", True),
            trading_real_confirmado=_env("TRADING_REAL") == "ENTIENDO_EL_RIESGO",
            comision_simulada_pct=_env_num("COMISION_SIMULADA_PCT", 0.1),
            mercado=mercado if mercado in MERCADOS else "yahoo",
            perfil=perfil if perfil in PERFILES else "moderado",
            cartera_objetivo=parsear_cartera(_env("CARTERA_OBJETIVO")),
            filtro_tendencia=_env_bool("FILTRO_TENDENCIA", True),
            media_tendencia_dias=int(_env_num("MEDIA_TENDENCIA_DIAS", 200)),
            umbral_rebalanceo_pct=_env_num("UMBRAL_REBALANCEO_PCT", 5.0),
            plan_cada_horas=_env_num("PLAN_CADA_HORAS", 24),
            reserva_efectivo_pct=_env_num("RESERVA_EFECTIVO_PCT", 3.0),
            max_posicion_pct=_env_num("MAX_POSICION_PCT", 50.0),
            min_orden_usd=_env_num("MIN_ORDEN_USD", 10.0),
            max_ordenes_dia=int(_env_num("MAX_ORDENES_DIA", 12)),
            stop_perdida_pct=_env_num("STOP_PERDIDA_PCT", 15.0),
            freno_perdida_pct=_env_num("FRENO_PERDIDA_PCT", 20.0),
            desvio_precio_max_pct=_env_num("DESVIO_PRECIO_MAX_PCT", 2.0),
            vigencia_propuesta_horas=_env_num("VIGENCIA_PROPUESTA_HORAS", 24),
            telegram_bot_token=_env("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_env("TELEGRAM_CHAT_ID"),
            gmail_usuario=usuario,
            gmail_app_password=_env("GMAIL_APP_PASSWORD").replace(" ", ""),
            notificar_a=_env("NOTIFY_EMAIL", usuario),
            smtp_host=_env("SMTP_HOST", "smtp.gmail.com"),
            smtp_puerto=int(_env_num("SMTP_PORT", 465)),
            tablero_url=_env("TABLERO_TRADING_URL"),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            llm_modelo=_env("LLM_MODELO", "claude-opus-5-5"),
            zona_horaria=_env("ZONA_HORARIA", "America/Santiago"),
            ruta_db=_env("TRADING_DB", "data/trading.db"),
        )

    # --- Cartera y universo ---
    @property
    def pesos_objetivo(self):
        """Pesos objetivo normalizados a 100, sin superar el máximo por posición."""
        base = self.cartera_objetivo or PERFILES[self.perfil]
        total = sum(base.values())
        return {s: min(p * 100 / total, self.max_posicion_pct) for s, p in base.items()}

    @property
    def universo(self):
        """Únicos símbolos que el agente puede comprar. La IA no puede agregar otros."""
        return sorted(set(self.pesos_objetivo) | {ACTIVO_REFUGIO})

    # --- Estado de conexiones ---
    @property
    def dinero_real(self):
        return self.broker == "alpaca" and not self.alpaca_paper

    @property
    def broker_listo(self):
        if self.broker == "simulado":
            return True, "Broker simulado: practicas con dinero ficticio y precios reales."
        if not (self.alpaca_api_key and self.alpaca_secret_key):
            return False, "Faltan ALPACA_API_KEY / ALPACA_SECRET_KEY."
        if self.dinero_real and not self.trading_real_confirmado:
            return False, "ALPACA_PAPER=false requiere TRADING_REAL=ENTIENDO_EL_RIESGO para operar con dinero real."
        return True, "Alpaca con dinero REAL." if self.dinero_real else "Alpaca en modo paper (dinero ficticio)."

    @property
    def telegram_configurado(self):
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def correo_configurado(self):
        return bool(self.gmail_usuario and self.gmail_app_password and self.notificar_a)

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
        pass
