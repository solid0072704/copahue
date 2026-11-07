"""Configuración del agente de trading (variables de entorno o .env)."""

import os
from dataclasses import dataclass, field

from agentes_ebook.config import _env, _env_bool, _env_int, _env_lista

MODOS = ("simulado", "alpaca_papel", "alpaca_real")

# ETFs líquidos y diversificados: acciones de EE.UU., internacionales, oro y bonos.
UNIVERSO_DEFECTO = "SPY,QQQ,IWM,EFA,EEM,GLD,TLT,IEF"


def _env_float(nombre, defecto):
    try:
        return float(_env(nombre, str(defecto)))
    except ValueError:
        return defecto


@dataclass
class ConfigTrading:
    # --- Modo de operación ---
    # simulado      = cartera ficticia que vive en un archivo JSON (recomendado para empezar)
    # alpaca_papel  = cuenta de práctica de Alpaca (dinero ficticio, mercado real)
    # alpaca_real   = cuenta real de Alpaca: SOLO si además CONFIRMO_DINERO_REAL=true
    modo: str = "simulado"
    confirmo_dinero_real: bool = False
    alpaca_key: str = ""
    alpaca_secret: str = ""

    # --- Capital y objetivo ---
    capital_inicial: float = 10_000.0
    meta_usd: float = 1_000.0

    # --- Estrategia (momentum con filtro de tendencia) ---
    universo: list = field(default_factory=lambda: UNIVERSO_DEFECTO.split(","))
    referencia: str = "SPY"
    max_posiciones: int = 3
    ventana_momentum: int = 126      # ~6 meses de ruedas
    ventana_tendencia: int = 200     # media móvil de 200 ruedas
    ventana_volatilidad: int = 20
    rebalanceo_dias: int = 21        # ~1 mes
    stop_loss_pct: float = 0.08      # vende una posición que cae 8% desde su compra
    max_caida_cartera_pct: float = 0.15  # con -15% desde el máximo, todo a efectivo
    comision_pct: float = 0.0005     # 0,05% por operación (comisión + deslizamiento)
    reserva_efectivo_pct: float = 0.02

    # --- Datos ---
    años_historia: int = 10
    ruta_datos: str = "data/trading"

    # --- Avisos por correo (mismas credenciales que el tablero del ebook) ---
    gmail_usuario: str = ""
    gmail_app_password: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_puerto: int = 465
    notificar_a: str = ""
    firma: str = "Agente de trading"
    empresa_nombre: str = "Copahue"
    avisar_solo_con_operaciones: bool = False

    @classmethod
    def desde_entorno(cls):
        modo = _env("TRADING_MODO", "simulado").lower()
        if modo not in MODOS:
            modo = "simulado"
        usuario = _env("GMAIL_USER")
        return cls(
            modo=modo,
            confirmo_dinero_real=_env_bool("CONFIRMO_DINERO_REAL", False),
            alpaca_key=_env("ALPACA_API_KEY"),
            alpaca_secret=_env("ALPACA_SECRET_KEY"),
            capital_inicial=_env_float("CAPITAL_INICIAL", 10_000.0),
            meta_usd=_env_float("META_USD", 1_000.0),
            universo=_env_lista("UNIVERSO", UNIVERSO_DEFECTO),
            referencia=_env("REFERENCIA", "SPY"),
            max_posiciones=_env_int("MAX_POSICIONES", 3),
            ventana_momentum=_env_int("VENTANA_MOMENTUM", 126),
            ventana_tendencia=_env_int("VENTANA_TENDENCIA", 200),
            ventana_volatilidad=_env_int("VENTANA_VOLATILIDAD", 20),
            rebalanceo_dias=_env_int("REBALANCEO_DIAS", 21),
            stop_loss_pct=_env_float("STOP_LOSS_PCT", 0.08),
            max_caida_cartera_pct=_env_float("MAX_CAIDA_CARTERA_PCT", 0.15),
            comision_pct=_env_float("COMISION_PCT", 0.0005),
            reserva_efectivo_pct=_env_float("RESERVA_EFECTIVO_PCT", 0.02),
            años_historia=_env_int("ANIOS_HISTORIA", 10),
            ruta_datos=_env("TRADING_DATOS", "data/trading"),
            gmail_usuario=usuario,
            gmail_app_password=_env("GMAIL_APP_PASSWORD").replace(" ", ""),
            smtp_host=_env("SMTP_HOST", "smtp.gmail.com"),
            smtp_puerto=_env_int("SMTP_PORT", 465),
            notificar_a=_env("NOTIFY_EMAIL", usuario),
            firma=_env("FIRMA", "Agente de trading"),
            empresa_nombre=_env("EMPRESA_NOMBRE", "Copahue"),
            avisar_solo_con_operaciones=_env_bool("AVISAR_SOLO_CON_OPERACIONES", False),
        )

    @property
    def gmail_configurado(self):
        return bool(self.gmail_usuario and self.gmail_app_password)

    @property
    def dinero_real(self):
        """True solo cuando el dueño de la cuenta lo confirmó de forma explícita."""
        return self.modo == "alpaca_real" and self.confirmo_dinero_real

    @property
    def alpaca_configurado(self):
        return bool(self.alpaca_key and self.alpaca_secret)

    @property
    def ruta_cartera(self):
        return os.path.join(self.ruta_datos, "cartera.json")

    @property
    def ruta_precios(self):
        return os.path.join(self.ruta_datos, "precios")

    def validar(self):
        """Devuelve una lista de problemas de configuración (vacía si todo está bien)."""
        problemas = []
        if self.modo.startswith("alpaca") and not self.alpaca_configurado:
            problemas.append("Faltan ALPACA_API_KEY / ALPACA_SECRET_KEY para operar con Alpaca.")
        if self.modo == "alpaca_real" and not self.confirmo_dinero_real:
            problemas.append("TRADING_MODO=alpaca_real requiere CONFIRMO_DINERO_REAL=true. Mientras tanto se usa la cuenta de práctica.")
        if self.max_posiciones < 1:
            problemas.append("MAX_POSICIONES debe ser al menos 1.")
        if self.referencia not in self.universo:
            problemas.append(f"La referencia {self.referencia} debe estar dentro de UNIVERSO.")
        return problemas
