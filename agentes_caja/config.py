"""Configuración del agente de flujo de caja (variables de entorno o .env)."""

from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from agentes_ebook.config import _env, _env_bool, _env_int, _env_lista

ESTADOS_FORPAY_DEFECTO = "Pendiente,Pagada,En Proceso de Cobro,Repactada"


def _env_mapa(nombre, convertir=str):
    """'Empresa A=valor,Empresa B=valor2' -> {'Empresa A': valor, 'Empresa B': valor2}."""
    mapa = {}
    for par in _env_lista(nombre):
        clave, separador, valor = par.partition("=")
        if separador and clave.strip() and valor.strip():
            try:
                mapa[clave.strip()] = convertir(valor.strip())
            except ValueError:
                pass
    return mapa


def _env_float(nombre, defecto):
    try:
        return float(_env(nombre, str(defecto)).replace(",", "."))
    except ValueError:
        return defecto


@dataclass
class ConfigCaja:
    # --- Cuentas por cobrar: archivos CSV/Excel (planilla, contabilidad, banco) y, opcionalmente, la API de Forpay ---
    cobros_archivos: list = field(default_factory=list)  # rutas de CSV/Excel con las cuotas o facturas por cobrar
    cobros_campos: dict = field(default_factory=dict)  # {campo interno: columna del archivo} si la detección automática falla
    forpay_url: str = "https://t.forpayservices.cl/api-empresas-obtener-cuotas-v5"
    forpay_empresas: dict = field(default_factory=dict)  # {nombre de la sociedad: token}
    forpay_estados: list = field(default_factory=lambda: ESTADOS_FORPAY_DEFECTO.split(","))
    historial_meses: int = 6  # meses de cuotas pagadas que se consideran para medir la cobranza

    # --- Egresos y saldos ---
    egresos_archivo: str = "data/egresos.csv"
    saldo_inicial: float = 0.0
    saldos_por_empresa: dict = field(default_factory=dict)
    caja_minima: float = 0.0

    # --- Proyección ---
    horizonte_semanas: int = 13
    tasa_cobro_puntual: float = 0.85  # se reemplaza por la histórica cuando hay datos
    tasa_recuperacion_mora: float = 0.25
    semanas_recuperacion_mora: int = 4
    incluir_pipeline: bool = False  # sumar ventas comprometidas de HubSpot (pie ponderado por probabilidad)
    pie_porcentaje: float = 10.0
    pipeline_prob_minima: float = 0.5

    # --- Cobranza ---
    top_cobranza: int = 25
    moneda: str = "CLP"

    # --- Avisos (misma cuenta Gmail del resto de agentes) ---
    gmail_usuario: str = ""
    gmail_app_password: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_puerto: int = 465
    notificar_a: str = ""
    notificar_cc: list = field(default_factory=list)
    empresa_nombre: str = "Mi empresa"
    firma: str = "Agente de caja"
    zona_horaria: str = "America/Santiago"

    # --- IA opcional ---
    anthropic_api_key: str = ""
    llm_modelo: str = "claude-opus-5-5"

    # --- Memoria e informes ---
    ruta_db: str = "data/caja.db"
    carpeta_informes: str = "data/informes"

    @classmethod
    def desde_entorno(cls):
        usuario = _env("GMAIL_USER")
        return cls(
            forpay_url=_env("FORPAY_URL", "https://t.forpayservices.cl/api-empresas-obtener-cuotas-v5"),
            forpay_empresas=_env_mapa("FORPAY_EMPRESAS"),
            forpay_estados=_env_lista("FORPAY_ESTADOS", ESTADOS_FORPAY_DEFECTO),
            cobros_archivos=_env_lista("COBROS_ARCHIVOS"),
            cobros_campos=_env_mapa("COBROS_CAMPOS"),
            historial_meses=_env_int("CAJA_HISTORIAL_MESES", 6),
            egresos_archivo=_env("EGRESOS_ARCHIVO", "data/egresos.csv"),
            saldo_inicial=_env_float("CAJA_SALDO_INICIAL", 0.0),
            saldos_por_empresa=_env_mapa("CAJA_SALDOS", float),
            caja_minima=_env_float("CAJA_MINIMA", 0.0),
            horizonte_semanas=_env_int("CAJA_HORIZONTE_SEMANAS", 13),
            tasa_cobro_puntual=_env_float("CAJA_TASA_COBRO", 0.85),
            tasa_recuperacion_mora=_env_float("CAJA_TASA_MORA", 0.25),
            semanas_recuperacion_mora=_env_int("CAJA_SEMANAS_MORA", 4),
            incluir_pipeline=_env_bool("CAJA_INCLUIR_PIPELINE", False),
            pie_porcentaje=_env_float("CAJA_PIE_PORCENTAJE", 10.0),
            pipeline_prob_minima=_env_float("CAJA_PIPELINE_PROB_MINIMA", 0.5),
            top_cobranza=_env_int("CAJA_TOP_COBRANZA", 25),
            moneda=_env("CAJA_MONEDA", "CLP"),
            gmail_usuario=usuario,
            gmail_app_password=_env("GMAIL_APP_PASSWORD").replace(" ", ""),
            smtp_host=_env("SMTP_HOST", "smtp.gmail.com"),
            smtp_puerto=_env_int("SMTP_PORT", 465),
            notificar_a=_env("CAJA_NOTIFY_EMAIL", _env("NOTIFY_EMAIL", usuario)),
            notificar_cc=_env_lista("CAJA_NOTIFY_CC"),
            empresa_nombre=_env("EMPRESA_NOMBRE", "Mi empresa"),
            firma=_env("FIRMA", "Agente de caja"),
            zona_horaria=_env("ZONA_HORARIA", "America/Santiago"),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            llm_modelo=_env("LLM_MODELO", "claude-opus-5-5"),
            ruta_db=_env("CAJA_DB", "data/caja.db"),
            carpeta_informes=_env("CAJA_INFORMES", "data/informes"),
        )

    @property
    def cobros_configurado(self):
        return bool(self.cobros_archivos) or bool(self.forpay_empresas)

    @property
    def gmail_configurado(self):
        return bool(self.gmail_usuario and self.gmail_app_password)

    @property
    def ia_configurada(self):
        return bool(self.anthropic_api_key)

    @property
    def saldo_inicial_total(self):
        return self.saldo_inicial or sum(self.saldos_por_empresa.values())

    @property
    def tz(self):
        return ZoneInfo(self.zona_horaria)
