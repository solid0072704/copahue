"""Configuración de la tienda y de sus agentes.

Todo se lee desde variables de entorno (o un archivo .env en desarrollo, o los Secrets
de GitHub / Streamlit). Nunca escribas claves de API en el código.
"""

import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

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


def _env_float(nombre, defecto):
    try:
        return float(_env(nombre, str(defecto)).replace(",", "."))
    except ValueError:
        return defecto


def _env_bool(nombre, defecto=False):
    valor = _env(nombre, "")
    if not valor:
        return defecto
    return valor.lower() in ("1", "true", "si", "sí", "yes", "on")


def _env_lista(nombre, defecto="", sep=","):
    return [x.strip() for x in _env(nombre, defecto).split(sep) if x.strip()]


MODOS_VENTA = ("afiliado", "dropshipping")

# Nicho visible en la tienda : palabras con que se busca en los proveedores (en inglés rinden más).
NICHOS_DEFECTO = (
    "Hogar y cocina: kitchen gadgets, home organizer;"
    "Mascotas: pet supplies, dog accessories;"
    "Bienestar: massager, personal care;"
    "Accesorios para auto: car accessories;"
    "Tecnología útil: phone accessories, smart gadgets;"
    "Fitness en casa: home fitness, resistance bands"
)


def parsear_nichos(texto):
    """'Cocina: kitchen gadget, organizer; Mascotas' -> [{'nombre': 'Cocina', 'busquedas': [...]}, ...]"""
    nichos = []
    for bloque in texto.split(";"):
        bloque = bloque.strip()
        if not bloque:
            continue
        nombre, _, busquedas = bloque.partition(":")
        nombre = nombre.strip()
        lista = [b.strip() for b in busquedas.split(",") if b.strip()] or [nombre]
        if nombre:
            nichos.append({"nombre": nombre, "busquedas": lista})
    return nichos


@dataclass
class Config:
    # --- La tienda ---
    nombre: str = "Copahue Market"
    eslogan: str = "Productos que resuelven problemas, con envío a todo Chile"
    url: str = ""  # https://tu-usuario.github.io/copahue  o  https://www.tudominio.cl
    dominio_propio: str = ""  # se escribe en el archivo CNAME de GitHub Pages
    pais: str = "CL"
    idioma: str = "es-CL"
    moneda: str = "CLP"
    correo_contacto: str = ""
    whatsapp: str = ""
    razon_social: str = ""
    rut: str = ""
    color: str = "#0f766e"
    color_acento: str = "#ea580c"

    # --- Modelo de negocio ---
    modo: str = "afiliado"  # afiliado = comisión por venta en el proveedor · dropshipping = vendes tú
    multiplicador_precio: float = 2.6
    margen_minimo: float = 0.30
    iva: float = 0.19
    comision_pasarela: float = 0.035
    recargo_importacion: float = 0.0
    envio_usd_defecto: float = 2.5
    terminacion_precio: int = 990
    tipo_cambio_defecto: float = 950.0
    pedidos_correo: str = ""  # correo que recibe los pedidos (FormSubmit) en modo dropshipping
    mercadopago_token: str = ""

    # --- Agente buscador ---
    nichos: list = field(default_factory=lambda: parsear_nichos(NICHOS_DEFECTO))
    score_minimo: int = 60
    max_por_nicho: int = 24
    max_nuevos_por_ciclo: int = 30
    resultados_por_busqueda: int = 50
    precio_min_usd: float = 3.0
    precio_max_usd: float = 80.0
    excluir_palabras: list = field(default_factory=list)

    # --- Proveedores ---
    aliexpress_app_key: str = ""
    aliexpress_app_secret: str = ""
    aliexpress_tracking_id: str = ""
    cj_api_key: str = ""

    # --- Agente de posicionamiento (SEO) ---
    sugerencias_google: bool = True
    max_guias_por_ciclo: int = 2
    indexnow_clave: str = ""
    gsc_credenciales: str = ""  # JSON de una cuenta de servicio con acceso a Search Console
    gsc_propiedad: str = ""
    google_verificacion: str = ""
    bing_verificacion: str = ""
    ga_id: str = ""

    # --- IA (opcional) ---
    anthropic_api_key: str = ""
    llm_modelo: str = "claude-opus-5-5"
    llm_max_productos_por_ciclo: int = 20

    # --- Archivos ---
    carpeta_datos: str = "tienda_datos"
    carpeta_sitio: str = "_sitio"

    @classmethod
    def desde_entorno(cls):
        modo = _env("TIENDA_MODO", "afiliado").lower()
        if modo not in MODOS_VENTA:
            modo = "afiliado"
        nichos = parsear_nichos(_env("TIENDA_NICHOS", NICHOS_DEFECTO)) or parsear_nichos(NICHOS_DEFECTO)
        return cls(
            nombre=_env("TIENDA_NOMBRE", "Copahue Market"),
            eslogan=_env("TIENDA_ESLOGAN", "Productos que resuelven problemas, con envío a todo Chile"),
            url=_env("TIENDA_URL").rstrip("/"),
            dominio_propio=_env("TIENDA_DOMINIO"),
            pais=_env("TIENDA_PAIS", "CL").upper(),
            idioma=_env("TIENDA_IDIOMA", "es-CL"),
            moneda=_env("TIENDA_MONEDA", "CLP").upper(),
            correo_contacto=_env("TIENDA_CORREO"),
            whatsapp=_env("TIENDA_WHATSAPP"),
            razon_social=_env("TIENDA_RAZON_SOCIAL"),
            rut=_env("TIENDA_RUT"),
            color=_env("TIENDA_COLOR", "#0f766e"),
            color_acento=_env("TIENDA_COLOR_ACENTO", "#ea580c"),
            modo=modo,
            multiplicador_precio=_env_float("PRECIO_MULTIPLICADOR", 2.6),
            margen_minimo=_env_float("MARGEN_MINIMO", 0.30),
            iva=_env_float("IVA", 0.19),
            comision_pasarela=_env_float("COMISION_PASARELA", 0.035),
            recargo_importacion=_env_float("RECARGO_IMPORTACION", 0.0),
            envio_usd_defecto=_env_float("ENVIO_USD_DEFECTO", 2.5),
            terminacion_precio=_env_int("PRECIO_TERMINACION", 990),
            tipo_cambio_defecto=_env_float("TIPO_CAMBIO", 950.0),
            pedidos_correo=_env("PEDIDOS_CORREO", _env("TIENDA_CORREO")),
            mercadopago_token=_env("MERCADOPAGO_ACCESS_TOKEN"),
            nichos=nichos,
            score_minimo=_env_int("SCORE_MINIMO", 60),
            max_por_nicho=_env_int("MAX_PRODUCTOS_POR_NICHO", 24),
            max_nuevos_por_ciclo=_env_int("MAX_NUEVOS_POR_CICLO", 30),
            resultados_por_busqueda=min(50, _env_int("RESULTADOS_POR_BUSQUEDA", 50)),
            precio_min_usd=_env_float("PRECIO_MIN_USD", 3.0),
            precio_max_usd=_env_float("PRECIO_MAX_USD", 80.0),
            excluir_palabras=_env_lista("EXCLUIR_PALABRAS"),
            aliexpress_app_key=_env("ALIEXPRESS_APP_KEY"),
            aliexpress_app_secret=_env("ALIEXPRESS_APP_SECRET"),
            aliexpress_tracking_id=_env("ALIEXPRESS_TRACKING_ID"),
            cj_api_key=_env("CJ_API_KEY"),
            sugerencias_google=_env_bool("SUGERENCIAS_GOOGLE", True),
            max_guias_por_ciclo=_env_int("MAX_GUIAS_POR_CICLO", 2),
            indexnow_clave=_env("INDEXNOW_CLAVE"),
            gsc_credenciales=_env("GSC_CREDENCIALES"),
            gsc_propiedad=_env("GSC_PROPIEDAD"),
            google_verificacion=_env("GOOGLE_SITE_VERIFICATION"),
            bing_verificacion=_env("BING_SITE_VERIFICATION"),
            ga_id=_env("GA_MEASUREMENT_ID"),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            llm_modelo=_env("LLM_MODELO", "claude-opus-5-5"),
            llm_max_productos_por_ciclo=_env_int("LLM_MAX_PRODUCTOS_POR_CICLO", 20),
            carpeta_datos=_env("TIENDA_DATOS", "tienda_datos"),
            carpeta_sitio=_env("TIENDA_SITIO", "_sitio"),
        )

    # --- Estado de conexiones ---
    @property
    def aliexpress_configurado(self):
        return bool(self.aliexpress_app_key and self.aliexpress_app_secret)

    @property
    def cj_configurado(self):
        return bool(self.cj_api_key)

    @property
    def ia_configurada(self):
        return bool(self.anthropic_api_key)

    @property
    def gsc_configurado(self):
        return bool(self.gsc_credenciales)

    @property
    def url_base(self):
        """URL pública sin barra final (o localhost si aún no publicas)."""
        return self.url or "http://localhost:8000"

    @property
    def ruta_base(self):
        """Prefijo de las rutas: '/copahue' en https://usuario.github.io/copahue, '' en un dominio propio."""
        return urlparse(self.url_base).path.rstrip("/")

    @property
    def es_afiliado(self):
        return self.modo == "afiliado"


def aplicar_secrets_streamlit():
    """Copia los Secrets de Streamlit al entorno (para Streamlit Cloud)."""
    try:
        import streamlit as st

        for clave, valor in st.secrets.items():
            if isinstance(valor, (str, int, float, bool)) and clave not in os.environ:
                os.environ[clave] = str(valor)
    except Exception:
        pass
