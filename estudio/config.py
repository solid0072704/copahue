"""Configuración de la plataforma del estudio contable.

Todo se lee desde variables de entorno (o un archivo .env en desarrollo, o los
Secrets de Streamlit Cloud). Nunca escribas contraseñas en el código.
"""

import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
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


def a_numero(texto, defecto=0.0):
    """Convierte montos escritos a la chilena: '39.500,25' → 39500.25 · '1.234.567' → 1234567 · '2,5' → 2.5."""
    if isinstance(texto, (int, float)):
        return defecto if texto != texto else float(texto)  # NaN → defecto
    limpio = re.sub(r"[^0-9,.\-]", "", str(texto or ""))
    if not limpio:
        return defecto
    if "," in limpio:
        limpio = limpio.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", limpio):
        limpio = limpio.replace(".", "")
    try:
        return float(limpio)
    except ValueError:
        return defecto


def _env_float(nombre, defecto):
    return a_numero(_env(nombre), defecto)


def _env_lista(nombre, defecto=""):
    return [x.strip() for x in _env(nombre, defecto).split(",") if x.strip()]


def _env_fechas(nombre):
    fechas = []
    for texto in _env_lista(nombre):
        try:
            fechas.append(date.fromisoformat(texto))
        except ValueError:
            continue
    return fechas


@dataclass
class Config:
    # --- Tu estudio ---
    estudio_nombre: str = "Mi Estudio Contable"
    firma: str = "Equipo del estudio"
    equipo: list = field(default_factory=list)
    zona_horaria: str = "America/Santiago"

    # --- Acceso ---
    clave: str = ""

    # --- Vencimientos ---
    dias_alerta: int = 7
    feriados_extra: list = field(default_factory=list)

    # --- Honorarios ---
    valor_uf: float = 0.0
    honorarios_dia_vence: int = 5
    datos_pago: str = ""

    # --- Almacenamiento ---
    ruta_db: str = "data/estudio.db"
    ruta_db_ebook: str = "data/tablero_ebook.db"

    @classmethod
    def desde_entorno(cls):
        return cls(
            estudio_nombre=_env("ESTUDIO_NOMBRE", "Mi Estudio Contable"),
            firma=_env("ESTUDIO_FIRMA", "Equipo del estudio"),
            equipo=_env_lista("ESTUDIO_EQUIPO"),
            zona_horaria=_env("ZONA_HORARIA", "America/Santiago"),
            clave=_env("ESTUDIO_CLAVE"),
            dias_alerta=max(1, _env_int("ESTUDIO_DIAS_ALERTA", 7)),
            feriados_extra=_env_fechas("ESTUDIO_FERIADOS_EXTRA"),
            valor_uf=_env_float("ESTUDIO_VALOR_UF", 0.0),
            honorarios_dia_vence=min(28, max(1, _env_int("ESTUDIO_HONORARIOS_DIA_VENCE", 5))),
            datos_pago=_env("ESTUDIO_DATOS_PAGO").replace("\\n", "\n"),
            ruta_db=_env("ESTUDIO_DB", "data/estudio.db"),
            ruta_db_ebook=_env("TABLERO_DB", "data/tablero_ebook.db"),
        )

    @property
    def tz(self):
        return ZoneInfo(self.zona_horaria)

    def hoy(self):
        """Fecha de hoy en la zona horaria del estudio."""
        return datetime.now(self.tz).date()


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
