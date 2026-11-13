"""Configuración del agente de ayuno intermitente.

Todo se lee desde variables de entorno (o un archivo .env en tu PC, o los
Secrets/Variables de GitHub Actions). Nunca escribas tokens en el código.
"""

import os
import unicodedata
from dataclasses import dataclass, field
from datetime import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv es opcional
    pass

DIAS = ("lun", "mar", "mie", "jue", "vie", "sab", "dom")
NOMBRES_DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")

# Orden en que ocurren dentro de un día de ayuno.
AVISOS = ("manana", "apertura_previa", "apertura", "cierre_previo", "inicio_ayuno")
DESCRIPCION_AVISOS = {
    "manana": "Buenos días: cuánto llevas y cuánto falta",
    "apertura_previa": "Falta poco para abrir tu ventana de comida",
    "apertura": "Se abre tu ventana de comida",
    "cierre_previo": "Falta poco para que cierre tu ventana",
    "inicio_ayuno": "Comienza tu ayuno",
}


class ErrorConfig(ValueError):
    """Un valor de configuración no tiene el formato esperado."""


def _env(nombre, defecto=""):
    valor = os.environ.get(nombre)
    return valor.strip() if valor and valor.strip() else defecto


def _env_int(nombre, defecto):
    texto = _env(nombre, str(defecto))
    try:
        return int(texto)
    except ValueError:
        raise ErrorConfig(f"{nombre} debe ser un número entero (recibí '{texto}').") from None


def _sin_tildes(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def leer_hora(texto, nombre="hora"):
    """'12:00', '8:30' o '20' -> time."""
    limpio = texto.strip().lower().replace("h", ":").replace(".", ":").rstrip(":")
    partes = limpio.split(":") if limpio else []
    try:
        if len(partes) == 1:
            return time(int(partes[0]), 0)
        if len(partes) == 2:
            return time(int(partes[0]), int(partes[1]))
    except ValueError:
        pass
    raise ErrorConfig(f"{nombre} debe tener formato HH:MM, por ejemplo 12:00 (recibí '{texto}').")


def _dia(texto):
    clave = texto.strip()[:3]
    if clave not in DIAS:
        raise ErrorConfig(f"AYUNO_DIAS: no reconozco el día '{texto.strip()}'. Usa lun, mar, mie, jue, vie, sab, dom.")
    return DIAS.index(clave)


def leer_dias(texto):
    """'lun,mar,mie' / 'lunes a viernes' / 'lun-vie,dom' / 'todos' -> {0, 1, 2, ...} (0 = lunes)."""
    limpio = _sin_tildes(texto.lower()).strip()
    if limpio in ("", "todos", "todos los dias", "diario"):
        return set(range(7))
    dias = set()
    for parte in limpio.replace(";", ",").split(","):
        parte = parte.strip()
        if not parte:
            continue
        separador = " a " if " a " in parte else "-" if "-" in parte else None
        if separador:  # rango, por ejemplo "lunes a viernes" o "vie-lun"
            inicio, _, fin = parte.partition(separador)
            i, f = _dia(inicio), _dia(fin)
            dias.update((i + k) % 7 for k in range((f - i) % 7 + 1))
        else:
            dias.update(_dia(p) for p in parte.split())
    if not dias:
        raise ErrorConfig("AYUNO_DIAS no tiene ningún día válido.")
    return dias


def leer_avisos(texto):
    limpio = _sin_tildes(texto.lower()).strip()
    if limpio in ("", "todos"):
        return list(AVISOS)
    pedidos = {p.strip() for p in limpio.replace(";", ",").split(",") if p.strip()}
    desconocidos = pedidos - set(AVISOS)
    if desconocidos:
        raise ErrorConfig(
            f"AYUNO_AVISOS: no reconozco {', '.join(sorted(desconocidos))}. Opciones: {', '.join(AVISOS)}."
        )
    return [a for a in AVISOS if a in pedidos]


@dataclass
class Config:
    # --- Tu plan de ayuno ---
    zona_horaria: str = "America/Santiago"
    ventana_inicio: time = time(12, 0)  # abres tu ventana de comida
    ventana_fin: time = time(20, 0)  # cierras tu ventana (empieza el ayuno)
    hora_manana: time = time(8, 0)  # mensaje de buenos días (si cae dentro del ayuno)
    aviso_previo_min: int = 30  # minutos de anticipación de los avisos "previos"
    dias: set = field(default_factory=lambda: set(range(7)))  # días en que termina un ayuno
    avisos: list = field(default_factory=lambda: list(AVISOS))
    nombre: str = ""
    tolerancia_min: int = 60  # si un aviso se atrasa más que esto, ya no se envía

    # --- Canales (configura al menos uno) ---
    telegram_token: str = ""
    telegram_chat_id: str = ""
    whatsapp_telefono: str = ""
    callmebot_apikey: str = ""
    ntfy_tema: str = ""
    ntfy_servidor: str = "https://ntfy.sh"

    # --- IA opcional (Claude) ---
    anthropic_api_key: str = ""
    llm_modelo: str = "claude-opus-5-5"

    # --- Memoria (qué avisos ya se enviaron) ---
    ruta_estado: str = "data/ayuno_estado.json"

    @classmethod
    def desde_entorno(cls):
        cfg = cls(
            zona_horaria=_env("ZONA_HORARIA", "America/Santiago"),
            ventana_inicio=leer_hora(_env("AYUNO_VENTANA_INICIO", "12:00"), "AYUNO_VENTANA_INICIO"),
            ventana_fin=leer_hora(_env("AYUNO_VENTANA_FIN", "20:00"), "AYUNO_VENTANA_FIN"),
            hora_manana=leer_hora(_env("AYUNO_HORA_MANANA", "08:00"), "AYUNO_HORA_MANANA"),
            aviso_previo_min=_env_int("AYUNO_AVISO_PREVIO_MIN", 30),
            dias=leer_dias(_env("AYUNO_DIAS", "todos")),
            avisos=leer_avisos(_env("AYUNO_AVISOS", "todos")),
            nombre=_env("AYUNO_NOMBRE"),
            tolerancia_min=_env_int("AYUNO_TOLERANCIA_MIN", 60),
            telegram_token=_env("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_env("TELEGRAM_CHAT_ID"),
            whatsapp_telefono=_env("WHATSAPP_TELEFONO").replace(" ", ""),
            callmebot_apikey=_env("CALLMEBOT_APIKEY"),
            ntfy_tema=_env("NTFY_TEMA"),
            ntfy_servidor=_env("NTFY_SERVIDOR", "https://ntfy.sh").rstrip("/"),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            llm_modelo=_env("AYUNO_LLM_MODELO", "claude-opus-5-5"),
            ruta_estado=_env("AYUNO_ESTADO", "data/ayuno_estado.json"),
        )
        cfg.validar()
        return cfg

    def validar(self):
        try:
            ZoneInfo(self.zona_horaria)
        except (ZoneInfoNotFoundError, ValueError):
            raise ErrorConfig(f"ZONA_HORARIA '{self.zona_horaria}' no existe (ejemplo: America/Santiago).") from None
        ventana = self.ventana_min
        if ventana == 0:
            raise ErrorConfig("AYUNO_VENTANA_INICIO y AYUNO_VENTANA_FIN no pueden ser iguales.")
        ayuno = 24 * 60 - ventana
        if not 0 <= self.aviso_previo_min < min(ventana, ayuno):
            raise ErrorConfig(
                "AYUNO_AVISO_PREVIO_MIN debe ser menor que la duración de tu ventana y de tu ayuno."
            )
        if self.tolerancia_min < 1:
            raise ErrorConfig("AYUNO_TOLERANCIA_MIN debe ser al menos 1.")
        return self

    # --- Datos derivados ---
    @property
    def tz(self):
        return ZoneInfo(self.zona_horaria)

    @property
    def ventana_min(self):
        inicio = self.ventana_inicio.hour * 60 + self.ventana_inicio.minute
        fin = self.ventana_fin.hour * 60 + self.ventana_fin.minute
        return (fin - inicio) % (24 * 60)

    @property
    def protocolo(self):
        """'16:8', '18:6', '14:10'... (horas redondeadas)."""
        ventana_h = round(self.ventana_min / 60)
        return f"{24 - ventana_h}:{ventana_h}"

    @property
    def canales_configurados(self):
        canales = []
        if self.telegram_token and self.telegram_chat_id:
            canales.append("telegram")
        if self.whatsapp_telefono and self.callmebot_apikey:
            canales.append("whatsapp")
        if self.ntfy_tema:
            canales.append("ntfy")
        return canales

    @property
    def ia_configurada(self):
        return bool(self.anthropic_api_key)
