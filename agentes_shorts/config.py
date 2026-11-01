"""Configuración del agente de Shorts.

Todo se lee desde variables de entorno (o un archivo .env). Nunca escribas
credenciales en el código: el client_secret y el token de YouTube viven en `data/`
(ignorado por git) o en los Secrets de GitHub Actions.
"""

import os
from dataclasses import dataclass, field

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
        return float(_env(nombre, str(defecto)))
    except ValueError:
        return defecto


def _env_bool(nombre, defecto=False):
    valor = _env(nombre, "")
    if not valor:
        return defecto
    return valor.lower() in ("1", "true", "si", "sí", "yes", "on")


def _env_lista(nombre, defecto=""):
    return [x.strip() for x in _env(nombre, defecto).split(",") if x.strip()]


# Qué videos puede reutilizar el agente.
#   creativeCommon = solo videos con licencia Creative Commons BY (reutilizables con atribución)
#   cualquiera     = cualquier video viral (⚠️ el reclamo por derechos de autor es casi seguro)
LICENCIAS = ("creativeCommon", "cualquiera")
PRIVACIDADES = ("private", "unlisted", "public")
ESTILOS = ("difuminado", "recorte")
# Costo en unidades de cuota de cada llamada a la YouTube Data API v3.
COSTO_CUOTA = {"videos.list": 1, "search.list": 100, "channels.list": 1, "playlistItems.list": 1, "videos.insert": 1600}


@dataclass
class Config:
    # --- Credenciales de YouTube ---
    youtube_api_key: str = ""  # opcional: solo lectura (tendencias y búsquedas)
    client_secret: str = "data/youtube_client_secret.json"  # OAuth de tu cuenta (subidas)
    token: str = "data/youtube_token.json"
    cuota_diaria: int = 10000

    # --- Qué buscar ---
    regiones: list = field(default_factory=lambda: ["CL", "MX", "ES", "US"])
    idiomas: list = field(default_factory=lambda: ["es"])
    temas: list = field(default_factory=list)  # palabras clave de tu nicho (cada búsqueda cuesta 100 unidades)
    canales: list = field(default_factory=list)  # canales con permiso (tuyos): no se filtra la licencia
    categorias_excluidas: list = field(default_factory=lambda: ["10"])  # 10 = Música
    licencia: str = "creativeCommon"
    busquedas_max: int = 2
    duracion_min_s: int = 90
    duracion_max_s: int = 2700
    umbral_viral: int = 50
    videos_por_ciclo: int = 2
    publicado_hace_dias: int = 7

    # --- Cómo cortar ---
    cortes_por_video: int = 2
    corte_duracion_s: int = 45
    corte_min_s: int = 20
    corte_max_s: int = 60
    estilo: str = "difuminado"
    subtitulos: bool = True
    rotulo: bool = True
    calidad_max: int = 1080
    preset_ffmpeg: str = "veryfast"
    cookies: str = ""  # archivo de cookies de yt-dlp si YouTube pide "confirma que no eres un robot"
    borrar_original: bool = True

    # --- Cómo publicar ---
    privacidad: str = "private"
    max_subidas_dia: int = 3
    programar_cada_horas: float = 0  # >0: programa la publicación escalonada (requiere privacidad=private)
    etiquetas_fijas: list = field(default_factory=lambda: ["Shorts"])
    firma_descripcion: str = ""
    made_for_kids: bool = False

    # --- IA opcional (Claude) ---
    anthropic_api_key: str = ""
    llm_modelo: str = "claude-opus-5-5"

    # --- Archivos ---
    db_path: str = "data/shorts.db"
    carpeta_trabajo: str = "data/shorts"

    @classmethod
    def desde_entorno(cls):
        cfg = cls(
            youtube_api_key=_env("YOUTUBE_API_KEY"),
            client_secret=_env("YOUTUBE_CLIENT_SECRET", "data/youtube_client_secret.json"),
            token=_env("YOUTUBE_TOKEN", "data/youtube_token.json"),
            cuota_diaria=_env_int("YOUTUBE_CUOTA_DIARIA", 10000),
            regiones=[r.upper() for r in _env_lista("SHORTS_REGIONES", "CL,MX,ES,US")],
            idiomas=[i.lower() for i in _env_lista("SHORTS_IDIOMAS", "es")],
            temas=_env_lista("SHORTS_TEMAS"),
            canales=_env_lista("SHORTS_CANALES"),
            categorias_excluidas=_env_lista("SHORTS_CATEGORIAS_EXCLUIDAS", "10"),
            licencia=_env("SHORTS_LICENCIA", "creativeCommon"),
            busquedas_max=_env_int("SHORTS_BUSQUEDAS_MAX", 2),
            duracion_min_s=_env_int("SHORTS_DURACION_MIN", 90),
            duracion_max_s=_env_int("SHORTS_DURACION_MAX", 2700),
            umbral_viral=_env_int("SHORTS_UMBRAL_VIRAL", 50),
            videos_por_ciclo=_env_int("SHORTS_VIDEOS_POR_CICLO", 2),
            publicado_hace_dias=_env_int("SHORTS_PUBLICADO_HACE_DIAS", 7),
            cortes_por_video=_env_int("SHORTS_CORTES_POR_VIDEO", 2),
            corte_duracion_s=_env_int("SHORTS_CORTE_DURACION", 45),
            corte_min_s=_env_int("SHORTS_CORTE_MIN", 20),
            corte_max_s=_env_int("SHORTS_CORTE_MAX", 60),
            estilo=_env("SHORTS_ESTILO", "difuminado"),
            subtitulos=_env_bool("SHORTS_SUBTITULOS", True),
            rotulo=_env_bool("SHORTS_ROTULO", True),
            calidad_max=_env_int("SHORTS_CALIDAD_MAX", 1080),
            preset_ffmpeg=_env("SHORTS_FFMPEG_PRESET", "veryfast"),
            cookies=_env("SHORTS_COOKIES"),
            borrar_original=_env_bool("SHORTS_BORRAR_ORIGINAL", True),
            privacidad=_env("SHORTS_PRIVACIDAD", "private"),
            max_subidas_dia=_env_int("SHORTS_MAX_SUBIDAS_DIA", 3),
            programar_cada_horas=_env_float("SHORTS_PROGRAMAR_CADA_HORAS", 0),
            etiquetas_fijas=_env_lista("SHORTS_ETIQUETAS", "Shorts"),
            firma_descripcion=_env("SHORTS_FIRMA"),
            made_for_kids=_env_bool("SHORTS_PARA_NINOS", False),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            llm_modelo=_env("LLM_MODELO", "claude-opus-5-5"),
            db_path=_env("SHORTS_DB", "data/shorts.db"),
            carpeta_trabajo=_env("SHORTS_CARPETA", "data/shorts"),
        )
        cfg.normalizar()
        return cfg

    def normalizar(self):
        if self.licencia not in LICENCIAS:
            self.licencia = "creativeCommon"
        if self.privacidad not in PRIVACIDADES:
            self.privacidad = "private"
        if self.estilo not in ESTILOS:
            self.estilo = "difuminado"
        # YouTube acepta Shorts de hasta 3 minutos; mantenemos los cortes dentro de ese límite.
        self.corte_max_s = max(5, min(self.corte_max_s, 180))
        self.corte_min_s = max(5, min(self.corte_min_s, self.corte_max_s))
        self.corte_duracion_s = max(self.corte_min_s, min(self.corte_duracion_s, self.corte_max_s))
        self.duracion_min_s = max(self.duracion_min_s, self.corte_min_s + 5)
        if self.programar_cada_horas > 0:
            self.privacidad = "private"  # YouTube exige privado para programar una publicación
        return self

    # --- Atajos ---
    @property
    def ia_configurada(self):
        return bool(self.anthropic_api_key)

    @property
    def lectura_configurada(self):
        """¿Podemos consultar tendencias? Basta la API key o el token OAuth."""
        return bool(self.youtube_api_key) or os.path.exists(self.token)

    @property
    def subida_configurada(self):
        return os.path.exists(self.token) or os.path.exists(self.client_secret)

    @property
    def solo_canales_permitidos(self):
        return bool(self.canales) and not self.temas and not self.regiones
