"""Cliente de la YouTube Data API v3: tendencias, búsquedas, detalles y subida de videos.

Lectura: basta una API key (YOUTUBE_API_KEY). Subida: OAuth de tu cuenta
(`python run_shorts.py auth` crea el token una sola vez).
Cada llamada descuenta su costo de la cuota diaria en la base de datos.
"""

import logging
import os
import re
import time
from datetime import timedelta

from .config import COSTO_CUOTA
from .db import ahora, iso

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.readonly"]
_RE_DURACION = re.compile(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def parsear_duracion(iso8601):
    """'PT1H2M3S' -> 3723 segundos."""
    m = _RE_DURACION.fullmatch(iso8601 or "")
    if not m:
        return 0
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s


def video_desde_api(item, fuente=""):
    sn, st, cd, status = item.get("snippet", {}), item.get("statistics", {}), item.get("contentDetails", {}), item.get("status", {})
    return {
        "video_id": item["id"] if isinstance(item["id"], str) else item["id"].get("videoId", ""),
        "titulo": sn.get("title", ""),
        "canal": sn.get("channelTitle", ""),
        "canal_id": sn.get("channelId", ""),
        "publicado_en": sn.get("publishedAt"),
        "duracion_s": parsear_duracion(cd.get("duration")),
        "vistas": int(st.get("viewCount") or 0),
        "likes": int(st.get("likeCount") or 0),
        "comentarios": int(st.get("commentCount") or 0),
        "licencia": status.get("license", ""),
        "categoria": sn.get("categoryId", ""),
        "idioma": (sn.get("defaultAudioLanguage") or sn.get("defaultLanguage") or "").lower(),
        "en_vivo": sn.get("liveBroadcastContent", "none") != "none",
        "fuente": fuente,
    }


class YouTube:
    """Envuelve googleapiclient. `db` se usa solo para contabilizar la cuota."""

    def __init__(self, cfg, db=None):
        self.cfg = cfg
        self.db = db
        self._lectura = None
        self._escritura = None

    # ---------- autenticación ----------
    def credenciales(self, interactivo=False):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        creds = None
        if os.path.exists(self.cfg.token):
            creds = Credentials.from_authorized_user_file(self.cfg.token, SCOPES)
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                self._guardar_token(creds)
            except Exception as e:  # el refresh token caduca a los 7 días si la app OAuth está "en pruebas"
                log.warning("No se pudo renovar el token de YouTube: %s", e)
                creds = None
        if creds and creds.valid:
            return creds
        if not interactivo:
            raise RuntimeError(
                f"No hay un token válido de YouTube en {self.cfg.token}. Ejecuta `python run_shorts.py auth` en tu PC."
            )
        if not os.path.exists(self.cfg.client_secret):
            raise RuntimeError(
                f"Falta {self.cfg.client_secret}. Descárgalo desde Google Cloud → APIs y servicios → Credenciales "
                "(ID de cliente OAuth, tipo 'Aplicación de escritorio')."
            )
        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(self.cfg.client_secret, SCOPES)
        creds = flow.run_local_server(port=0, authorization_prompt_message="Abre este enlace para autorizar al agente: {url}")
        self._guardar_token(creds)
        return creds

    def _guardar_token(self, creds):
        carpeta = os.path.dirname(self.cfg.token)
        if carpeta:
            os.makedirs(carpeta, exist_ok=True)
        with open(self.cfg.token, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    def _servicio_lectura(self):
        if self._lectura is None:
            from googleapiclient.discovery import build

            if self.cfg.youtube_api_key:
                self._lectura = build("youtube", "v3", developerKey=self.cfg.youtube_api_key, cache_discovery=False)
            else:
                self._lectura = self._servicio_escritura()
        return self._lectura

    def _servicio_escritura(self):
        if self._escritura is None:
            from googleapiclient.discovery import build

            self._escritura = build("youtube", "v3", credentials=self.credenciales(), cache_discovery=False)
        return self._escritura

    # ---------- cuota ----------
    def _cobrar(self, metodo):
        if self.db is not None:
            self.db.sumar_cuota(COSTO_CUOTA.get(metodo, 1))

    def cuota_disponible(self, metodo):
        if self.db is None:
            return True
        return self.db.cuota_usada() + COSTO_CUOTA.get(metodo, 1) <= self.cfg.cuota_diaria

    # ---------- lectura ----------
    def tendencias(self, region, max_resultados=50):
        """Videos más populares de una región (1 unidad por página)."""
        yt = self._servicio_lectura()
        self._cobrar("videos.list")
        resp = yt.videos().list(
            part="snippet,statistics,contentDetails,status",
            chart="mostPopular",
            regionCode=region,
            maxResults=min(50, max_resultados),
        ).execute()
        return [video_desde_api(i, fuente=f"tendencias:{region}") for i in resp.get("items", [])]

    def buscar(self, consulta="", region=None, idioma=None, dias=7, solo_cc=True, canal_id=None, max_resultados=25):
        """Videos más vistos que coinciden con la búsqueda (100 unidades) + sus detalles (1 unidad)."""
        yt = self._servicio_lectura()
        self._cobrar("search.list")
        params = dict(
            part="id",
            type="video",
            order="viewCount",
            maxResults=min(50, max_resultados),
            publishedAfter=iso(ahora() - timedelta(days=dias)).replace("+00:00", "Z"),
            videoDuration="medium" if not canal_id else "any",  # medium = 4-20 min: ideal para recortar
            safeSearch="moderate",
        )
        if consulta:
            params["q"] = consulta
        if region:
            params["regionCode"] = region
        if idioma:
            params["relevanceLanguage"] = idioma
        if solo_cc:
            params["videoLicense"] = "creativeCommon"
        if canal_id:
            params["channelId"] = canal_id
        resp = yt.search().list(**params).execute()
        ids = [i["id"]["videoId"] for i in resp.get("items", []) if i.get("id", {}).get("videoId")]
        etiqueta = f"busqueda:{consulta or canal_id or 'cc'}"
        return self.detalles(ids, fuente=etiqueta)

    def detalles(self, ids, fuente=""):
        yt = self._servicio_lectura()
        videos = []
        for i in range(0, len(ids), 50):
            self._cobrar("videos.list")
            resp = yt.videos().list(part="snippet,statistics,contentDetails,status", id=",".join(ids[i : i + 50])).execute()
            videos += [video_desde_api(it, fuente=fuente) for it in resp.get("items", [])]
        return videos

    def videos_de_canal(self, canal, max_resultados=25):
        """Últimos videos de un canal por su playlist de subidas (2 unidades, en vez de 100 de una búsqueda)."""
        yt = self._servicio_lectura()
        self._cobrar("channels.list")
        if canal.startswith("@"):
            resp = yt.channels().list(part="contentDetails", forHandle=canal).execute()
        else:
            resp = yt.channels().list(part="contentDetails", id=canal).execute()
        items = resp.get("items", [])
        if not items:
            return []
        playlist = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        self._cobrar("playlistItems.list")
        resp = yt.playlistItems().list(part="contentDetails", playlistId=playlist, maxResults=min(50, max_resultados)).execute()
        ids = [i["contentDetails"]["videoId"] for i in resp.get("items", [])]
        return self.detalles(ids, fuente=f"canal:{canal}")

    # ---------- escritura ----------
    def subir(self, archivo, titulo, descripcion, etiquetas, categoria="22", privacidad="private", publicar_en=None, made_for_kids=False):
        """Sube un video (1600 unidades). Devuelve el ID de YouTube."""
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload

        yt = self._servicio_escritura()
        cuerpo = {
            "snippet": {
                "title": titulo[:100],
                "description": descripcion[:4900],
                "tags": [e[:60] for e in etiquetas][:30],
                "categoryId": str(categoria or "22"),
            },
            "status": {"privacyStatus": privacidad, "selfDeclaredMadeForKids": bool(made_for_kids)},
        }
        if publicar_en:
            cuerpo["status"]["privacyStatus"] = "private"
            cuerpo["status"]["publishAt"] = publicar_en
        self._cobrar("videos.insert")
        media = MediaFileUpload(archivo, mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
        solicitud = yt.videos().insert(part="snippet,status", body=cuerpo, media_body=media)
        respuesta, intentos = None, 0
        while respuesta is None:
            try:
                _, respuesta = solicitud.next_chunk()
            except HttpError as e:
                if e.resp.status in (500, 502, 503, 504) and intentos < 5:
                    intentos += 1
                    time.sleep(2**intentos)
                    continue
                raise
        return respuesta["id"]
