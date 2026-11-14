"""Avisar a los buscadores y leer cómo te encuentra Google.

- IndexNow: avisa al instante a Bing, Yandex, Seznam y Naver (y a los buscadores con IA que usan
  el índice de Bing) qué páginas son nuevas o cambiaron. Gratis, sin cuenta: solo una clave.
- Google Search Console (opcional): consultas reales, impresiones, clics y posición de cada página,
  para que el agente reoptimice lo que Google ya muestra pero nadie cliquea.
"""

import json
import logging
from datetime import date, timedelta
from urllib.parse import quote, urlparse

import requests

log = logging.getLogger(__name__)

INDEXNOW_URL = "https://api.indexnow.org/indexnow"


def avisar_indexnow(cfg, urls):
    if not cfg.indexnow_clave or not cfg.url or not urls:
        return 0, "sin clave, sin URL pública o sin cambios"
    host = urlparse(cfg.url).netloc
    enviadas = 0
    for i in range(0, len(urls), 10_000):
        lote = urls[i : i + 10_000]
        r = requests.post(
            INDEXNOW_URL,
            json={"host": host, "key": cfg.indexnow_clave, "keyLocation": f"{cfg.url}/{cfg.indexnow_clave}.txt", "urlList": lote},
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=30,
        )
        if r.status_code not in (200, 202):
            return enviadas, f"IndexNow respondió {r.status_code}: {r.text[:200]}"
        enviadas += len(lote)
    return enviadas, "ok"


def consultas_search_console(cfg, dias=28):
    """Filas {keys: [página, consulta], clicks, impressions, ctr, position} de los últimos `dias`."""
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2 import service_account

    info = json.loads(cfg.gsc_credenciales)
    credenciales = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/webmasters.readonly"]
    )
    sesion = AuthorizedSession(credenciales)
    propiedad = cfg.gsc_propiedad or (cfg.url.rstrip("/") + "/")
    fin = date.today() - timedelta(days=3)  # los datos de Search Console llegan con ~2 días de retraso
    r = sesion.post(
        f"https://www.googleapis.com/webmasters/v3/sites/{quote(propiedad, safe='')}/searchAnalytics/query",
        json={
            "startDate": (fin - timedelta(days=dias)).isoformat(),
            "endDate": fin.isoformat(),
            "dimensions": ["page", "query"],
            "rowLimit": 5000,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("rows", [])
