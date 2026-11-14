"""CJ Dropshipping (API 2.0): catálogo de un proveedor que despacha a Chile sin mínimo de compra.

Crea tu API Key en https://cjdropshipping.com → My CJ → Authorization → API. La API
permite 1 consulta por segundo, por eso el cliente espera entre llamadas.
"""

import logging
import time

import requests

from ..texto import a_float

log = logging.getLogger(__name__)

BASE = "https://developers.cjdropshipping.com/api2.0/v1"


class ClienteCJ:
    def __init__(self, cfg, sesion=None):
        self.cfg = cfg
        self.sesion = sesion or requests.Session()
        self._token = None
        self._ultima = 0.0

    def _esperar(self):
        pausa = 1.1 - (time.monotonic() - self._ultima)
        if pausa > 0:
            time.sleep(pausa)
        self._ultima = time.monotonic()

    def token(self):
        if self._token:
            return self._token
        self._esperar()
        r = self.sesion.post(f"{BASE}/authentication/getAccessToken", json={"apiKey": self.cfg.cj_api_key}, timeout=30)
        r.raise_for_status()
        datos = r.json()
        if not datos.get("result"):
            raise RuntimeError(f"CJ: {datos.get('message')}")
        self._token = (datos.get("data") or {}).get("accessToken")
        return self._token

    def buscar(self, palabras, por_pagina=50):
        try:
            token = self.token()
            self._esperar()
            r = self.sesion.get(
                f"{BASE}/product/list",
                params={"productNameEn": palabras, "pageNum": 1, "pageSize": por_pagina},
                headers={"CJ-Access-Token": token},
                timeout=30,
            )
            r.raise_for_status()
            datos = r.json()
        except Exception as e:
            log.warning("CJ (%s): %s", palabras, e)
            return []
        lista = ((datos.get("data") or {}).get("list")) or []
        return [p for p in (normalizar(x, palabras, self.cfg) for x in lista) if p]


def normalizar(p, busqueda, cfg):
    pid = str(p.get("pid") or p.get("id") or "").strip()
    costo = a_float(p.get("sellPrice") or p.get("nowPrice"))
    if not pid or not costo:
        return None
    peso_g = a_float(p.get("productWeight"), 0) or 0
    # Estimación de envío a Chile por peso (CJPacket). Ajusta ENVIO_USD_DEFECTO según tus cotizaciones.
    envio = round(max(cfg.envio_usd_defecto, 2.0 + peso_g / 1000 * 12), 2)
    imagen = p.get("productImage") or p.get("bigImage")
    imagenes = imagen if isinstance(imagen, list) else [imagen]
    return {
        "fuente": "cj",
        "id_fuente": pid,
        "titulo_original": (p.get("productNameEn") or p.get("nameEn") or "").strip(),
        "categoria_fuente": p.get("categoryName") or "",
        "imagenes": [i for i in imagenes if i][:8],
        "video": None,
        "costo_usd": round(costo, 2),
        "precio_lista_usd": None,
        "envio_usd": envio,
        "pedidos": None,
        "listados": int(a_float(p.get("listedNum"), 0) or 0),
        "valoracion": None,
        "comision": None,
        "dias_envio": None,
        "peso_kg": round(peso_g / 1000, 3) if peso_g else None,
        "url_proveedor": f"https://cjdropshipping.com/product/-p-{pid}.html",
        "url_afiliado": None,
        "busqueda": busqueda,
    }
