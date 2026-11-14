"""AliExpress Open Platform (API de afiliados): búsqueda de productos, más vendidos y actualización.

Requiere una app de afiliado en https://openservice.aliexpress.com (App Key + App Secret) y tu
Tracking ID del portal https://portals.aliexpress.com. Los enlaces que devuelve la API
(`promotion_link`) ya incluyen tu código de afiliado: cada venta te deja comisión.
"""

import hashlib
import logging
import time

import requests

from ..texto import a_float

log = logging.getLogger(__name__)

URL = "https://api-sg.aliexpress.com/sync"


def firmar(secreto, parametros):
    """Firma MD5 del protocolo de la plataforma: MD5(secreto + clave1valor1clave2valor2... + secreto) en mayúsculas."""
    base = secreto + "".join(f"{k}{parametros[k]}" for k in sorted(parametros)) + secreto
    return hashlib.md5(base.encode("utf-8")).hexdigest().upper()


class ClienteAliExpress:
    def __init__(self, cfg, sesion=None):
        self.cfg = cfg
        self.sesion = sesion or requests.Session()

    def _llamar(self, metodo, **parametros):
        sistema = {
            "app_key": self.cfg.aliexpress_app_key,
            "method": metodo,
            "sign_method": "md5",
            "timestamp": str(int(time.time() * 1000)),
            "format": "json",
            "v": "2.0",
        }
        app = {k: str(v) for k, v in parametros.items() if v not in (None, "")}
        sistema["sign"] = firmar(self.cfg.aliexpress_app_secret, {**sistema, **app})
        r = self.sesion.post(URL, params=sistema, data=app, timeout=30)
        r.raise_for_status()
        datos = r.json()
        if "error_response" in datos:
            e = datos["error_response"]
            raise RuntimeError(f"AliExpress {e.get('code')}: {e.get('msg')} {e.get('sub_msg', '')}".strip())
        clave = metodo.replace(".", "_") + "_response"
        resultado = (datos.get(clave) or {}).get("resp_result") or {}
        if str(resultado.get("resp_code")) != "200":
            # 405 = sin resultados para esa búsqueda; no es un error.
            return []
        productos = ((resultado.get("result") or {}).get("products") or {}).get("product") or []
        return productos if isinstance(productos, list) else [productos]

    def _comunes(self):
        return {
            "target_currency": "USD",
            "target_language": "ES",
            "tracking_id": self.cfg.aliexpress_tracking_id,
        }

    def buscar(self, palabras, por_pagina=50):
        """Más vendidos para la búsqueda (orden por volumen) + productos 'hot' (más comisión)."""
        salida = []
        for metodo in ("aliexpress.affiliate.hotproduct.query", "aliexpress.affiliate.product.query"):
            try:
                crudos = self._llamar(
                    metodo,
                    keywords=palabras,
                    page_no=1,
                    page_size=por_pagina,
                    sort="LAST_VOLUME_DESC",
                    ship_to_country=self.cfg.pais,
                    **self._comunes(),
                )
            except Exception as e:  # una búsqueda fallida no detiene a las demás
                log.warning("AliExpress (%s, %s): %s", metodo, palabras, e)
                continue
            salida += [normalizar(p, palabras) for p in crudos]
        return [p for p in salida if p]

    def detalle(self, ids):
        """Precio, ventas y disponibilidad actualizados de hasta 50 productos por llamada."""
        salida = []
        for i in range(0, len(ids), 50):
            lote = ids[i : i + 50]
            crudos = self._llamar(
                "aliexpress.affiliate.productdetail.get",
                product_ids=",".join(lote),
                country=self.cfg.pais,
                **self._comunes(),
            )
            salida += [normalizar(p) for p in crudos]
        return [p for p in salida if p]


def _imagenes(p):
    imagenes = [p.get("product_main_image_url")]
    pequenas = p.get("product_small_image_urls") or {}
    if isinstance(pequenas, dict):
        pequenas = pequenas.get("string") or []
    if isinstance(pequenas, str):
        pequenas = [pequenas]
    imagenes += list(pequenas)
    vistas, salida = set(), []
    for url in imagenes:
        if url and url not in vistas:
            vistas.add(url)
            salida.append(url if not url.startswith("//") else "https:" + url)
    return salida[:8]


def normalizar(p, busqueda=""):
    pid = str(p.get("product_id") or "").strip()
    costo = a_float(p.get("target_sale_price"))
    if not pid or not costo:
        return None
    comision = a_float(p.get("hot_product_commission_rate")) or a_float(p.get("commission_rate"))
    categoria = " > ".join(x for x in (p.get("first_level_category_name"), p.get("second_level_category_name")) if x)
    return {
        "fuente": "aliexpress",
        "id_fuente": pid,
        "titulo_original": (p.get("product_title") or "").strip(),
        "categoria_fuente": categoria,
        "imagenes": _imagenes(p),
        "video": p.get("product_video_url") or None,
        "costo_usd": round(costo, 2),
        "precio_lista_usd": a_float(p.get("target_original_price")),
        "envio_usd": None,  # el precio para Chile ya suele incluir envío estándar; si no, se usa ENVIO_USD_DEFECTO
        "pedidos": int(a_float(p.get("lastest_volume"), 0) or 0),
        "valoracion": a_float(p.get("evaluate_rate")),
        "comision": comision,
        "dias_envio": int(a_float(p.get("ship_to_days"), 0) or 0) or None,
        "url_proveedor": p.get("product_detail_url") or f"https://www.aliexpress.com/item/{pid}.html",
        "url_afiliado": p.get("promotion_link") or None,
        "busqueda": busqueda,
    }
