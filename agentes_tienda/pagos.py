"""Links de pago de Mercado Pago (modo dropshipping): uno por producto, creado al publicar o al cambiar el precio."""

import logging

import requests

log = logging.getLogger(__name__)

URL = "https://api.mercadopago.com/checkout/preferences"


def crear_link(cfg, p, url_gracias):
    r = requests.post(
        URL,
        headers={"Authorization": f"Bearer {cfg.mercadopago_token}"},
        json={
            "items": [
                {
                    "id": p["id"],
                    "title": p["nombre"][:250],
                    "quantity": 1,
                    "unit_price": int(p["precio_clp"]),
                    "currency_id": cfg.moneda,
                    "picture_url": (p.get("imagenes") or [None])[0],
                }
            ],
            "external_reference": p["id"],
            "back_urls": {"success": url_gracias, "pending": url_gracias, "failure": url_gracias},
            "auto_return": "approved",
            "statement_descriptor": cfg.nombre[:22],
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("init_point")


def asegurar_links(almacen, cfg, usar_red=True):
    """Crea o renueva el link de pago de cada producto publicado cuyo precio cambió."""
    if cfg.es_afiliado or not cfg.mercadopago_token or not usar_red or not cfg.url:
        return []
    creados = []
    for p in almacen.publicados():
        pago = p.get("pago") or {}
        if pago.get("link") and pago.get("precio") == p.get("precio_clp") or p.get("demo"):
            continue
        try:
            link = crear_link(cfg, p, f"{cfg.url}/gracias/?p={p.get('slug', '')}&pago=ok")
        except Exception as e:
            almacen.log("Pagos", "Error creando link de pago", str(e), p["id"])
            break
        if link:
            p["pago"] = {"link": link, "precio": p["precio_clp"]}
            creados.append(p["id"])
    if creados:
        almacen.log("Pagos", "Links de Mercado Pago creados", f"{len(creados)} productos")
    return creados
