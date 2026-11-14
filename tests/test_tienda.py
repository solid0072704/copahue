"""Pruebas de la tienda y sus agentes (sin conexión a internet ni claves reales)."""

import dataclasses
import hashlib
import json
import os
import re

import pytest

from agentes_tienda import buscador, indexacion, keywords, seo
from agentes_tienda.almacen import Almacen
from agentes_tienda.config import Config, parsear_nichos
from agentes_tienda.fuentes import aliexpress, archivo, cj
from agentes_tienda.orquestador import avisar_buscadores, ejecutar_ciclo, preparar_demo
from agentes_tienda.texto import a_float, limpiar_titulo, redondear_precio, slugify, truncar

RESPUESTA_ALIEXPRESS = {
    "aliexpress_affiliate_product_query_response": {
        "resp_result": {
            "resp_code": 200,
            "resp_msg": "Call succeeds",
            "result": {
                "current_record_count": 2,
                "products": {
                    "product": [
                        {
                            "product_id": 1005006123456,
                            "product_title": "2024 Nuevo Organizador de cocina giratorio, estante de especias 360, envío gratis",
                            "product_detail_url": "https://www.aliexpress.com/item/1005006123456.html",
                            "promotion_link": "https://s.click.aliexpress.com/e/_ejemplo1",
                            "product_main_image_url": "https://ae01.alicdn.com/kf/ejemplo1.jpg",
                            "product_small_image_urls": {"string": ["https://ae01.alicdn.com/kf/ejemplo1.jpg", "https://ae01.alicdn.com/kf/ejemplo2.jpg"]},
                            "target_sale_price": "12.40",
                            "target_original_price": "20.00",
                            "evaluate_rate": "97.2%",
                            "lastest_volume": 8400,
                            "commission_rate": "7.0%",
                            "hot_product_commission_rate": "9.0%",
                            "first_level_category_name": "Hogar y jardín",
                        },
                        {
                            "product_id": 1005006999999,
                            "product_title": "Audífonos AirPods Pro réplica 1:1",
                            "promotion_link": "https://s.click.aliexpress.com/e/_ejemplo2",
                            "product_main_image_url": "https://ae01.alicdn.com/kf/ejemplo3.jpg",
                            "target_sale_price": "8.00",
                            "evaluate_rate": "90%",
                            "lastest_volume": 50000,
                            "commission_rate": "5%",
                        },
                    ]
                },
            },
        }
    }
}


@pytest.fixture
def cfg(tmp_path):
    base = Config.desde_entorno()
    return dataclasses.replace(
        base,
        url="https://tienda.example/copahue",
        dominio_propio="",
        modo="afiliado",
        aliexpress_app_key="",
        aliexpress_app_secret="",
        cj_api_key="",
        anthropic_api_key="",
        gsc_credenciales="",
        indexnow_clave="",
        mercadopago_token="",
        ga_id="",
        sugerencias_google=False,
        nichos=parsear_nichos("Hogar y cocina: kitchen organizer;Mascotas: pet supplies"),
        carpeta_datos=str(tmp_path / "datos"),
        carpeta_sitio=str(tmp_path / "sitio"),
    )


@pytest.fixture
def almacen(cfg):
    return Almacen(cfg.carpeta_datos)


def producto_crudo(i, nicho="Hogar y cocina", **extra):
    base = {
        "fuente": "archivo",
        "id_fuente": f"X{i:03d}",
        "titulo_original": f"Organizador modelo {i} para cocina, gran capacidad, fácil de limpiar",
        "categoria_fuente": nicho,
        "imagenes": [f"https://cdn.example/img{i}.jpg", f"https://cdn.example/img{i}b.jpg"],
        "video": None,
        "costo_usd": 9.0 + i,
        "precio_lista_usd": None,
        "envio_usd": None,
        "pedidos": 5000 + i * 100,
        "valoracion": 97.0,
        "comision": 8.0,
        "dias_envio": 12,
        "url_proveedor": f"https://proveedor.example/p/{i}",
        "url_afiliado": f"https://s.click.example/{i}",
        "nicho": nicho,
        "busqueda": "",
    }
    base.update(extra)
    return base


class SesionFalsa:
    def __init__(self, respuesta):
        self.respuesta = respuesta
        self.llamadas = []

    def post(self, url, params=None, data=None, json=None, timeout=None):
        self.llamadas.append({"url": url, "params": params, "data": data})

        class R:
            def raise_for_status(_):
                pass

            def json(_):
                return self.respuesta

        return R()


# ---------------------------------------------------------------------------
# Texto y precios
# ---------------------------------------------------------------------------
def test_slug_y_titulos():
    assert slugify("Báscula de Cocina — Digital 5kg!") == "bascula-de-cocina-digital-5kg"
    assert limpiar_titulo("2024 Nuevo Masajeador eléctrico de cuello, almohada Shiatsu, envío gratis") == "Masajeador eléctrico de cuello"
    assert not truncar("Cuerda para saltar inteligente con contador digital de calorías", 52, "").endswith(" de")
    assert redondear_precio(23417) == 23990
    assert redondear_precio(23990) == 23990
    assert redondear_precio(24001) == 24990
    assert a_float("US $12,50") == 12.5
    assert a_float("96.5%") == 96.5
    assert a_float("1.23-4.56") == 1.23


# ---------------------------------------------------------------------------
# Proveedores
# ---------------------------------------------------------------------------
def test_firma_aliexpress_es_md5_ordenado():
    params = {"method": "aliexpress.affiliate.product.query", "app_key": "123", "keywords": "lamp"}
    esperado = hashlib.md5(b"secretoapp_key123keywordslampmethodaliexpress.affiliate.product.querysecreto").hexdigest().upper()
    assert aliexpress.firmar("secreto", params) == esperado


def test_cliente_aliexpress_firma_y_normaliza(cfg):
    cfg = dataclasses.replace(cfg, aliexpress_app_key="123", aliexpress_app_secret="secreto", aliexpress_tracking_id="mitienda")
    sesion = SesionFalsa(RESPUESTA_ALIEXPRESS)
    cliente = aliexpress.ClienteAliExpress(cfg, sesion)
    productos = cliente._llamar("aliexpress.affiliate.product.query", keywords="kitchen", **cliente._comunes())
    llamada = sesion.llamadas[0]
    firma = llamada["params"].pop("sign")
    assert firma == aliexpress.firmar("secreto", {**llamada["params"], **llamada["data"]})
    assert llamada["data"]["tracking_id"] == "mitienda"
    p = aliexpress.normalizar(productos[0], "kitchen")
    assert p["id_fuente"] == "1005006123456"
    assert p["costo_usd"] == 12.4 and p["valoracion"] == 97.2 and p["comision"] == 9.0 and p["pedidos"] == 8400
    assert p["url_afiliado"].startswith("https://s.click.aliexpress.com")
    assert len(p["imagenes"]) == 2


def test_cj_normaliza_y_estima_envio(cfg):
    p = cj.normalizar(
        {"pid": "ABC-1", "productNameEn": "Pet hair remover brush", "sellPrice": "3.20-4.10", "productImage": "https://cj.example/a.jpg",
         "productWeight": "250", "listedNum": 1200, "categoryName": "Pet"},
        "pet supplies", cfg,
    )
    assert p["costo_usd"] == 3.2 and p["listados"] == 1200 and p["envio_usd"] >= cfg.envio_usd_defecto


def test_importar_csv_en_pesos(cfg):
    csv = (
        "Nombre;Precio;Imagen;Rating;Ventas;Link\n"
        "Lámpara LED recargable, sensor de movimiento;9500;https://cdn.example/l.jpg;4,8;1200;https://prov.example/l\n"
        ";1000;;;;\n"
    ).encode("utf-8")
    tabla = archivo.leer(csv, "productos.csv")
    productos = archivo.productos_desde_tabla(tabla, moneda="CLP", tipo_cambio=950, nicho="Hogar y cocina")
    assert len(productos) == 1
    p = productos[0]
    assert p["costo_usd"] == 10.0 and p["valoracion"] == 96.0 and p["pedidos"] == 1200 and p["nicho"] == "Hogar y cocina"


# ---------------------------------------------------------------------------
# Agente Buscador
# ---------------------------------------------------------------------------
def test_filtros_de_riesgo(cfg):
    assert "marca" in buscador.motivo_exclusion(producto_crudo(1, titulo_original="Zapatillas Nike Air"), cfg)
    assert "réplica" in buscador.motivo_exclusion(producto_crudo(1, titulo_original="Reloj réplica 1:1 de lujo"), cfg)
    assert "restringido" in buscador.motivo_exclusion(producto_crudo(1, titulo_original="Power bank 20000 mAh"), cfg)
    assert buscador.motivo_exclusion(producto_crudo(1, titulo_original="Pistola de masaje muscular"), cfg) is None
    assert "afiliado" in buscador.motivo_exclusion(producto_crudo(1, url_afiliado=None), cfg)
    assert "bajo" in buscador.motivo_exclusion(producto_crudo(1, costo_usd=1.0), cfg)


def test_precio_dropshipping_respeta_margen_minimo(cfg):
    cfg = dataclasses.replace(cfg, modo="dropshipping", multiplicador_precio=1.2, margen_minimo=0.35)
    r = buscador.calcular_precio(producto_crudo(1, costo_usd=10.0), cfg, 950)
    assert r["margen"] >= 0.35
    assert str(r["precio_clp"]).endswith("990")
    neto = r["precio_clp"] / 1.19
    assert r["ganancia_clp"] == pytest.approx(neto - r["costo_clp"] - r["precio_clp"] * cfg.comision_pasarela, abs=1)


def test_precio_estable_ante_cambios_menores_del_dolar(cfg):
    p = dict(producto_crudo(1, costo_usd=10.0))
    buscador.aplicar_precio(p, cfg, 950)
    precio = p["precio_clp"]
    buscador.aplicar_precio(p, cfg, 960)  # +1 %
    assert p["precio_clp"] == precio
    buscador.aplicar_precio(p, cfg, 1050)  # +10 %
    assert p["precio_clp"] > precio


def test_score_premia_demanda_y_valoracion(cfg):
    bueno = dict(producto_crudo(1), **buscador.calcular_precio(producto_crudo(1), cfg, 950))
    malo = dict(producto_crudo(2, pedidos=3, valoracion=85.0, dias_envio=35))
    malo.update(buscador.calcular_precio(malo, cfg, 950))
    assert buscador.puntuar(bueno, cfg)[0] > buscador.puntuar(malo, cfg)[0] + 25


def test_seleccion_limita_por_nicho_y_respeta_decisiones_manuales(almacen, cfg):
    cfg = dataclasses.replace(cfg, max_por_nicho=3, score_minimo=40)
    buscador.incorporar(almacen, cfg, [producto_crudo(i) for i in range(1, 7)], 950)
    almacen.productos["ar-X006"]["manual"] = "descartado"
    publicados, _ = buscador.seleccionar(almacen, cfg)
    assert len(publicados) == 3
    assert almacen.productos["ar-X006"]["estado"] == "descartado"
    # Un producto mucho mejor reemplaza al peor publicado (que queda pausado, con su página viva).
    estrella = producto_crudo(99, pedidos=900000, valoracion=99.5, costo_usd=25.0, comision=12.0, dias_envio=7)
    buscador.incorporar(almacen, cfg, [estrella], 950)
    peor = almacen.publicados()[-1]
    peor["score"] = 45
    publicados, pausados = buscador.seleccionar(almacen, cfg)
    assert publicados == ["ar-X099"] and pausados == [peor["id"]]
    assert almacen.productos[peor["id"]]["estado"] == "pausado"


def test_producto_que_deja_de_existir_se_pausa(almacen, cfg):
    buscador.incorporar(almacen, cfg, [producto_crudo(1, fuente="aliexpress")], 950)
    p = almacen.productos["ae-X001"]
    p["estado"] = "publicado"

    class ClienteSinResultados:
        def detalle(self, ids):
            return []

    for _ in range(2):
        buscador._actualizar_publicados_aliexpress(almacen, ClienteSinResultados())
    assert p["estado"] == "pausado" and "no disponible" in p["motivo"]


# ---------------------------------------------------------------------------
# Agente Posicionador
# ---------------------------------------------------------------------------
def test_keywords_eligen_busqueda_real():
    class InvestigadorFalso:
        def sugerencias(self, q):
            return {"organizador giratorio": ["organizador giratorio", "organizador giratorio de especias", "organizador giratorio cocina"]}.get(q, [])

    kw = keywords.investigar(InvestigadorFalso(), "Organizador giratorio 360 para especias")
    assert kw["principal"] == "organizador giratorio"
    assert "organizador giratorio de especias" in kw["secundarias"]


def test_textos_por_reglas_cumplen_largos(cfg):
    p = dict(producto_crudo(1), nombre="Organizador giratorio para especias de cocina con base antideslizante", precio_clp=9990, nicho="Hogar y cocina")
    t = seo.textos_por_reglas(p, {"principal": "organizador giratorio"}, cfg)
    assert len(t["titulo_seo"]) <= 60 and len(t["meta_descripcion"]) <= 160
    assert "organizador giratorio" in t["intro"]
    assert t["faq"] and t["descripcion"]


def test_redactor_ia_se_usa_y_define_el_slug(almacen, cfg):
    class RedactorFalso:
        def producto(self, p, kw):
            return {
                "nombre": "Especiero giratorio 360°", "titulo_seo": "Especiero giratorio 360° con envío a Chile",
                "meta_descripcion": "Ordena tus especias con este especiero giratorio. " * 4, "keyword_principal": "especiero giratorio",
                "intro": "Ordena tu cocina.", "beneficios": ["Gira 360°"], "descripcion": ["Párrafo."], "usos": ["Cocina"],
                "faq": [{"pregunta": "¿Gira?", "respuesta": "Sí."}], "alt_imagen": "Especiero",
            }

        def nicho(self, *a):
            return None

        def guia(self, *a):
            return None

    buscador.incorporar(almacen, cfg, [producto_crudo(1)], 950)
    almacen.productos["ar-X001"]["estado"] = "publicado"
    r = seo.posicionar(almacen, cfg, RedactorFalso(), usar_red=False)
    p = almacen.productos["ar-X001"]
    assert r["ia"] == 1 and p["seo"]["origen"] == "ia"
    assert p["slug"].startswith("especiero-giratorio-360") and len(p["seo"]["meta_descripcion"]) <= 160


def test_reoptimiza_con_search_console(almacen, cfg, monkeypatch):
    cfg = dataclasses.replace(cfg, gsc_credenciales="{}")
    buscador.incorporar(almacen, cfg, [producto_crudo(1)], 950)
    p = almacen.productos["ar-X001"]
    p["estado"] = "publicado"
    seo.posicionar(almacen, cfg, None, usar_red=False)
    url = f"{cfg.url}/p/{p['slug']}/"
    filas = [
        {"keys": [url, "especiero giratorio"], "clicks": 0, "impressions": 120, "ctr": 0, "position": 8.2},
        {"keys": [url, "organizador de especias"], "clicks": 1, "impressions": 40, "ctr": 0.025, "position": 11.0},
    ]
    monkeypatch.setattr(indexacion, "consultas_search_console", lambda cfg: filas)
    hechos = seo.reoptimizar_con_search_console(almacen, cfg)
    assert hechos == ["ar-X001"]
    assert p["seo"]["titulo_seo"].lower().startswith("especiero giratorio") and p["gsc"]["impresiones"] == 160
    assert seo.reoptimizar_con_search_console(almacen, cfg) == []  # espera 21 días antes de volver a tocarla


def test_auditoria_detecta_problemas():
    html = "<html><head><title>Corto</title></head><body><main><h1>a</h1><h1>b</h1><p>poco texto</p></main></body></html>"
    puntaje, problemas, _ = seo.auditar_html(html, "producto")
    assert puntaje < 50
    assert any("H1" in x for x in problemas) and any("meta" in x for x in problemas) and any("escaso" in x for x in problemas)


# ---------------------------------------------------------------------------
# Sitio y ciclo completo
# ---------------------------------------------------------------------------
def _leer(cfg, ruta):
    with open(os.path.join(cfg.carpeta_sitio, ruta), encoding="utf-8") as f:
        return f.read()


def _json_ld(html):
    return [json.loads(b) for b in re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)]


def test_demo_completa_no_se_indexa(almacen, cfg):
    r = preparar_demo(almacen, cfg)
    assert r.en_vitrina >= 20 and r.paginas > 30
    assert "Disallow: /" in _leer(cfg, "robots.txt")
    assert 'content="noindex' in _leer(cfg, "index.html")
    assert r.cambiadas == []  # nada que avisar a los buscadores en modo demo
    assert any("airpods" in (p.get("motivo") or "") for p in almacen.productos.values())


def test_sitio_real_afiliado(almacen, cfg):
    crudos = [producto_crudo(i) for i in range(1, 5)] + [producto_crudo(i, nicho="Mascotas") for i in range(5, 8)]
    buscador.incorporar(almacen, cfg, crudos, 950)
    r = ejecutar_ciclo(almacen, cfg, buscar=False, usar_ia=False, usar_red=False)
    assert r.en_vitrina == 7 and r.auditoria >= 90, almacen.estado["auditoria"]
    p = almacen.publicados()[0]
    html = _leer(cfg, f"p/{p['slug']}/index.html")
    assert 'rel="sponsored nofollow noopener"' in html and p["url_afiliado"] in html
    assert "noindex" not in html
    assert f'<link rel="canonical" href="{cfg.url}/p/{p["slug"]}/">' in html
    assert 'href="/copahue/c/' in html  # rutas con el prefijo de GitHub Pages
    producto_ld = next(b for b in _json_ld(html) if b["@type"] == "Product")
    assert producto_ld["offers"]["price"] == str(p["precio_clp"]) and producto_ld["offers"]["priceCurrency"] == "CLP"
    sitemap = _leer(cfg, "sitemap.xml")
    assert f"{cfg.url}/p/{p['slug']}/" in sitemap and "/buscar/" not in sitemap
    assert "Sitemap: https://tienda.example/copahue/sitemap.xml" in _leer(cfg, "robots.txt")
    assert os.path.exists(os.path.join(cfg.carpeta_sitio, "guias", "mejores-hogar-y-cocina", "index.html"))
    indice = json.loads(_leer(cfg, "buscar.json"))
    assert len(indice) == 7
    # Las URLs cambiadas se avisan una vez; si nada cambia, la siguiente construcción no avisa nada.
    assert len(r.cambiadas) > 10
    r2 = ejecutar_ciclo(almacen, cfg, buscar=False, usar_ia=False, usar_red=False)
    assert r2.cambiadas == []


def test_producto_pausado_conserva_pagina_sin_indexar(almacen, cfg):
    buscador.incorporar(almacen, cfg, [producto_crudo(i) for i in range(1, 5)], 950)
    ejecutar_ciclo(almacen, cfg, buscar=False, usar_ia=False, usar_red=False)
    p = almacen.publicados()[0]
    p["manual"] = "pausado"
    ejecutar_ciclo(almacen, cfg, buscar=False, usar_ia=False, usar_red=False)
    html = _leer(cfg, f"p/{p['slug']}/index.html")
    assert 'content="noindex' in html and "no está disponible" in html
    assert p["slug"] not in _leer(cfg, "sitemap.xml")


def test_sitio_dropshipping_tiene_pedido_y_feed(almacen, cfg):
    cfg = dataclasses.replace(cfg, modo="dropshipping", pedidos_correo="pedidos@tienda.example", whatsapp="+56 9 1234 5678")
    buscador.incorporar(almacen, cfg, [producto_crudo(i, url_afiliado=None) for i in range(1, 5)], 950)
    ejecutar_ciclo(almacen, cfg, buscar=False, usar_ia=False, usar_red=False)
    p = almacen.publicados()[0]
    html = _leer(cfg, f"p/{p['slug']}/index.html")
    assert 'action="https://formsubmit.co/pedidos@tienda.example"' in html and "wa.me/56912345678" in html
    oferta = next(b for b in _json_ld(html) if b["@type"] == "Product")["offers"]
    assert oferta["hasMerchantReturnPolicy"]["merchantReturnDays"] == 10
    feed = _leer(cfg, "feeds/google-merchant.xml")
    assert f"<g:price>{p['precio_clp']} CLP</g:price>" in feed
    assert os.path.exists(os.path.join(cfg.carpeta_sitio, "gracias", "index.html"))


def test_ciclo_con_proveedor_falso(almacen, cfg):
    class ProveedorFalso:
        def buscar(self, palabras, por_pagina=50):
            return [aliexpress.normalizar(x, palabras) for x in RESPUESTA_ALIEXPRESS["aliexpress_affiliate_product_query_response"]["resp_result"]["result"]["products"]["product"]]

        def detalle(self, ids):
            return []

    cfg = dataclasses.replace(cfg, score_minimo=50)
    r = ejecutar_ciclo(almacen, cfg, usar_ia=False, usar_red=False, clientes={"aliexpress": ProveedorFalso()})
    assert "ae-1005006123456" in r.publicados
    assert almacen.productos["ae-1005006999999"]["estado"] == "descartado"
    assert os.path.exists(os.path.join(cfg.carpeta_datos, "productos.json"))
    assert Almacen(cfg.carpeta_datos).productos["ae-1005006123456"]["slug"]  # persistido en disco


def test_indexnow(almacen, cfg, monkeypatch):
    cfg = dataclasses.replace(cfg, indexnow_clave="clave123")
    enviados = {}

    class R:
        status_code = 200
        text = ""

    def post(url, json=None, headers=None, timeout=None):
        enviados.update(json)
        return R()

    monkeypatch.setattr(indexacion.requests, "post", post)
    buscador.incorporar(almacen, cfg, [producto_crudo(i) for i in range(1, 4)], 950)
    ejecutar_ciclo(almacen, cfg, buscar=False, usar_ia=False, usar_red=False)
    assert _leer(cfg, "clave123.txt") == "clave123"
    n, estado = avisar_buscadores(almacen, cfg)
    assert estado == "ok" and n == len(enviados["urlList"]) > 0
    assert enviados["host"] == "tienda.example" and enviados["keyLocation"] == f"{cfg.url}/clave123.txt"


def test_url_publicada_nunca_desaparece(almacen, cfg):
    buscador.incorporar(almacen, cfg, [producto_crudo(i) for i in range(1, 5)], 950)
    ejecutar_ciclo(almacen, cfg, buscar=False, usar_ia=False, usar_red=False)
    p = almacen.publicados()[0]
    p["manual"] = "descartado"
    ejecutar_ciclo(almacen, cfg, buscar=False, usar_ia=False, usar_red=False)
    assert p["estado"] == "descartado"
    html = _leer(cfg, f"p/{p['slug']}/index.html")
    assert 'content="noindex' in html and "Alternativas similares" in html
