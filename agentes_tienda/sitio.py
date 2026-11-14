"""Generador del sitio: HTML estático, rápido y fácil de indexar (ideal para GitHub Pages).

Crea la portada, categorías, una página por producto, guías de compra, buscador, páginas legales,
sitemap.xml, robots.txt, datos estructurados (schema.org) y el feed de Google Merchant Center.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import seo
from .almacen import ahora
from .texto import cantidad_corta, miles, precio_texto, slugify, truncar

CARPETA_PLANTILLAS = os.path.join(os.path.dirname(__file__), "plantillas")

ICONOS = {
    "cocina": '<path d="M4 10h16v6a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4z"/><path d="M2 10h20"/><path d="M9 6.5c0-1.4 1.3-2.5 3-2.5s3 1.1 3 2.5"/>',
    "mascota": '<circle cx="6.5" cy="9" r="1.8"/><circle cx="10" cy="5.5" r="1.8"/><circle cx="14" cy="5.5" r="1.8"/><circle cx="17.5" cy="9" r="1.8"/><path d="M12 11c-3 0-5.5 3.2-5.5 5.6 0 1.6 1.2 2.4 2.6 2.4 1.2 0 1.8-.6 2.9-.6s1.7.6 2.9.6c1.4 0 2.6-.8 2.6-2.4C17.5 14.2 15 11 12 11z"/>',
    "bienestar": '<path d="M12 21s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 11c0 5.6-7 10-7 10z"/>',
    "auto": '<path d="M3 16v-3l2.2-5a2 2 0 0 1 1.8-1.2h10a2 2 0 0 1 1.8 1.2L21 13v3"/><path d="M3 13h18"/><circle cx="7" cy="17" r="2"/><circle cx="17" cy="17" r="2"/>',
    "tecnologia": '<path d="M13 2 4 14h7l-1 8 9-12h-7z"/>',
    "fitness": '<path d="M6 7v10M18 7v10M3 10v4M21 10v4M6 12h12"/>',
    "otro": '<path d="M5 8h14l-1.2 12H6.2z"/><path d="M9 8V6.5a3 3 0 0 1 6 0V8"/>',
}
PALETA = ["#0f766e", "#7c3aed", "#db2777", "#2563eb", "#ca8a04", "#059669", "#ea580c", "#0891b2"]


def icono_de(nombre):
    n = slugify(nombre)
    for clave, palabras in (
        ("fitness", ("fitness", "deporte", "gym", "ejercicio")),
        ("mascota", ("mascota", "perro", "gato")),
        ("bienestar", ("bienestar", "belleza", "salud", "cuidado")),
        ("auto", ("auto", "vehiculo", "moto")),
        ("tecnologia", ("tecno", "gadget", "electron", "celular")),
        ("cocina", ("cocina", "hogar", "casa")),
    ):
        if any(p in n for p in palabras):
            return clave
    return "otro"


def svg_placeholder(nombre, nicho, color):
    icono = ICONOS[icono_de(nicho)]
    inicial = (nombre or "?")[:1].upper()
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600" role="img" aria-label="{nombre}">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{color}" stop-opacity=".16"/><stop offset="1" stop-color="{color}" stop-opacity=".38"/></linearGradient></defs>
<rect width="600" height="600" fill="#f8fafc"/><rect width="600" height="600" fill="url(#g)"/>
<circle cx="300" cy="285" r="150" fill="#fff" fill-opacity=".75"/>
<g transform="translate(180 165) scale(10)" fill="none" stroke="{color}" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">{icono}</g>
<text x="300" y="530" text-anchor="middle" font-family="system-ui,sans-serif" font-size="30" font-weight="700" fill="{color}" fill-opacity=".55">{inicial} · producto de ejemplo</text>
</svg>"""


class Constructor:
    def __init__(self, almacen, cfg, salida=None):
        self.almacen = almacen
        self.cfg = cfg
        self.salida = salida or cfg.carpeta_sitio
        self.base = cfg.ruta_base
        self.url = cfg.url_base
        self.demo = almacen.hay_demo()
        self.paginas = {}  # ruta -> (contenido, tipo, indexable)
        self.env = Environment(loader=FileSystemLoader(CARPETA_PLANTILLAS), autoescape=select_autoescape(["html", "xml"]))
        self.env.filters.update(precio=precio_texto, miles=miles, corto=cantidad_corta, truncar=truncar)
        self.env.globals.update(u=self.u, abs_url=self.abs, json_ld=self.json_ld)
        self.anio = datetime.now(timezone.utc).year

    # --- URLs ---
    def u(self, ruta):
        return f"{self.base}{ruta}"

    def abs(self, ruta):
        return f"{self.url}{ruta}"

    @staticmethod
    def json_ld(datos):
        return json.dumps(datos, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    # --- Vistas ---
    def _color_nicho(self, slug):
        orden = sorted({p.get("nicho_slug") for p in self.almacen.productos.values() if p.get("nicho_slug")})
        return PALETA[orden.index(slug) % len(PALETA)] if slug in orden else PALETA[0]

    def vista_producto(self, p):
        cfg, seo_p = self.cfg, p.get("seo") or {}
        imagenes = p.get("imagenes") or [self.u(f"/img/p/{p['slug']}.svg")]
        if self.demo and p.get("demo"):
            cta = {"url": "#", "texto": "Ver oferta (demo)", "externo": False}
        elif cfg.es_afiliado:
            proveedor = {"aliexpress": "AliExpress", "cj": "CJ"}.get(p.get("fuente"), "la tienda")
            cta = {"url": p.get("url_afiliado") or p.get("url_proveedor"), "texto": f"Ver oferta en {proveedor}", "externo": True}
        else:
            cta = {"url": "#pedido", "texto": "Comprar ahora", "externo": False}
        return {
            **p,
            "url": self.u(f"/p/{p['slug']}/"),
            "url_abs": self.abs(f"/p/{p['slug']}/"),
            "nicho_url": self.u(f"/c/{p.get('nicho_slug')}/"),
            "imagenes": imagenes,
            "imagen": imagenes[0],
            "color": self._color_nicho(p.get("nicho_slug")),
            "cta": cta,
            "seo": seo_p,
            "disponible": p.get("estado") == "publicado",
        }

    # --- Datos estructurados (schema.org) ---
    def _img_abs(self, img):
        return img if img.startswith("http") else self.abs(img[len(self.base):])

    def ld_migas(self, items):
        return {
            "@context": "https://schema.org", "@type": "BreadcrumbList",
            "itemListElement": [
                {"@type": "ListItem", "position": i, "name": nombre, **({"item": self.abs(ruta)} if ruta else {})}
                for i, (nombre, ruta) in enumerate(items, start=1)
            ],
        }

    def ld_lista(self, productos, nombre):
        return {
            "@context": "https://schema.org", "@type": "ItemList", "name": nombre,
            "itemListElement": [
                {"@type": "ListItem", "position": i, "url": v["url_abs"], "name": v["nombre"]} for i, v in enumerate(productos, start=1)
            ],
        }

    def ld_producto(self, v):
        seo_p = v["seo"]
        oferta = {
            "@type": "Offer", "url": v["url_abs"], "price": str(v.get("precio_clp") or 0), "priceCurrency": self.cfg.moneda,
            "availability": "https://schema.org/InStock" if v["disponible"] else "https://schema.org/OutOfStock",
            "itemCondition": "https://schema.org/NewCondition",
        }
        if not self.cfg.es_afiliado:
            dias = v.get("dias_envio") or 15
            oferta["shippingDetails"] = {
                "@type": "OfferShippingDetails",
                "shippingRate": {"@type": "MonetaryAmount", "value": "0", "currency": self.cfg.moneda},
                "shippingDestination": {"@type": "DefinedRegion", "addressCountry": self.cfg.pais},
                "deliveryTime": {
                    "@type": "ShippingDeliveryTime",
                    "handlingTime": {"@type": "QuantitativeValue", "minValue": 1, "maxValue": 3, "unitCode": "DAY"},
                    "transitTime": {"@type": "QuantitativeValue", "minValue": max(3, dias - 3), "maxValue": dias + 5, "unitCode": "DAY"},
                },
            }
            oferta["hasMerchantReturnPolicy"] = {
                "@type": "MerchantReturnPolicy", "applicableCountry": self.cfg.pais,
                "returnPolicyCategory": "https://schema.org/MerchantReturnFiniteReturnWindow",
                "merchantReturnDays": 10, "returnMethod": "https://schema.org/ReturnByMail",
            }
        return {
            "@context": "https://schema.org", "@type": "Product", "name": v["nombre"],
            "image": [self._img_abs(i) for i in v["imagenes"][:5]],
            "description": seo_p.get("meta_descripcion") or seo_p.get("intro") or v["nombre"],
            "sku": v["id"], "category": v.get("nicho"), "url": v["url_abs"], "offers": oferta,
        }

    def contexto_base(self):
        cfg = self.cfg
        nichos = []
        for slug, datos in sorted(self.almacen.nichos.items(), key=lambda kv: kv[1].get("nombre", kv[0])):
            cantidad = len(self.almacen.publicados(slug))
            if cantidad:
                nichos.append(
                    {"slug": slug, "nombre": datos.get("nombre", slug), "url": self.u(f"/c/{slug}/"), "cantidad": cantidad,
                     "icono": ICONOS[icono_de(datos.get("nombre", slug))], "color": self._color_nicho(slug)}
                )
        guias = [
            {"titulo": g["titulo"], "url": self.u(f"/guias/{g['slug']}/"), "nicho": g.get("nombre_nicho"), "slug": g["slug"]}
            for g in sorted(self.almacen.guias.values(), key=lambda g: g.get("nombre_nicho", ""))
            if self.almacen.publicados(g.get("nicho_slug"))
        ]
        return {
            "tienda": {
                "nombre": cfg.nombre, "eslogan": cfg.eslogan, "modo": cfg.modo, "afiliado": cfg.es_afiliado,
                "whatsapp": "".join(c for c in cfg.whatsapp if c.isdigit()), "correo": cfg.correo_contacto,
                "razon_social": cfg.razon_social, "rut": cfg.rut, "color": cfg.color, "acento": cfg.color_acento,
                "ga_id": cfg.ga_id, "google_verificacion": cfg.google_verificacion, "bing_verificacion": cfg.bing_verificacion,
                "idioma": cfg.idioma, "moneda": cfg.moneda, "pedidos_correo": cfg.pedidos_correo, "url": self.url,
            },
            "nichos": nichos,
            "guias": guias,
            "demo": self.demo,
            "anio": self.anio,
            "version": self._version_assets(),
        }

    def _version_assets(self):
        h = hashlib.sha1()
        for nombre in ("estilos.css", "app.js"):
            with open(os.path.join(CARPETA_PLANTILLAS, nombre), "rb") as f:
                h.update(f.read())
        return h.hexdigest()[:8]

    # --- Render ---
    def render(self, ruta, plantilla, tipo="otra", indexable=True, **contexto):
        ctx = {**self.ctx, **contexto, "ruta": ruta, "canonical": self.abs(ruta), "indexable": indexable and not self.demo}
        html = self.env.get_template(plantilla).render(**ctx)
        self.paginas[ruta] = (html, tipo, indexable)

    def construir(self):
        self.ctx = self.contexto_base()
        a = self.almacen
        visibles = sorted((self.vista_producto(p) for p in a.visibles()), key=lambda v: (-v.get("score", 0), v["id"]))
        publicados = [v for v in visibles if v["disponible"]]
        por_nicho = {}
        for v in publicados:
            por_nicho.setdefault(v.get("nicho_slug"), []).append(v)
        guia_de = {g["nicho_slug"]: g for g in a.guias.values() if g.get("nicho_slug") in por_nicho}

        # Portada
        destacados = publicados[:8]
        filas = [(n, por_nicho.get(n["slug"], [])[:4]) for n in self.ctx["nichos"]]
        ld_inicio = [
            {"@context": "https://schema.org", "@type": "WebSite", "name": self.cfg.nombre, "url": self.abs("/")},
            {"@context": "https://schema.org", "@type": "Organization", "name": self.cfg.razon_social or self.cfg.nombre,
             "url": self.abs("/"), "logo": self.abs("/favicon.svg"),
             **({"email": self.cfg.correo_contacto} if self.cfg.correo_contacto else {})},
        ]
        self.render("/", "inicio.html", "inicio", destacados=destacados, filas=filas, total=len(publicados), ld=ld_inicio)

        # Categorías
        for n in self.ctx["nichos"]:
            datos = a.nichos.get(n["slug"], {})
            guia = guia_de.get(n["slug"])
            ld = [self.ld_migas([("Inicio", "/"), (n["nombre"], None)]), self.ld_lista(por_nicho.get(n["slug"], []), n["nombre"])]
            self.render(
                f"/c/{n['slug']}/", "categoria.html", "categoria", nicho=n, datos=datos, productos=por_nicho.get(n["slug"], []), ld=ld,
                guia={"titulo": guia["titulo"], "url": self.u(f"/guias/{guia['slug']}/")} if guia else None,
            )

        # Productos (los pausados conservan su página, sin indexar, con alternativas)
        for v in visibles:
            relacionados = [x for x in por_nicho.get(v.get("nicho_slug"), []) if x["id"] != v["id"]][:4]
            guia = guia_de.get(v.get("nicho_slug"))
            ld = [self.ld_producto(v), self.ld_migas([("Inicio", "/"), (v.get("nicho"), f"/c/{v.get('nicho_slug')}/"), (v["nombre"], None)])]
            self.render(
                f"/p/{v['slug']}/", "producto.html", "producto", indexable=v["disponible"], p=v, relacionados=relacionados, ld=ld,
                guia={"titulo": guia["titulo"], "url": self.u(f"/guias/{guia['slug']}/")} if guia else None,
            )

        # Guías de compra
        vistas_por_id = {v["id"]: v for v in publicados}
        for g in a.guias.values():
            if g.get("nicho_slug") not in por_nicho:
                continue
            items = []
            for item in g.get("productos", []):
                v = vistas_por_id.get(item["id"])
                if v:
                    items.append({**item, "p": v})
            if items:
                ld = [
                    {"@context": "https://schema.org", "@type": "Article", "headline": g["titulo"], "inLanguage": self.cfg.idioma,
                     "datePublished": (g.get("creado") or ahora())[:10], "dateModified": (g.get("actualizado") or ahora())[:10],
                     "author": {"@type": "Organization", "name": self.cfg.nombre}, "mainEntityOfPage": self.abs(f"/guias/{g['slug']}/")},
                    self.ld_lista([i["p"] for i in items], g["titulo"]),
                    self.ld_migas([("Inicio", "/"), ("Guías de compra", "/guias/"), (g.get("nombre_nicho"), None)]),
                ]
                self.render(f"/guias/{g['slug']}/", "guia.html", "guia", g=g, items=items, ld=ld,
                            nicho_url=self.u(f"/c/{g['nicho_slug']}/"), actualizado=(g.get("actualizado") or ahora())[:10])
        self.render("/guias/", "guias.html", "listado", indexable=bool(self.ctx["guias"]))

        # Buscador, páginas legales y de ayuda
        self.render("/buscar/", "buscar.html", "buscar", indexable=False)
        self.render("/terminos/", "terminos.html", "legal")
        self.render("/privacidad/", "privacidad.html", "legal")
        self.render("/como-comprar/", "como_comprar.html", "legal")
        self.render("/contacto/", "contacto.html", "legal")
        if not self.cfg.es_afiliado:
            self.render("/gracias/", "gracias.html", "gracias", indexable=False)
        self.render("/404.html", "404.html", "404", indexable=False)

        return self._escribir(visibles, publicados)

    # --- Archivos ---
    def _escribir(self, visibles, publicados):
        cfg, a = self.cfg, self.almacen
        if os.path.isdir(self.salida):
            shutil.rmtree(self.salida)
        os.makedirs(self.salida)

        def escribir(ruta_rel, contenido, binario=False):
            destino = os.path.join(self.salida, ruta_rel.lstrip("/"))
            if destino.endswith(os.sep) or ruta_rel.endswith("/"):
                destino = os.path.join(destino, "index.html")
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            with open(destino, "wb" if binario else "w", encoding=None if binario else "utf-8") as f:
                f.write(contenido)

        # Detección de cambios: lastmod real para el sitemap y lista de URLs para IndexNow.
        hashes = a.estado.setdefault("hashes", {})
        lastmod = a.estado.setdefault("lastmod", {})
        hoy = ahora()[:10]
        cambiadas, auditoria = [], []
        for ruta, (html, tipo, indexable) in sorted(self.paginas.items()):
            escribir(ruta, html)
            h = hashlib.sha1(html.encode("utf-8")).hexdigest()[:16]
            if hashes.get(ruta) != h:
                hashes[ruta] = h
                lastmod[ruta] = hoy
                if indexable and not self.demo and ruta != "/404.html":
                    cambiadas.append(self.abs(ruta))
            if tipo in ("inicio", "categoria", "producto", "guia", "listado", "legal") and indexable:
                puntaje, problemas, info = seo.auditar_html(html, tipo)
                auditoria.append({"ruta": ruta, "tipo": tipo, "puntaje": puntaje, "problemas": problemas, **info})

        # Estáticos
        for nombre in ("estilos.css", "app.js"):
            with open(os.path.join(CARPETA_PLANTILLAS, nombre), encoding="utf-8") as f:
                escribir(f"/assets/{nombre}", f.read())
        escribir("/favicon.svg", self.env.get_template("favicon.svg").render(**self.ctx))
        for v in visibles:
            if not (a.productos[v["id"]].get("imagenes")):
                escribir(f"/img/p/{v['slug']}.svg", svg_placeholder(v["nombre"], v.get("nicho", ""), v["color"]))
        escribir("/.nojekyll", "")
        if cfg.dominio_propio:
            escribir("/CNAME", cfg.dominio_propio.strip() + "\n")
        if cfg.indexnow_clave:
            escribir(f"/{cfg.indexnow_clave}.txt", cfg.indexnow_clave)

        # Índice del buscador interno
        indice = [
            {"n": v["nombre"], "u": v["url"], "i": v["imagen"], "p": v.get("precio_clp"), "c": v.get("nicho"),
             "k": " ".join((v["seo"].get("keywords") or {}).get("secundarias", [])[:5]), "s": v.get("slug"),
             "v": v.get("valoracion"), "o": v.get("pedidos"), "pg": (v.get("pago") or {}).get("link")}
            for v in publicados
        ]
        escribir("/buscar.json", json.dumps(indice, ensure_ascii=False, separators=(",", ":")))

        # sitemap.xml y robots.txt
        urls = [
            {"loc": self.abs(r), "lastmod": lastmod.get(r, hoy)}
            for r, (_, tipo, indexable) in sorted(self.paginas.items())
            if indexable and tipo not in ("404", "buscar", "gracias")
        ]
        escribir("/sitemap.xml", self.env.get_template("sitemap.xml").render(urls=urls))
        if self.demo:
            robots = "# Modo demostración: el sitio no se indexa hasta cargar productos reales.\nUser-agent: *\nDisallow: /\n"
        else:
            robots = f"User-agent: *\nAllow: /\nDisallow: {self.u('/buscar/')}\nDisallow: {self.u('/gracias/')}\n\nSitemap: {self.abs('/sitemap.xml')}\n"
        escribir("/robots.txt", robots)

        # Feed de productos para Google Merchant Center (listados gratuitos en Google Shopping) — solo venta directa.
        if not cfg.es_afiliado and not self.demo:
            escribir("/feeds/google-merchant.xml", self.env.get_template("merchant.xml").render(productos=publicados, **self.ctx))

        a.estado["auditoria"] = {
            "fecha": ahora(),
            "paginas": len(auditoria),
            "promedio": round(sum(x["puntaje"] for x in auditoria) / len(auditoria), 1) if auditoria else 0,
            "con_problemas": [x for x in sorted(auditoria, key=lambda x: x["puntaje"]) if x["problemas"]][:100],
        }
        # Páginas de producto con contenido escaso: se reescriben en el próximo ciclo.
        for x in auditoria:
            if x["tipo"] == "producto" and any(pr.startswith("contenido escaso") for pr in x["problemas"]):
                slug = x["ruta"].strip("/").split("/")[-1]
                for p in a.productos.values():
                    if p.get("slug") == slug and p.get("seo") and p["seo"].get("origen") != "ia":
                        p["seo"]["rehacer"] = True
        a.estado["ultima_construccion"] = ahora()
        return {"paginas": len(self.paginas), "productos": len(publicados), "cambiadas": cambiadas,
                "auditoria": a.estado["auditoria"]["promedio"], "salida": self.salida}


def construir(almacen, cfg, salida=None):
    return Constructor(almacen, cfg, salida).construir()
