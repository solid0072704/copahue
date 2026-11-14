"""Agente Posicionador (SEO): palabras clave, textos de cada página, guías de compra,
reoptimización con datos reales de Google y auditoría de cada página generada."""

import json
import logging
import re
from datetime import datetime, timezone
from html.parser import HTMLParser

from . import indexacion
from .almacen import ahora, dias_desde
from .keywords import Investigador, investigar
from .texto import caracteristicas_de_titulo, capitalizar, cantidad_corta, contar_palabras, normalizar, precio_texto, slugify, truncar

log = logging.getLogger(__name__)

MIN_PRODUCTOS_GUIA = 3


def _anio():
    return datetime.now(timezone.utc).year


def _primero_que_cabe(opciones, maximo):
    for o in opciones:
        if o and len(o) <= maximo:
            return o
    return truncar(opciones[-1], maximo)


def _envio_texto(p):
    dias = p.get("dias_envio")
    if not dias:
        return "entre 10 y 20 días hábiles"
    return f"entre {max(5, dias - 3)} y {dias + 5} días hábiles"


# ---------------------------------------------------------------------------
# Textos por reglas (sin IA): honestos, basados solo en los datos del proveedor
# ---------------------------------------------------------------------------
def textos_por_reglas(p, kw, cfg):
    nombre = p.get("nombre") or "Producto"
    keyword = kw.get("principal") or normalizar(nombre)
    precio = precio_texto(p.get("precio_clp"))
    val = p.get("valoracion")
    pedidos = p.get("pedidos")
    envio = _envio_texto(p)
    caracteristicas = caracteristicas_de_titulo(p.get("titulo_original"))

    datos = []
    if pedidos:
        datos.append(f"más de {cantidad_corta(pedidos)} pedidos en el mundo")
    if val:
        datos.append(f"{val:.0f}% de valoraciones positivas")
    datos_txt = " y ".join(datos)

    titulo = _primero_que_cabe(
        [f"{nombre} – Envío a todo Chile", f"{nombre} | {cfg.nombre}", f"{nombre} en Chile", nombre], 60
    )
    if cfg.es_afiliado:
        meta = (
            f"{nombre}: precio en Chile ({precio} aprox.), características y tiempo de envío"
            + (f". {capitalizar(datos_txt)}." if datos_txt else ".")
        )
        compra = (
            "La compra se realiza en la tienda del proveedor, con su protección al comprador: si el producto no "
            "llega o no coincide con la descripción, puedes pedir el reembolso desde tu cuenta."
        )
    else:
        meta = f"Compra {nombre[:1].lower() + nombre[1:]} con envío a todo Chile por {precio}" + (
            f". {capitalizar(datos_txt)}." if datos_txt else ". Pago seguro."
        )
        compra = (
            "Pagas de forma segura y te enviamos el número de seguimiento apenas sale tu pedido. Tienes 10 días "
            "desde que lo recibes para ejercer tu derecho a retracto y 6 meses de garantía legal."
        )
    beneficios = caracteristicas[:5] or []
    if pedidos and len(beneficios) < 6:
        beneficios.append(f"Probado por compradores: más de {cantidad_corta(pedidos)} pedidos")
    if len(beneficios) < 6:
        beneficios.append(f"Envío a todo Chile, {envio}")
    intro = (
        f"¿Buscas {keyword}? {nombre}"
        + (f", con {datos_txt}" if datos_txt else "")
        + f". Llega a todo Chile {envio}."
    )
    descripcion = [
        f"{nombre} es una de las opciones mejor evaluadas de nuestra sección {p.get('nicho', '')}. "
        f"Lo seleccionamos porque combina buena valoración de compradores, demanda constante y un precio "
        f"accesible: hoy está en {precio}{' aproximadamente' if cfg.es_afiliado else ', IVA incluido'}.",
        f"Se despacha desde la bodega del proveedor internacional y llega a cualquier región de Chile {envio}. "
        "Recibirás el código de seguimiento para revisar el estado del envío en todo momento.",
        compra,
    ]
    faq = [
        {"pregunta": "¿Cuánto demora en llegar a Chile?", "respuesta": f"El tiempo estimado es {envio}, según la región y el operador logístico."},
        {
            "pregunta": "¿Cómo se compra?",
            "respuesta": "Presiona «Ver oferta»: te llevaremos a la página del proveedor, donde eliges la variante y pagas con tarjeta."
            if cfg.es_afiliado
            else "Presiona «Comprar», completa tus datos de envío y paga de forma segura. Te confirmamos el pedido por correo.",
        },
        {
            "pregunta": "¿Puedo devolverlo?",
            "respuesta": "Sí, según la política de protección al comprador del proveedor, que permite reclamar si no llega o no es lo descrito."
            if cfg.es_afiliado
            else "Sí. Tienes 10 días desde la recepción para retractarte de la compra y 6 meses de garantía legal si presenta fallas.",
        },
    ]
    return {
        "titulo_seo": titulo,
        "meta_descripcion": truncar(meta, 158),
        "keyword_principal": keyword,
        "intro": intro,
        "beneficios": beneficios,
        "descripcion": descripcion,
        "usos": [],
        "caracteristicas": caracteristicas,
        "faq": faq,
        "alt_imagen": f"{nombre} – foto del producto",
    }


def _asegurar_slug(almacen, p):
    if p.get("slug"):
        return
    usados = {x.get("slug") for x in almacen.productos.values() if x.get("slug")}
    base = slugify(p.get("nombre") or p.get("titulo_original"), 60)
    sufijo = re.sub(r"[^a-z0-9]", "", p["id_fuente"].lower())[-5:] or "1"
    slug = f"{base}-{sufijo}"
    n = 2
    while slug in usados:
        slug = f"{base}-{sufijo}{n}"
        n += 1
    p["slug"] = slug


# ---------------------------------------------------------------------------
# Ciclo del agente
# ---------------------------------------------------------------------------
def posicionar(almacen, cfg, redactor=None, usar_red=True):
    r = {"textos": [], "ia": 0, "guias": [], "nichos": [], "reoptimizados": [], "errores": []}
    investigador = Investigador(almacen, cfg, usar_red=usar_red)
    presupuesto = cfg.llm_max_productos_por_ciclo if redactor else 0

    # 1) Textos de producto: nuevos primero, luego los que quedaron pendientes de IA.
    publicados = sorted(
        (p for p in almacen.productos.values() if p.get("estado") in ("publicado", "pausado")),
        key=lambda p: (bool(p.get("seo")), -p.get("score", 0)),
    )
    for p in publicados:
        seo = p.get("seo")
        quiere_ia = redactor and presupuesto > 0 and (not seo or seo.get("origen") != "ia") and not p.get("demo")
        if seo and not quiere_ia:
            _asegurar_slug(almacen, p)
            continue
        kw = (seo or {}).get("keywords") or investigar(investigador, p.get("nombre", ""))
        textos = None
        if quiere_ia:
            presupuesto -= 1
            textos = redactor.producto(p, kw)
            if textos:
                r["ia"] += 1
                if not p.get("slug"):
                    p["nombre"] = truncar(textos.pop("nombre"), 70, "")
                else:
                    textos.pop("nombre", None)  # la URL ya existe: el nombre visible puede cambiar, el slug no
                textos["caracteristicas"] = caracteristicas_de_titulo(p.get("titulo_original"))
                textos["titulo_seo"] = truncar(textos["titulo_seo"], 62)
                textos["meta_descripcion"] = truncar(textos["meta_descripcion"], 158)
                textos["origen"] = "ia"
        if not textos:
            textos = textos_por_reglas(p, kw, cfg)
            textos["origen"] = "reglas"
        textos["keywords"] = kw
        textos["actualizado"] = ahora()
        textos["version"] = (seo or {}).get("version", 0) + 1
        if seo:  # conserva lo aprendido de Search Console
            for campo in ("faq_gsc", "reoptimizado", "consultas_gsc"):
                if campo in seo:
                    textos[campo] = seo[campo]
        p["seo"] = textos
        _asegurar_slug(almacen, p)
        r["textos"].append(p["id"])
        almacen.log("Posicionador", "Textos SEO " + ("con IA" if textos["origen"] == "ia" else "por reglas"), f"{p['nombre']} · «{kw.get('principal')}»", p["id"])

    # 2) Textos de cada categoría (nicho).
    nichos_unicos = {}
    for nombre in [n["nombre"] for n in cfg.nichos] + [p.get("nicho") or "Otros" for p in publicados]:
        nichos_unicos.setdefault(slugify(nombre), nombre)
    for slug, nombre_nicho in nichos_unicos.items():
        nicho = {"nombre": nombre_nicho}
        productos = almacen.publicados(slug)
        if not productos:
            continue
        actual = almacen.nichos.get(slug) or {}
        if actual.get("intro") and (actual.get("origen") == "ia" or not redactor):
            continue
        kw = investigar(investigador, nicho["nombre"])
        datos = redactor.nicho(nicho["nombre"], productos, kw) if redactor else None
        if datos:
            datos["origen"] = "ia"
        else:
            datos = {
                "h1": f"{nicho['nombre']}: productos con envío a todo Chile",
                "titulo_seo": _primero_que_cabe([f"{nicho['nombre']} con envío a Chile | {cfg.nombre}", f"{nicho['nombre']} con envío a Chile"], 60),
                "meta_descripcion": truncar(
                    f"Los productos de {nicho['nombre'].lower()} mejor evaluados por compradores, con precio en pesos y "
                    "envío a todo Chile. Selección actualizada cada día.", 158
                ),
                "intro": [
                    f"Seleccionamos cada día los productos de {nicho['nombre'].lower()} con mejores valoraciones y más pedidos, "
                    "para que compres sin perder tiempo comparando.",
                    "Cada producto muestra su precio en pesos chilenos, el tiempo estimado de envío a tu región y la valoración "
                    "de compradores reales. Si un producto deja de estar disponible o empeora su evaluación, lo retiramos de la vitrina.",
                ],
                "origen": "reglas",
            }
        datos.update(nombre=nicho["nombre"], keywords=kw, actualizado=ahora())
        almacen.nichos[slug] = datos
        r["nichos"].append(slug)
        almacen.log("Posicionador", "Textos de categoría", nicho["nombre"], slug)

    # 3) Guías de compra («los mejores … en Chile»): capturan búsquedas informativas y enlazan a los productos.
    hechas = 0
    for slug, datos_nicho in sorted(almacen.nichos.items()):
        productos = almacen.publicados(slug)[:8]
        if len(productos) < MIN_PRODUCTOS_GUIA or hechas >= cfg.max_guias_por_ciclo:
            continue
        ids = [p["id"] for p in productos]
        guia = almacen.guias.get(slug)
        cambio = not guia or set(ids) != set(guia.get("ids", [])) or guia.get("anio") != _anio()
        mejorar_con_ia = redactor and guia and guia.get("origen") != "ia" and not any(p.get("demo") for p in productos)
        if not cambio and not mejorar_con_ia:
            continue
        nombre = datos_nicho.get("nombre", slug)
        guia_nueva = _guia(almacen, cfg, redactor, investigador, slug, nombre, productos)
        if guia:
            guia_nueva["creado"] = guia.get("creado", ahora())
        almacen.guias[slug] = guia_nueva
        hechas += 1
        r["guias"].append(slug)
        almacen.log("Posicionador", "Guía de compra " + ("creada" if not guia else "actualizada"), guia_nueva["titulo"], slug)

    # 4) Aprende de Google: reoptimiza páginas con impresiones pero pocos clics.
    if cfg.gsc_configurado and usar_red:
        try:
            r["reoptimizados"] = reoptimizar_con_search_console(almacen, cfg, redactor)
        except Exception as e:
            r["errores"].append(f"Search Console: {e}")
            almacen.log("Posicionador", "Error leyendo Search Console", str(e))
    return r


def _guia(almacen, cfg, redactor, investigador, slug, nombre, productos):
    anio = _anio()
    kw = investigar(investigador, f"mejores {nombre}")
    datos = None
    if redactor and not any(p.get("demo") for p in productos):
        datos = redactor.guia(nombre, productos, kw, anio)
        if datos:
            por_id = {x.get("id"): x for x in datos.get("productos", [])}
            datos["productos"] = [
                {"id": p["id"], "por_que": por_id.get(p["id"], {}).get("por_que", ""), "ideal_para": por_id.get(p["id"], {}).get("ideal_para", "")}
                for p in productos
            ]
            datos["titulo_seo"] = truncar(datos["titulo_seo"], 62)
            datos["meta_descripcion"] = truncar(datos["meta_descripcion"], 158)
            datos["origen"] = "ia"
    if not datos:
        items = []
        for p in productos:
            partes = []
            if p.get("valoracion"):
                partes.append(f"{p['valoracion']:.0f}% de valoraciones positivas")
            if p.get("pedidos"):
                partes.append(f"más de {cantidad_corta(p['pedidos'])} pedidos")
            items.append(
                {
                    "id": p["id"],
                    "por_que": (capitalizar(" y ".join(partes)) + ". " if partes else "")
                    + f"Cuesta {precio_texto(p.get('precio_clp'))} y llega {_envio_texto(p)}.",
                    "ideal_para": "",
                }
            )
        datos = {
            "titulo": f"Los mejores productos de {nombre.lower()} en Chile ({anio})",
            "titulo_seo": _primero_que_cabe([f"Mejores productos de {nombre.lower()} en Chile {anio}", f"{nombre}: los mejores de {anio}"], 60),
            "meta_descripcion": truncar(
                f"Comparamos los productos de {nombre.lower()} mejor evaluados por compradores: precios en pesos, "
                f"tiempos de envío a Chile y cuál conviene según tu uso. Actualizado {anio}.", 158
            ),
            "intro": [
                f"Elegir bien en {nombre.lower()} no es fácil: hay miles de opciones y muchas se ven iguales. Por eso "
                "revisamos a diario los datos de compradores reales —valoraciones, cantidad de pedidos y tiempos de envío— "
                "y dejamos solo los que destacan.",
                f"Esta guía resume los {len(productos)} mejores de nuestra selección actual, con su precio en pesos chilenos "
                "y lo que debes mirar antes de comprar.",
            ],
            "criterios": [
                {"titulo": "Valoración de compradores", "texto": "Prioriza productos sobre 95% de valoraciones positivas: es la señal más confiable de que cumplen lo que prometen."},
                {"titulo": "Cantidad de pedidos", "texto": "Miles de pedidos indican un producto probado y un proveedor con stock constante."},
                {"titulo": "Tiempo de envío", "texto": "Revisa el plazo estimado a tu región; si lo necesitas para una fecha, compra con margen."},
                {"titulo": "Precio total", "texto": "Compara el precio final en pesos, no solo el de lista. Todos los precios de esta guía están en CLP."},
            ],
            "productos": items,
            "faq": [
                {"pregunta": "¿Cada cuánto se actualiza esta guía?", "respuesta": "Los precios y la disponibilidad se revisan todos los días; la selección cambia cuando aparece un producto claramente mejor."},
                {"pregunta": "¿Cómo eligen los productos?", "respuesta": "Con un puntaje que combina valoración de compradores, demanda, precio y tiempo de envío a Chile."},
            ],
            "conclusion": f"Si tienes dudas, parte por el primero de la lista: es el que mejor combina evaluación, demanda y precio en {nombre.lower()}.",
            "origen": "reglas",
        }
    datos.update(slug=f"mejores-{slug}", nicho_slug=slug, nombre_nicho=nombre, ids=[p["id"] for p in productos], anio=anio, keywords=kw, actualizado=ahora())
    datos.setdefault("creado", ahora())
    if any(p.get("demo") for p in productos):
        datos["demo"] = True
    return datos


def reoptimizar_con_search_console(almacen, cfg, redactor=None):
    """Páginas con impresiones (Google ya las muestra) pero CTR bajo: reescribe título y meta con
    las consultas reales. Espera 21 días entre cambios de una misma página para medir el efecto."""
    filas = indexacion.consultas_search_console(cfg)
    por_pagina = {}
    for fila in filas:
        pagina, consulta = fila["keys"][0], fila["keys"][1]
        d = por_pagina.setdefault(pagina, {"clics": 0, "impresiones": 0, "pos_pond": 0.0, "consultas": []})
        d["clics"] += fila.get("clicks", 0)
        d["impresiones"] += fila.get("impressions", 0)
        d["pos_pond"] += fila.get("position", 0) * fila.get("impressions", 0)
        d["consultas"].append({"consulta": consulta, "impresiones": fila.get("impressions", 0), "posicion": round(fila.get("position", 0), 1)})
    por_slug = {p.get("slug"): p for p in almacen.productos.values() if p.get("slug")}
    hechos = []
    for pagina, d in por_pagina.items():
        m = re.search(r"/p/([^/]+)/?", pagina)
        p = por_slug.get(m.group(1)) if m else None
        if not p:
            continue
        impresiones = d["impresiones"]
        ctr = d["clics"] / impresiones if impresiones else 0
        posicion = d["pos_pond"] / impresiones if impresiones else 0
        consultas = sorted(d["consultas"], key=lambda c: -c["impresiones"])[:8]
        p["gsc"] = {"clics": d["clics"], "impresiones": impresiones, "ctr": round(ctr, 4), "posicion": round(posicion, 1), "fecha": ahora()[:10]}
        seo = p.get("seo") or {}
        if impresiones < 30 or ctr >= 0.02 or not 3 <= posicion <= 20 or dias_desde(seo.get("reoptimizado")) < 21:
            continue
        datos = redactor.reoptimizar(p, consultas) if redactor else None
        if datos:
            seo["titulo_seo"] = truncar(datos["titulo_seo"], 62)
            seo["meta_descripcion"] = truncar(datos["meta_descripcion"], 158)
            seo["faq_gsc"] = datos.get("faq_extra", [])[:2]
        else:
            top = capitalizar(consultas[0]["consulta"])
            seo["titulo_seo"] = _primero_que_cabe([f"{top} | {p['nombre']}", f"{top} – Envío a Chile", top], 60)
            seo["meta_descripcion"] = truncar(f"{top}: {seo.get('meta_descripcion', '')}", 158)
        seo["consultas_gsc"] = consultas
        seo["reoptimizado"] = ahora()
        p["seo"] = seo
        hechos.append(p["id"])
        almacen.log("Posicionador", "Reoptimizado con Search Console", f"{p['nombre']}: {impresiones} impresiones, CTR {ctr:.1%}, pos. {posicion:.1f}", p["id"])
    return hechos


# ---------------------------------------------------------------------------
# Auditoría de las páginas generadas
# ---------------------------------------------------------------------------
class _Lector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.titulo, self.meta, self.canonical, self.robots = "", None, None, ""
        self.h1, self.imgs_sin_alt, self.enlaces_internos = 0, 0, 0
        self.jsonld, self._en, self._texto_main, self._dentro_main = [], None, [], 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title":
            self._en = "title"
        elif tag == "meta" and a.get("name") == "description":
            self.meta = a.get("content", "")
        elif tag == "meta" and a.get("name") == "robots":
            self.robots = a.get("content", "")
        elif tag == "link" and a.get("rel") == "canonical":
            self.canonical = a.get("href")
        elif tag == "h1":
            self.h1 += 1
        elif tag == "img" and not (a.get("alt") or "").strip():
            self.imgs_sin_alt += 1
        elif tag == "a" and (a.get("href") or "").startswith("/"):
            self.enlaces_internos += 1
        elif tag == "script" and a.get("type") == "application/ld+json":
            self._en = "jsonld"
            self.jsonld.append("")
        elif tag == "main":
            self._dentro_main += 1
        elif tag in ("script", "style"):
            self._en = "ignorar"

    def handle_endtag(self, tag):
        if tag in ("title", "script", "style"):
            self._en = None
        elif tag == "main":
            self._dentro_main -= 1

    def handle_data(self, data):
        if self._en == "title":
            self.titulo += data
        elif self._en == "jsonld":
            self.jsonld[-1] += data
        elif self._en is None and self._dentro_main > 0:
            self._texto_main.append(data)


def auditar_html(html, tipo):
    """Devuelve (puntaje 0-100, [problemas]) para una página."""
    lector = _Lector()
    lector.feed(html)
    problemas = []
    titulo = lector.titulo.strip()
    if not titulo:
        problemas.append("sin <title>")
    elif not 25 <= len(titulo) <= 65:
        problemas.append(f"título de {len(titulo)} caracteres (ideal 30-60)")
    if lector.meta is None or not lector.meta.strip():
        problemas.append("sin meta descripción")
    elif not 70 <= len(lector.meta) <= 165:
        problemas.append(f"meta descripción de {len(lector.meta)} caracteres (ideal 120-160)")
    if lector.h1 != 1:
        problemas.append(f"{lector.h1} encabezados H1 (debe haber 1)")
    if not lector.canonical:
        problemas.append("sin URL canónica")
    if lector.imgs_sin_alt:
        problemas.append(f"{lector.imgs_sin_alt} imágenes sin texto alternativo")
    palabras = contar_palabras(" ".join(lector._texto_main))
    minimo = {"producto": 150, "guia": 400, "categoria": 120}.get(tipo, 0)
    if palabras < minimo:
        problemas.append(f"contenido escaso ({palabras} palabras, mínimo {minimo})")
    if tipo in ("producto", "guia", "categoria") and lector.enlaces_internos < 5:
        problemas.append("pocos enlaces internos")
    for bloque in lector.jsonld:
        try:
            json.loads(bloque)
        except json.JSONDecodeError:
            problemas.append("datos estructurados (JSON-LD) inválidos")
    if tipo in ("producto", "guia", "categoria") and not lector.jsonld:
        problemas.append("sin datos estructurados")
    puntaje = max(0, 100 - 15 * len(problemas))
    return puntaje, problemas, {"titulo": titulo, "palabras": palabras, "noindex": "noindex" in lector.robots}
