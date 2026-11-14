"""Agente Buscador: encuentra productos ganadores, calcula precio y margen, los puntúa y decide cuáles publicar.

Score 0-100:
    Demanda (pedidos en el proveedor o tiendas que lo venden)  hasta 30
    Margen (dropshipping) o comisión por venta (afiliado)      hasta 25
    Valoración de compradores                                  hasta 15
    Precio de compra impulsiva ($9.990 – $49.990)              hasta 15
    Rapidez de envío a Chile                                   hasta 10
    Calidad del material (fotos, video)                        hasta 5
"""

import logging
import math
import re

from . import divisas
from .almacen import ahora
from .fuentes import clientes_activos, demo
from .texto import limpiar_titulo, normalizar, redondear_precio, slugify

log = logging.getLogger(__name__)

PREFIJO = {"aliexpress": "ae", "cj": "cj", "archivo": "ar", "demo": "demo"}

# Marcas que casi siempre son falsificaciones en estos catálogos (vender réplicas es ilegal).
MARCAS = [
    "nike", "adidas", "gucci", "louis vuitton", "chanel", "rolex", "disney", "marvel", "pokemon", "lego",
    "airpods", "dyson", "stanley", "jbl", "beats", "ray ban", "rayban", "supreme", "balenciaga", "prada",
    "hermes", "cartier", "pandora", "swarovski", "hello kitty", "barbie", "harry potter", "star wars",
    "playstation", "nintendo", "xbox", "apple watch", "michael kors", "the north face", "under armour",
]
REPLICAS = ["replica", "1:1", "aaa", "imitacion", "clon", "copia", "fake", "copy"]
# Productos con restricciones de importación, regulación sanitaria o riesgo legal en Chile.
RESTRINGIDOS = [
    "vape", "vaper", "cigarrillo electronico", "e-cigarette", "cbd", "thc", "medicamento", "pastillas",
    "suplemento", "inyeccion", "jeringa", "pistola de aire", "airsoft", "cuchillo tactico", "navaja",
    "punal", "taser", "electroshock", "puntero laser", "laser pointer", "dron", "drone", "power bank",
    "bateria de litio", "lithium battery", "camara espia", "spy camera", "inhibidor", "jammer",
    "gas pimienta", "pepper spray", "semillas", "seeds", "lentes de contacto", "blanqueador dental",
    "sexual", "adult toy", "pirotecnia", "fuegos artificiales",
]


def _contiene(texto, termino):
    termino = normalizar(termino)
    if re.fullmatch(r"[a-z0-9 ]+", termino):
        return re.search(rf"\b{re.escape(termino)}\b", texto) is not None
    return termino in texto


def motivo_exclusion(p, cfg):
    """Devuelve por qué NO conviene vender este producto, o None si pasa los filtros."""
    texto = normalizar(f"{p.get('titulo_original', '')} {p.get('categoria_fuente', '')}")
    for termino in MARCAS:
        if _contiene(texto, termino):
            return f"marca registrada «{termino}» (riesgo de falsificación)"
    for termino in REPLICAS:
        if _contiene(texto, termino):
            return f"posible réplica («{termino}»)"
    for termino in RESTRINGIDOS + list(cfg.excluir_palabras):
        if _contiene(texto, termino):
            return f"producto restringido o riesgoso («{termino}»)"
    costo = p.get("costo_usd") or 0
    if costo < cfg.precio_min_usd:
        return f"precio de costo muy bajo (US$ {costo:.2f}): poca ganancia por venta"
    if costo > cfg.precio_max_usd:
        return f"precio de costo muy alto (US$ {costo:.2f}): difícil de vender por impulso"
    if not p.get("imagenes") and not p.get("demo"):
        return "sin fotos"
    if cfg.es_afiliado and not p.get("url_afiliado") and not p.get("demo"):
        return "sin enlace de afiliado (en modo afiliado no deja comisión)"
    return None


def calcular_precio(p, cfg, tc):
    """Precio de venta en CLP, ganancia estimada por venta y margen sobre el precio neto."""
    costo_usd = (p.get("costo_usd") or 0) + (p.get("envio_usd") if p.get("envio_usd") is not None else cfg.envio_usd_defecto)
    if cfg.es_afiliado:
        precio = int(round((p.get("costo_usd") or 0) * tc, -1))  # precio referencial del proveedor
        ganancia = precio * (p.get("comision") or 0) / 100
        return {"precio_clp": precio, "ganancia_clp": int(round(ganancia)), "margen": None, "costo_clp": precio}
    costo = costo_usd * tc * (1 + cfg.recargo_importacion)
    precio = redondear_precio(costo * cfg.multiplicador_precio, cfg.terminacion_precio)

    def margen_de(pr):
        neto = pr / (1 + cfg.iva)
        return (neto - costo - pr * cfg.comision_pasarela) / neto if neto else 0

    intentos = 0
    while margen_de(precio) < cfg.margen_minimo and intentos < 20:
        precio = redondear_precio(precio * 1.08, cfg.terminacion_precio)
        intentos += 1
    neto = precio / (1 + cfg.iva)
    ganancia = neto - costo - precio * cfg.comision_pasarela
    return {"precio_clp": precio, "ganancia_clp": int(round(ganancia)), "margen": round(margen_de(precio), 3), "costo_clp": int(round(costo))}


def aplicar_precio(p, cfg, tc):
    """Actualiza precio y ganancia. Si el cambio es menor al 3% se mantiene el precio publicado
    (evita que todas las páginas 'cambien' cada día por el dólar), salvo que el margen quede bajo el mínimo."""
    anterior = p.get("precio_clp")
    nuevo = calcular_precio(p, cfg, tc)
    if anterior and nuevo["precio_clp"] and abs(nuevo["precio_clp"] - anterior) / anterior < 0.03:
        fijo = dict(nuevo, precio_clp=anterior)
        if cfg.es_afiliado:
            fijo["ganancia_clp"] = int(round(anterior * (p.get("comision") or 0) / 100))
            p.update(fijo)
            return
        neto = anterior / (1 + cfg.iva)
        margen = (neto - nuevo["costo_clp"] - anterior * cfg.comision_pasarela) / neto
        if margen >= cfg.margen_minimo:
            fijo.update(margen=round(margen, 3), ganancia_clp=int(round(neto * margen)))
            p.update(fijo)
            return
    p.update(nuevo)


def puntuar(p, cfg):
    d = {}
    pedidos, listados = p.get("pedidos"), p.get("listados")
    if pedidos:
        d["demanda"] = min(30, round(7.5 * math.log10(pedidos + 1)))  # 10→8 · 100→15 · 1.000→23 · 10.000→30
    elif listados:
        d["demanda"] = 30 if listados >= 2000 else 22 if listados >= 500 else 14 if listados >= 100 else 7 if listados >= 20 else 2
    else:
        d["demanda"] = 10  # sin datos: neutral

    if cfg.es_afiliado:
        g = p.get("ganancia_clp") or 0
        if p.get("comision") is None:
            d["ganancia"] = 12  # comisión desconocida (p. ej. CSV sin esa columna): neutral
        else:
            d["ganancia"] = 25 if g >= 2500 else 20 if g >= 1500 else 14 if g >= 900 else 8 if g >= 500 else 3
    else:
        m = p.get("margen") or 0
        d["margen"] = 25 if m >= 0.45 else 20 if m >= 0.38 else 14 if m >= 0.32 else 8 if m >= 0.25 else 0

    v = p.get("valoracion")
    d["valoracion"] = 6 if v is None else 15 if v >= 97 else 12 if v >= 95.5 else 9 if v >= 94 else 5 if v >= 91 else 0

    precio = p.get("precio_clp") or 0
    d["precio"] = 15 if 9_990 <= precio <= 49_990 else 9 if 49_990 < precio <= 89_990 else 6 if 4_990 <= precio < 9_990 else 2

    dias = p.get("dias_envio")
    d["envio"] = 5 if not dias else 10 if dias <= 10 else 8 if dias <= 14 else 5 if dias <= 20 else 2 if dias <= 30 else 0

    fotos = len(p.get("imagenes") or [])
    d["fotos"] = min(5, (2 if fotos >= 1 else 0) + (2 if fotos >= 4 else 0) + (1 if p.get("video") else 0))
    if p.get("demo"):
        d["fotos"] = 3
    return sum(d.values()), d


def incorporar(almacen, cfg, crudos, tc, nicho=None, agente="Buscador"):
    """Guarda (o actualiza) productos normalizados. Devuelve los ids nuevos."""
    nuevos = []
    slugs_nicho = {n["nombre"]: slugify(n["nombre"]) for n in cfg.nichos}
    for c in crudos:
        pid = f"{PREFIJO.get(c['fuente'], c['fuente'])}-{c['id_fuente']}"
        nombre_nicho = c.get("nicho") or nicho or "Otros"
        existente = almacen.productos.get(pid)
        if existente:
            p = existente
            if c.get("costo_usd") and c["costo_usd"] != p.get("costo_usd"):
                historial = p.setdefault("historial_costo", [])
                historial.append([ahora()[:10], c["costo_usd"]])
                del historial[:-30]
            for campo in ("costo_usd", "envio_usd", "pedidos", "listados", "valoracion", "comision", "dias_envio",
                          "url_afiliado", "url_proveedor", "video", "precio_lista_usd"):
                if c.get(campo) is not None:
                    p[campo] = c[campo]
            if c.get("imagenes") and len(c["imagenes"]) >= len(p.get("imagenes") or []):
                p["imagenes"] = c["imagenes"]
            p["no_visto"] = 0
            if p.get("estado") == "pausado" and p.get("motivo", "").startswith("no disponible"):
                p["estado"], p["motivo"] = "publicado", ""
                almacen.log(agente, "Vuelve a estar disponible", p.get("nombre"), pid)
        else:
            p = dict(c)
            p.update(
                id=pid,
                nicho=nombre_nicho,
                nicho_slug=slugs_nicho.get(nombre_nicho, slugify(nombre_nicho)),
                nombre=limpiar_titulo(c["titulo_original"]),
                estado="candidato",
                creado=ahora(),
                no_visto=0,
            )
            almacen.productos[pid] = p
            nuevos.append(pid)
        p["visto"] = ahora()
        aplicar_precio(p, cfg, tc)
        motivo = motivo_exclusion(p, cfg)
        if motivo and not p.get("manual"):
            if p.get("estado") == "publicado":
                p["estado"] = "pausado"  # su página sigue viva, pero sale de la vitrina
            elif p.get("estado") != "pausado":
                p["estado"] = "descartado"
            p["motivo"] = motivo
        elif p.get("estado") == "descartado" and not p.get("manual"):
            p["estado"], p["motivo"] = "candidato", ""
        p["score"], p["score_detalle"] = puntuar(p, cfg)
    return nuevos


def _actualizar_publicados_aliexpress(almacen, cliente, agente="Buscador"):
    """Revisa precio y disponibilidad de lo que ya está a la venta (evita vender con pérdida)."""
    publicados = [p for p in almacen.productos.values() if p.get("fuente") == "aliexpress" and p.get("estado") == "publicado"]
    if not publicados:
        return [], []
    frescos = cliente.detalle([p["id_fuente"] for p in publicados])
    vistos = {f["id_fuente"] for f in frescos}
    pausados = []
    for p in publicados:
        if p["id_fuente"] in vistos:
            continue
        p["no_visto"] = p.get("no_visto", 0) + 1
        if p["no_visto"] >= 2:
            p["estado"], p["motivo"] = "pausado", "no disponible en el proveedor"
            pausados.append(p["id"])
            almacen.log(agente, "Pausado: ya no está disponible", p.get("nombre"), p["id"])
    return frescos, pausados


def buscar(almacen, cfg, tc=None, clientes=None, usar_red=True):
    """Recorre cada nicho en cada proveedor configurado. Devuelve (nuevos, errores)."""
    tc = tc or divisas.tipo_cambio(almacen, cfg, usar_red)
    clientes = clientes if clientes is not None else clientes_activos(cfg)
    nuevos, errores = [], []
    for nombre_fuente, cliente in clientes.items():
        if nombre_fuente == "aliexpress":
            try:
                frescos, _ = _actualizar_publicados_aliexpress(almacen, cliente)
                incorporar(almacen, cfg, frescos, tc)
            except Exception as e:
                errores.append(f"Actualización AliExpress: {e}")
        for nicho in cfg.nichos:
            for busqueda in nicho["busquedas"]:
                try:
                    crudos = cliente.buscar(busqueda, cfg.resultados_por_busqueda)
                except Exception as e:
                    errores.append(f"{nombre_fuente} «{busqueda}»: {e}")
                    continue
                ids = incorporar(almacen, cfg, crudos, tc, nicho["nombre"])
                nuevos += ids
                almacen.log("Buscador", f"Búsqueda en {nombre_fuente}", f"«{busqueda}» ({nicho['nombre']}): {len(crudos)} resultados, {len(ids)} nuevos")
    # Recalcula precios de todo el catálogo con el dólar del día.
    for p in almacen.productos.values():
        if p.get("estado") in ("publicado", "candidato", "pausado"):
            aplicar_precio(p, cfg, tc)
            margen_ok = cfg.es_afiliado or (p.get("margen") or 0) >= cfg.margen_minimo
            if not margen_ok and p.get("estado") == "publicado":
                p["estado"], p["motivo"] = "pausado", "margen bajo el mínimo"
            elif margen_ok and p.get("estado") == "pausado" and p.get("motivo") == "margen bajo el mínimo":
                p["estado"], p["motivo"] = "publicado", ""
            p["score"], p["score_detalle"] = puntuar(p, cfg)
    return nuevos, errores


def cargar_demo(almacen, cfg):
    tc = almacen.estado.get("tipo_cambio") or cfg.tipo_cambio_defecto
    return incorporar(almacen, cfg, demo.productos(), tc, agente="Demo")


def seleccionar(almacen, cfg):
    """Decide qué se publica. Lo ya publicado se mantiene (estabilidad para Google) salvo que algo
    mucho mejor ocupe su lugar. Las decisiones manuales del tablero siempre mandan."""
    publicados_ahora, pausados_ahora = [], []
    cupo_ciclo = cfg.max_nuevos_por_ciclo
    for p in almacen.productos.values():
        if p.get("manual") in ("publicado", "pausado", "descartado") and p.get("estado") != p["manual"]:
            p["estado"] = p["manual"]
        elif (
            not p.get("manual") and p.get("estado") == "pausado" and p.get("score", 0) >= cfg.score_minimo
            and str(p.get("motivo", "")).startswith(("score bajo", "reemplazado"))
        ):
            p["estado"] = "candidato"  # vuelve a competir por un lugar (conserva su URL)
    nichos = {}
    for p in almacen.productos.values():
        nichos.setdefault(p.get("nicho_slug"), []).append(p)
    # Reparto justo: ninguna categoría se lleva todo el cupo de productos nuevos del ciclo.
    cupo_nicho = max(1, math.ceil(cfg.max_nuevos_por_ciclo / max(1, len(nichos))))
    for slug, productos in sorted(nichos.items(), key=lambda kv: str(kv[0])):
        cupo_ciclo_nicho = cupo_nicho
        publicados = [p for p in productos if p.get("estado") == "publicado"]
        candidatos = sorted(
            (p for p in productos if p.get("estado") == "candidato" and p.get("score", 0) >= cfg.score_minimo),
            key=lambda p: -p.get("score", 0),
        )
        for c in candidatos:
            if cupo_ciclo <= 0 or cupo_ciclo_nicho <= 0:
                break
            if len(publicados) < cfg.max_por_nicho:
                publicados.append(c)
            else:
                reemplazables = [p for p in publicados if not p.get("manual")]
                peor = min(reemplazables, key=lambda p: p.get("score", 0), default=None)
                if not peor or c["score"] < peor.get("score", 0) + 15:
                    continue
                peor["estado"], peor["motivo"] = "pausado", f"reemplazado por un producto mejor ({c['id']})"
                publicados.remove(peor)
                pausados_ahora.append(peor["id"])
                publicados.append(c)
            c["estado"], c["motivo"] = "publicado", ""
            c.setdefault("publicado_en", ahora())
            publicados_ahora.append(c["id"])
            cupo_ciclo -= 1
            cupo_ciclo_nicho -= 1
            almacen.log("Buscador", "Publicado", f"{c.get('nombre')} · score {c['score']}", c["id"])
        # Histéresis: lo publicado solo baja si cae claramente bajo el mínimo.
        for p in publicados:
            if not p.get("manual") and p.get("score", 0) < cfg.score_minimo - 10 and p.get("estado") == "publicado":
                p["estado"], p["motivo"] = "pausado", f"score bajo ({p['score']})"
                pausados_ahora.append(p["id"])
    return publicados_ahora, pausados_ahora
