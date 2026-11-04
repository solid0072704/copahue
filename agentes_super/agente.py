"""El agente de compras: cotiza tus listas, optimiza, detecta ofertas y te avisa."""

import calendar
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

from . import conectores as cx
from .config import ConfigSuper
from .db import BaseSuper
from .llm import crear_filtro
from .optimizador import comparar_tiendas, evaluar, mejores_opciones, optimizar
from .unidades import NOMBRE_BASE, pesos, precio_por_unidad, texto_formato

log = logging.getLogger(__name__)


def abrir_db(cfg=None):
    cfg = cfg or ConfigSuper.desde_entorno()
    db = BaseSuper(cfg.ruta_db)
    if not db.get_meta("tiendas_sembradas"):
        for t in cx.TIENDAS_SUGERIDAS:
            db.guardar_tienda(t["nombre"], t["tipo"], t["url"], activa=False, despacho=t["despacho"],
                              gratis_desde=t["gratis_desde"], minimo=t["minimo"])
        db.set_meta("tiendas_sembradas", "1")
    return db


def avanzar_fecha(fecha, frecuencia):
    if frecuencia == "semanal":
        return fecha + timedelta(days=7)
    if frecuencia == "quincenal":
        return fecha + timedelta(days=14)
    if frecuencia == "mensual":
        mes = fecha.month % 12 + 1
        anio = fecha.year + (fecha.month == 12)
        return date(anio, mes, min(fecha.day, calendar.monthrange(anio, mes)[1]))
    return fecha


# ---------------------------------------------------------------------------
# Cotización
# ---------------------------------------------------------------------------
@dataclass
class Cotizacion:
    items: list
    plan: object
    comparacion: list
    opciones: dict  # item_id -> [Opcion] (todas las tiendas, de la mejor a la peor)
    ofertas: dict  # tienda -> {item_id: Opcion}
    errores: list = field(default_factory=list)


class Cotizador:
    """Busca en los supermercados activos con caché por ciclo y guarda el historial de precios."""

    def __init__(self, db, cfg, conectores=None, usar_ia=True, fecha=None):
        self.db = db
        self.cfg = cfg
        self.fecha = (fecha or date.today()).isoformat()
        self.tiendas_cfg = {t["nombre"]: t for t in db.tiendas()}
        self.conectores = conectores if conectores is not None else self._conectores()
        self.filtro = crear_filtro(cfg, db) if usar_ia else None
        self.cache = {}
        self.errores = []

    def _conectores(self):
        activas = [t for t in self.db.tiendas(solo_activas=True)]
        lista = None
        if any(t["tipo"] == "csv" for t in activas):
            try:
                lista = cx.leer_lista_precios(cx.ruta_lista_precios())
            except (OSError, ValueError) as e:
                self.errores.append(f"Lista de precios: {e}")
                lista = []
        return [cx.crear_conector(t, lista) for t in activas]

    def buscar(self, conector, consulta):
        clave = (conector.nombre, consulta.lower())
        if clave not in self.cache:
            try:
                productos = conector.buscar(consulta, self.cfg.resultados_por_busqueda)
                if conector.tipo == "vtex" and self.cfg.pausa_entre_consultas:
                    time.sleep(self.cfg.pausa_entre_consultas)
                self.db.guardar_precios(self.fecha, consulta, productos, tienda=conector.nombre)
            except Exception as e:  # noqa: BLE001 - un supermercado caído no detiene al resto
                productos = []
                mensaje = f"{conector.nombre}: {e}"
                if mensaje not in self.errores:
                    self.errores.append(mensaje)
                    self.db.log("cotizador", "error", mensaje)
            self.cache[clave] = productos
        return self.cache[clave]

    def cotizar(self, items, max_tiendas=None):
        ofertas, opciones = {}, {i["id"]: [] for i in items}
        for conector in self.conectores:
            ofertas[conector.nombre] = {}
            for item in items:
                productos = self.buscar(conector, item["consulta"])
                if self.filtro and productos:
                    productos = self.filtro.filtrar(item, productos)
                mejores = mejores_opciones(item, productos)
                if mejores:
                    ofertas[conector.nombre][item["id"]] = mejores[0]
                    opciones[item["id"]] += mejores[:5]
        for lista in opciones.values():
            lista.sort(key=lambda o: (o.comparable, o.costo))
        plan = optimizar(items, ofertas, self.tiendas_cfg, max_tiendas or self.cfg.max_tiendas)
        if plan:
            por_tienda = {}
            for opcion in plan.asignacion.values():
                por_tienda.setdefault(opcion.tienda, []).append((opcion.producto, opcion.envases))
            for conector in self.conectores:
                if conector.nombre in por_tienda:
                    url = conector.url_carrito(por_tienda[conector.nombre])
                    if url:
                        plan.urls_carrito[conector.nombre] = url
        return Cotizacion(items, plan, comparar_tiendas(items, ofertas, self.tiendas_cfg), opciones, ofertas,
                          list(self.errores))


def detalle_plan(cotizacion):
    """Convierte el plan en un dict guardable (y fácil de mostrar o enviar por correo)."""
    plan = cotizacion.plan
    nombres = {i["id"]: i for i in cotizacion.items}
    detalle = {"tiendas": [], "faltantes": [], "comparacion": cotizacion.comparacion, "errores": cotizacion.errores}
    if not plan:
        detalle["faltantes"] = [i["consulta"] for i in cotizacion.items]
        return detalle
    for tienda in plan.tiendas:
        lineas = []
        for item_id, o in plan.asignacion.items():
            if o.tienda != tienda:
                continue
            p = o.producto
            item = nombres[item_id]
            alternativas = [a for a in cotizacion.opciones.get(item_id, []) if a.tienda != tienda][:1]
            lineas.append(
                {
                    "item": item["consulta"],
                    "pedido": f"{item['cantidad']:g} × {item['formato']}" if item.get("formato") else f"{item['cantidad']:g}",
                    "producto": p.nombre,
                    "marca": p.marca,
                    "envases": o.envases,
                    "precio": p.precio,
                    "precio_normal": p.precio_normal,
                    "costo": o.costo,
                    "precio_unidad": precio_por_unidad(p.precio, p.cantidad, p.unidad),
                    "unidad": NOMBRE_BASE.get(p.unidad, ""),
                    "formato": texto_formato(p.cantidad, p.unidad),
                    "url": p.url,
                    "sku": p.sku,
                    "alternativa": (
                        f"{alternativas[0].tienda}: {pesos(alternativas[0].costo)}" if alternativas else ""
                    ),
                }
            )
        detalle["tiendas"].append(
            {
                "tienda": tienda,
                "lineas": sorted(lineas, key=lambda l: l["item"]),
                "subtotal": plan.subtotales[tienda],
                "despacho": plan.despachos[tienda],
                "carrito": plan.urls_carrito.get(tienda, ""),
            }
        )
    detalle["faltantes"] = [nombres[i]["consulta"] for i in plan.faltantes]
    detalle["total"] = plan.total
    detalle["productos"] = plan.productos_total
    detalle["despacho"] = plan.despacho_total
    detalle["ahorro_vs_promedio"] = plan.ahorro_vs_promedio
    detalle["ahorro_ofertas"] = plan.ahorro_ofertas
    return detalle


def cotizar_lista(db, cfg, lista_id, cotizador=None, fecha_objetivo=None, guardar=True):
    lista = db.lista(lista_id)
    items = db.items(lista_id)
    cotizador = cotizador or Cotizador(db, cfg)
    cotizacion = cotizador.cotizar(items, lista.get("max_tiendas") or cfg.max_tiendas)
    compra_id = None
    if guardar and cotizacion.plan:
        detalle = detalle_plan(cotizacion)
        fecha = fecha_objetivo or lista["proxima_fecha"] or date.today().isoformat()
        compra_id = db.guardar_compra(lista_id, fecha, cotizacion.plan.total, cotizacion.plan.ahorro_vs_promedio,
                                      cotizacion.plan.tiendas, detalle)
        db.log("optimizador", "cotización", f"{lista['nombre']}: {pesos(cotizacion.plan.total)} en "
               f"{', '.join(cotizacion.plan.tiendas)} (ahorro {pesos(cotizacion.plan.ahorro_vs_promedio)})")
    return compra_id, cotizacion


def marcar_comprada(db, compra_id):
    compra = db.compra(compra_id)
    db.actualizar_compra(compra_id, estado="comprada")
    lista = db.lista(compra["lista_id"])
    if lista and lista["frecuencia"] != "unica" and lista["proxima_fecha"] <= compra["fecha_objetivo"]:
        siguiente = avanzar_fecha(date.fromisoformat(compra["fecha_objetivo"]), lista["frecuencia"])
        db.actualizar_lista(lista["id"], proxima_fecha=siguiente)
    db.log("usuario", "compra realizada", f"{lista['nombre'] if lista else ''}: {pesos(compra['total'])}")


# ---------------------------------------------------------------------------
# Ofertas: precio de hoy contra la mediana de los últimos 30 días
# ---------------------------------------------------------------------------
def _producto_desde_fila(f):
    return cx.Producto(tienda=f["tienda"], sku=f["sku"], nombre=f["nombre"], precio=f["precio"], marca=f["marca"] or "",
                       precio_normal=f["precio_normal"], url=f["url"] or "", cantidad=f["cantidad"], unidad=f["unidad"])


def detectar_ofertas(db, cfg, cotizacion, hoy=None):
    hoy = hoy or date.today()
    desde = (hoy - timedelta(days=30)).isoformat()
    nuevas = []
    for item in cotizacion.items:
        opciones = cotizacion.opciones.get(item["id"])
        if not opciones:
            continue
        mejor = opciones[0]
        por_dia = {}
        for fila in db.historial(item["consulta"], desde):
            if fila["fecha"] >= hoy.isoformat():
                continue
            o = evaluar(item, _producto_desde_fila(fila))
            if o and o.envases:
                por_dia[fila["fecha"]] = min(por_dia.get(fila["fecha"], float("inf")), o.comparable)
        if len(por_dia) < 3:
            continue
        mediana = statistics.median(por_dia.values())
        if mejor.comparable <= mediana * (1 - cfg.umbral_oferta):
            if db.registrar_alerta(hoy.isoformat(), item["consulta"], mejor.tienda, mejor.producto.nombre,
                                   mejor.comparable, mediana, mejor.producto.url):
                nuevas.append(item["consulta"])
                db.log("vigía de ofertas", "oferta", f"{item['consulta']}: {mejor.producto.nombre} en {mejor.tienda} "
                       f"{pesos(mejor.comparable)} vs {pesos(mediana)} habitual")
    return nuevas


# ---------------------------------------------------------------------------
# Ciclo completo
# ---------------------------------------------------------------------------
@dataclass
class ResultadoCiclo:
    cotizadas: list = field(default_factory=list)
    ofertas: list = field(default_factory=list)
    notificaciones: int = 0
    errores: list = field(default_factory=list)

    def texto(self):
        partes = [f"{len(self.cotizadas)} compras preparadas", f"{len(self.ofertas)} ofertas"]
        if self.notificaciones:
            partes.append("aviso enviado")
        if self.errores:
            partes.append(f"{len(self.errores)} errores")
        return " · ".join(partes)


def ejecutar_ciclo(db, cfg, notificar=True, hoy=None, conectores=None, usar_ia=True):
    hoy = hoy or date.today()
    resultado = ResultadoCiclo()
    cotizador = Cotizador(db, cfg, conectores=conectores, usar_ia=usar_ia, fecha=hoy)
    if not cotizador.conectores:
        resultado.errores.append("No hay supermercados activos.")
        return resultado

    for lista in db.listas(solo_activas=True):
        items = db.items(lista["id"])
        if not items:
            continue
        proxima = date.fromisoformat(lista["proxima_fecha"]) if lista["proxima_fecha"] else hoy
        if lista["frecuencia"] != "unica":
            while proxima < hoy:  # compras pasadas: se programa la siguiente
                proxima = avanzar_fecha(proxima, lista["frecuencia"])
            if proxima.isoformat() != lista["proxima_fecha"]:
                db.actualizar_lista(lista["id"], proxima_fecha=proxima)
        toca = lista["frecuencia"] != "unica" and proxima - timedelta(days=cfg.dias_anticipacion) <= hoy
        compra_id, cotizacion = cotizar_lista(db, cfg, lista["id"], cotizador, proxima.isoformat(), guardar=toca)
        if compra_id:
            resultado.cotizadas.append(compra_id)
        resultado.ofertas += detectar_ofertas(db, cfg, cotizacion, hoy)

    resultado.errores = list(cotizador.errores)
    db.set_meta("ultimo_ciclo", hoy.isoformat())
    if notificar:
        from .notificador import notificar_pendientes

        try:
            resultado.notificaciones = notificar_pendientes(db, cfg)
        except Exception as e:  # noqa: BLE001
            resultado.errores.append(f"Correo: {e}")
    return resultado


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
LISTAS_DEMO = {
    ("Despensa semanal", "semanal"): [
        ("leche entera", 6, "1 L", "", "chocolate"), ("pan", 1, "1 kg", "", ""), ("huevos", 1, "12 un", "", ""),
        ("tomate", 1, "1 kg", "", "cherry"), ("palta hass", 1, "1 kg", "", ""), ("platano", 1, "1 kg", "", ""),
        ("yogurt", 1, "750 g", "", "griego"), ("pechuga de pollo", 1, "1 kg", "", ""), ("queso", 1, "250 g", "", ""),
    ],
    ("Aseo y abarrotes del mes", "mensual"): [
        ("arroz", 2, "1 kg", "", ""), ("aceite", 2, "1 L", "", "oliva"), ("azucar", 1, "1 kg", "", ""),
        ("fideos", 4, "400 g", "", ""), ("atun", 4, "", "", ""), ("cafe", 1, "", "", ""),
        ("papel higienico", 1, "12 un", "", ""), ("detergente liquido", 1, "3 L", "", ""),
        ("lavalozas", 1, "750 ml", "", ""), ("pasta dental", 2, "90 g", "", ""),
    ],
}


def preparar_demo(db, cfg, semanas_historial=8):
    """Crea supermercados ficticios, dos listas recurrentes y 8 semanas de historial de precios."""
    for nombre, (_, despacho, gratis, minimo) in cx.TIENDAS_DEMO.items():
        db.guardar_tienda(nombre, "demo", activa=True, despacho=despacho, gratis_desde=gratis, minimo=minimo)
    existentes = {l["nombre"] for l in db.listas()}
    hoy = date.today()
    for (nombre, frecuencia), items in LISTAS_DEMO.items():
        if nombre in existentes:
            continue
        lista_id = db.crear_lista(nombre, frecuencia, hoy + timedelta(days=1 if frecuencia == "semanal" else 6))
        for consulta, cantidad, formato, marcas, excluir in items:
            db.agregar_item(lista_id, consulta, cantidad, formato, marcas, excluir)
    semana_actual = int(time.time() // (7 * 86400))
    consultas = {i["consulta"] for l in db.listas() for i in db.items(l["id"])}
    for atras in range(semanas_historial, 0, -1):
        fecha = (hoy - timedelta(days=7 * atras)).isoformat()
        for nombre in cx.TIENDAS_DEMO:
            conector = cx.ConectorDemo(nombre, parametros={"semana": semana_actual - atras})
            for consulta in consultas:
                db.guardar_precios(fecha, consulta, conector.buscar(consulta), tienda=nombre)
    db.log("demo", "datos de demostración", "4 supermercados ficticios, 2 listas y 8 semanas de precios")
    resultado = ejecutar_ciclo(db, cfg, notificar=False, usar_ia=False)
    # En la demo también se deja lista la compra mensual, para verla de inmediato.
    cotizador = Cotizador(db, cfg, usar_ia=False)
    for lista in db.listas(solo_activas=True):
        if not db.compras(lista["id"]):
            compra_id, _ = cotizar_lista(db, cfg, lista["id"], cotizador)
            if compra_id:
                resultado.cotizadas.append(compra_id)
    return resultado
