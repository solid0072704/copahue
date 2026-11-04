"""El optimizador: elige qué producto comprar en cada supermercado y cómo repartir la compra.

1. Para cada ítem y supermercado busca la opción más barata que cubra la cantidad pedida,
   comparando por precio por kg/L/unidad cuando indicas un formato (6 × 1 L de leche puede
   salir más barato en una caja de 6 o en 3 botellas de 2 L).
2. Prueba todas las combinaciones de hasta ``max_tiendas`` supermercados, suma despacho y
   respeta montos mínimos de compra, y se queda con la más barata que trae más ítems.
"""

import itertools
import math
from dataclasses import dataclass, field

from .emparejar import candidatos
from .unidades import parsear_formato

TOLERANCIA_FALTA = 0.9  # acepta llevar hasta 10 % menos de lo pedido (ej. 900 ml en vez de 1 L)
TOLERANCIA_EXCESO = 1.5  # y hasta 50 % más (ej. caja de 6 cuando pediste 4)


@dataclass
class Opcion:
    item_id: int
    tienda: str
    producto: object
    envases: int
    costo: float  # lo que pagas
    comparable: float  # costo llevado exactamente a la cantidad pedida (para comparar)
    cantidad_total: float = None
    unidad: str = None

    @property
    def ahorro_oferta(self):
        return (self.producto.precio_normal - self.producto.precio) * self.envases


def necesidad(item):
    """Cantidad pedida en unidad base, o (None, None) si se cuenta por envases."""
    if item.get("formato"):
        cantidad, unidad = parsear_formato(item["formato"])
        if cantidad:
            return cantidad * float(item.get("cantidad") or 1), unidad
    return None, None


def evaluar(item, producto):
    """Devuelve una Opcion o None si el producto no sirve para este ítem."""
    total, unidad = necesidad(item)
    if total is None:
        envases = max(1, int(math.ceil(float(item.get("cantidad") or 1))))
        costo = producto.precio * envases
        return Opcion(item["id"], producto.tienda, producto, envases, costo, costo)
    if producto.unidad != unidad or not producto.cantidad:
        return None
    envases = math.floor(total / producto.cantidad + 1e-9)
    if envases < 1 or envases * producto.cantidad < total * TOLERANCIA_FALTA:
        envases = math.ceil(total / producto.cantidad - 1e-9)
    comprado = envases * producto.cantidad
    if comprado > total * TOLERANCIA_EXCESO or comprado < total * TOLERANCIA_FALTA:
        return None
    costo = producto.precio * envases
    return Opcion(item["id"], producto.tienda, producto, envases, costo, costo / comprado * total, comprado, unidad)


def mejores_opciones(item, productos):
    """Opciones válidas de un supermercado, de la más conveniente a la menos."""
    opciones = [o for o in (evaluar(item, p) for p in candidatos(item, productos)) if o]
    return sorted(opciones, key=lambda o: (o.comparable, o.costo))


@dataclass
class Plan:
    tiendas: list
    asignacion: dict  # item_id -> Opcion
    subtotales: dict
    despachos: dict
    faltantes: list
    total: float = 0
    ahorro_vs_promedio: float = 0
    ahorro_ofertas: float = 0
    urls_carrito: dict = field(default_factory=dict)

    @property
    def productos_total(self):
        return sum(self.subtotales.values())

    @property
    def despacho_total(self):
        return sum(self.despachos.values())


def costo_despacho(tienda_cfg, subtotal):
    if not tienda_cfg:
        return 0
    gratis = tienda_cfg.get("gratis_desde") or 0
    if gratis and subtotal >= gratis:
        return 0
    return tienda_cfg.get("despacho") or 0


def _evaluar_combinacion(combo, items, ofertas, tiendas_cfg, con_despacho):
    asignacion, faltantes = {}, []
    for item in items:
        mejor = None
        for tienda in combo:
            opcion = ofertas.get(tienda, {}).get(item["id"])
            if opcion and (mejor is None or opcion.comparable < mejor.comparable):
                mejor = opcion
        if mejor:
            asignacion[item["id"]] = mejor
        else:
            faltantes.append(item["id"])
    subtotales = {t: 0.0 for t in combo}
    for opcion in asignacion.values():
        subtotales[opcion.tienda] += opcion.costo
    if any(v == 0 for v in subtotales.values()):
        return None  # una tienda sin productos: ya se evalúa la combinación más chica
    for tienda, subtotal in subtotales.items():
        minimo = (tiendas_cfg.get(tienda) or {}).get("minimo") or 0
        if con_despacho and subtotal < minimo:
            return None
    despachos = {t: costo_despacho(tiendas_cfg.get(t), s) if con_despacho else 0 for t, s in subtotales.items()}
    plan = Plan(list(combo), asignacion, subtotales, despachos, faltantes)
    plan.total = plan.productos_total + plan.despacho_total
    return plan


def optimizar(items, ofertas, tiendas_cfg=None, max_tiendas=2, con_despacho=True):
    """``ofertas``: {tienda: {item_id: Opcion}}. Devuelve el mejor Plan (o None).

    Criterio: primero que falten menos productos, después el menor total (productos + despacho).
    """
    tiendas_cfg = tiendas_cfg or {}
    tiendas = sorted(t for t, o in ofertas.items() if o)
    mejor = None
    for n in range(1, min(max_tiendas, len(tiendas)) + 1):
        for combo in itertools.combinations(tiendas, n):
            plan = _evaluar_combinacion(combo, items, ofertas, tiendas_cfg, con_despacho)
            if plan and (mejor is None or (len(plan.faltantes), plan.total) < (len(mejor.faltantes), mejor.total)):
                mejor = plan
    if mejor:
        _ahorros(mejor, ofertas)
    return mejor


def _ahorros(plan, ofertas):
    ahorro = 0.0
    for item_id, elegida in plan.asignacion.items():
        precios = [o[item_id].comparable for o in ofertas.values() if item_id in o]
        if len(precios) > 1:
            ahorro += sum(precios) / len(precios) - elegida.comparable
    plan.ahorro_vs_promedio = max(0.0, ahorro)
    plan.ahorro_ofertas = sum(o.ahorro_oferta for o in plan.asignacion.values())


def comparar_tiendas(items, ofertas, tiendas_cfg=None):
    """Cuánto costaría comprar todo en un solo supermercado (para la tabla comparativa)."""
    filas = []
    for tienda in sorted(ofertas):
        plan = _evaluar_combinacion((tienda,), items, ofertas, tiendas_cfg or {}, con_despacho=False)
        if not plan:
            continue
        despacho = costo_despacho((tiendas_cfg or {}).get(tienda), plan.productos_total)
        filas.append(
            {
                "tienda": tienda,
                "encontrados": len(plan.asignacion),
                "faltan": len(plan.faltantes),
                "productos": plan.productos_total,
                "despacho": despacho,
                "total": plan.productos_total + despacho,
            }
        )
    return sorted(filas, key=lambda f: (f["faltan"], f["total"]))
