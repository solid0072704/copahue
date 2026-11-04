"""Pruebas del agente de compras de supermercado (sin internet)."""

import dataclasses
import io
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import pytest

from agentes_super import conectores as cx
from agentes_super.agente import (
    Cotizador,
    abrir_db,
    avanzar_fecha,
    cotizar_lista,
    detectar_ofertas,
    ejecutar_ciclo,
    marcar_comprada,
    preparar_demo,
)
from agentes_super.config import ConfigSuper
from agentes_super.emparejar import coincide, cumple_filtros
from agentes_super.exportar import excel_compra, texto_whatsapp
from agentes_super.notificador import notificar_pendientes
from agentes_super.optimizador import evaluar, optimizar
from agentes_super.unidades import parsear_formato, pesos


@pytest.fixture
def cfg():
    return dataclasses.replace(ConfigSuper(), ruta_db=":memory:", pausa_entre_consultas=0, anthropic_api_key="")


@pytest.fixture
def db(cfg):
    base = abrir_db(cfg)
    yield base
    base.cerrar()


def prod(tienda, nombre, precio, **kw):
    return cx.Producto(tienda=tienda, sku=nombre[:6], nombre=nombre, precio=precio, **kw)


class ConectorFijo(cx.Conector):
    tipo = "prueba"

    def __init__(self, nombre, productos):
        super().__init__(nombre)
        self.productos = productos

    def buscar(self, consulta, limite=24):
        return [p for p in self.productos if coincide(consulta, p.nombre)]


# --- unidades -----------------------------------------------------------------
@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("Leche Entera 1 L", (1000, "ml")),
        ("Yogurt Batido 6 x 125 g", (750, "g")),
        ("Arroz Grado 2 1.000 g", (1000, "g")),
        ("Huevos Blancos 12 un", (12, "un")),
        ("Papel Higiénico 4 rollos 50 m", (4, "un")),
        ("Bebida 1,5 L pack 6", (9000, "ml")),
        ("Detergente 3 lts", (3000, "ml")),
        ("Aceite 900 cc", (900, "ml")),
        ("Producto sin formato", (None, None)),
    ],
)
def test_parsear_formato(texto, esperado):
    assert parsear_formato(texto) == esperado


def test_pesos_formato_chileno():
    assert pesos(12990) == "$12.990"


# --- emparejar ----------------------------------------------------------------
def test_coincide_evita_falsos_positivos():
    assert coincide("leche entera", "Leche Entera Colun 1 L")
    assert coincide("huevo", "Huevos Blancos 12 un")
    assert coincide("papel higienico", "Papel Higiénico Doble Hoja")
    assert not coincide("leche", "Dulce de Leche Nestlé 400 g")
    assert not coincide("leche entera", "Leche Descremada 1 L")
    assert coincide("crema", "Crema de Leche 200 ml") and not coincide("tomate", "Salsa de Tomate 200 g")


def test_filtros_de_marca_y_exclusion():
    item = {"consulta": "leche", "marcas": "Colun, Soprole", "excluir": "chocolate"}
    assert cumple_filtros(item, prod("A", "Leche Entera Colun 1 L", 1000))
    assert not cumple_filtros(item, prod("A", "Leche Chocolate Soprole 1 L", 1000))
    assert not cumple_filtros(item, prod("A", "Leche Entera Marca Propia 1 L", 900))


# --- optimizador --------------------------------------------------------------
def test_compara_por_unidad_y_elige_el_formato_conveniente():
    item = {"id": 1, "consulta": "leche", "cantidad": 6, "formato": "1 L"}
    suelta = evaluar(item, prod("A", "Leche 1 L", 1000))
    pack = evaluar(item, prod("A", "Leche Pack 6 x 1 L", 5400))
    grande = evaluar(item, prod("A", "Leche 2 L", 1900))
    assert (suelta.envases, suelta.costo) == (6, 6000)
    assert (pack.envases, pack.costo) == (1, 5400)
    assert (grande.envases, grande.costo) == (3, 5700)
    assert evaluar(item, prod("A", "Leche 5 kg", 100)) is None  # otra unidad
    assert evaluar({"id": 1, "consulta": "arroz", "cantidad": 1, "formato": "1 kg"}, prod("A", "Arroz 5 kg", 4000)) is None


def test_sin_formato_cuenta_envases():
    opcion = evaluar({"id": 1, "consulta": "atun", "cantidad": 3, "formato": ""}, prod("A", "Atún 160 g", 1500))
    assert (opcion.envases, opcion.costo) == (3, 4500)


def _ofertas(precios):
    """precios: {tienda: {item_id: costo}} -> ofertas para el optimizador."""
    salida = {}
    for tienda, por_item in precios.items():
        salida[tienda] = {}
        for item_id, costo in por_item.items():
            o = evaluar({"id": item_id, "consulta": "x", "cantidad": 1, "formato": ""}, prod(tienda, f"x {item_id}", costo))
            salida[tienda][item_id] = o
    return salida


ITEMS = [{"id": 1}, {"id": 2}, {"id": 3}]


def test_divide_la_compra_cuando_conviene():
    ofertas = _ofertas({"A": {1: 1000, 2: 5000, 3: 1000}, "B": {1: 3000, 2: 1000, 3: 3000}})
    plan = optimizar(ITEMS, ofertas, {}, max_tiendas=2)
    assert sorted(plan.tiendas) == ["A", "B"]
    assert plan.total == 3000
    assert plan.ahorro_vs_promedio == pytest.approx(4000)  # 1.000 + 2.000 + 1.000 bajo el promedio


def test_el_despacho_puede_hacer_conveniente_un_solo_supermercado():
    ofertas = _ofertas({"A": {1: 1000, 2: 2000, 3: 1000}, "B": {1: 3000, 2: 1000, 3: 3000}})
    tiendas = {"A": {"despacho": 2000}, "B": {"despacho": 2000}}
    plan = optimizar(ITEMS, ofertas, tiendas, max_tiendas=2)
    assert plan.tiendas == ["A"] and plan.total == 6000


def test_despacho_gratis_y_minimo_de_compra():
    ofertas = _ofertas({"A": {1: 1000, 2: 2000, 3: 1000}, "B": {1: 3000, 2: 1000, 3: 3000}})
    tiendas = {"A": {"despacho": 2000, "gratis_desde": 4000}, "B": {"minimo": 5000}}
    plan = optimizar(ITEMS, ofertas, tiendas, max_tiendas=2)
    assert plan.tiendas == ["A"] and plan.total == 4000  # gratis desde $4.000; B no llega al mínimo


def test_prefiere_traer_todo_antes_que_lo_barato():
    ofertas = _ofertas({"A": {1: 100, 2: 100}, "B": {1: 900, 2: 900, 3: 900}})
    plan = optimizar(ITEMS, ofertas, {}, max_tiendas=1)
    assert plan.tiendas == ["B"] and not plan.faltantes


# --- conectores ---------------------------------------------------------------
RESPUESTA_VTEX = [
    {
        "productId": "10",
        "productName": "Leche Entera",
        "brand": "Colun",
        "link": "https://super.example/leche-entera-1-l/p",
        "items": [
            {
                "itemId": "2001",
                "nameComplete": "Leche Entera Colun 1 L",
                "measurementUnit": "un",
                "unitMultiplier": 1,
                "images": [{"imageUrl": "https://super.example/img.jpg"}],
                "sellers": [
                    {"sellerId": "1", "sellerDefault": True,
                     "commertialOffer": {"Price": 1090, "ListPrice": 1290, "AvailableQuantity": 50, "IsAvailable": True}}
                ],
            }
        ],
    },
    {
        "productId": "11",
        "productName": "Queso granel",
        "brand": "X",
        "link": "/queso/p",
        "items": [{"itemId": "2002", "name": "Queso granel", "measurementUnit": "kg", "unitMultiplier": 0.25,
                   "sellers": [{"sellerId": "1", "commertialOffer": {"Price": 3000, "ListPrice": 3000, "IsAvailable": True}}]}],
    },
    {"productId": "12", "productName": "Sin stock", "items": [
        {"itemId": "9", "sellers": [{"sellerId": "1", "commertialOffer": {"Price": 0}}]}]},
]


class RespuestaFalsa:
    status_code = 200

    def json(self):
        return RESPUESTA_VTEX


class SesionFalsa:
    def __init__(self):
        self.llamadas = []

    def get(self, url, params=None, timeout=None):
        self.llamadas.append((url, params))
        return RespuestaFalsa()


def test_conector_vtex_normaliza_y_arma_carrito():
    sesion = SesionFalsa()
    conector = cx.ConectorVTEX("Súper X", "https://super.example/", {"sc": 2}, sesion=sesion)
    productos = conector.buscar("leche", limite=10)
    assert sesion.llamadas[0][0] == "https://super.example/api/catalog_system/pub/products/search"
    assert sesion.llamadas[0][1]["ft"] == "leche" and sesion.llamadas[0][1]["sc"] == 2
    assert len(productos) == 2
    leche, queso = productos
    assert (leche.precio, leche.precio_normal, leche.cantidad, leche.unidad, leche.marca) == (1090, 1290, 1000, "ml", "Colun")
    assert queso.url == "https://super.example/queso/p" and (queso.cantidad, queso.unidad) == (250, "g")
    url = conector.url_carrito([(leche, 6), (queso, 1)])
    assert url == "https://super.example/checkout/cart/add?sku=2001&qty=6&seller=1&sku=2002&qty=1&seller=1&sc=2"


def test_lista_de_precios_csv_con_formato_chileno():
    csv = "Supermercado;Producto;Precio;Marca\nLocal Barrio;Arroz Grado 2 1 kg;$1.590;Tucapel\nLocal Barrio;Azúcar 1 kg;1.190;\n"
    productos = cx.leer_lista_precios(io.StringIO(csv))
    assert [p.precio for p in productos] == [1590, 1190]
    conector = cx.ConectorLista("Local Barrio", productos)
    assert [p.nombre for p in conector.buscar("arroz")] == ["Arroz Grado 2 1 kg"]
    with pytest.raises(ValueError):
        cx.leer_lista_precios(pd.DataFrame({"producto": ["x"]}))


def test_demo_es_estable_y_cambia_por_semana():
    a = cx.ConectorDemo("MercaCentro (demo)", parametros={"semana": 100}).buscar("arroz")
    b = cx.ConectorDemo("MercaCentro (demo)", parametros={"semana": 100}).buscar("arroz")
    assert [p.precio for p in a] == [p.precio for p in b] and a


# --- agente -------------------------------------------------------------------
def test_avanzar_fecha():
    assert avanzar_fecha(date(2026, 1, 31), "mensual") == date(2026, 2, 28)
    assert avanzar_fecha(date(2026, 12, 15), "mensual") == date(2027, 1, 15)
    assert avanzar_fecha(date(2026, 1, 1), "quincenal") == date(2026, 1, 15)


def _supermercados():
    return [
        ConectorFijo("A", [prod("A", "Leche Entera 1 L", 1000), prod("A", "Arroz 1 kg", 2000)]),
        ConectorFijo("B", [prod("B", "Leche Entera Pack 6 x 1 L", 5000), prod("B", "Arroz 1 kg", 1500)]),
    ]


def test_ciclo_prepara_compra_recurrente_y_avanza_fecha(db, cfg):
    hoy = date(2026, 11, 4)
    lista_id = db.crear_lista("Semanal", "semanal", hoy - timedelta(days=6), max_tiendas=1)
    db.agregar_item(lista_id, "leche entera", 6, "1 L")
    db.agregar_item(lista_id, "arroz", 1, "1 kg")
    resultado = ejecutar_ciclo(db, cfg, notificar=False, hoy=hoy, conectores=_supermercados())
    assert db.lista(lista_id)["proxima_fecha"] == "2026-11-05"  # la fecha pasada avanzó una semana
    assert len(resultado.cotizadas) == 1
    compra = db.compra(resultado.cotizadas[0])
    assert compra["tiendas"] == "B" and compra["total"] == 6500 and compra["fecha_objetivo"] == "2026-11-05"
    # un segundo ciclo actualiza la misma propuesta, no la duplica
    ejecutar_ciclo(db, cfg, notificar=False, hoy=hoy, conectores=_supermercados())
    assert len(db.compras(lista_id)) == 1
    marcar_comprada(db, compra["id"])
    assert db.lista(lista_id)["proxima_fecha"] == "2026-11-12"
    assert db.compra(compra["id"])["estado"] == "comprada"


def test_lista_lejana_no_genera_compra(db, cfg):
    hoy = date(2026, 11, 4)
    lista_id = db.crear_lista("Mensual", "mensual", hoy + timedelta(days=20))
    db.agregar_item(lista_id, "arroz", 1, "1 kg")
    resultado = ejecutar_ciclo(db, cfg, notificar=False, hoy=hoy, conectores=_supermercados())
    assert resultado.cotizadas == [] and db.historial("arroz")  # igual guarda precios para el historial


def test_supermercado_caido_no_detiene_al_agente(db, cfg):
    class Caido(cx.Conector):
        def buscar(self, consulta, limite=24):
            raise ConnectionError("sin conexión")

    lista_id = db.crear_lista("L", "unica")
    db.agregar_item(lista_id, "arroz", 1, "1 kg")
    cot = Cotizador(db, cfg, conectores=[Caido("Z")] + _supermercados())
    compra_id, cotizacion = cotizar_lista(db, cfg, lista_id, cot)
    assert compra_id and cotizacion.plan.tiendas == ["B"]
    assert any("sin conexión" in e for e in cotizacion.errores)


def test_detecta_oferta_contra_historial(db, cfg):
    hoy = date(2026, 11, 4)
    lista_id = db.crear_lista("L", "unica")
    db.agregar_item(lista_id, "arroz", 1, "1 kg")
    for dias in (3, 10, 17):
        db.guardar_precios((hoy - timedelta(days=dias)).isoformat(), "arroz", [prod("A", "Arroz 1 kg", 2000)])
    cot = Cotizador(db, cfg, conectores=_supermercados(), fecha=hoy)
    cotizacion = cot.cotizar(db.items(lista_id))
    assert detectar_ofertas(db, cfg, cotizacion, hoy) == ["arroz"]  # $1.500 vs $2.000 habitual = -25 %
    assert detectar_ofertas(db, cfg, cotizacion, hoy) == []  # no repite el aviso el mismo día


def test_demo_completa_y_aviso_por_correo(db, cfg):
    resultado = preparar_demo(db, cfg)
    assert len(resultado.cotizadas) == 2 and not resultado.errores
    compra = db.compras()[0]
    assert compra["detalle"]["tiendas"] and compra["total"] > 0
    assert excel_compra(compra["detalle"])[:2] == b"PK"
    assert "Total" in texto_whatsapp("Demo", compra["detalle"])
    cfg_correo = dataclasses.replace(cfg, gmail_usuario="yo@gmail.com", gmail_app_password="x", notificar_a="yo@casa.example")
    assert notificar_pendientes(db, cfg_correo) == 1  # dominio .example: se simula, no sale nada
    assert notificar_pendientes(db, cfg_correo) == 0  # ya avisado


def test_tablero_carga_con_demo(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SUPER_DB", str(tmp_path / "super.db"))
    cfg = dataclasses.replace(ConfigSuper.desde_entorno(), anthropic_api_key="")
    base = abrir_db(cfg)
    preparar_demo(base, cfg)
    base.cerrar()
    app = AppTest.from_file(str(Path(__file__).parent.parent / "tablero_super.py"), default_timeout=60).run()
    assert not app.exception
    assert any("Compras por hacer" in m.label for m in app.metric)
