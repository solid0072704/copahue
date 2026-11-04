"""Conectores a supermercados: buscan productos y devuelven precios normalizados.

- ``vtex``: tiendas en línea hechas con VTEX (API pública de catálogo). Además permite
  armar el carrito por URL, así el agente te deja la compra lista para pagar.
- ``csv``: una lista de precios propia (archivo CSV/Excel con tienda, producto y precio).
- ``demo``: supermercados ficticios para probar sin internet.

Agregar otro supermercado = escribir una clase con ``buscar(consulta, limite)``.
"""

import hashlib
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from urllib.parse import urlencode

from .unidades import normalizar, parsear_formato, precio_por_unidad

log = logging.getLogger(__name__)
AGENTE_HTTP = "Mozilla/5.0 (compatible; CopahueCompras/1.0; comparador de precios personal)"


@dataclass
class Producto:
    tienda: str
    sku: str
    nombre: str
    precio: float
    marca: str = ""
    precio_normal: float = 0.0
    url: str = ""
    imagen: str = ""
    vendedor: str = "1"
    disponible: bool = True
    cantidad: float = None  # en unidad base (g, ml, un)
    unidad: str = None

    def __post_init__(self):
        if self.cantidad is None:
            self.cantidad, self.unidad = parsear_formato(self.nombre)
        if not self.precio_normal or self.precio_normal < self.precio:
            self.precio_normal = self.precio

    @property
    def precio_unidad(self):
        return precio_por_unidad(self.precio, self.cantidad, self.unidad)

    @property
    def descuento(self):
        return 1 - self.precio / self.precio_normal if self.precio_normal else 0

    def dict(self):
        return asdict(self)


class Conector:
    tipo = "base"
    arma_carrito = False

    def __init__(self, nombre, url="", parametros=None):
        self.nombre = nombre
        self.url = (url or "").rstrip("/")
        self.parametros = parametros or {}

    def buscar(self, consulta, limite=24):  # pragma: no cover - interfaz
        raise NotImplementedError

    def url_carrito(self, lineas):
        """lineas: [(Producto, envases)] -> URL que agrega todo al carrito, o None."""
        return None


# ---------------------------------------------------------------------------
# VTEX
# ---------------------------------------------------------------------------
class ConectorVTEX(Conector):
    tipo = "vtex"
    arma_carrito = True

    def __init__(self, nombre, url, parametros=None, sesion=None):
        super().__init__(nombre, url, parametros)
        self.sesion = sesion

    def _http(self):
        if self.sesion is None:
            import requests

            self.sesion = requests.Session()
            self.sesion.headers.update({"User-Agent": AGENTE_HTTP, "Accept": "application/json"})
        return self.sesion

    def buscar(self, consulta, limite=24):
        params = {"ft": consulta, "_from": 0, "_to": max(0, min(limite, 50) - 1)}
        if self.parametros.get("sc"):
            params["sc"] = self.parametros["sc"]
        respuesta = self._http().get(f"{self.url}/api/catalog_system/pub/products/search", params=params, timeout=20)
        if respuesta.status_code not in (200, 206):
            raise RuntimeError(f"{self.nombre} respondió {respuesta.status_code}")
        return self.parsear(respuesta.json())

    def parsear(self, datos):
        productos = []
        for p in datos or []:
            for item in (p.get("items") or [])[:1]:
                vendedor = next(
                    (s for s in item.get("sellers") or [] if s.get("sellerDefault")), (item.get("sellers") or [None])[0]
                )
                if not vendedor:
                    continue
                oferta = vendedor.get("commertialOffer") or {}
                precio = oferta.get("Price") or 0
                if precio <= 0:
                    continue
                nombre = item.get("nameComplete") or p.get("productName") or item.get("name") or ""
                cantidad, unidad = parsear_formato(nombre)
                if cantidad is None:
                    cantidad, unidad = _formato_vtex(item)
                link = p.get("link") or ""
                if link.startswith("/"):
                    link = self.url + link
                imagenes = item.get("images") or []
                productos.append(
                    Producto(
                        tienda=self.nombre,
                        sku=str(item.get("itemId") or p.get("productId")),
                        nombre=nombre,
                        marca=p.get("brand") or "",
                        precio=float(precio),
                        precio_normal=float(oferta.get("ListPrice") or precio),
                        url=link,
                        imagen=imagenes[0].get("imageUrl", "") if imagenes else "",
                        vendedor=str(vendedor.get("sellerId") or "1"),
                        disponible=bool(oferta.get("IsAvailable", (oferta.get("AvailableQuantity") or 0) > 0)),
                        cantidad=cantidad,
                        unidad=unidad,
                    )
                )
        return productos

    def url_carrito(self, lineas):
        if not lineas:
            return None
        pares = []
        for producto, envases in lineas:
            pares += [("sku", producto.sku), ("qty", int(envases)), ("seller", producto.vendedor or "1")]
        pares.append(("sc", self.parametros.get("sc", 1)))
        return f"{self.url}/checkout/cart/add?{urlencode(pares)}"


def _formato_vtex(item):
    unidad = (item.get("measurementUnit") or "").lower()
    multiplicador = item.get("unitMultiplier") or 0
    if unidad in ("kg", "g", "l", "ml", "un") and multiplicador and unidad != "un":
        from .unidades import UNIDADES

        base, factor = UNIDADES[unidad]
        return float(multiplicador) * factor, base
    return None, None


# ---------------------------------------------------------------------------
# Lista de precios propia (CSV / Excel)
# ---------------------------------------------------------------------------
COLUMNAS_CSV = {
    "tienda": ("tienda", "supermercado", "local"),
    "nombre": ("producto", "nombre", "descripcion", "descripción"),
    "precio": ("precio", "precio_oferta", "valor"),
    "precio_normal": ("precio_normal", "precio_lista", "normal"),
    "marca": ("marca",),
    "url": ("url", "link", "enlace"),
    "sku": ("sku", "codigo", "código", "id"),
}


def leer_lista_precios(origen):
    """Lee un CSV/Excel y devuelve [Producto]. ``origen``: ruta, archivo subido o DataFrame."""
    import pandas as pd

    if isinstance(origen, pd.DataFrame):
        df = origen
    else:
        nombre = getattr(origen, "name", str(origen)).lower()
        df = pd.read_excel(origen) if nombre.endswith((".xlsx", ".xls")) else pd.read_csv(origen, sep=None, engine="python")
    columnas = {normalizar(c): c for c in df.columns}
    mapa = {}
    for campo, alias in COLUMNAS_CSV.items():
        for a in alias:
            if normalizar(a) in columnas:
                mapa[campo] = columnas[normalizar(a)]
                break
    faltan = {"tienda", "nombre", "precio"} - set(mapa)
    if faltan:
        raise ValueError(f"Faltan columnas: {', '.join(sorted(faltan))} (se esperan tienda, producto, precio)")
    productos = []
    for i, fila in df.iterrows():
        precio = _a_numero(fila[mapa["precio"]])
        if not precio:
            continue
        valor = lambda campo: "" if campo not in mapa or pd.isna(fila[mapa[campo]]) else str(fila[mapa[campo]]).strip()  # noqa: E731
        productos.append(
            Producto(
                tienda=valor("tienda"),
                sku=valor("sku") or f"csv-{i}",
                nombre=valor("nombre"),
                marca=valor("marca"),
                precio=precio,
                precio_normal=_a_numero(valor("precio_normal")) or precio,
                url=valor("url"),
            )
        )
    return productos


def _a_numero(valor):
    if valor is None:
        return 0.0
    if isinstance(valor, (int, float)):
        return 0.0 if math.isnan(valor) else float(valor)
    texto = "".join(c for c in str(valor) if c.isdigit() or c in ",.")
    if not texto:
        return 0.0
    # Formato chileno: $1.990 o 1.990,50
    if "," in texto:
        texto = texto.replace(".", "").replace(",", ".")
    elif texto.count(".") >= 1 and len(texto.rsplit(".", 1)[1]) == 3:
        texto = texto.replace(".", "")
    try:
        return float(texto)
    except ValueError:
        return 0.0


class ConectorLista(Conector):
    """Busca dentro de una lista de precios ya cargada (CSV/Excel o productos guardados)."""

    tipo = "csv"

    def __init__(self, nombre, productos, parametros=None):
        super().__init__(nombre, "", parametros)
        self.productos = [p for p in productos if p.tienda == nombre]

    def buscar(self, consulta, limite=24):
        from .emparejar import coincide

        return [p for p in self.productos if coincide(consulta, p.nombre)][:limite]


# ---------------------------------------------------------------------------
# Demo (supermercados ficticios, precios inventados y estables)
# ---------------------------------------------------------------------------
TIENDAS_DEMO = {
    # nombre: (nivel de precios, despacho, despacho gratis desde, mínimo de compra)
    "Súper Ahorro (demo)": (0.95, 2990, 50000, 15000),
    "MercaCentro (demo)": (1.00, 1990, 40000, 10000),
    "Almacén Express (demo)": (1.08, 990, 25000, 5000),
    "FreshMarket (demo)": (1.04, 3490, 60000, 20000),
}
# producto base: [(variante, precio de referencia CLP)]
CATALOGO_DEMO = {
    "leche": [("Leche Entera Colun 1 L", 1190), ("Leche Semidescremada Soprole 1 L", 1150),
              ("Leche Entera Pack 6 x 1 L Marca Propia", 5990), ("Leche Descremada Loncoleche 1 L", 1090),
              ("Leche Chocolate Soprole 1 L", 1490)],
    "pan": [("Pan de Molde Blanco Ideal 750 g", 2690), ("Pan Hallulla 1 kg", 2290), ("Pan Marraqueta 1 kg", 2390)],
    "arroz": [("Arroz Grado 2 Tucapel 1 kg", 1590), ("Arroz Grado 1 Miraflores 1 kg", 1890),
              ("Arroz Grado 2 Marca Propia 5 kg", 6490)],
    "huevos": [("Huevos Blancos Grandes 12 un", 3390), ("Huevos Color Extra 30 un", 7690), ("Huevos de Campo 6 un", 2290)],
    "aceite": [("Aceite Maravilla Belmont 1 L", 2590), ("Aceite Vegetal Chef 900 ml", 2190),
               ("Aceite de Oliva Extra Virgen 500 ml", 5990)],
    "azucar": [("Azúcar Granulada Iansa 1 kg", 1390), ("Azúcar Marca Propia 1 kg", 1150)],
    "fideos": [("Fideos Spaghetti 5 Carozzi 400 g", 990), ("Fideos Corbatas Lucchetti 400 g", 1090),
               ("Fideos Tallarines Marca Propia 1 kg", 1790)],
    "atun": [("Atún Lomitos en Agua Van Camps 160 g", 1990), ("Atún Desmenuzado en Aceite 170 g", 1290)],
    "cafe": [("Café Instantáneo Nescafé Tradición 170 g", 6990), ("Café Grano Molido 250 g", 4990)],
    "yogurt": [("Yogurt Batido Frutilla 6 x 125 g", 2490), ("Yogurt Natural Griego 1 kg", 4290)],
    "queso": [("Queso Gauda Laminado 250 g", 3290), ("Queso Mantecoso Granulado 500 g", 5590)],
    "pollo": [("Pechuga de Pollo Deshuesada 1 kg", 6990), ("Trutro Entero de Pollo 1 kg", 3290)],
    "carne molida": [("Carne Molida 10% Grasa 500 g", 4690), ("Carne Molida 4% Grasa 500 g", 5490)],
    "tomate": [("Tomate Larga Vida 1 kg", 1990), ("Tomate Cherry 250 g", 1690)],
    "palta": [("Palta Hass 1 kg", 4990), ("Palta Hass Malla 500 g", 2790)],
    "platano": [("Plátano 1 kg", 1390)],
    "manzana": [("Manzana Roja 1 kg", 1590), ("Manzana Verde 1 kg", 1690)],
    "papa": [("Papa Lavada Malla 2 kg", 2490), ("Papa 1 kg", 1290)],
    "cebolla": [("Cebolla Malla 1 kg", 1190)],
    "papel higienico": [("Papel Higiénico Doble Hoja 12 rollos", 6990), ("Papel Higiénico Elite 4 rollos", 2590),
                        ("Papel Higiénico Noble 24 rollos", 11990)],
    "detergente": [("Detergente Líquido Omo 3 L", 9990), ("Detergente en Polvo Ariel 2,7 kg", 10990),
                   ("Detergente Líquido Marca Propia 3 L", 6490)],
    "lavalozas": [("Lavalozas Quix Limón 750 ml", 1990), ("Lavalozas Marca Propia 1 L", 1490)],
    "cloro": [("Cloro Gel Clorox 900 ml", 1690), ("Cloro Tradicional 2 L", 1990)],
    "shampoo": [("Shampoo Sedal 340 ml", 2990), ("Shampoo Head & Shoulders 375 ml", 5490)],
    "pasta dental": [("Pasta Dental Colgate Total 90 g", 2690), ("Pasta Dental Pack 3 x 90 g", 5990)],
    "bebida": [("Bebida Coca-Cola Original 3 L", 3290), ("Bebida Sprite 1,5 L", 1790), ("Agua Mineral Sin Gas 1,6 L", 990)],
    "mantequilla": [("Mantequilla con Sal Colun 250 g", 2690), ("Margarina Pote 500 g", 1890)],
    "harina": [("Harina Sin Polvos de Hornear 1 kg", 1190), ("Harina Con Polvos de Hornear 1 kg", 1290)],
    "porotos": [("Porotos Hallados 1 kg", 2990), ("Porotos Negros Conserva 380 g", 1290)],
    "pañales": [("Pañales Babysec Talla G 60 un", 15990), ("Pañales Huggies Talla G 52 un", 17990)],
}


def _ruido(*partes):
    h = hashlib.sha256("|".join(map(str, partes)).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def precio_demo(tienda, nombre, referencia, semana=0):
    """Precio ficticio estable: depende de la tienda, del producto y de la semana (ofertas que rotan)."""
    nivel = TIENDAS_DEMO[tienda][0]
    variacion = 0.88 + 0.24 * _ruido(tienda, nombre)
    normal = referencia * nivel * variacion
    oferta = _ruido(tienda, nombre, "oferta", semana) < 0.18
    precio = normal * (0.72 + 0.12 * _ruido(tienda, nombre, semana)) if oferta else normal
    redondear = lambda v: max(190, int(round(v / 10.0)) * 10 - 10)  # noqa: E731  ($1.990 al estilo chileno)
    return redondear(precio), redondear(normal)


class ConectorDemo(Conector):
    tipo = "demo"
    arma_carrito = False

    def __init__(self, nombre, url="", parametros=None):
        super().__init__(nombre, url, parametros)
        self.semana = int(self.parametros.get("semana", int(time.time() // (7 * 86400))))

    def buscar(self, consulta, limite=24):
        from .emparejar import coincide

        resultados = []
        for base, variantes in CATALOGO_DEMO.items():
            for nombre, referencia in variantes:
                if not coincide(consulta, nombre):
                    continue
                if _ruido(self.nombre, nombre, "surtido") < 0.12:  # no todas las tiendas tienen todo
                    continue
                precio, normal = precio_demo(self.nombre, nombre, referencia, self.semana)
                sku = hashlib.md5(f"{self.nombre}{nombre}".encode()).hexdigest()[:8]
                resultados.append(
                    Producto(tienda=self.nombre, sku=sku, nombre=nombre, marca=_marca_demo(nombre),
                             precio=precio, precio_normal=normal, url="")
                )
        return resultados[:limite]


def _marca_demo(nombre):
    for marca in ("Colun", "Soprole", "Loncoleche", "Tucapel", "Iansa", "Carozzi", "Lucchetti", "Nescafé", "Omo",
                  "Ariel", "Quix", "Clorox", "Sedal", "Colgate", "Coca-Cola", "Sprite", "Elite", "Noble",
                  "Babysec", "Huggies", "Belmont", "Chef", "Van Camps", "Ideal", "Miraflores"):
        if marca.lower() in nombre.lower():
            return marca
    return "Marca Propia" if "Marca Propia" in nombre else ""


# ---------------------------------------------------------------------------
# Fábrica
# ---------------------------------------------------------------------------
# Sugerencias que vienen desactivadas: actívalas y pruébalas con "Probar conexión".
# Los supermercados cambian sus sitios seguido; si uno deja de responder, desactívalo
# o carga sus precios por CSV.
TIENDAS_SUGERIDAS = [
    {"nombre": "Jumbo", "tipo": "vtex", "url": "https://www.jumbo.cl", "despacho": 3990, "gratis_desde": 0, "minimo": 0},
    {"nombre": "Santa Isabel", "tipo": "vtex", "url": "https://www.santaisabel.cl", "despacho": 3490, "gratis_desde": 0, "minimo": 0},
    {"nombre": "Unimarc", "tipo": "vtex", "url": "https://www.unimarc.cl", "despacho": 2990, "gratis_desde": 0, "minimo": 0},
    {"nombre": "Alvi", "tipo": "vtex", "url": "https://www.alvi.cl", "despacho": 2990, "gratis_desde": 0, "minimo": 0},
]


def crear_conector(tienda, productos_lista=None):
    """``tienda``: fila de la tabla ``tiendas`` (dict)."""
    tipo = tienda["tipo"]
    parametros = tienda.get("parametros") or {}
    if tipo == "vtex":
        return ConectorVTEX(tienda["nombre"], tienda["url"], parametros)
    if tipo == "demo":
        return ConectorDemo(tienda["nombre"], parametros=parametros)
    if tipo == "csv":
        return ConectorLista(tienda["nombre"], productos_lista or [], parametros)
    raise ValueError(f"Tipo de supermercado desconocido: {tipo}")


def probar_conector(conector, consulta="leche"):
    """Devuelve (ok, mensaje) para mostrar en el tablero."""
    inicio = time.time()
    try:
        productos = conector.buscar(consulta, limite=5)
    except Exception as e:  # noqa: BLE001 - se informa al usuario
        return False, f"No respondió: {e}"
    if not productos:
        return False, f"Respondió pero no encontró «{consulta}». Revisa la URL."
    p = productos[0]
    return True, f"OK en {time.time() - inicio:.1f} s · ej.: {p.nombre} a ${p.precio:,.0f}".replace(",", ".")


def ruta_lista_precios():
    return os.environ.get("SUPER_LISTA_PRECIOS", "data/lista_precios.csv")
