"""Importa productos desde un CSV o Excel exportado de cualquier proveedor o herramienta
(Dropi, AutoDS, Zendrop, CJ, una planilla propia...). Las columnas se reconocen por nombre."""

import hashlib
import io
import re

import pandas as pd

from ..texto import a_float, normalizar

ALIAS = {
    "id_fuente": ("id", "sku", "product_id", "id_producto", "codigo", "código"),
    "titulo_original": ("titulo", "título", "title", "nombre", "name", "producto", "product_name"),
    "costo_usd": ("costo", "cost", "precio", "price", "precio_proveedor", "costo_unitario", "sell_price"),
    "envio_usd": ("envio", "envío", "shipping", "costo_envio", "shipping_cost"),
    "imagenes": ("imagen", "imagenes", "imágenes", "image", "images", "image_url", "foto", "fotos"),
    "url_proveedor": ("url", "link", "enlace", "product_url", "url_proveedor"),
    "url_afiliado": ("url_afiliado", "affiliate_link", "link_afiliado", "promotion_link"),
    "pedidos": ("pedidos", "orders", "ventas", "sold", "vendidos"),
    "valoracion": ("valoracion", "valoración", "rating", "calificacion", "calificación"),
    "comision": ("comision", "comisión", "commission", "commission_rate"),
    "dias_envio": ("dias_envio", "días_envío", "shipping_days", "delivery_days", "dias"),
    "categoria_fuente": ("categoria", "categoría", "category"),
    "nicho": ("nicho", "niche", "coleccion", "colección"),
}


def _columna(df, campo):
    nombres = {normalizar(c).replace(" ", "_"): c for c in df.columns}
    for alias in (campo,) + ALIAS[campo]:
        clave = normalizar(alias).replace(" ", "_")
        if clave in nombres:
            return nombres[clave]
    return None


def leer(contenido, nombre_archivo="productos.csv"):
    datos = io.BytesIO(contenido) if isinstance(contenido, bytes) else contenido
    if str(nombre_archivo).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(datos, dtype=str)
    return pd.read_csv(datos, dtype=str, sep=None, engine="python")


def productos_desde_tabla(df, moneda="USD", tipo_cambio=950.0, nicho=None):
    """Convierte cada fila en un producto normalizado. moneda='CLP' si los costos vienen en pesos."""
    columnas = {campo: _columna(df, campo) for campo in ALIAS}
    salida = []
    for _, fila in df.iterrows():
        def val(campo):
            col = columnas.get(campo)
            v = fila.get(col) if col else None
            return None if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "" else str(v).strip()

        titulo = val("titulo_original")
        costo = a_float(val("costo_usd"))
        if not titulo or not costo:
            continue
        envio = a_float(val("envio_usd"))
        if moneda.upper() == "CLP":
            costo = costo / tipo_cambio
            envio = envio / tipo_cambio if envio else envio
        valoracion = a_float(val("valoracion"))
        if valoracion is not None and valoracion <= 5:
            valoracion = valoracion * 20  # 4,8 estrellas -> 96 %
        imagenes = [i.strip() for i in re.split(r"[|;\n ]+|,(?=\s*https?:)", val("imagenes") or "") if i.strip().startswith("http")]
        pid = val("id_fuente") or hashlib.sha1(titulo.encode("utf-8")).hexdigest()[:12]
        salida.append(
            {
                "fuente": "archivo",
                "id_fuente": re.sub(r"[^A-Za-z0-9_-]+", "", pid)[:40] or hashlib.sha1(pid.encode()).hexdigest()[:12],
                "titulo_original": titulo,
                "categoria_fuente": val("categoria_fuente") or "",
                "imagenes": imagenes[:8],
                "video": None,
                "costo_usd": round(costo, 2),
                "precio_lista_usd": None,
                "envio_usd": round(envio, 2) if envio else None,
                "pedidos": int(a_float(val("pedidos"), 0) or 0) or None,
                "valoracion": valoracion,
                "comision": a_float(val("comision")),
                "dias_envio": int(a_float(val("dias_envio"), 0) or 0) or None,
                "url_proveedor": val("url_proveedor") or "",
                "url_afiliado": val("url_afiliado"),
                "nicho": val("nicho") or nicho,
                "busqueda": "",
            }
        )
    return salida
