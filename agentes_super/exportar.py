"""Exporta una compra a Excel o a texto para WhatsApp."""

import io

import pandas as pd

from .unidades import pesos


def tabla_compra(detalle):
    filas = []
    for t in detalle.get("tiendas", []):
        for l in t["lineas"]:
            filas.append(
                {
                    "Supermercado": t["tienda"],
                    "Ítem": l["item"],
                    "Producto": l["producto"],
                    "Marca": l["marca"],
                    "Envases": l["envases"],
                    "Precio": l["precio"],
                    "Total": l["costo"],
                    "Precio por unidad": round(l["precio_unidad"]) if l.get("precio_unidad") else None,
                    "Unidad": l.get("unidad", ""),
                    "Enlace": l["url"],
                }
            )
    return pd.DataFrame(filas)


def excel_compra(detalle):
    salida = io.BytesIO()
    with pd.ExcelWriter(salida, engine="xlsxwriter") as xl:
        tabla_compra(detalle).to_excel(xl, sheet_name="Compra", index=False)
        pd.DataFrame(detalle.get("comparacion", [])).to_excel(xl, sheet_name="Comparación", index=False)
    return salida.getvalue()


def texto_whatsapp(nombre, detalle):
    lineas = [f"*🛒 {nombre}*"]
    for t in detalle.get("tiendas", []):
        lineas.append(f"\n*{t['tienda']}* ({pesos(t['subtotal'])})")
        lineas += [f"▫️ {l['envases']} × {l['producto']} — {pesos(l['costo'])}" for l in t["lineas"]]
    lineas.append(f"\n*Total: {pesos(detalle.get('total'))}*")
    return "\n".join(lineas)
