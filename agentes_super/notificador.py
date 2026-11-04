"""Avisos por correo: compra recurrente lista para revisar y ofertas para abastecerse."""

from html import escape

from agentes_ebook.gmail import enviar_correo

from .unidades import pesos


def texto_compra(lista, compra):
    d = compra["detalle"]
    lineas = [f"🛒 {lista['nombre']} — compra del {compra['fecha_objetivo']}", ""]
    for t in d.get("tiendas", []):
        lineas.append(f"== {t['tienda']} · {pesos(t['subtotal'])} + despacho {pesos(t['despacho'])}")
        for l in t["lineas"]:
            lineas.append(f"  [ ] {l['envases']} × {l['producto']} — {pesos(l['costo'])}")
        if t.get("carrito"):
            lineas.append(f"  Carrito listo: {t['carrito']}")
        lineas.append("")
    if d.get("faltantes"):
        lineas.append("No encontrados: " + ", ".join(d["faltantes"]))
    lineas.append(f"TOTAL {pesos(compra['total'])} · ahorro vs. precio promedio {pesos(compra['ahorro'])}")
    return "\n".join(lineas)


def _html_compra(lista, compra):
    d = compra["detalle"]
    partes = [f"<h2 style='margin:0 0 4px'>🛒 {escape(lista['nombre'])}</h2>",
              f"<p style='color:#666;margin:0 0 12px'>Compra del {compra['fecha_objetivo']} · "
              f"total <b>{pesos(compra['total'])}</b> · ahorras <b>{pesos(compra['ahorro'])}</b> vs. el precio promedio</p>"]
    for t in d.get("tiendas", []):
        filas = "".join(
            f"<tr><td style='padding:4px 8px'>{l['envases']} ×</td>"
            f"<td style='padding:4px 8px'>{'<a href=' + chr(34) + escape(l['url']) + chr(34) + '>' if l['url'] else ''}"
            f"{escape(l['producto'])}{'</a>' if l['url'] else ''}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{pesos(l['costo'])}</td></tr>"
            for l in t["lineas"]
        )
        boton = (
            f"<p><a href='{escape(t['carrito'])}' style='background:#2a78d6;color:#fff;padding:8px 14px;"
            f"border-radius:6px;text-decoration:none'>Abrir carrito en {escape(t['tienda'])}</a></p>"
            if t.get("carrito") else ""
        )
        partes.append(
            f"<h3 style='margin:16px 0 4px'>{escape(t['tienda'])} · {pesos(t['subtotal'])}"
            f" <span style='color:#666;font-weight:normal'>+ despacho {pesos(t['despacho'])}</span></h3>"
            f"<table style='border-collapse:collapse;font-size:14px'>{filas}</table>{boton}"
        )
    if d.get("faltantes"):
        partes.append(f"<p>⚠️ No encontrados: {escape(', '.join(d['faltantes']))}</p>")
    return "".join(partes)


def _html_ofertas(alertas):
    filas = "".join(
        f"<li><b>{escape(a['consulta'])}</b>: {escape(a['nombre'])} en {escape(a['tienda'])} a "
        f"{pesos(a['precio_unidad'])} (lo habitual es {pesos(a['mediana'])}, "
        f"{round(100 * (1 - a['precio_unidad'] / a['mediana']))} % menos)</li>"
        for a in alertas
    )
    return f"<h2>🔥 Buen momento para abastecerse</h2><ul>{filas}</ul>"


def notificar_pendientes(db, cfg):
    """Envía un correo con las compras nuevas y las ofertas. Devuelve 1 si envió, 0 si no."""
    if not cfg.gmail_configurado or not cfg.notificar_a:
        return 0
    compras = [c for c in db.compras(estados=["propuesta"]) if not c["notificada"]]
    alertas = db.alertas(solo_pendientes=True)
    if not compras and not alertas:
        return 0
    html, texto = [], []
    for c in compras:
        lista = db.lista(c["lista_id"])
        html.append(_html_compra(lista, c))
        texto.append(texto_compra(lista, c))
    if alertas:
        html.append(_html_ofertas(alertas))
        texto.append("Ofertas: " + "; ".join(f"{a['consulta']} en {a['tienda']}" for a in alertas))
    asunto = (
        f"🛒 Tu compra está lista: {pesos(sum(c['total'] for c in compras))}" if compras
        else f"🔥 {len(alertas)} ofertas para abastecerte"
    )
    cuerpo = "<div style='font-family:Arial,sans-serif;max-width:640px'>" + "<hr>".join(html) + "</div>"
    enviar_correo(cfg, cfg.notificar_a, asunto, "\n\n".join(texto), html=cuerpo)
    for c in compras:
        db.actualizar_compra(c["id"], notificada=1)
    db.marcar_alertas_notificadas([a["id"] for a in alertas])
    db.log("notificador", "correo", asunto)
    return 1
