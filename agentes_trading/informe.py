"""Informe diario en texto y HTML, y envío por correo."""

import html

AVISO = (
    "Este agente opera según reglas automáticas y NO garantiza ganancias: puede perder dinero. "
    "Las cifras históricas no aseguran resultados futuros."
)


def _usd(v):
    return f"{v:,.2f} USD"


def construir_informe(cartera, precios, cfg, operaciones_hoy, pendiente, fallidos=(), nombre_broker="cartera simulada"):
    patrimonio = cartera.patrimonio(precios)
    ganancia = patrimonio - cartera.capital_inicial
    progreso = ganancia / cfg.meta_usd if cfg.meta_usd else 0.0
    lineas = [
        f"Fecha de datos: {cartera.ultima_fecha or '-'} · Modo: {nombre_broker}",
        f"Patrimonio: {_usd(patrimonio)} (capital inicial {_usd(cartera.capital_inicial)})",
        f"Ganancia acumulada: {ganancia:+,.2f} USD · Meta {_usd(cfg.meta_usd)} · Progreso {progreso:.0%}",
        f"Efectivo: {_usd(cartera.efectivo)}",
    ]
    if cartera.posiciones:
        lineas.append("Posiciones:")
        for t, pos in sorted(cartera.posiciones.items()):
            precio = precios.get(t, pos["precio_promedio"])
            var = precio / pos["precio_promedio"] - 1 if pos["precio_promedio"] else 0.0
            lineas.append(f"  · {t}: {pos['cantidad']:.4f} uds · compra {pos['precio_promedio']:.2f} · ahora {precio:.2f} ({var:+.1%})")
    else:
        lineas.append("Posiciones: ninguna (todo en efectivo).")
    if operaciones_hoy:
        lineas.append("Operaciones de este ciclo:")
        for op in operaciones_hoy:
            lineas.append(f"  · {op['lado'].upper()} {op['ticker']} {op['cantidad']:.4f} uds a {op['precio']:.2f} ({_usd(op['importe'])}) — {op['motivo']}")
    else:
        lineas.append("Operaciones de este ciclo: ninguna.")
    if pendiente:
        detalle = []
        if pendiente.get("vender"):
            detalle.append("vender " + ", ".join(pendiente["vender"]))
        if pendiente.get("objetivo") is not None:
            detalle.append("objetivo " + (", ".join(f"{t} {p:.0%}" for t, p in pendiente["objetivo"].items()) or "100% efectivo"))
        lineas.append(f"Órdenes para la próxima apertura: {'; '.join(detalle)}. Motivo: {pendiente.get('motivo', '')}")
    señal = cartera.ultima_señal
    if señal.get("ranking"):
        ranking = ", ".join(f"{r['ticker']} {r['momentum']:+.1%}{'' if r['tendencia'] else ' (bajo su media)'}" for r in señal["ranking"])
        lineas.append(f"Ranking de momentum ({señal.get('fecha', '')}): {ranking}")
    if fallidos:
        lineas.append(f"Sin datos para: {', '.join(fallidos)}")
    lineas.append("")
    lineas.append(AVISO)
    return "\n".join(lineas)


def informe_html(texto):
    cuerpo = "<br>".join(html.escape(l).replace("  ", "&nbsp;&nbsp;") for l in texto.splitlines())
    return f'<div style="font-family:Arial,sans-serif;font-size:14px;line-height:1.5;color:#222">{cuerpo}</div>'


def asunto_informe(cartera, precios, operaciones_hoy):
    ganancia = cartera.patrimonio(precios) - cartera.capital_inicial
    estado = f"{len(operaciones_hoy)} operación(es)" if operaciones_hoy else "sin cambios"
    return f"Agente de trading · {ganancia:+,.0f} USD · {estado}"


def enviar_informe(cfg, asunto, texto):
    from agentes_ebook.gmail import enviar_correo

    if not cfg.notificar_a:
        return "sin destinatario"
    return enviar_correo(cfg, cfg.notificar_a, asunto, texto, informe_html(texto))
