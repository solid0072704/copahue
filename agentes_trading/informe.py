"""Informe diario en texto y HTML, y envío por correo."""

import html

AVISO = (
    "Este agente opera según reglas automáticas y NO garantiza ganancias: puede perder dinero. "
    "Las cifras históricas no aseguran resultados futuros."
)


def _usd(v):
    return f"{v:,.2f} USD"


def _clp(v):
    return f"{v:,.0f} CLP".replace(",", ".")


def construir_informe(cartera, precios, cfg, operaciones_hoy, pendiente, fallidos=(), nombre_broker="cartera simulada", usdclp=None, aporte=None):
    patrimonio = cartera.patrimonio(precios)
    ganancia = patrimonio - cartera.capital_aportado
    en_pesos = bool(usdclp) and bool(cfg.aporte_mensual_clp or cfg.meta_mensual_clp)
    lineas = [f"Fecha de datos: {cartera.ultima_fecha or '-'} · Modo: {nombre_broker}"]
    if en_pesos:
        lineas += [
            f"Tipo de cambio: {usdclp:,.0f} CLP por USD".replace(",", "."),
            f"Patrimonio: {_clp(patrimonio * usdclp)} ({_usd(patrimonio)})",
            f"Aportado hasta hoy: {_clp(cartera.capital_aportado * usdclp)} en {len(cartera.aportes)} aporte(s)",
            f"Ganancia acumulada: {ganancia * usdclp:+,.0f} CLP ({ganancia / cartera.capital_aportado:+.1%})".replace(",", ".") if cartera.capital_aportado else "Ganancia acumulada: 0 CLP",
            f"Efectivo: {_clp(cartera.efectivo * usdclp)}",
        ]
        if aporte:
            lineas.append(f"Aporte de este mes: {_clp(aporte.get('clp') or 0)} → {_usd(aporte['importe'])}")
        if cfg.meta_mensual_clp:
            from .proyeccion import capital_para_renta

            capital_meta = capital_para_renta(cfg.meta_mensual_clp, cfg.tasa_retiro)
            renta_hoy = patrimonio * usdclp * cfg.tasa_retiro / 12
            lineas.append(
                f"Meta: renta de {_clp(cfg.meta_mensual_clp)} al mes → requiere {_clp(capital_meta)} de capital. "
                f"Hoy la cartera daría {_clp(renta_hoy)} al mes ({patrimonio * usdclp / capital_meta:.1%} de la meta)."
            )
    else:
        progreso = ganancia / cfg.meta_usd if cfg.meta_usd else 0.0
        lineas += [
            f"Patrimonio: {_usd(patrimonio)} (aportado {_usd(cartera.capital_aportado)})",
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


def asunto_informe(cartera, precios, operaciones_hoy, usdclp=None):
    ganancia = cartera.patrimonio(precios) - cartera.capital_aportado
    estado = f"{len(operaciones_hoy)} operación(es)" if operaciones_hoy else "sin cambios"
    if usdclp:
        return f"Agente de trading · {ganancia * usdclp:+,.0f} CLP · {estado}".replace(",", ".")
    return f"Agente de trading · {ganancia:+,.0f} USD · {estado}"


def enviar_informe(cfg, asunto, texto):
    from agentes_ebook.gmail import enviar_correo

    if not cfg.notificar_a:
        return "sin destinatario"
    return enviar_correo(cfg, cfg.notificar_a, asunto, texto, informe_html(texto))
