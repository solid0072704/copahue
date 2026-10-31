"""Informe de caja: Markdown para guardar, HTML para el correo y texto plano."""

import os
import re
from html import escape

from agentes_ebook.gmail import enviar_correo
from agentes_hubspot.informe import C_SEC, C_TEXTO, C_FONDO, _h2, _h3, _kpi, _lista, _tabla_html, _tabla_md

from .analista import fecha_corta, fmt_monto

NIVEL_ICONO = {"media": "🟡", "alta": "🟠", "crítica": "🔴"}


def _pct(v):
    return "—" if v is None else f"{v:.0f}%"


def _rango(f):
    return f"{fecha_corta(f['inicio'])} al {fecha_corta(f['fin'])}"


def construir(diag, cfg, lectura_ia=None, ahora_local=None):
    """Devuelve (título, markdown, html, texto)."""
    m = diag["moneda"]
    cb, py = diag["cobranza"], diag["proyeccion"]
    fecha = ahora_local.strftime("%d/%m/%Y") if ahora_local else diag["fecha"][:10]
    titulo = f"Flujo de caja · {cfg.empresa_nombre} · {fecha}"
    base = py["escenarios"]["base"]
    kpis = [
        (fmt_monto(cb["total_pendiente"], m), "cartera pendiente"),
        (fmt_monto(cb["total_moroso"], m), f"moroso ({cb['pct_mora']:.0f}%)"),
        (fmt_monto(py["proximas_4"]["ingresos"], m), "ingresos esperados 4 sem."),
        (fmt_monto(py["proximas_4"]["egresos"], m), "egresos 4 sem."),
    ]
    if py["saldo_inicial"] or py["caja_minima"]:
        kpis.insert(2, (fmt_monto(base["saldo_minimo"], m), f"saldo mínimo proyectado ({fecha_corta(base['semana_minima'])})"))
    contexto = dict(diag=diag, cfg=cfg, titulo=titulo, kpis=kpis, lectura_ia=lectura_ia, m=m, cb=cb, py=py)
    return titulo, _markdown(**contexto), _html(**contexto), _texto(**contexto)


def _filas_proyeccion(py, m, con_ventas):
    filas = []
    for f in py["semanas"]:
        fila = [f"S{f['semana']} · {_rango(f)}", fmt_monto(f["programado"], m), fmt_monto(f["esperado"], m), fmt_monto(f["mora"], m)]
        if con_ventas:
            fila.append(fmt_monto(f["ventas"], m))
        fila += [fmt_monto(f["egresos"], m), fmt_monto(f["neto"], m), fmt_monto(f["saldo"], m)]
        filas.append(fila)
    return filas


def _cabeceras_proyeccion(con_ventas):
    cab = ["Semana", "Cobros programados", "Esperados", "Rec. mora"]
    if con_ventas:
        cab.append("Ventas")
    return cab + ["Egresos", "Neto", "Saldo"]


def _filas_escenarios(py, m):
    return [
        (nombre.capitalize(), fmt_monto(e["ingresos"], m), fmt_monto(e["egresos"], m), f"{fmt_monto(e['saldo_minimo'], m)} ({fecha_corta(e['semana_minima'])})", fmt_monto(e["saldo_final"], m))
        for nombre, e in py["escenarios"].items()
    ]


def _filas_empresas(cb, m):
    return [
        (e["empresa"], fmt_monto(e["pendiente"], m), fmt_monto(e["moroso"], m), _pct(e["pct_mora"]), e["compromisos_morosos"], fmt_monto(e["vence_30d"], m), fmt_monto(e["recaudado_30d"], m))
        for e in cb["por_empresa"]
    ]


def _filas_tramos(cb, m):
    return [(t["tramo"], t["cuotas"], fmt_monto(t["monto"], m), _pct(t["pct"]), t["accion"]) for t in cb["tramos"]]


def _filas_prioritarios(cb, m, maximo):
    filas = []
    for p in cb["prioritarios"][:maximo]:
        quien = p["cliente"] or p["compromiso"] or "—"
        if p["compromiso"] and p["cliente"]:
            quien = f"{p['cliente']} · {p['compromiso']}"
        filas.append((f"{NIVEL_ICONO.get(p['nivel'], '')} {quien}".strip(), p["empresa"], p["proyecto"] or "—", fmt_monto(p["monto"], m), p["cuotas"], p["dias_mora"], p["accion"], p["vencen_30d"]))
    return filas


def _filas_efectividad(cb, m):
    return [(f["mes"], fmt_monto(f["vencido"], m), fmt_monto(f["cobrado"], m), _pct(f["puntualidad"]), fmt_monto(f["recaudado"], m)) for f in cb["efectividad"]]


# --------------------------------------------------------------------------- Markdown
def _markdown(diag, cfg, titulo, kpis, lectura_ia, m, cb, py):
    con_ventas = any(f["ventas"] for f in py["semanas"])
    s = [f"# {titulo}", "", f"_Tasa de cobro usada: {py['tasa_cobro'] * 100:.0f}% · recuperación de mora: {py['tasa_mora'] * 100:.0f}% en {cfg.semanas_recuperacion_mora} semanas · {diag['cuotas_total']} cuotas de {len(diag['empresas'])} sociedades._", ""]
    s.append(_tabla_md([e for _, e in kpis], [[v for v, _ in kpis]]))
    if lectura_ia:
        s += ["", "## Lectura", "", lectura_ia]
    s += ["", "## Hallazgos", ""] + [f"- {h}" for h in diag["hallazgos"]]
    if diag["acciones"]:
        s += ["", "## Qué hacer esta semana", ""] + [f"{i}. {a}" for i, a in enumerate(diag["acciones"], 1)]
    s += ["", f"## Proyección de caja · {py['horizonte_semanas']} semanas (escenario base)", ""]
    if py["saldo_inicial"]:
        s.append(f"Saldo inicial: {fmt_monto(py['saldo_inicial'], m)} · caja mínima: {fmt_monto(py['caja_minima'], m)}")
        s.append("")
    s.append(_tabla_md(_cabeceras_proyeccion(con_ventas), _filas_proyeccion(py, m, con_ventas)))
    s += ["", "### Escenarios", "", _tabla_md(["Escenario", "Ingresos", "Egresos", "Saldo mínimo (semana)", "Saldo final"], _filas_escenarios(py, m))]
    s += ["", "## Por sociedad", "", _tabla_md(["Sociedad", "Pendiente", "Moroso", "% mora", "Compromisos morosos", "Vence 30 d", "Recaudado 30 d"], _filas_empresas(cb, m))]
    s += ["", "## Morosidad por antigüedad", "", _tabla_md(["Tramo", "Cuotas", "Monto", "%", "Acción"], _filas_tramos(cb, m))]
    s += ["", f"## Cobranza prioritaria ({len(cb['prioritarios'])} de {cb['total_prioritarios']} compromisos en mora)", ""]
    if cb["prioritarios"]:
        s.append(_tabla_md(["Cliente · compromiso", "Sociedad", "Proyecto", "Moroso", "Cuotas", "Días", "Acción", "Vencen 30 d"], _filas_prioritarios(cb, m, cfg.top_cobranza)))
    else:
        s.append("No hay cuotas en mora.")
    if cb["en_cobro"]["monto"] or cb["repactadas"]["monto"]:
        s += ["", f"En proceso de cobro: {fmt_monto(cb['en_cobro']['monto'], m)} ({cb['en_cobro']['cuotas']} cuotas) · Repactado: {fmt_monto(cb['repactadas']['monto'], m)} ({cb['repactadas']['cuotas']} cuotas)."]
    if cb["efectividad"]:
        s += ["", "## Efectividad de cobranza", "", _tabla_md(["Mes", "Vencía", "Cobrado de lo vencido", "Puntualidad", "Recaudado en el mes"], _filas_efectividad(cb, m))]
    if diag["egresos_categorias_4s"]:
        s += ["", "## Egresos de las próximas 4 semanas", "", _tabla_md(["Categoría", "Monto"], [(e["categoria"], fmt_monto(e["monto"], m)) for e in diag["egresos_categorias_4s"]])]
        s += ["", _tabla_md(["Fecha", "Concepto", "Sociedad", "Monto"], [(fecha_corta(e["fecha"]), e["concepto"], e["empresa"] or "—", fmt_monto(e["monto"], m)) for e in diag["egresos_mayores_4s"]])]
    comp = diag.get("comparacion")
    if comp:
        s += ["", f"## Frente a la foto del {comp['fecha'][:10]}", "", f"- Morosidad: {'+' if comp['moroso'] > 0 else ''}{fmt_monto(comp['moroso'], m)} ({'+' if comp['pct_mora'] > 0 else ''}{comp['pct_mora']:.1f} puntos)", f"- Cartera pendiente: {'+' if comp['pendiente'] > 0 else ''}{fmt_monto(comp['pendiente'], m)}", f"- Saldo mínimo proyectado: {'+' if comp['saldo_minimo'] > 0 else ''}{fmt_monto(comp['saldo_minimo'], m)}"]
    if diag["errores_fuentes"]:
        s += ["", "## Fuentes con problemas", ""] + [f"- {e}" for e in diag["errores_fuentes"]]
    s += ["", f"_Generado por el agente de caja de {cfg.empresa_nombre}._"]
    return "\n".join(s)


# --------------------------------------------------------------------------- HTML
def _envoltorio(titulo, contenido, cfg):
    return f"""<!doctype html><html><body style="margin:0;background:{C_FONDO};">
<div style="max-width:860px;margin:0 auto;padding:24px 16px;font-family:system-ui,-apple-system,'Segoe UI',Arial,sans-serif;color:{C_TEXTO};">
<p style="margin:0 0 4px;font-size:12px;color:{C_SEC};letter-spacing:.04em;text-transform:uppercase;">Agente de caja · {escape(cfg.empresa_nombre)}</p>
<h1 style="margin:0 0 16px;font-size:22px;line-height:1.3;">{escape(titulo)}</h1>
{contenido}
<p style="margin:24px 0 0;font-size:12px;color:{C_SEC};">Este informe lo generó tu agente de caja automáticamente a partir de Forpay y del archivo de egresos. No modifica ningún sistema.</p>
</div></body></html>"""


def _esc(fila):
    return tuple(escape(str(c)) for c in fila)


def _html(diag, cfg, titulo, kpis, lectura_ia, m, cb, py):
    con_ventas = any(f["ventas"] for f in py["semanas"])
    partes = [
        '<table role="presentation" style="width:100%;border-collapse:separate;border-spacing:6px;margin:0 -6px 8px;"><tr>' + "".join(_kpi(v, e) for v, e in kpis) + "</tr></table>",
        f"<p style='font-size:12px;color:{C_SEC};margin:0;'>Tasa de cobro usada: {py['tasa_cobro'] * 100:.0f}% · recuperación de mora: {py['tasa_mora'] * 100:.0f}% en {cfg.semanas_recuperacion_mora} semanas · {diag['cuotas_total']} cuotas de {len(diag['empresas'])} sociedades.</p>",
    ]
    if lectura_ia:
        partes += [_h2("Lectura"), f"<p style='font-size:14px;margin:0;'>{escape(lectura_ia)}</p>"]
    partes += [_h2("Hallazgos"), _lista(diag["hallazgos"])]
    if diag["acciones"]:
        partes += [_h2("Qué hacer esta semana"), _lista(diag["acciones"], numerada=True)]
    partes.append(_h2(f"Proyección de caja · {py['horizonte_semanas']} semanas (escenario base)"))
    if py["saldo_inicial"]:
        partes.append(f"<p style='font-size:13px;margin:0 0 6px;'>Saldo inicial: {escape(fmt_monto(py['saldo_inicial'], m))} · caja mínima: {escape(fmt_monto(py['caja_minima'], m))}</p>")
    n = len(_cabeceras_proyeccion(con_ventas))
    partes.append(_tabla_html(_cabeceras_proyeccion(con_ventas), [_esc(f) for f in _filas_proyeccion(py, m, con_ventas)], derecha=tuple(range(1, n))))
    partes += [_h3("Escenarios"), _tabla_html(["Escenario", "Ingresos", "Egresos", "Saldo mínimo (semana)", "Saldo final"], [_esc(f) for f in _filas_escenarios(py, m)], derecha=(1, 2, 3, 4))]
    partes += [_h2("Por sociedad"), _tabla_html(["Sociedad", "Pendiente", "Moroso", "% mora", "Compromisos morosos", "Vence 30 d", "Recaudado 30 d"], [_esc(f) for f in _filas_empresas(cb, m)], derecha=(1, 2, 3, 4, 5, 6))]
    partes += [_h2("Morosidad por antigüedad"), _tabla_html(["Tramo", "Cuotas", "Monto", "%", "Acción"], [_esc(f) for f in _filas_tramos(cb, m)], derecha=(1, 2, 3))]
    partes.append(_h2(f"Cobranza prioritaria ({len(cb['prioritarios'])} de {cb['total_prioritarios']} compromisos en mora)"))
    if cb["prioritarios"]:
        partes.append(_tabla_html(["Cliente · compromiso", "Sociedad", "Proyecto", "Moroso", "Cuotas", "Días", "Acción", "Vencen 30 d"], [_esc(f) for f in _filas_prioritarios(cb, m, cfg.top_cobranza)], derecha=(3, 4, 5, 7)))
    else:
        partes.append("<p style='font-size:14px;margin:0;'>No hay cuotas en mora.</p>")
    if cb["en_cobro"]["monto"] or cb["repactadas"]["monto"]:
        partes.append(f"<p style='font-size:13px;margin:6px 0 0;'>En proceso de cobro: {escape(fmt_monto(cb['en_cobro']['monto'], m))} ({cb['en_cobro']['cuotas']} cuotas) · Repactado: {escape(fmt_monto(cb['repactadas']['monto'], m))} ({cb['repactadas']['cuotas']} cuotas).</p>")
    if cb["efectividad"]:
        partes += [_h2("Efectividad de cobranza"), _tabla_html(["Mes", "Vencía", "Cobrado de lo vencido", "Puntualidad", "Recaudado en el mes"], [_esc(f) for f in _filas_efectividad(cb, m)], derecha=(1, 2, 3, 4))]
    if diag["egresos_categorias_4s"]:
        partes += [_h2("Egresos de las próximas 4 semanas"), _tabla_html(["Categoría", "Monto"], [_esc((e["categoria"], fmt_monto(e["monto"], m))) for e in diag["egresos_categorias_4s"]], derecha=(1,))]
        partes.append(_tabla_html(["Fecha", "Concepto", "Sociedad", "Monto"], [_esc((fecha_corta(e["fecha"]), e["concepto"], e["empresa"] or "—", fmt_monto(e["monto"], m))) for e in diag["egresos_mayores_4s"]], derecha=(3,)))
    comp = diag.get("comparacion")
    if comp:
        partes += [_h2(f"Frente a la foto del {comp['fecha'][:10]}"), _lista([f"Morosidad: {'+' if comp['moroso'] > 0 else ''}{fmt_monto(comp['moroso'], m)} ({'+' if comp['pct_mora'] > 0 else ''}{comp['pct_mora']:.1f} puntos)", f"Cartera pendiente: {'+' if comp['pendiente'] > 0 else ''}{fmt_monto(comp['pendiente'], m)}", f"Saldo mínimo proyectado: {'+' if comp['saldo_minimo'] > 0 else ''}{fmt_monto(comp['saldo_minimo'], m)}"])]
    if diag["errores_fuentes"]:
        partes += [_h2("Fuentes con problemas"), _lista(diag["errores_fuentes"])]
    return _envoltorio(titulo, "".join(partes), cfg)


# --------------------------------------------------------------------------- texto plano
def _texto(diag, cfg, titulo, kpis, lectura_ia, m, cb, py):
    lineas = [titulo, " · ".join(f"{e}: {v}" for v, e in kpis), ""]
    if lectura_ia:
        lineas += [lectura_ia, ""]
    lineas += ["Hallazgos:"] + [f"- {h}" for h in diag["hallazgos"]]
    if diag["acciones"]:
        lineas += ["", "Qué hacer esta semana:"] + [f"{i}. {a}" for i, a in enumerate(diag["acciones"], 1)]
    lineas += ["", "La proyección semanal, la cobranza prioritaria y el detalle por sociedad están en la versión HTML de este correo."]
    return "\n".join(lineas)


# --------------------------------------------------------------------------- guardar y enviar
def guardar(markdown, cfg, ahora_local):
    os.makedirs(cfg.carpeta_informes, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", cfg.empresa_nombre.lower()).strip("-") or "empresa"
    ruta = os.path.join(cfg.carpeta_informes, f"caja_{slug}_{ahora_local:%Y-%m-%d}.md")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(os.path.join(cfg.carpeta_informes, "ultimo_informe_caja.md"), "w", encoding="utf-8") as f:
        f.write(markdown)
    return ruta


def enviar(cfg, titulo, texto, html):
    if not (cfg.gmail_configurado and cfg.notificar_a):
        return "sin_configurar"
    try:
        enviar_correo(cfg, cfg.notificar_a, f"💵 {titulo}", texto, html, cc=cfg.notificar_cc)
        return "enviado"
    except Exception as e:
        return f"error: {e}"
