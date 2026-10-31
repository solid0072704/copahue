"""Informe del embudo: Markdown para guardar, HTML para el correo y texto plano de resumen."""

import os
import re
from html import escape

from agentes_ebook.gmail import enviar_correo

from .desatascador import ESTADOS, NOMBRES_CATEGORIA
from .embudo import _signo

C_TEXTO, C_SEC, C_ACENTO, C_BORDE, C_FONDO = "#0b0b0b", "#52514e", "#2a78d6", "#e1e0d9", "#f9f9f7"
MODO_TXT = {"lectura": "solo lectura: propone, no escribe en HubSpot", "tareas": "crea tareas en HubSpot"}


def url_negocio(portal_id, negocio_id):
    return f"https://app.hubspot.com/contacts/{portal_id}/record/0-3/{negocio_id}" if portal_id else ""


def _n(valor):
    return f"{valor:,.0f}".replace(",", ".")


def _pct(valor):
    return "—" if valor is None else f"{valor:.0f}%"


def _md(texto):
    """Texto seguro dentro de una celda Markdown."""
    return str(texto).replace("|", "/").replace("\n", " ")


def _slug(texto):
    return re.sub(r"[^a-z0-9]+", "-", (texto or "").lower()).strip("-") or "pipeline"


def _nombre_propietario(propietarios, pid):
    if not pid:
        return "Sin propietario"
    return propietarios.get(pid, {}).get("nombre") or f"Propietario {pid}"


def _filas_etapas(diag):
    return [
        (e["nombre"], e["abiertos"], e["mediana_dias"], e["p75_dias"], e["sin_proxima_actividad"], e["sin_contacto"], e["alcanzaron"], e["conversion"])
        for e in diag["etapas"]
        if e["abiertos"] or e["alcanzaron"]
    ]


def _filas_estancados(grupo, propietarios, portal_id, maximo):
    filas = []
    for est in grupo[:maximo]:
        n = est["negocio"]
        filas.append(
            {
                "nombre": n["nombre"],
                "url": url_negocio(portal_id, n["id"]),
                "etapa": est["etapa"]["nombre"],
                "dias": est["dias"],
                "sin_contacto": est["dias_sin_contacto"],
                "propietario": _nombre_propietario(propietarios, n.get("propietario_id")),
                "accion": est["accion"],
                "estado": ESTADOS.get(est["estado"], est["estado"]),
            }
        )
    return filas


def construir(diag, resultado, cfg, portal_id="", propietarios=None, lectura_ia=None, ahora_local=None):
    """Devuelve (título, markdown, html, texto) del informe."""
    propietarios = propietarios or {}
    fecha = ahora_local.strftime("%d/%m/%Y") if ahora_local else diag["fecha"][:10]
    titulo = f"Embudo HubSpot · {diag['pipeline'] or cfg.pipeline} · {fecha}"
    grupos = resultado.por_categoria()
    por_grupo = max(5, cfg.max_estancados_informe // max(1, len(grupos)))
    kpis = [
        (_n(diag["total_abiertos"]), "negocios abiertos"),
        (_n(diag["sin_proxima_total"]), "sin próxima actividad"),
        (_n(len(resultado.estancados)), "estancados"),
        (_n(diag["creados_periodo"]), f"creados en {diag['periodo_dias']} días"),
        (_n(diag["perdidos_periodo"]), f"perdidos en {diag['periodo_dias']} días"),
        (_n(diag["ganados_periodo"]), f"ganados en {diag['periodo_dias']} días"),
    ]
    if diag["monto_abierto"]:
        kpis.insert(1, (_n(diag["monto_abierto"]), "monto abierto"))
    contexto = dict(
        diag=diag, resultado=resultado, cfg=cfg, titulo=titulo, kpis=kpis, grupos=grupos, por_grupo=por_grupo,
        propietarios=propietarios, portal_id=portal_id, lectura_ia=lectura_ia,
    )
    return titulo, _markdown(**contexto), _html(**contexto), _texto(**contexto)


# --------------------------------------------------------------------------- Markdown
def _tabla_md(cabeceras, filas):
    lineas = ["| " + " | ".join(cabeceras) + " |", "|" + "---|" * len(cabeceras)]
    lineas += ["| " + " | ".join(_md(c) for c in fila) + " |" for fila in filas]
    return "\n".join(lineas)


def _markdown(diag, resultado, cfg, titulo, kpis, grupos, por_grupo, propietarios, portal_id, lectura_ia):
    salida = [f"# {titulo}", "", f"_Modo: {MODO_TXT.get(resultado.modo, resultado.modo)}._", ""]
    salida.append(_tabla_md([etiqueta for _, etiqueta in kpis], [[valor for valor, _ in kpis]]))
    if lectura_ia:
        salida += ["", "## Lectura", "", lectura_ia]
    salida += ["", "## Hallazgos", ""] + [f"- {h}" for h in diag["hallazgos"]]
    if diag["recomendaciones"]:
        salida += ["", "## Qué hacer esta semana", ""] + [f"{i}. {r}" for i, r in enumerate(diag["recomendaciones"], 1)]
    salida += [
        "",
        "## Por etapa",
        "",
        _tabla_md(
            [
                "Etapa", "Abiertos", "Mediana días", "P75 días", "Sin próx. actividad",
                f"Sin contacto ≥{cfg.sin_contacto_dias} d", f"Llegaron ({diag['periodo_dias']} d)", "Conversión",
            ],
            [
                (nombre, abiertos, f"{mediana:.0f}", f"{p75:.0f}", sin_prox, sin_cont, alcanzaron, _pct(conv))
                for nombre, abiertos, mediana, p75, sin_prox, sin_cont, alcanzaron, conv in _filas_etapas(diag)
            ],
        ),
    ]
    salida += ["", f"## Negocios estancados ({len(resultado.estancados)})", ""]
    if not resultado.estancados:
        salida.append("Ningún negocio supera los umbrales de su etapa.")
    for categoria, grupo in grupos.items():
        salida += [f"### {NOMBRES_CATEGORIA[categoria]} ({len(grupo)})", ""]
        filas = []
        for f in _filas_estancados(grupo, propietarios, portal_id, por_grupo):
            nombre = f"[{_md(f['nombre'])}]({f['url']})" if f["url"] else _md(f["nombre"])
            sin_contacto = "—" if f["sin_contacto"] is None else f"{f['sin_contacto']:.0f}"
            filas.append((nombre, f["etapa"], f"{f['dias']:.0f}", sin_contacto, f["propietario"], f["accion"], f["estado"]))
        salida.append(_tabla_md(["Negocio", "Etapa", "Días en etapa", "Sin contacto (d)", "Propietario", "Acción propuesta", "Estado"], filas))
        if len(grupo) > por_grupo:
            salida.append(f"\n_… y {len(grupo) - por_grupo} más en esta categoría._")
        salida.append("")
    if resultado.escalados:
        salida += ["## Para escalar", ""]
        salida += [
            f"- {_md(e['negocio']['nombre'])} · {e['etapa']['nombre']} · {e['dias']:.0f} días · "
            f"{_nombre_propietario(propietarios, e['negocio'].get('propietario_id'))}"
            for e in resultado.escalados[:20]
        ] + [""]
    if diag["razones_perdida"]:
        total = diag["perdidos_periodo"]
        salida += [
            f"## Razones de pérdida ({diag['periodo_dias']} días)",
            "",
            _tabla_md(["Razón", "Negocios", "%"], [(r, c, _pct(100 * c / total) if total else "—") for r, c in diag["razones_perdida"]]),
            "",
        ]
    if diag["propietarios"]:
        salida += [
            "## Por propietario",
            "",
            _tabla_md(
                ["Propietario", "Abiertos", "Sin próx. actividad", "Estancados"],
                [(x["nombre"], x["abiertos"], x["sin_proxima"], x["estancados"]) for x in diag["propietarios"][:10]],
            ),
            "",
        ]
    comp = diag.get("comparacion")
    if comp:
        salida += [
            f"## Frente a la foto del {comp['fecha'][:10]}",
            "",
            f"- Negocios abiertos: {_signo(comp['abiertos'])}",
            f"- Sin próxima actividad: {_signo(comp['sin_proxima'])}",
        ]
        cambios = sorted(comp["por_etapa"].items(), key=lambda kv: -abs(kv[1]))[:6]
        salida += [f"- {nombre}: {_signo(delta)}" for nombre, delta in cambios] + [""]
    if resultado.errores:
        salida += ["## Errores", ""] + [f"- {e}" for e in resultado.errores] + [""]
    salida.append(f"_Generado por los agentes del embudo de {cfg.empresa_nombre}._")
    return "\n".join(salida)


# --------------------------------------------------------------------------- HTML
def _h2(texto):
    return f'<h2 style="font-size:17px;margin:18px 0 6px;">{escape(texto)}</h2>'


def _h3(texto):
    return f'<h3 style="font-size:15px;margin:14px 0 4px;color:{C_SEC};">{escape(texto)}</h3>'


def _lista(items, numerada=False):
    tag = "ol" if numerada else "ul"
    cuerpo = "".join(f"<li style='margin:0 0 4px;'>{escape(i)}</li>" for i in items)
    return f"<{tag} style='padding-left:18px;margin:0;font-size:14px;'>{cuerpo}</{tag}>"


def _tabla_html(cabeceras, filas, derecha=()):
    def celda(i, contenido, encabezado=False):
        estilo = f"padding:6px 4px;border-bottom:1px solid {C_BORDE};text-align:{'right' if i in derecha else 'left'};"
        if encabezado:
            estilo += f"font-size:12px;color:{C_SEC};border-bottom:2px solid {C_BORDE};"
        tag = "th" if encabezado else "td"
        return f"<{tag} style='{estilo}'>{contenido}</{tag}>"

    cab = "".join(celda(i, escape(str(c)), True) for i, c in enumerate(cabeceras))
    cuerpo = "".join("<tr>" + "".join(celda(i, c) for i, c in enumerate(fila)) + "</tr>" for fila in filas)
    return f"<table style='width:100%;border-collapse:collapse;font-size:13px;'><tr>{cab}</tr>{cuerpo}</table>"


def _kpi(valor, etiqueta):
    return (
        f'<td style="padding:10px 12px;background:#ffffff;border:1px solid {C_BORDE};border-radius:8px;">'
        f'<div style="font-size:22px;font-weight:600;">{escape(str(valor))}</div>'
        f'<div style="font-size:12px;color:{C_SEC};">{escape(etiqueta)}</div></td>'
    )


def _envoltorio(titulo, contenido, cfg, modo):
    return f"""<!doctype html><html><body style="margin:0;background:{C_FONDO};">
<div style="max-width:760px;margin:0 auto;padding:24px 16px;font-family:system-ui,-apple-system,'Segoe UI',Arial,sans-serif;color:{C_TEXTO};">
<p style="margin:0 0 4px;font-size:12px;color:{C_SEC};letter-spacing:.04em;text-transform:uppercase;">Agentes del embudo · {escape(cfg.empresa_nombre)}</p>
<h1 style="margin:0 0 6px;font-size:22px;line-height:1.3;">{escape(titulo)}</h1>
<p style="margin:0 0 16px;font-size:13px;color:{C_SEC};">Modo: {escape(MODO_TXT.get(modo, modo))}.</p>
{contenido}
<p style="margin:24px 0 0;font-size:12px;color:{C_SEC};">Este informe lo generaron tus agentes automáticamente a partir de HubSpot. En modo lectura no modifican nada en el CRM.</p>
</div></body></html>"""


def _html(diag, resultado, cfg, titulo, kpis, grupos, por_grupo, propietarios, portal_id, lectura_ia):
    partes = [
        '<table role="presentation" style="width:100%;border-collapse:separate;border-spacing:6px;margin:0 -6px 8px;"><tr>'
        + "".join(_kpi(v, e) for v, e in kpis)
        + "</tr></table>"
    ]
    if lectura_ia:
        partes += [_h2("Lectura"), f"<p style='font-size:14px;margin:0;'>{escape(lectura_ia)}</p>"]
    partes += [_h2("Hallazgos"), _lista(diag["hallazgos"])]
    if diag["recomendaciones"]:
        partes += [_h2("Qué hacer esta semana"), _lista(diag["recomendaciones"], numerada=True)]
    partes += [
        _h2("Por etapa"),
        _tabla_html(
            ["Etapa", "Abiertos", "Mediana d", "P75 d", "Sin próx. act.", f"Sin contacto ≥{cfg.sin_contacto_dias} d", "Llegaron", "Conv."],
            [
                (escape(nombre), abiertos, f"{mediana:.0f}", f"{p75:.0f}", sin_prox, sin_cont, alcanzaron, _pct(conv))
                for nombre, abiertos, mediana, p75, sin_prox, sin_cont, alcanzaron, conv in _filas_etapas(diag)
            ],
            derecha=(1, 2, 3, 4, 5, 6, 7),
        ),
    ]
    partes.append(_h2(f"Negocios estancados ({len(resultado.estancados)})"))
    if not resultado.estancados:
        partes.append("<p style='font-size:14px;margin:0;'>Ningún negocio supera los umbrales de su etapa.</p>")
    for categoria, grupo in grupos.items():
        partes.append(_h3(f"{NOMBRES_CATEGORIA[categoria]} ({len(grupo)})"))
        filas = []
        for f in _filas_estancados(grupo, propietarios, portal_id, por_grupo):
            nombre = (
                f"<a href='{escape(f['url'])}' style='color:{C_ACENTO};'>{escape(f['nombre'])}</a>" if f["url"] else escape(f["nombre"])
            )
            sin_contacto = "—" if f["sin_contacto"] is None else f"{f['sin_contacto']:.0f}"
            filas.append((nombre, escape(f["etapa"]), f"{f['dias']:.0f}", sin_contacto, escape(f["propietario"]), escape(f["accion"]), escape(f["estado"])))
        partes.append(_tabla_html(["Negocio", "Etapa", "Días", "Sin contacto", "Propietario", "Acción propuesta", "Estado"], filas, derecha=(2, 3)))
        if len(grupo) > por_grupo:
            partes.append(f"<p style='font-size:12px;color:{C_SEC};margin:4px 0 0;'>… y {len(grupo) - por_grupo} más en esta categoría.</p>")
    if resultado.escalados:
        partes += [
            _h2("Para escalar"),
            _lista(
                f"{e['negocio']['nombre']} · {e['etapa']['nombre']} · {e['dias']:.0f} días · "
                f"{_nombre_propietario(propietarios, e['negocio'].get('propietario_id'))}"
                for e in resultado.escalados[:20]
            ),
        ]
    if diag["razones_perdida"]:
        total = diag["perdidos_periodo"]
        partes += [
            _h2(f"Razones de pérdida ({diag['periodo_dias']} días)"),
            _tabla_html(["Razón", "Negocios", "%"], [(escape(str(r)), c, _pct(100 * c / total) if total else "—") for r, c in diag["razones_perdida"]], derecha=(1, 2)),
        ]
    if diag["propietarios"]:
        partes += [
            _h2("Por propietario"),
            _tabla_html(
                ["Propietario", "Abiertos", "Sin próx. act.", "Estancados"],
                [(escape(x["nombre"]), x["abiertos"], x["sin_proxima"], x["estancados"]) for x in diag["propietarios"][:10]],
                derecha=(1, 2, 3),
            ),
        ]
    comp = diag.get("comparacion")
    if comp:
        items = [f"Negocios abiertos: {_signo(comp['abiertos'])}", f"Sin próxima actividad: {_signo(comp['sin_proxima'])}"]
        items += [f"{n}: {_signo(d)}" for n, d in sorted(comp["por_etapa"].items(), key=lambda kv: -abs(kv[1]))[:6]]
        partes += [_h2(f"Frente a la foto del {comp['fecha'][:10]}"), _lista(items)]
    if resultado.errores:
        partes += [_h2("Errores"), _lista(resultado.errores)]
    return _envoltorio(titulo, "".join(partes), cfg, resultado.modo)


# --------------------------------------------------------------------------- texto plano
def _texto(diag, resultado, cfg, titulo, kpis, grupos, por_grupo, propietarios, portal_id, lectura_ia):
    lineas = [titulo, " · ".join(f"{e}: {v}" for v, e in kpis), ""]
    if lectura_ia:
        lineas += [lectura_ia, ""]
    lineas += ["Hallazgos:"] + [f"- {h}" for h in diag["hallazgos"]]
    if diag["recomendaciones"]:
        lineas += ["", "Qué hacer esta semana:"] + [f"{i}. {r}" for i, r in enumerate(diag["recomendaciones"], 1)]
    if grupos:
        lineas += ["", "Estancados: " + " · ".join(f"{NOMBRES_CATEGORIA[c]} {len(g)}" for c, g in grupos.items())]
    lineas += ["", "El detalle por etapa y por negocio está en la versión HTML de este correo."]
    return "\n".join(lineas)


# --------------------------------------------------------------------------- guardar y enviar
def guardar(markdown, cfg, ahora_local, pipeline=""):
    os.makedirs(cfg.carpeta_informes, exist_ok=True)
    ruta = os.path.join(cfg.carpeta_informes, f"embudo_{_slug(pipeline or cfg.pipeline)}_{ahora_local:%Y-%m-%d}.md")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(os.path.join(cfg.carpeta_informes, "ultimo_informe.md"), "w", encoding="utf-8") as f:
        f.write(markdown)
    return ruta


def enviar(cfg, titulo, texto, html):
    """Envía el informe por correo. Devuelve 'enviado', 'sin_configurar' o 'error: ...'."""
    if not (cfg.gmail_configurado and cfg.notificar_a):
        return "sin_configurar"
    try:
        enviar_correo(cfg, cfg.notificar_a, f"📊 {titulo}", texto, html, cc=cfg.notificar_cc)
        return "enviado"
    except Exception as e:
        return f"error: {e}"
