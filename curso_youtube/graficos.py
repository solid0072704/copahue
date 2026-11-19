"""Gráficos SVG de las láminas, dibujados a partir de las cifras del caso."""

from __future__ import annotations

from . import caso

TINTA, TINTA3, ORO, VERDE, ROJO, AZUL, GRIS, LINEA = "#0B1B2B", "#1D3C5A", "#E3A72B", "#1B8553", "#C23A29", "#2A66CF", "#8792A0", "#D3CBB9"


def _num(v: float, signo: bool = False) -> str:
    s = caso.miles(abs(v))
    if v < 0:
        return "−" + s
    return ("+" + s) if signo else s


def cascada(barras: list[dict], ancho: int = 1660, alto: int = 684, anotaciones: list[dict] | None = None) -> str:
    """Gráfico de cascada. Cada barra: etq, valor, tipo ('total' | 'delta' | 'subtotal'), p (paso en que aparece)."""
    sup, inf, izq, der = 54, 150, 30, 30
    acum, tramos = 0.0, []
    for b in barras:
        if b["tipo"] == "delta":
            a, z = acum, acum + b["valor"]
            acum = z
        else:  # total o subtotal: desde cero hasta el acumulado
            if b["tipo"] == "total" and not tramos:
                acum = b["valor"]
            a, z = 0.0, acum
        tramos.append((a, z))
    vmax = max(0.0, max(max(t) for t in tramos))
    vmin = min(0.0, min(min(t) for t in tramos))
    plot_h = alto - sup - inf
    y = lambda v: sup + (vmax - v) / (vmax - vmin) * plot_h  # noqa: E731
    paso_x = (ancho - izq - der) / len(barras)
    w = paso_x * 0.62
    out = [f'<svg viewBox="0 0 {ancho} {alto}" xmlns="http://www.w3.org/2000/svg">']
    out.append(f'<line x1="{izq}" y1="{y(0):.1f}" x2="{ancho - der}" y2="{y(0):.1f}" stroke="{TINTA}" stroke-width="2.5"/>')
    for i, (b, (a, z)) in enumerate(zip(barras, tramos)):
        x = izq + i * paso_x + (paso_x - w) / 2
        y0, y1 = sorted((y(a), y(z)))
        h = max(y1 - y0, 4)
        if b["tipo"] == "delta":
            color = VERDE if b["valor"] >= 0 else ROJO
            texto = _num(b["valor"], signo=True)
        else:
            color = TINTA3 if z >= 0 else "#8E2A1D"
            texto = _num(z)
        arriba = z >= a if b["tipo"] == "delta" else z >= 0
        ty = (y0 - 14) if arriba else (y1 + 40)
        g = [f'<g data-p="{b["p"]}">']
        if i and barras[i]["tipo"] == "delta" or (i and b["tipo"] != "delta"):
            # conector desde el acumulado anterior
            prev = tramos[i - 1][1]
            x_prev = izq + (i - 1) * paso_x + (paso_x - w) / 2 + w
            g.append(f'<line x1="{x_prev:.1f}" y1="{y(prev):.1f}" x2="{x:.1f}" y2="{y(prev):.1f}" stroke="{GRIS}" '
                     f'stroke-width="2" stroke-dasharray="6 6"/>')
        g.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" rx="7" fill="{color}"/>')
        g.append(f'<text x="{x + w / 2:.1f}" y="{ty:.1f}" text-anchor="middle" font-size="31" font-weight="750" '
                 f'fill="{color}">{texto}</text>')
        for j, linea in enumerate(b["etq"].split("\n")):
            peso = "750" if b["tipo"] != "delta" else "550"
            g.append(f'<text x="{x + w / 2:.1f}" y="{alto - inf + 84 + j * 28:.1f}" text-anchor="middle" font-size="23" '
                     f'font-weight="{peso}" fill="{TINTA if b["tipo"] != "delta" else "#4E5C6B"}">{linea}</text>')
        g.append("</g>")
        out.append("".join(g))
    for an in anotaciones or []:
        out.append(f'<g data-p="{an["p"]}"><rect x="{an["x"]}" y="{an["y"]}" width="{an["w"]}" height="{an.get("h", 108)}" rx="16" '
                   f'fill="{an.get("fondo", "#FBF0D4")}" stroke="{an.get("borde", "#EFD89A")}" stroke-width="2"/>'
                   + "".join(f'<text x="{an["x"] + 26}" y="{an["y"] + 44 + k * 40}" font-size="{30 if k == 0 else 27}" '
                             f'font-weight="{750 if k == 0 else 500}" fill="{TINTA if k == 0 else "#4E5C6B"}">{t}</text>'
                             for k, t in enumerate(an["lineas"])) + "</g>")
    out.append("</svg>")
    return "".join(out)


def puente_caja() -> str:
    f = caso.flujo_2025()
    otros = f["var_otros_ac"] + f["var_otros_pc"]
    deuda = f["var_deuda_cp"] + f["var_deuda_lp"]
    barras = [
        dict(etq="Utilidad\nneta", valor=f["utilidad"], tipo="total", p=0),
        dict(etq="Depreciación", valor=f["depreciacion"], tipo="delta", p=1),
        dict(etq="Cuentas\npor cobrar", valor=f["var_cxc"], tipo="delta", p=2),
        dict(etq="Inventarios", valor=f["var_inventarios"], tipo="delta", p=3),
        dict(etq="Proveedores", valor=f["var_proveedores"], tipo="delta", p=4),
        dict(etq="Otros", valor=otros, tipo="delta", p=4),
        dict(etq="Flujo de la\noperación", valor=f["operacional"], tipo="subtotal", p=5),
        dict(etq="Inversión", valor=f["capex"], tipo="delta", p=6),
        dict(etq="Dividendos", valor=f["dividendos"], tipo="delta", p=6),
        dict(etq="Deuda\nbancaria", valor=deuda, tipo="delta", p=7),
        dict(etq="Variación\nde la caja", valor=f["var_neta"], tipo="subtotal", p=7),
    ]
    anot = [dict(p=5, x=1050, y=20, w=560, lineas=["Ganó 485 · la operación consumió 685",
                                                     f"En 2024 había generado {caso.flujo_2024()['operacional']}"])]
    return cascada(barras, anotaciones=anot)


def puente_margen() -> str:
    d = caso.desviaciones()
    barras = [
        dict(etq="Margen bruto\npresupuestado", valor=d["margen_bruto"]["ppto"], tipo="total", p=0),
        dict(etq="Efecto volumen\n(vendió más)", valor=round(d["efecto_volumen"]), tipo="delta", p=1),
        dict(etq="Efecto margen\n(vendió más barato)", valor=round(d["efecto_margen"]), tipo="delta", p=2),
        dict(etq="Margen bruto\nreal", valor=d["margen_bruto"]["real"], tipo="subtotal", p=3),
    ]
    return _cascada_truncada(barras, base=2700, ancho=1042, alto=684)


def _cascada_truncada(barras: list[dict], base: float, ancho: int, alto: int) -> str:
    """Cascada con el eje cortado en `base`, para que se vean variaciones pequeñas sobre totales grandes."""
    sup, inf, izq, der = 60, 132, 20, 20
    acum, tramos = 0.0, []
    for b in barras:
        if b["tipo"] == "delta":
            a, z = acum, acum + b["valor"]
            acum = z
        else:
            if not tramos:
                acum = b["valor"]
            a, z = base, acum
        tramos.append((a, z))
    vmax = max(max(t) for t in tramos) + 20
    plot_h = alto - sup - inf
    y = lambda v: sup + (vmax - v) / (vmax - base) * plot_h  # noqa: E731
    paso_x = (ancho - izq - der) / len(barras)
    w = paso_x * 0.6
    out = [f'<svg viewBox="0 0 {ancho} {alto}" xmlns="http://www.w3.org/2000/svg">']
    out.append(f'<line x1="{izq}" y1="{y(base):.1f}" x2="{ancho - der}" y2="{y(base):.1f}" stroke="{TINTA}" stroke-width="2.5"/>')
    for i, (b, (a, z)) in enumerate(zip(barras, tramos)):
        x = izq + i * paso_x + (paso_x - w) / 2
        y0, y1 = sorted((y(a), y(z)))
        h = max(y1 - y0, 4)
        if b["tipo"] == "delta":
            color, texto = (VERDE if b["valor"] >= 0 else ROJO), _num(b["valor"], signo=True)
        else:
            color, texto = TINTA3, _num(z)
        g = [f'<g data-p="{b["p"]}">']
        if i:
            prev = tramos[i - 1][1]
            x_prev = izq + (i - 1) * paso_x + (paso_x - w) / 2 + w
            g.append(f'<line x1="{x_prev:.1f}" y1="{y(prev):.1f}" x2="{x:.1f}" y2="{y(prev):.1f}" stroke="{GRIS}" '
                     f'stroke-width="2" stroke-dasharray="6 6"/>')
        g.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" rx="7" fill="{color}"/>')
        if b["tipo"] != "delta":  # marca de eje cortado
            g.append(f'<path d="M{x - 6:.1f} {y(base) - 26:.1f} l{w + 12:.1f} -14 v10 l-{w + 12:.1f} 14 z" fill="#FFFFFF"/>')
        g.append(f'<text x="{x + w / 2:.1f}" y="{y0 - 16:.1f}" text-anchor="middle" font-size="36" font-weight="750" '
                 f'fill="{color}">{texto}</text>')
        for j, linea in enumerate(b["etq"].split("\n")):
            g.append(f'<text x="{x + w / 2:.1f}" y="{alto - inf + 50 + j * 32:.1f}" text-anchor="middle" '
                     f'font-size="{26 if j == 0 else 23}" font-weight="{700 if j == 0 else 500}" '
                     f'fill="{TINTA if j == 0 else "#4E5C6B"}">{linea}</text>')
        g.append("</g>")
        out.append("".join(g))
    out.append("</svg>")
    return "".join(out)


def ciclo_caja() -> str:
    """Ciclo de conversión de efectivo de los dos años, como línea de tiempo en días."""
    ancho, alto = 1660, 684
    x0, escala = 170, 9.6  # px por día
    out = [f'<svg viewBox="0 0 {ancho} {alto}" xmlns="http://www.w3.org/2000/svg">']
    for d in range(0, 151, 30):
        x = x0 + d * escala
        out.append(f'<line x1="{x:.0f}" y1="34" x2="{x:.0f}" y2="{alto - 60}" stroke="#ECE6D9" stroke-width="2"/>'
                   f'<text x="{x:.0f}" y="{alto - 24}" text-anchor="middle" font-size="24" fill="{GRIS}" font-weight="600">'
                   f'{"día " + str(d) if d else "día 0"}</text>')

    def barra(x_ini, dias, yb, color, etiqueta, valor, p, color_texto="#FFFFFF", alto_b=86):
        x = x0 + x_ini * escala
        w = dias * escala
        return (f'<g data-p="{p}"><rect x="{x:.1f}" y="{yb}" width="{w - 5:.1f}" height="{alto_b}" rx="12" fill="{color}"/>'
                f'<text x="{x + 20:.1f}" y="{yb + 36}" font-size="25" font-weight="600" fill="{color_texto}" '
                f'opacity=".92">{etiqueta}</text>'
                f'<text x="{x + 20:.1f}" y="{yb + 72}" font-size="33" font-weight="750" fill="{color_texto}">{valor} días</text></g>')

    for i, anio in enumerate(("2024", "2025")):
        r = caso.ratios(anio)
        inv, cob, pag = round(r["dias_inventario"]), round(r["dias_cobro"]), round(r["dias_pago"])
        cic = round(r["ciclo_caja"])
        yb = 56 + i * 300
        out.append(f'<text x="24" y="{yb + 106}" font-size="46" font-weight="780" fill="{TINTA}">{anio}</text>')
        out.append(barra(0, inv, yb, AZUL, "Inventario", inv, 0))
        out.append(barra(inv, cob, yb, TINTA3, "Cobro a clientes", cob, 1))
        out.append(barra(0, pag, yb + 98, GRIS, "Pago a proveedores", pag, 2))
        out.append(barra(pag, inv + cob - pag, yb + 98, ORO, "Ciclo de conversión de efectivo", cic, 3, color_texto=TINTA))
    x_fin_24 = x0 + (55 + 50) * escala
    x_fin_25 = x0 + (68 + 72) * escala
    out.append(f'<g data-p="4"><line x1="{x_fin_24:.0f}" y1="262" x2="{x_fin_24:.0f}" y2="334" stroke="{ROJO}" stroke-width="3" stroke-dasharray="7 7"/>'
               f'<line x1="{x_fin_24:.0f}" y1="298" x2="{x_fin_25 - 8:.0f}" y2="298" stroke="{ROJO}" stroke-width="4"/>'
               f'<polygon points="{x_fin_25 - 8:.0f},298 {x_fin_25 - 26:.0f},288 {x_fin_25 - 26:.0f},308" fill="{ROJO}"/>'
               f'<text x="{(x_fin_24 + x_fin_25) / 2:.0f}" y="284" text-anchor="middle" font-size="30" font-weight="780" fill="{ROJO}">'
               f'+37 días que financiar</text></g>')
    out.append("</svg>")
    return "".join(out)


def dupont_html() -> str:
    x24, x25 = caso.ratios("2024"), caso.ratios("2025")

    def fila(anio, x, p, flechas=None):
        f = flechas or ["", "", "", ""]
        celdas = [
            ("Margen neto", caso.pct(x["margen_neto"]), f[0]),
            ("Rotación de activos", caso.miles(x["rotacion_activos"], 2), f[1]),
            ("Apalancamiento", caso.miles(x["multiplicador"], 2), f[2]),
        ]
        partes = [f'<div class="an">{anio}</div>']
        for i, (nombre, valor, fl) in enumerate(celdas):
            partes.append(f'<div class="fa"><span>{nombre}</span><b>{valor}{fl}</b></div>')
            partes.append(f'<i>{"×" if i < 2 else "="}</i>')
        partes.append(f'<div class="fa res"><span>Rentabilidad del patrimonio (ROE)</span><b>{caso.pct(x["roe"])}{f[3]}</b></div>')
        return f'<div class="dfila" data-p="{p}">{"".join(partes)}</div>'

    baja, sube = '<em class="neg">▼</em>', '<em class="amb">▲</em>'
    return ('<div class="dupont">' + fila("2024", x24, 0) + fila("2025", x25, 1, [baja, baja, sube, baja])
            + '<div class="dconc" data-p="2"><b>El negocio rinde menos.</b> La rentabilidad se sostiene porque la '
              'empresa usa más deuda: es una rentabilidad de peor calidad.</div></div>')


def proyeccion_caja() -> str:
    """Saldo de caja semanal a 13 semanas: sin medidas y con el plan."""
    pr = caso.proyeccion_13_semanas()
    ancho, alto = 1042, 684
    izq, der, sup, inf = 92, 30, 40, 70
    vmax, vmin = 280.0, -140.0
    x = lambda s: izq + s * (ancho - izq - der) / 13  # noqa: E731
    y = lambda v: sup + (vmax - v) / (vmax - vmin) * (alto - sup - inf)  # noqa: E731
    out = [f'<svg viewBox="0 0 {ancho} {alto}" xmlns="http://www.w3.org/2000/svg">']
    out.append(f'<rect x="{izq}" y="{y(0):.1f}" width="{ancho - izq - der}" height="{y(vmin) - y(0):.1f}" fill="#FBE5E1" opacity=".7"/>')
    for v in (-100, 0, 100, 200):
        out.append(f'<line x1="{izq}" y1="{y(v):.1f}" x2="{ancho - der}" y2="{y(v):.1f}" stroke="{TINTA if v == 0 else "#E4DDCF"}" '
                   f'stroke-width="{2.5 if v == 0 else 1.5}"/><text x="{izq - 14}" y="{y(v) + 9:.1f}" text-anchor="end" '
                   f'font-size="24" font-weight="600" fill="{GRIS}">{_num(v)}</text>')
    for s in range(1, 14):
        marca = s in caso.SEMANAS_SUELDOS
        out.append(f'<text x="{x(s):.1f}" y="{alto - 28}" text-anchor="middle" font-size="23" font-weight="{750 if marca else 500}" '
                   f'fill="{TINTA if marca else GRIS}">{s}</text>')
    out.append(f'<text x="{izq}" y="{alto - 28}" text-anchor="middle" font-size="23" font-weight="600" fill="{GRIS}">Semana</text>')

    def linea(serie, color, p):
        pts = " ".join(f"{x(s):.1f},{y(v):.1f}" for s, v in enumerate(serie))
        circ = "".join(f'<circle cx="{x(s):.1f}" cy="{y(v):.1f}" r="7" fill="{color}"/>' for s, v in enumerate(serie))
        return (f'<g data-p="{p}"><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="5" stroke-linejoin="round"/>'
                f'{circ}</g>')

    def leyenda(fila, color, p, texto):
        yl = sup + 6 + fila * 40
        return (f'<g data-p="{p}"><rect x="{izq + 150}" y="{yl - 20}" width="34" height="8" rx="4" fill="{color}"/>'
                f'<text x="{izq + 198}" y="{yl - 6}" font-size="26" font-weight="700" fill="{color}">{texto}</text></g>')

    out.append(linea(pr["base"], ROJO, 1))
    out.append(linea(pr["plan"], VERDE, 3))
    out.append(leyenda(0, ROJO, 1, f"Sin medidas: termina en {_num(round(pr['base'][-1]))}"))
    out.append(leyenda(1, VERDE, 3, f"Con el plan: termina en {_num(round(pr['plan'][-1]))}"))
    s_neg = pr["primera_negativa"]
    out.append(f'<g data-p="2"><line x1="{x(s_neg):.1f}" y1="{y(pr["base"][s_neg]) + 14:.1f}" x2="{x(s_neg):.1f}" y2="{y(vmin) - 34:.1f}" '
               f'stroke="{ROJO}" stroke-width="3" stroke-dasharray="6 6"/><text x="{x(s_neg):.1f}" y="{y(vmin) - 10:.1f}" '
               f'text-anchor="middle" font-size="25" font-weight="750" fill="{ROJO}">Semana {s_neg}: caja negativa</text></g>')
    out.append(f'<text x="{x(0) + 10:.1f}" y="{y(caso.balance("2025")["efectivo"]) + 40:.1f}" font-size="25" font-weight="750" '
               f'fill="{TINTA}">Hoy: 65</text>')
    out.append("</svg>")
    return "".join(out)
