"""Genera el HTML de cada lámina del curso a partir de su definición.

Cada lámina es un diccionario con un `tipo` y una lista de `pasos` (la narración).
Los elementos visuales llevan el número de paso en que aparecen (`p`) o se enfocan.
"""

from __future__ import annotations

import html
import math
import re

CURSO = "Análisis de estados financieros con Claude"
MARCA = "Rodrigo Barceló"
# Datos de contacto del autor: se muestran en el cierre del video y van en la descripción de YouTube
LINKEDIN_URL = "https://www.linkedin.com/in/rodrigobarcelo"
CORREO = "r.diaz.barcelo@gmail.com"

# Logo de LinkedIn (el recuadro con «in»): acompaña al nombre del autor para indicar dónde encontrarlo
LINKEDIN = ('<svg class="logo-in" viewBox="0 0 24 24"><rect x="2" y="2" width="20" height="20" rx="2" fill="#fff"/>'
            '<path fill="#0A66C2" d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 '
            '2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 '
            '7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 '
            '2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 '
            '24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>')

ICONOS = {
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "x": '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    "doc": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "barras": '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>',
    "sube": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "baja": '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
    "peso": '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "escudo": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "alerta": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "reloj": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "lupa": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "diana": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "carpeta": '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>',
    "candado": '<rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "persona": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "personas": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "mensaje": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "capas": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "lapiz": '<path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>',
    "rayo": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "duda": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "descarga": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "calendario": '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    "play": '<polygon points="5 3 19 12 5 21 5 3"/>',
    "tabla": '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/>',
    "imagen": '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>',
    "ojo": '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>',
    "maletin": '<rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
    "ciclo": '<polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/>',
    "camion": '<rect x="1" y="3" width="15" height="13"/><polygon points="16 8 20 8 23 11 23 16 16 16 16 8"/><circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/>',
    "porciento": '<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
    "caja": '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>',
    "tarjeta": '<rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/>',
    "flecha": '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
    "bandera": '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/>',
    "ajustes": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "ok": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "banco": '<line x1="3" y1="21" x2="21" y2="21"/><line x1="5" y1="10" x2="5" y2="18"/><line x1="10" y1="10" x2="10" y2="18"/><line x1="14" y1="10" x2="14" y2="18"/><line x1="19" y1="10" x2="19" y2="18"/><polygon points="12 3 21 8 3 8 12 3"/>',
    "enviar": '<line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/>',
    "correo": '<path d="M4 4h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"/><polyline points="22 6 12 13 2 6"/>',
    "libro": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
}

# El método de 8 pasos (número, verbo, tema, ícono)
METODO = [
    (1, "Prepara", "Datos y contexto", "carpeta"),
    (2, "Verifica", "Que todo cuadre", "ok"),
    (3, "Entiende", "Estructura y tendencia", "capas"),
    (4, "Mide", "Ratios", "porciento"),
    (5, "Sigue la caja", "Flujo de efectivo", "peso"),
    (6, "Compara", "Presupuesto", "diana"),
    (7, "Cuestiona", "Señales de alerta", "alerta"),
    (8, "Decide", "Plan e informe", "bandera"),
]


def ico(nombre: str) -> str:
    if nombre == "linkedin":
        return LINKEDIN
    return f'<svg class="ico" viewBox="0 0 24 24">{ICONOS[nombre]}</svg>'


def esc(t: str) -> str:
    return html.escape(str(t), quote=False)


def rico(t: str) -> str:
    """Marcado mínimo: **negrita**, ==resaltado==, {neg:texto} {pos:texto} {amb:texto} {tenue:texto}."""
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"==(.+?)==", r"<mark>\1</mark>", t)
    t = re.sub(r"\{(pos|neg|amb|tenue):(.+?)\}", r'<span class="\1">\2</span>', t)
    return t.replace("\n", "<br>")


def _p(d: dict) -> str:
    return f' data-p="{d["p"]}"' if d.get("p") is not None else ""


# --- Bloques comunes ------------------------------------------------------
def _cab(s: dict, ctx: dict) -> str:
    return f'<div class="cab"><div class="kicker">{esc(ctx["kicker"])}</div><h1>{rico(s["titulo"])}</h1></div>'


def _pie(ctx: dict) -> str:
    barras = []
    for i in range(ctx["n_modulos"]):
        if i < ctx["modulo"]:
            barras.append('<i class="hecho"></i>')
        elif i == ctx["modulo"]:
            barras.append(f'<i class="actual" style="--w:{ctx["avance"] * 100:.0f}%"></i>')
        else:
            barras.append("<i></i>")
    return (f'<div class="pie"><span class="curso">{esc(CURSO)}</span>'
            f'<div class="avance">{"".join(barras)}</div><span class="marca-h">{LINKEDIN}{esc(MARCA)}</span></div>')


def _notas(notas: list[dict]) -> str:
    if not notas:
        return ""
    out = []
    for n in notas:
        gr = f'<span class="gr {n.get("tono", "")}">{rico(n["gr"])}</span>' if n.get("gr") else ""
        d = f'<span>{rico(n["d"])}</span>' if n.get("d") else ""
        out.append(f'<div class="nt {n.get("clase", "")}"{_p(n)}>{gr}<b>{rico(n["t"])}</b>{d}</div>')
    return f'<div class="notas">{"".join(out)}</div>'


def _lateral(tarjetas: list[dict]) -> str:
    if not tarjetas:
        return ""
    out = []
    for t in tarjetas:
        partes = []
        if t.get("eti"):
            partes.append(f'<div class="eti">{esc(t["eti"])}</div>')
        if t.get("gr"):
            partes.append(f'<div class="gr {t.get("tono", "")}">{rico(t["gr"])}</div>')
        if t.get("frase"):
            partes.append(f'<div class="frase">{rico(t["frase"])}</div>')
        for parrafo in t.get("ps", []):
            partes.append(f"<p>{rico(parrafo)}</p>")
        out.append(f'<div class="tarjeta {t.get("clase", "")}"{_p(t)}>{"".join(partes)}</div>')
    return f'<div class="lateral" style="width:{tarjetas[0].get("ancho", 560)}px">{"".join(out)}</div>'


# --- Tipos de lámina ------------------------------------------------------
def _puntos(s: dict) -> str:
    estilo = s.get("estilo", "numerado")
    clases = ["puntos"] + (["dos-col"] if s.get("columnas") == 2 else []) + (["compacto"] if s.get("compacto") else [])
    lis = []
    for i, it in enumerate(s["items"], 1):
        if it.get("icono"):
            marca = f'<span class="marca ic">{ico(it["icono"])}</span>'
        elif estilo == "check":
            marca = f'<span class="marca ok">{ico("check")}</span>'
        elif estilo == "cruz":
            marca = f'<span class="marca no">{ico("x")}</span>'
        else:
            marca = f'<span class="marca">{i:02d}</span>'
        d = f'<span>{rico(it["d"])}</span>' if it.get("d") else ""
        lis.append(f'<li{_p(it)}>{marca}<div class="txt"><b>{rico(it["t"])}</b>{d}</div></li>')
    return f'<ul class="{" ".join(clases)}">{"".join(lis)}</ul>' + _lateral(s.get("lateral", []))


def _dos(s: dict) -> str:
    paneles = []
    for pn in s["paneles"]:
        lis = []
        for it in pn["items"]:
            d = f'<span>{rico(it["d"])}</span>' if it.get("d") else ""
            lis.append(f'<li{_p(it)}>{ico(it.get("icono", pn.get("icono_item", "check")))}'
                       f'<div class="txt"><b>{rico(it["t"])}</b>{d}</div></li>')
        paneles.append(f'<div class="panel {pn.get("clase", "")}"{_p(pn)}><div class="ph">{ico(pn["icono"])}'
                       f'{esc(pn["titulo"])}</div><ul>{"".join(lis)}</ul></div>')
    return f'<div class="dos">{"".join(paneles)}</div>'


def _prompt_texto(t: str) -> str:
    t = esc(t)
    t = re.sub(r"\[([^\]\n]+)\]", r'<span class="var">[\1]</span>', t)
    return re.sub(r"(?m)^([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ ]{3,})$", r'<span class="tit">\1</span>', t)


def _prompt(s: dict) -> str:
    cuerpo = []
    for seg in s["segs"]:
        texto, foco = (seg, None) if isinstance(seg, str) else seg
        attr = f' data-foco="{foco}"' if foco is not None else ""
        cuerpo.append(f'<span class="seg"{attr}>{_prompt_texto(texto)}</span>')
    return (f'<div class="prompt"><div class="ph"><span class="pill">PROMPT {s["numero"]}</span>'
            f'<span class="nom">{esc(s["nombre"])}</span></div><pre>{"".join(cuerpo)}</pre>'
            f'<div class="pb"><span class="adj">{ico("carpeta")}Proyecto · Distribuidora Alerce</span>'
            f'<span class="env">{ico("enviar")}</span></div></div>'
            + _notas(s.get("notas", [])))


def _celda(c, clase_col: str = "") -> str:
    clase = clase_col
    if isinstance(c, tuple):
        c, extra = c
        clase = f"{clase} {extra}".strip()
    if isinstance(c, dict):
        contenido = c["html"]
    else:
        contenido = rico(c)
    return f'<td class="{clase}">{contenido}</td>' if clase else f"<td>{contenido}</td>"


def _tabla(s: dict) -> str:
    clases_col = s.get("clases_col", [""] * len(s["columnas"]))
    ths = "".join(f"<th>{rico(c)}</th>" for c in s["columnas"])
    trs = []
    for f in s["filas"]:
        attr = f' data-foco="{f["p"]}"' if f.get("p") is not None else ""
        if f.get("aparece") is not None:
            attr += f' data-p="{f["aparece"]}"'
        tds = "".join(_celda(c, clases_col[i] if i < len(clases_col) else "") for i, c in enumerate(f["c"]))
        trs.append(f'<tr class="{f.get("cls", "")}"{attr}>{tds}</tr>')
    estilo = f' style="flex:{s["flex"]}"' if s.get("flex") else ""
    variante = " ".join(v for v in ("compacta", "ratios", "texto") if s.get(v))
    caja = (f'<div class="tabla-caja"{estilo}><table class="tabla {variante}">'
            f'<thead><tr>{ths}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>')
    if s.get("previo"):
        pv = s["previo"]
        caja = (f'<div class="columna" style="gap:22px;flex:{s.get("flex", 1.9)}"><div class="mini-prompt"{_p(pv)}><span class="pill">{esc(pv.get("pill", "PROMPT"))}</span>'
                f'<span>{rico(pv["t"])}</span></div>{caja}</div>')
    return caja + _notas(s.get("notas", []))


def _respuesta(s: dict) -> str:
    bloques = []
    for b in s["bloques"]:
        h = f'<h3>{esc(b["h"])}</h3>' if b.get("h") else ""
        bloques.append(f'<div class="bl"{_p(b)}>{h}{b["html"]}</div>')
    etiqueta = s.get("etiqueta", "Respuesta de Claude")
    return (f'<div class="resp {"ancha" if not s.get("notas") else ""}"><div class="ph">{ico("mensaje")}{esc(etiqueta)}'
            f'<em>{esc(s.get("fuente", "ejemplo ilustrativo · datos del caso"))}</em></div>'
            f'<div class="rb">{"".join(bloques)}</div></div>' + _notas(s.get("notas", [])))


def _cifras(s: dict) -> str:
    tiles = []
    for t in s["tiles"]:
        antes = f'<div class="antes">{rico(t["antes"])}</div>' if t.get("antes") else ""
        sub = f'<div class="sub">{rico(t["sub"])}</div>' if t.get("sub") else ""
        tiles.append(f'<div class="cifra {t.get("tono", "")}"{_p(t)}><div class="eti">{esc(t["eti"])}</div>'
                     f'{antes}<div class="val">{t["val"]}</div>{sub}</div>')
    pie = ""
    if s.get("pie"):
        pie = f'<div class="frase-pie"{_p(s["pie"])}>{rico(s["pie"]["t"])}</div>'
    if s.get("sub_pie"):
        icono = ico(s["sub_pie"]["icono"]) if s["sub_pie"].get("icono") else ""
        pie += f'<div class="sub-pie"{_p(s["sub_pie"])}>{icono}{rico(s["sub_pie"]["t"])}</div>'
    sobre = ""
    if s.get("sobre"):
        sobre = f'<div class="sobre-cifras"{_p(s["sobre"])}>{rico(s["sobre"]["t"])}</div>'
    return (f'<div class="columna">{sobre}<div class="cifras c{s.get("cols", len(s["tiles"]))}" style="flex:none">'
            f'{"".join(tiles)}</div>{pie}</div>')


def _svg(s: dict) -> str:
    return f'<div class="grafico {"ancho" if not s.get("notas") else ""}">{s["svg"]}</div>' + _notas(s.get("notas", []))


def _metodo(s: dict) -> str:
    p0 = s.get("p0")
    out = []
    for i, (n, verbo, tema, icono) in enumerate(METODO):
        attr = f' data-p="{p0 + i}"' if p0 is not None else ""
        if s.get("foco0") is not None:
            attr += f' data-foco="{s["foco0"] + i}"'
        out.append(f'<div class="mp"{attr}><div class="n"><span>Paso {n}</span>{ico(icono)}</div>'
                   f'<b>{esc(verbo)}</b><span>{esc(tema)}</span></div>')
    return f'<div class="metodo">{"".join(out)}</div>'


def _alertas(s: dict) -> str:
    nombres = {"cr": "Crítica", "im": "Importante", "ob": "A observar"}
    out = []
    for a in s["items"]:
        out.append(f'<div class="al {a["nivel"]}"{_p(a)}><div class="niv">{nombres[a["nivel"]]}</div>'
                   f'<div class="at"><b>{rico(a["t"])}</b><span>{rico(a["d"])}</span></div></div>')
    return f'<div class="alertas">{"".join(out)}</div>'


def _memo(s: dict) -> str:
    secs = "".join(f'<div class="ms"{_p(x)}><h3>{esc(x["h"])}</h3><div>{x["html"]}</div></div>' for x in s["secciones"])
    return (f'<div class="memo"><div class="mh"><b>{esc(s["encabezado"])}</b><span>{esc(s["fecha"])}</span></div>'
            f'{secs}</div>')


def _html_libre(s: dict) -> str:
    return s["html"] + _notas(s.get("notas", []))


# --- Láminas a pantalla completa (tema oscuro) -----------------------------
_BOCETOS = {
    "balance": '<svg viewBox="0 0 316 150"><rect x="0" y="20" width="150" height="130" rx="8" fill="#2A66CF" opacity=".85"/><rect x="166" y="20" width="150" height="62" rx="8" fill="#E3A72B"/><rect x="166" y="90" width="150" height="60" rx="8" fill="#5B7895"/></svg>',
    "resultados": '<svg viewBox="0 0 316 150"><rect x="0" y="18" width="52" height="132" rx="6" fill="#5B7895"/><rect x="66" y="44" width="52" height="106" rx="6" fill="#5B7895"/><rect x="132" y="70" width="52" height="80" rx="6" fill="#5B7895"/><rect x="198" y="96" width="52" height="54" rx="6" fill="#5B7895"/><rect x="264" y="114" width="52" height="36" rx="6" fill="#E3A72B"/></svg>',
    "flujo": '<svg viewBox="0 0 316 150"><polyline points="0,40 60,58 120,36 180,84 240,110 316,132" fill="none" stroke="#E3A72B" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/><line x1="0" y1="148" x2="316" y2="148" stroke="#5B7895" stroke-width="3"/></svg>',
}


def _gancho(s: dict) -> str:
    docs = "".join(f'<div class="doc"><div class="dt">{esc(t)}</div>{_BOCETOS[b]}</div>' for t, b in s["docs"])
    remate = " <span class=\"ne\">≠</span> ".join(f"<b>{esc(x)}</b>" for x in s["remate"])
    return (f'<div class="centro"><div class="docs"{_p({"p": s.get("p_docs", 0)})}>{docs}</div>'
            f'<div class="pregunta"{_p({"p": s["p_pregunta"]})}>{s["pregunta"]}</div>'
            f'<div class="remate"{_p({"p": s["p_remate"]})}>{remate}</div></div>')


def _titulo_curso(s: dict) -> str:
    return (f'<div class="centro titulo-curso"><div class="sobre">{esc(s["sobre"])}</div><h2>{s["titulo_html"]}</h2>'
            f'<div class="sub">{esc(s["sub"])}</div><div class="quien">{LINKEDIN}{esc(s["quien"])}</div></div>')


def _chips(s: dict) -> str:
    chips = "".join(f'<div class="chip"{_p(c)}>{ico(c["icono"])}<b>{esc(c["t"])}</b><span>{esc(c["d"])}</span></div>'
                    for c in s["chips"])
    return f'<div class="centro"><div class="chips-tit">{s["titulo_html"]}</div><div class="chips">{chips}</div></div>'


def _seccion(s: dict, ctx: dict) -> str:
    lista = []
    for n, verbo, tema, _ in METODO:
        actual = s.get("paso")
        estado = "actual" if n == actual else ("hecho" if actual and n < actual else "")
        marca = ico("check") if estado == "hecho" else str(n)
        lista.append(f'<div class="it {estado}"><i>{marca}</i>{esc(verbo)} · {esc(tema)}</div>')
    return (f'<div class="seccion"><div class="iz"><div class="num">{esc(s["etiqueta"])}</div>'
            f'<h2>{rico(s["nombre"])}</h2><div class="baj"{_p({"p": s.get("p_bajada", 0)})}>{rico(s["bajada"])}</div></div>'
            f'<div class="de">{"".join(lista)}</div></div>')


def _cierre(s: dict) -> str:
    ops = "".join(f'<div class="op"{_p(o)}><div class="n">{esc(o["n"])}</div>{ico(o["icono"])}<b>{esc(o["t"])}</b>'
                  f'<span>{rico(o["d"])}</span></div>' for o in s["opciones"])
    gracias = f'<div class="gracias"{_p(s["gracias"])}>{s["gracias"]["html"]}</div>' if s.get("gracias") else ""
    asesoria = ""
    if s.get("asesoria"):            # mensaje de asesoría con los datos de contacto del autor
        a = s["asesoria"]
        contactos = (f'<span>{ico("linkedin")}{esc(LINKEDIN_URL.split("://www.")[-1])}</span>'
                     f'<span>{ico("correo")}{esc(CORREO)}</span>')
        asesoria = (f'<div class="asesoria"{_p(a)}><div class="as-t"><div class="as-k">{esc(a["k"])}</div>'
                    f'<b>{rico(a["t"])}</b><span>{rico(a["d"])}</span></div><div class="as-c">{contactos}</div></div>')
    return (f'<div class="cab"><div class="kicker">{esc(s["kicker"])}</div><h1>{rico(s["titulo"])}</h1></div>'
            f'<div class="cuerpo cierre"><div class="opciones">{ops}</div>{asesoria}</div>{gracias}')


_CON_CABECERA = {"puntos": _puntos, "dos": _dos, "prompt": _prompt, "tabla": _tabla, "respuesta": _respuesta,
                 "cifras": _cifras, "svg": _svg, "metodo": _metodo, "alertas": _alertas, "memo": _memo,
                 "html": _html_libre}
_OSCURAS = {"gancho", "titulo_curso", "chips", "seccion", "cierre"}
# Imagen de fondo de cada lámina de portada (recortes de las portadas de las presentaciones del autor en Canva).
# Los inicios de capítulo las van alternando.
_HEROES = {"gancho": "gestion", "titulo_curso": "datos", "chips": "automatizacion", "seccion": None, "cierre": "transformacion"}
_HEROES_DE_CAPITULO = ["datos", "transformacion", "automatizacion", "gestion"]

_JS = """
function irPaso(k){
  document.body.dataset.paso = k;
  document.querySelectorAll('[data-p]').forEach(function(el){
    var p = +el.dataset.p; el.classList.toggle('visto', p <= k); el.classList.toggle('actual', p === k);
  });
  document.querySelectorAll('[data-foco]').forEach(function(el){
    var ps = el.dataset.foco.split(',').map(Number);
    el.classList.toggle('foco', ps.indexOf(k) >= 0); el.classList.toggle('pasado', Math.max.apply(null, ps) < k);
  });
}
function ajustar(){
  var pre = document.querySelector('.prompt pre');
  if(pre){ var caja = pre.parentElement, t = 27;
    while((pre.scrollHeight > pre.clientHeight + 1 || caja.scrollHeight > caja.clientHeight + 1) && t > 20){ t -= 0.5; pre.style.fontSize = t + 'px'; } }
}
function revisar(){
  var L = document.querySelector('.lienzo').getBoundingClientRect(), fuera = [];
  var zona = document.querySelector('.cuerpo');
  var Z = zona ? zona.getBoundingClientRect() : L;
  document.querySelectorAll('.cuerpo *, .cab *, .centro *, .seccion *').forEach(function(el){
    if(el.closest('svg') && el.tagName.toLowerCase() !== 'svg') return;
    if(el.closest('.ondas')) return;
    if(el.closest('.pl')) return;
    var r = el.getBoundingClientRect(); if(r.width === 0 || r.height === 0) return;
    var enCuerpo = !!el.closest('.cuerpo');
    var lim = enCuerpo ? {l: Z.left - 2, t: Z.top - 2, r: Z.right + 2, b: Z.bottom + 2} : {l: 40, t: 30, r: 1880, b: 1000};
    if(r.left < lim.l || r.top < lim.t || r.right > lim.r || r.bottom > lim.b)
      fuera.push(el.tagName.toLowerCase() + '.' + (el.className.baseVal !== undefined ? el.className.baseVal : el.className) + ' [' + Math.round(r.left) + ',' + Math.round(r.top) + ',' + Math.round(r.right) + ',' + Math.round(r.bottom) + ']');
    if(el.scrollWidth > el.clientWidth + 2 && getComputedStyle(el).overflow !== 'visible')
      fuera.push('desborde-h ' + el.tagName.toLowerCase() + '.' + el.className);
  });
  var h1 = document.querySelector('.cab h1');
  if(h1 && h1.getBoundingClientRect().height > 80) fuera.push('TITULO-EN-DOS-LINEAS');
  document.querySelectorAll('.tabla-caja').forEach(function(c){ var t = c.querySelector('table');
    if(t && t.getBoundingClientRect().width > c.clientWidth + 1) fuera.push('TABLA-MAS-ANCHA-QUE-SU-CAJA ' + Math.round(t.getBoundingClientRect().width) + '>' + c.clientWidth); });
  return fuera.slice(0, 8);
}
ajustar(); irPaso(0);
"""


def _ondas(modulo: int) -> str:
    """Cinta de líneas finas que cruza el fondo de las láminas de contenido, como en las portadas.

    Es la misma en todas las láminas de un capítulo (el fondo no se mueve al pasar de una a otra) y cambia
    levemente de un capítulo al siguiente.
    """
    n, fase = 22, modulo * 0.83
    lineas = []
    for i in range(n):
        puntos = []
        for j in range(97):
            u = j / 96
            centro = 1018 - 380 * u ** 1.9
            ancho = (i - (n - 1) / 2) * (2.2 + 19 * u)
            y = centro + ancho * math.sin(1.9 * math.pi * u + fase + i * 0.075)
            puntos.append(f"{'M' if j == 0 else 'L'}{1920 * u:.0f} {y:.1f}")
        lineas.append(f'<path d="{" ".join(puntos)}"/>')
    return ('<svg class="ondas" viewBox="0 0 1920 1080" aria-hidden="true"><defs>'
            '<linearGradient id="ondas-g" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="1920" y2="0">'
            '<stop offset=".2" stop-color="#E3A62F" stop-opacity="0"/><stop offset=".62" stop-color="#E3A62F" stop-opacity=".5"/>'
            '<stop offset="1" stop-color="#F2D9A0" stop-opacity="1"/></linearGradient></defs>'
            f'<g fill="none" stroke="url(#ondas-g)" stroke-width="1.15">{"".join(lineas)}</g></svg>')


def pagina(s: dict, ctx: dict, css_href: str = "../../estilo.css") -> str:
    """HTML completo de una lámina. `ctx`: kicker, modulo, n_modulos, avance (0 a 1)."""
    tipo = s["tipo"]
    if tipo in _CON_CABECERA:
        alto = " alto" if s.get("alto") else ""
        bajo, estilo = "", ""
        if s.get("pie") and tipo != "cifras":
            bajo = f'<div class="bajo"><div class="frase-pie"{_p(s["pie"])}>{rico(s["pie"]["t"])}</div></div>'
            estilo = ' style="bottom:236px"'
        contenido = _CON_CABECERA[tipo](s)
        if s.get("previo") and tipo != "tabla":
            pv = s["previo"]
            contenido = (f'<div class="columna" style="gap:22px"><div class="mini-prompt"{_p(pv)}>'
                         f'<span class="pill">{esc(pv.get("pill", "PROMPT"))}</span><span>{rico(pv["t"])}</span></div>'
                         f'<div class="fila">{contenido}</div></div>')
        cuerpo = _cab(s, ctx) + f'<div class="cuerpo{alto}"{estilo}>{contenido}</div>{bajo}'
    elif tipo == "gancho":
        cuerpo = _gancho(s)
    elif tipo == "titulo_curso":
        cuerpo = _titulo_curso(s)
    elif tipo == "chips":
        cuerpo = _chips(s)
    elif tipo == "seccion":
        cuerpo = _seccion(s, ctx)
    elif tipo == "cierre":
        cuerpo = _cierre(dict(s, kicker=ctx["kicker"]))
    else:
        raise ValueError(f"Tipo de lámina desconocido: {tipo}")
    clase = "oscuro" if tipo in _OSCURAS else "claro"
    base = css_href.rsplit("/", 1)[0] if "/" in css_href else "."
    heroe = _ondas(ctx["modulo"])
    if tipo in _HEROES:
        nombre = _HEROES[tipo] or _HEROES_DE_CAPITULO[ctx["modulo"] % len(_HEROES_DE_CAPITULO)]
        fondo = " fondo" if tipo == "cierre" else ""
        heroe = f'<div class="heroe{fondo}"><i style="background-image:url(\'{base}/imagenes/heroe_{nombre}.jpg\')"></i></div>'
    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8"><title>{esc(s.get("id", ""))}</title>'
            f'<link rel="stylesheet" href="{css_href}"></head><body class="{clase} t-{tipo}">'
            f'<div class="lienzo">{heroe}{cuerpo}{_pie(ctx)}</div><script>{_JS}</script>'
            f'<script src="{base}/movimiento.js"></script></body></html>')
