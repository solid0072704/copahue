"""Ayudas para escribir el contenido: formato de cifras del caso y pequeños bloques HTML."""

from __future__ import annotations

from .. import caso
from ..laminas import esc, ico, rico

MENOS = "−"


def m(n: float, dec: int = 0) -> str:
    """Cifra con formato chileno y signo menos tipográfico."""
    return caso.miles(n, dec).replace("-", MENOS)


def con_signo(n: float, dec: int = 0) -> str:
    return ("+" if n > 0 else "") + m(n, dec)


def pc(x: float, dec: int = 1) -> str:
    return caso.pct(x, dec).replace("-", MENOS)


def var(x: float, dec: int = 1) -> str:
    return caso.pct(x, dec, signo=True).replace("-", MENOS)


def ratio(nombre: str, formula: str) -> dict:
    """Celda con el nombre del ratio y su fórmula debajo."""
    return {"html": f'{esc(nombre)}<span class="formula">{esc(formula)}</span>'}


def semaforo(color: str) -> dict:
    """Celda con un semáforo: 'v' verde, 'a' amarillo, 'r' rojo."""
    return {"html": f'<span class="semaforo {color}"></span>'}


def etiqueta(favorable: bool) -> dict:
    return {"html": f'<span class="etq {"fav" if favorable else "des"}">{"Favorable" if favorable else "Desfavorable"}</span>'}


def rank(filas: list[dict]) -> str:
    """Lista ordenada por impacto: n, t, d, v (valor), unidad, p, pos (bool)."""
    out = []
    for f in filas:
        unidad = f'<small>{esc(f["unidad"])}</small>' if f.get("unidad") else ""
        out.append(f'<div class="rk" data-p="{f["p"]}"><div class="n">{f["n"]}</div><div><b>{rico(f["t"])}</b>'
                   f'<span>{rico(f["d"])}</span></div><div class="v {"pos" if f.get("pos") else ""}">{f["v"]}{unidad}</div></div>')
    return f'<div class="rank">{"".join(out)}</div>'


def bloques(items: list[dict]) -> str:
    """Bloques numerados en fila: n, t, d, p."""
    out = [f'<div class="bq" data-p="{b["p"]}"><div class="n">{esc(b["n"])}</div><b>{rico(b["t"])}</b>'
           f'<span>{rico(b["d"])}</span></div>' for b in items]
    return f'<div class="bloques">{"".join(out)}</div>'


def versus(mal: str, bien: str, p: int) -> str:
    return (f'<div class="versus" data-p="{p}"><div class="vs mal">{ico("x")}<span>{mal}</span></div>'
            f'<div class="vs bien">{ico("check")}<span>{bien}</span></div></div>')


def formato(filas: list[dict]) -> str:
    """Ranking de formatos de archivo: estrellas (1 a 3), t, d, p."""
    out = []
    for f in filas:
        estrellas = "★" * f["estrellas"] + f'<s>{"★" * (3 - f["estrellas"])}</s>'
        out.append(f'<div class="fm" data-p="{f["p"]}"><div class="st">{estrellas}</div><div><b>{rico(f["t"])}</b>'
                   f'<span>{rico(f["d"])}</span></div></div>')
    return f'<div class="formato">{"".join(out)}</div>'


def proyecto(titulo: str, tarjetas: list[dict]) -> str:
    """Diagrama de un Proyecto de Claude: tarjetas con h, p y `html` o `items` [(icono, texto)]."""
    out = []
    for t in tarjetas:
        if t.get("items"):
            cuerpo = "<ul>" + "".join(f"<li>{ico(i)}{esc(x)}</li>" for i, x in t["items"]) + "</ul>"
        else:
            cuerpo = f'<p>{rico(t["t"])}</p>'
        out.append(f'<div class="pc" data-p="{t["p"]}"><h4>{esc(t["h"])}</h4>{cuerpo}</div>')
    return f'<div class="proyecto"><div class="pt">{ico("carpeta")}{esc(titulo)}</div>{"".join(out)}</div>'
