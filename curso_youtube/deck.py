"""Arma el curso completo: módulos -> láminas -> pasos de narración -> segmentos de audio.

Convenciones del contenido (carpeta `contenido/`):
- Cada lámina tiene `pasos`: la narración, un texto por cada cambio visual.
- Un paso que empieza con «+ » continúa el párrafo anterior (se une con un espacio);
  los demás empiezan párrafo nuevo, lo que deja una pausa natural en la voz.
- `une=True` hace que la lámina comparta el archivo de audio con la lámina anterior.

Uso:  python -m curso_youtube.deck      # resumen de duración por módulo
"""

from __future__ import annotations

from importlib import import_module

from .laminas import METODO
from .voz import CARACTERES_POR_SEGUNDO, a_voz

MODULOS = [
    dict(clave="m00_intro", capitulo="Introducción", kicker="Introducción"),
    dict(clave="m01_claude", capitulo="Qué puede y qué no puede hacer Claude", kicker="Antes de empezar"),
    dict(clave="m02_prepara", capitulo="Paso 1: preparar los datos y el contexto", paso=1),
    dict(clave="m03_verifica", capitulo="Paso 2: verificar que todo cuadre", paso=2),
    dict(clave="m04_entiende", capitulo="Paso 3: análisis vertical y horizontal", paso=3),
    dict(clave="m05_mide", capitulo="Paso 4: ratios financieros", paso=4),
    dict(clave="m06_caja", capitulo="Paso 5: flujo de efectivo y capital de trabajo", paso=5),
    dict(clave="m07_compara", capitulo="Paso 6: desviaciones contra el presupuesto", paso=6),
    dict(clave="m08_cuestiona", capitulo="Paso 7: señales de alerta", paso=7),
    dict(clave="m09_decide", capitulo="Paso 8: escenarios, plan de acción e informe", paso=8),
    dict(clave="m10_rutina", capitulo="Tu rutina mensual y los errores más comunes", kicker="Tu rutina mensual"),
    dict(clave="m11_cierre", capitulo="Resumen y próximos pasos", kicker="Cierre"),
]

PAUSA_ENTRE_SEGMENTOS = 0.55  # segundos de silencio entre un audio y el siguiente


def _kicker(m: dict) -> str:
    if m.get("paso"):
        n, verbo, tema, _ = METODO[m["paso"] - 1]
        return f"Paso {n} · {verbo}: {tema}"
    return m["kicker"]


def _paso(texto: str) -> tuple[str, bool]:
    """Devuelve (texto sin marca, continúa_párrafo)."""
    if texto.startswith("+ "):
        return texto[2:], True
    return texto, False


def cargar() -> list[dict]:
    """Lista de láminas con id, módulo, kicker, avance y segmento de audio asignados."""
    laminas, seg = [], -1
    for mi, m in enumerate(MODULOS):
        try:
            crudas = import_module(f"{__package__}.contenido.{m['clave']}").LAMINAS
        except ModuleNotFoundError:
            continue
        for si, s in enumerate(crudas):
            s = dict(s)
            if not s.get("une") or seg < 0:
                seg += 1
            s.update(id=f"m{mi:02d}_s{si + 1:02d}", modulo=mi, kicker=_kicker(m), n_modulos=len(MODULOS),
                     avance=(si + 1) / len(crudas), seg=seg, primera_del_modulo=(si == 0))
            assert s.get("pasos"), f"La lámina {s['id']} no tiene narración"
            laminas.append(s)
    return laminas


def segmentos(laminas: list[dict] | None = None) -> list[dict]:
    """Segmentos de audio: texto en pantalla, texto para la voz y pasos con su peso para sincronizar."""
    laminas = laminas or cargar()
    segs: dict[int, dict] = {}
    for s in laminas:
        g = segs.setdefault(s["seg"], dict(id=f"seg_{s['seg']:03d}", n=s["seg"], laminas=[], pasos=[], texto=""))
        g["laminas"].append(s["id"])
        for k, crudo in enumerate(s["pasos"]):
            texto, continua = _paso(crudo)
            if g["texto"]:
                g["texto"] += (" " if continua else "\n\n")
            g["texto"] += texto
            # peso: largo del texto hablado más la pausa que lo precede (más larga si empieza párrafo)
            g["pasos"].append(dict(lamina=s["id"], paso=k, texto=texto, continua=continua,
                                   peso=len(a_voz(texto)) + (4 if continua else 12)))
    out = []
    for n in sorted(segs):
        g = segs[n]
        g["voz"] = "\n\n".join(a_voz(par) for par in g["texto"].split("\n\n"))
        g["caracteres"] = len(g["voz"])
        g["segundos_estimados"] = g["caracteres"] / CARACTERES_POR_SEGUNDO
        out.append(g)
    return out


def resumen() -> str:
    laminas = cargar()
    segs = segmentos(laminas)
    por_seg = {g["n"]: g for g in segs}
    filas, total_car, total_seg, total_lam, total_pasos = [], 0, 0.0, 0, 0
    for mi, m in enumerate(MODULOS):
        lams = [s for s in laminas if s["modulo"] == mi]
        if not lams:
            continue
        ns = sorted({s["seg"] for s in lams})
        car = sum(por_seg[n]["caracteres"] for n in ns)
        dur = car / CARACTERES_POR_SEGUNDO + PAUSA_ENTRE_SEGMENTOS * len(ns)
        pasos = sum(len(s["pasos"]) for s in lams)
        filas.append(f"{m['capitulo']:<52}{len(lams):>4}{pasos:>6}{car:>8}{dur / 60:>7.1f} min")
        total_car += car
        total_seg += dur
        total_lam += len(lams)
        total_pasos += pasos
    cab = f"{'Módulo':<52}{'lám':>4}{'pasos':>6}{'caract':>8}{'dur':>11}"
    pie = f"{'TOTAL':<52}{total_lam:>4}{total_pasos:>6}{total_car:>8}{total_seg / 60:>7.1f} min   ({len(segs)} audios)"
    return "\n".join([cab, *filas, pie])


if __name__ == "__main__":
    print(resumen())
