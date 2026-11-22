"""Las evaluaciones: armar un intento, corregirlo y explicar el resultado.

Todo ocurre en el servidor. La página de la prueba nunca recibe las respuestas correctas: recibe las preguntas
elegidas y sus alternativas barajadas, y las explicaciones llegan recién al entregar.
"""

from __future__ import annotations

import random
import re

from django.utils import timezone

from .models import Evaluacion, Intento, Pregunta


def disponible(evaluacion: Evaluacion) -> bool:
    return evaluacion.preguntas.filter(activa=True).exists()


def armar(evaluacion: Evaluacion, usuario) -> Intento:
    """Un intento nuevo: preguntas tomadas al azar del banco (el diagnóstico las lleva todas) y alternativas barajadas."""
    banco = list(evaluacion.preguntas.filter(activa=True))
    elegidas = banco if evaluacion.tipo == "diagnostico" else random.sample(banco, min(evaluacion.preguntas_por_intento, len(banco)))
    items = []
    for q in elegidas:
        orden = list(range(len(q.opciones)))
        if q.tipo == "orden":
            while orden == sorted(orden):
                random.shuffle(orden)
        elif q.tipo != "numero":
            random.shuffle(orden)
        items.append(dict(id=q.id, orden=orden))
    return Intento.objects.create(usuario=usuario, evaluacion=evaluacion, preguntas=items, total=len(items))


def para_mostrar(intento: Intento) -> list[dict]:
    """Las preguntas del intento como las ve el alumno: sin respuestas, con las alternativas en el orden que le tocó."""
    por_id = Pregunta.objects.in_bulk([x["id"] for x in intento.preguntas])
    return [dict(n=n, pregunta=por_id[x["id"]], campo=f'p{x["id"]}', opciones=[(k, por_id[x["id"]].opciones[k]) for k in x["orden"]],
                 orden=",".join(str(k) for k in x["orden"]))
            for n, x in enumerate(intento.preguntas, 1) if x["id"] in por_id]


def numero(texto: str) -> float | None:
    """«18», «18,5 %», «1.234,5» o «1234.5» como número; None si no lo es."""
    t = re.sub(r"[^\d,.\-]", "", texto or "")
    if "," in t:
        t = t.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", t):
        t = t.replace(".", "")
    try:
        return float(t)
    except ValueError:
        return None


def leer(q: Pregunta, campo: str, post):
    """Lo que respondió el alumno a una pregunta, tal como llega del formulario."""
    if q.tipo == "unica":
        v = post.get(campo, "")
        return int(v) if v.isdigit() else None
    if q.tipo == "multiple":
        return sorted({int(v) for v in post.getlist(campo) if v.isdigit()})
    if q.tipo == "numero":
        return post.get(campo, "").strip()[:40]
    return [int(v) for v in post.get(campo, "").split(",") if v.strip().isdigit()]


def correcta(q: Pregunta, valor) -> bool:
    if q.tipo == "unica":
        return valor is not None and valor == q.respuesta
    if q.tipo == "multiple":
        return bool(valor) and sorted(valor) == sorted(q.respuesta)
    if q.tipo == "numero":
        n = numero(valor or "")
        return n is not None and abs(n - float(q.respuesta)) <= q.tolerancia + 1e-9
    return valor == list(range(len(q.opciones)))


def corregir(intento: Intento, post) -> Intento:
    """Corrige y guarda el intento. Entregarlo dos veces no cambia nada."""
    if intento.entregado:
        return intento
    por_id = Pregunta.objects.in_bulk([x["id"] for x in intento.preguntas])
    respuestas = {str(x["id"]): leer(por_id[x["id"]], f'p{x["id"]}', post) for x in intento.preguntas if x["id"] in por_id}
    intento.respuestas = respuestas
    intento.buenas = sum(correcta(por_id[int(k)], v) for k, v in respuestas.items())
    intento.total = len(respuestas)
    intento.nota = round(100 * intento.buenas / intento.total) if intento.total else 0
    intento.aprobado = intento.evaluacion.lleva_nota and intento.nota >= intento.evaluacion.aprobacion
    intento.entregado = timezone.now()
    intento.save()
    return intento


def _texto(q: Pregunta, valor) -> str:
    if valor in (None, "", []):
        return "Sin respuesta"
    if q.tipo == "unica":
        return q.opciones[valor] if 0 <= valor < len(q.opciones) else "Sin respuesta"
    if q.tipo == "multiple":
        return " · ".join(q.opciones[k] for k in valor if 0 <= k < len(q.opciones))
    if q.tipo == "numero":
        return f"{valor} {q.unidad}".strip()
    return "  ".join(f"{i}) {q.opciones[k]}" for i, k in enumerate(valor, 1) if 0 <= k < len(q.opciones))


def revision(intento: Intento) -> list[dict]:
    """Pregunta por pregunta: qué respondió el alumno, cuál era la correcta y por qué."""
    por_id = Pregunta.objects.select_related("leccion__modulo", "tema").in_bulk([x["id"] for x in intento.preguntas])
    out = []
    for n, x in enumerate(intento.preguntas, 1):
        q = por_id.get(x["id"])
        if not q:
            continue
        valor = intento.respuestas.get(str(q.id))
        buena = q.respuesta if q.tipo != "orden" else list(range(len(q.opciones)))
        buena = str(buena).replace(".", ",") if q.tipo == "numero" else buena
        out.append(dict(n=n, pregunta=q, ok=correcta(q, valor), tuya=_texto(q, valor), buena=_texto(q, buena)))
    return out


def recomendacion(evaluacion: Evaluacion, buenas: int) -> str:
    """Lo que el diagnóstico le sugiere al alumno según cuántas acertó."""
    texto = ""
    for desde, consejo in sorted(evaluacion.recomendaciones):
        if buenas >= desde:
            texto = consejo
    return texto
