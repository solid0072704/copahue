"""Carga en la academia un curso escrito como paquete de `academia/contenido`.

El paquete trae `CURSO` (título, dirección, instructor…), `programa` (bloques, módulos y lecciones) y `evaluaciones`
(diagnóstico y bancos de preguntas). Importar de nuevo actualiza lo que cambió sin tocar lo que hicieron los alumnos:
los módulos, las lecciones y las evaluaciones se reconocen por su clave. El precio (en dólares), la publicación y los
videos se manejan en el panel de administración y no se pisan.
"""

from __future__ import annotations

from importlib import import_module

from django.db import transaction

from .models import Curso, Evaluacion, Instructor, Leccion, Modulo, Pregunta


@transaction.atomic
def importar(nombre_del_paquete: str) -> Curso:
    paquete = import_module(nombre_del_paquete)
    c, programa, evaluaciones = paquete.CURSO, paquete.programa, paquete.evaluaciones
    evaluaciones.validar()
    instructor, _ = Instructor.objects.update_or_create(nombre=c["instructor"]["nombre"], defaults={k: v for k, v in c["instructor"].items() if k != "nombre"})
    curso, _ = Curso.objects.update_or_create(slug=c["slug"], defaults=dict(
        titulo=c["titulo"], bajada=c["bajada"], descripcion=c.get("descripcion", ""), categoria=c.get("categoria", ""), logros=c.get("logros", []),
        preguntas=[list(x) for x in c.get("preguntas", [])], instructor=instructor, muestra_titulo=c.get("muestra_titulo", "")))

    titulos = {b["clave"]: b["t"] for b in programa.BLOQUES}
    modulos, lecciones = {}, {}
    for m in programa.modulos():
        modulos[m["clave"]], _ = Modulo.objects.update_or_create(curso=curso, clave=m["clave"], defaults=dict(
            numero=m["n"], titulo=m["t"], descripcion=m["d"], logros=m["logras"], practica=m["practica"], grupo=titulos[m["bloque"]], etapa=m["etapa"]))
        for x in m["lecciones"]:
            lecciones[x["id"]], _ = Leccion.objects.update_or_create(modulo=modulos[m["clave"]], clave=x["id"], defaults=dict(
                numero=x["n"], titulo=x["t"], descripcion=x["d"], tipo=x["tipo"], minutos=x["min"]))

    reglas = programa.EVALUACION

    def cargar(evaluacion: Evaluacion, banco: list[dict]) -> None:
        """Las preguntas se reconocen por su enunciado: las que ya no están en el banco se desactivan, no se borran."""
        vistas = []
        for q in banco:
            p, _ = Pregunta.objects.update_or_create(evaluacion=evaluacion, enunciado=q["p"], defaults=dict(
                tipo=q["tipo"], opciones=q["o"], respuesta=q["r"], tolerancia=q.get("tol", 0), unidad=q.get("unidad", ""), explicacion=q["e"],
                leccion=lecciones.get(q.get("l") or ""), tema=modulos.get(q.get("tema") or ""), activa=True))
            vistas.append(p.id)
        evaluacion.preguntas.exclude(id__in=vistas).update(activa=False)

    diagnostico, _ = Evaluacion.objects.update_or_create(curso=curso, tipo="diagnostico", defaults=dict(
        preguntas_por_intento=len(evaluaciones.DIAGNOSTICO), aprobacion=0, recomendaciones=[list(x) for x in evaluaciones.RECOMENDACIONES]))
    cargar(diagnostico, evaluaciones.DIAGNOSTICO)
    for clave, banco in evaluaciones.BANCOS.items():
        e, _ = Evaluacion.objects.update_or_create(curso=curso, modulo=modulos[clave], defaults=dict(
            tipo="modulo", preguntas_por_intento=reglas["preguntas_por_modulo"], aprobacion=reglas["aprobacion"]))
        cargar(e, banco)
    return curso
