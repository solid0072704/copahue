"""Quién puede ver qué, cuánto lleva cada alumno y cuándo le corresponde el certificado."""

from __future__ import annotations

from .models import Avance, Certificado, Curso, Inscripcion, Intento, Leccion


def inscrito(usuario, curso: Curso) -> bool:
    """El equipo de la academia ve todos los cursos; los demás, los que compraron o recibieron."""
    return usuario.is_authenticated and (usuario.is_staff or Inscripcion.objects.filter(usuario=usuario, curso=curso).exists())


def puede_ver(usuario, leccion: Leccion) -> bool:
    return leccion.muestra or inscrito(usuario, leccion.curso)


def avance_del_curso(usuario, curso: Curso) -> dict:
    """Lecciones publicadas, cuáles completó el alumno y por cuál le conviene seguir."""
    lecciones = [x for x in curso.lecciones() if x.publicada]
    hechos = {a.leccion_id: a for a in Avance.objects.filter(usuario=usuario, leccion__in=lecciones)} if usuario.is_authenticated else {}
    completadas = [x for x in lecciones if x.id in hechos and hechos[x.id].completada]
    siguiente = next((x for x in lecciones if x not in completadas), None)
    return dict(publicadas=len(lecciones), completadas=len(completadas), porcentaje=round(100 * len(completadas) / len(lecciones)) if lecciones else 0,
                por_leccion=hechos, siguiente=siguiente)


def aprobadas(usuario, curso: Curso) -> set[int]:
    """Las evaluaciones del curso que el alumno ya aprobó."""
    return set(Intento.objects.filter(usuario=usuario, evaluacion__curso=curso, aprobado=True).values_list("evaluacion_id", flat=True))


def requisitos(usuario, curso: Curso) -> list[dict]:
    """Lo que falta y lo que ya está para el certificado, en el orden del curso."""
    if curso.certifica == "lecciones":
        a = avance_del_curso(usuario, curso)
        total = curso.lecciones().count()
        return [dict(texto=f"Completar las {total} lecciones del curso ({a['completadas']} completadas)", listo=total > 0 and a["completadas"] == total)]
    ok, out = aprobadas(usuario, curso), []
    con_prueba = {e.modulo_id: e for e in curso.evaluaciones.filter(tipo="modulo") if e.preguntas.filter(activa=True).exists()}
    for m in curso.modulos.all():
        e = con_prueba.get(m.id)
        out.append(dict(texto=f"Prueba del módulo {m.numero}" + ("" if e else " (se publica con el módulo)"), listo=bool(e and e.id in ok)))
    final = curso.evaluaciones.filter(tipo="final").first()
    publicado = bool(final and final.preguntas.filter(activa=True).exists())
    out.append(dict(texto="Examen final" + ("" if publicado else " (se publica al completarse el curso)"), listo=bool(publicado and final.id in ok)))
    return out


def certificado_de(usuario, curso: Curso) -> Certificado | None:
    """El certificado del alumno; se emite la primera vez que cumple todos los requisitos."""
    existente = Certificado.objects.filter(usuario=usuario, curso=curso).first()
    if existente or not all(r["listo"] for r in requisitos(usuario, curso)):
        return existente
    nombre = usuario.get_full_name() or usuario.first_name or usuario.email
    return Certificado.objects.create(usuario=usuario, curso=curso, nombre=nombre, horas=round(curso.resumen()["minutos"] / 60, 1))
