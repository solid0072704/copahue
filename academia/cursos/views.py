"""Las páginas de la academia: el catálogo y la página de cada curso (públicas), y el aula (mi aprendizaje, el curso por
dentro, las lecciones, las pruebas y el certificado), que exige estar inscrito."""

from __future__ import annotations

import json
import mimetypes
import re
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import redirect_to_login
from django.http import FileResponse, Http404, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from . import acceso, pruebas
from .formularios import Cuenta
from .models import Avance, Certificado, Curso, Evaluacion, Inscripcion, Intento, Leccion, Perfil


# --- Archivos ---------------------------------------------------------------
def _enviar(request, ruta: Path, tipo: str | None = None):
    """Entrega un archivo del disco aceptando rangos, que es lo que pide un reproductor de video para saltar."""
    if not ruta.is_file():
        raise Http404
    total = ruta.stat().st_size
    tipo = tipo or mimetypes.guess_type(ruta.name)[0] or "application/octet-stream"
    rango = re.fullmatch(r"bytes=(\d*)-(\d*)", request.headers.get("Range", "").strip())
    if rango and (rango[1] or rango[2]):
        a = int(rango[1]) if rango[1] else max(0, total - int(rango[2]))
        b = min(int(rango[2]), total - 1) if rango[1] and rango[2] else total - 1
        if a > b or a >= total:
            return HttpResponse(status=416, headers={"Content-Range": f"bytes */{total}"})

        def trozos(f, quedan: int):
            with f:
                while quedan > 0:
                    datos = f.read(min(256 * 1024, quedan))
                    if not datos:
                        break
                    quedan -= len(datos)
                    yield datos
        f = ruta.open("rb")
        f.seek(a)
        r = StreamingHttpResponse(trozos(f, b - a + 1), status=206, content_type=tipo)
        r["Content-Range"], r["Content-Length"] = f"bytes {a}-{b}/{total}", str(b - a + 1)
    else:
        r = FileResponse(ruta.open("rb"), content_type=tipo)
        r["Content-Length"] = str(total)
    r["Accept-Ranges"] = "bytes"
    return r


def _archivo(request, campo, tipo: str | None = None):
    """Un archivo subido: desde el disco en local, o con un enlace firmado que caduca si está en un almacenamiento externo."""
    if not campo:
        raise Http404
    return redirect(campo.url) if settings.MEDIOS_FIRMADOS else _enviar(request, Path(campo.path), tipo)


def medio_publico(request, ruta: str):
    """Imágenes y videos de muestra: lo único de los medios que cualquiera puede pedir por su dirección."""
    base = (Path(settings.MEDIA_ROOT) / "publico").resolve()
    destino = (base / ruta).resolve()
    if base not in destino.parents:
        raise Http404
    return _enviar(request, destino)


# --- Lo público -------------------------------------------------------------
def _grupos(curso: Curso) -> list[dict]:
    """Los módulos del curso con sus lecciones, agrupados por bloque en el orden del programa."""
    grupos: list[dict] = []
    for m in curso.modulos.prefetch_related("lecciones"):
        if not grupos or grupos[-1]["titulo"] != m.grupo:
            grupos.append(dict(titulo=m.grupo, etapa=m.etapa, modulos=[]))
        grupos[-1]["modulos"].append(m)
    return grupos


def catalogo(request):
    cursos = list(Curso.objects.filter(publicado=True).select_related("instructor"))
    mios = set(Inscripcion.objects.filter(usuario=request.user).values_list("curso_id", flat=True)) if request.user.is_authenticated else set()
    for c in cursos:
        c.datos, c.mio = c.resumen(), c.id in mios
    return render(request, "cursos/catalogo.html", dict(cursos=cursos, destacado=next((c for c in cursos if c.destacado), None),
                                                        de_pago=any(not c.gratuito and not c.mio for c in cursos)))


def curso(request, slug: str):
    c = get_object_or_404(Curso.objects.select_related("instructor"), slug=slug)
    if not c.publicado and not request.user.is_staff:
        raise Http404
    con_prueba = c.evaluaciones.filter(tipo="modulo").first()
    return render(request, "cursos/curso.html", dict(
        curso=c, grupos=_grupos(c), resumen=c.resumen(), inscrito=acceso.inscrito(request.user, c),
        diagnostico=c.evaluaciones.filter(tipo="diagnostico").first(), regla=con_prueba, n_evaluaciones=c.evaluaciones.count()))


# --- El aula ----------------------------------------------------------------
@login_required
def mi_aprendizaje(request):
    cursos = [i.curso for i in request.user.inscripciones.select_related("curso__instructor")]
    for c in cursos:
        c.datos, c.avance = c.resumen(), acceso.avance_del_curso(request.user, c)
    return render(request, "cursos/mi_aprendizaje.html", dict(cursos=cursos, pendientes=request.user.ordenes.filter(estado="pendiente")[:3]))


@login_required
def cuenta(request):
    """Los datos del alumno. Quien llega desde un aviso («indica tu país») vuelve a la página en que estaba."""
    siguiente = request.POST.get("siguiente") or request.GET.get("siguiente", "")
    if not url_has_allowed_host_and_scheme(siguiente, allowed_hosts={request.get_host()}, require_https=request.is_secure()):
        siguiente = ""
    datos = dict(nombre=request.user.get_full_name() or request.user.first_name, pais=Perfil.pais_de(request.user))
    form = Cuenta(request.POST or None, initial=datos)
    if request.method == "POST" and form.is_valid():
        form.guardar(request.user)
        messages.success(request, "Guardamos tus datos.")
        return redirect(siguiente or "cuenta")
    return render(request, "cursos/cuenta.html", dict(form=form, siguiente=siguiente))


def _aula(request, slug: str) -> Curso | None:
    """El curso, si quien pide la página está inscrito."""
    c = get_object_or_404(Curso.objects.select_related("instructor"), slug=slug)
    return c if acceso.inscrito(request.user, c) else None


@login_required
def aprender(request, slug: str):
    c = _aula(request, slug)
    if not c:
        messages.info(request, "Para entrar al curso primero tienes que inscribirte.")
        return redirect("curso", slug=slug)
    avance, ok = acceso.avance_del_curso(request.user, c), acceso.aprobadas(request.user, c)
    mejores: dict[int, int] = {}
    for i in Intento.objects.filter(usuario=request.user, evaluacion__curso=c, entregado__isnull=False):
        mejores[i.evaluacion_id] = max(mejores.get(i.evaluacion_id, 0), i.nota)
    evaluaciones = {e.modulo_id: e for e in c.evaluaciones.filter(tipo="modulo") if pruebas.disponible(e)}
    grupos = _grupos(c)
    for g in grupos:
        for m in g["modulos"]:
            m.filas = [dict(leccion=x, hecha=x.id in avance["por_leccion"] and avance["por_leccion"][x.id].completada) for x in m.lecciones.all()]
            m.con_video = sum(1 for f in m.filas if f["leccion"].publicada)
            m.prueba = evaluaciones.get(m.id)
            m.prueba_aprobada = bool(m.prueba and m.prueba.id in ok)
            m.mejor_nota = mejores.get(m.prueba.id) if m.prueba else None
    diagnostico = c.evaluaciones.filter(tipo="diagnostico").first()
    ultimo = diagnostico and Intento.objects.filter(usuario=request.user, evaluacion=diagnostico, entregado__isnull=False).first()
    return render(request, "cursos/aprender.html", dict(
        curso=c, grupos=grupos, avance=avance, diagnostico=diagnostico, ultimo_diagnostico=ultimo,
        recomendacion=pruebas.recomendacion(diagnostico, ultimo.buenas) if ultimo else "", aprobadas=len(ok), con_prueba=len(evaluaciones)))


def leccion(request, slug: str, pk: int):
    x = get_object_or_404(Leccion.objects.select_related("modulo__curso"), pk=pk, modulo__curso__slug=slug)
    if not acceso.puede_ver(request.user, x):
        if not request.user.is_authenticated:
            return redirect_to_login(request.get_full_path())
        messages.info(request, "Esta lección es parte del curso. Inscríbete para verla.")
        return redirect("curso", slug=slug)
    hermanas = list(x.modulo.lecciones.all())
    i = hermanas.index(x)
    avance = Avance.objects.filter(usuario=request.user, leccion=x).first() if request.user.is_authenticated else None
    evaluacion = getattr(x.modulo, "evaluacion", None)
    return render(request, "cursos/leccion.html", dict(
        curso=x.curso, modulo=x.modulo, leccion=x, hermanas=hermanas, anterior=hermanas[i - 1] if i else None,
        siguiente=hermanas[i + 1] if i + 1 < len(hermanas) else None, avance=avance, inscrito=acceso.inscrito(request.user, x.curso),
        prueba=evaluacion if evaluacion and pruebas.disponible(evaluacion) else None))


def video(request, pk: int):
    x = get_object_or_404(Leccion.objects.select_related("modulo__curso"), pk=pk)
    if not acceso.puede_ver(request.user, x):
        raise Http404
    return _archivo(request, x.video, "video/mp4")


def subtitulos(request, pk: int):
    x = get_object_or_404(Leccion.objects.select_related("modulo__curso"), pk=pk)
    if not acceso.puede_ver(request.user, x):
        raise Http404
    return _archivo(request, x.subtitulos, "text/vtt")


@login_required
def recurso(request, pk: int):
    from .models import Recurso
    r = get_object_or_404(Recurso.objects.select_related("curso"), pk=pk)
    if not acceso.inscrito(request.user, r.curso):
        raise Http404
    return _archivo(request, r.archivo)


@login_required
@require_POST
def guardar_avance(request, pk: int):
    """El reproductor avisa por dónde va el alumno; al pasar el 90 % la lección queda completada."""
    x = get_object_or_404(Leccion.objects.select_related("modulo__curso"), pk=pk)
    if not acceso.puede_ver(request.user, x):
        raise Http404
    try:
        datos = json.loads(request.body or b"{}")
        segundos = max(0, int(datos.get("segundos", 0)))
    except (ValueError, TypeError, AttributeError):
        return JsonResponse(dict(error="datos"), status=400)
    a, _ = Avance.objects.get_or_create(usuario=request.user, leccion=x)
    a.segundos = segundos
    a.completada = a.completada or bool(datos.get("completada"))
    a.save()
    return JsonResponse(dict(completada=a.completada))


@login_required
@require_POST
def marcar(request, pk: int):
    """El alumno marca o desmarca la lección a mano."""
    x = get_object_or_404(Leccion.objects.select_related("modulo__curso"), pk=pk)
    if not acceso.puede_ver(request.user, x):
        raise Http404
    a, _ = Avance.objects.get_or_create(usuario=request.user, leccion=x)
    a.completada = not a.completada
    a.save()
    return redirect("leccion", slug=x.curso.slug, pk=x.pk)


# --- Pruebas ----------------------------------------------------------------
@login_required
def prueba(request, slug: str, pk: int):
    e = get_object_or_404(Evaluacion.objects.select_related("curso", "modulo"), pk=pk, curso__slug=slug)
    # el diagnóstico lo puede rendir cualquiera con cuenta; las pruebas de módulo y el examen, solo los inscritos
    if e.tipo != "diagnostico" and not acceso.inscrito(request.user, e.curso):
        messages.info(request, "Las pruebas son parte del curso. Inscríbete para rendirlas.")
        return redirect("curso", slug=slug)
    if not pruebas.disponible(e):
        raise Http404
    if request.method == "POST":
        return redirect("intento", pk=pruebas.armar(e, request.user).pk)
    anteriores = Intento.objects.filter(usuario=request.user, evaluacion=e, entregado__isnull=False)[:5]
    banco = e.preguntas.filter(activa=True).count()
    return render(request, "cursos/prueba.html", dict(curso=e.curso, evaluacion=e, anteriores=anteriores, banco=banco,
                                                      n=banco if e.tipo == "diagnostico" else min(e.preguntas_por_intento, banco),
                                                      inscrito=acceso.inscrito(request.user, e.curso)))


@login_required
def intento(request, pk: int):
    i = get_object_or_404(Intento.objects.select_related("evaluacion__curso", "evaluacion__modulo"), pk=pk, usuario=request.user)
    e = i.evaluacion
    if request.method == "POST":
        pruebas.corregir(i, request.POST)
        return redirect("intento", pk=i.pk)
    contexto = dict(curso=e.curso, evaluacion=e, intento=i, inscrito=acceso.inscrito(request.user, e.curso))
    if not i.entregado:
        return render(request, "cursos/intento.html", dict(contexto, preguntas=pruebas.para_mostrar(i)))
    revision = pruebas.revision(i)
    reforzar = sorted({r["pregunta"].tema for r in revision if not r["ok"] and r["pregunta"].tema}, key=lambda m: m.numero)
    return render(request, "cursos/resultado.html", dict(contexto, revision=revision, reforzar=reforzar,
                                                         recomendacion=pruebas.recomendacion(e, i.buenas) if e.tipo == "diagnostico" else ""))


# --- Certificado ------------------------------------------------------------
@login_required
def certificado(request, slug: str):
    c = _aula(request, slug)
    if not c:
        return redirect("curso", slug=slug)
    return render(request, "cursos/certificado.html", dict(curso=c, requisitos=acceso.requisitos(request.user, c),
                                                           certificado=acceso.certificado_de(request.user, c), resumen=c.resumen()))


def verificar(request, codigo: str):
    """Página pública: cualquiera con el código puede comprobar que un certificado es auténtico."""
    cert = Certificado.objects.select_related("curso__instructor").filter(codigo=codigo.upper()).first()
    return render(request, "cursos/verificar.html", dict(certificado=cert, codigo=codigo.upper()), status=200 if cert else 404)
