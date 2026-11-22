"""La compra de un curso: elegir cómo pagar, ir a la pasarela, recibir su aviso y mostrarle al alumno cómo quedó."""

from __future__ import annotations

import logging

import requests
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from academia.cursos.acceso import inscrito
from academia.cursos.models import Curso, Inscripcion

from . import pasarelas
from .models import Orden

registro = logging.getLogger(__name__)


@login_required
def comprar(request, slug: str):
    curso = get_object_or_404(Curso, slug=slug, publicado=True)
    if inscrito(request.user, curso) and not request.user.is_staff:
        return redirect("aprender", slug=curso.slug)
    disponibles = pasarelas.activas()
    if request.method == "POST":
        if curso.gratuito:
            Inscripcion.objects.get_or_create(usuario=request.user, curso=curso, defaults=dict(origen="gratis"))
            return redirect("aprender", slug=curso.slug)
        elegida = pasarelas.pasarela(request.POST.get("pasarela", "")) or (disponibles[0] if len(disponibles) == 1 else None)
        if not elegida:
            messages.error(request, "Elige un medio de pago para continuar.")
            return redirect("comprar", slug=curso.slug)
        orden = Orden.objects.create(usuario=request.user, curso=curso, monto=curso.precio, moneda=settings.MONEDA, pasarela=elegida.clave)
        try:
            return redirect(elegida.iniciar(orden, request))
        except (requests.RequestException, KeyError, ValueError) as e:
            registro.exception("No se pudo iniciar el pago de la orden %s", orden.codigo)
            orden.cerrar("anulada", dict(error=str(e)[:300]))
            messages.error(request, f"No pudimos conectar con {elegida.nombre}. No se hizo ningún cobro: inténtalo de nuevo en unos minutos.")
            return redirect("comprar", slug=curso.slug)
    return render(request, "pagos/comprar.html", dict(curso=curso, pasarelas=disponibles, resumen=curso.resumen()))


@csrf_exempt
def confirmacion(request, clave: str):
    """El aviso de la pasarela al servidor. Si no se pudo comprobar el pago, se responde con error para que reintente."""
    p = pasarelas.pasarela(clave)
    if not p:
        raise Http404
    try:
        p.confirmar(request)
    except requests.RequestException:
        registro.exception("No se pudo comprobar un aviso de %s", clave)
        return HttpResponse(status=502)
    return HttpResponse("ok")


@csrf_exempt
def retorno(request, clave: str):
    """El comprador vuelve desde la pasarela: se le muestra su orden."""
    p = pasarelas.pasarela(clave)
    if not p:
        raise Http404
    try:
        orden = p.de_retorno(request)
    except requests.RequestException:
        orden = None
    if not orden:
        messages.info(request, "Estamos confirmando tu pago. Si ya pagaste, el curso aparecerá aquí en unos minutos.")
        return redirect("mi_aprendizaje")
    return redirect("orden", codigo=orden.codigo)


@login_required
def orden(request, codigo: str):
    return render(request, "pagos/orden.html", dict(orden=get_object_or_404(Orden, codigo=codigo, usuario=request.user)))


@login_required
def pago_de_prueba(request, codigo: str):
    """La página de la pasarela de prueba: aprobar o rechazar el pago."""
    if not pasarelas.pasarela("prueba"):
        raise Http404
    return render(request, "pagos/prueba.html", dict(orden=get_object_or_404(Orden, codigo=codigo, usuario=request.user, pasarela="prueba", estado="pendiente")))


@login_required
@require_POST
def resolver_prueba(request):
    p = pasarelas.pasarela("prueba")
    if not p or not Orden.objects.filter(codigo=request.POST.get("orden", ""), usuario=request.user).exists():
        raise Http404
    resuelta = p.confirmar(request)
    return redirect("orden", codigo=resuelta.codigo)
