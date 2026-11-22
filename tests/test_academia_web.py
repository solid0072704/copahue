"""La academia como aplicación: cuentas, catálogo, compra, aula, pruebas y certificado."""

import hashlib
import hmac
import json
from unittest import mock

import pytest

pytest.importorskip("django")
pytest.importorskip("allauth")

from django.core import mail  # noqa: E402
from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402
from django.http import QueryDict  # noqa: E402
from django.urls import reverse  # noqa: E402

from academia.cursos import acceso, pruebas  # noqa: E402
from academia.cursos.importar import importar  # noqa: E402
from academia.cursos.models import Avance, Certificado, Curso, Inscripcion, Intento, Leccion, Pregunta  # noqa: E402
from academia.pagos.models import Orden  # noqa: E402
from academia.pagos.pasarelas import flow, mercadopago  # noqa: E402

pytestmark = pytest.mark.django_db
PAQUETE = "academia.contenido.claude_finanzas"
CLAVE = "una-clave-larga-y-rara-2026"


@pytest.fixture
def curso(settings, tmp_path) -> Curso:
    settings.MEDIA_ROOT = tmp_path
    settings.PASARELAS = ["prueba"]
    c = importar(PAQUETE)
    c.precio, c.publicado = 50000, True
    c.save()
    return c


@pytest.fixture
def alumno(django_user_model):
    return django_user_model.objects.create_user(username="ana", email="ana@example.com", password=CLAVE, first_name="Ana Soto")


def _respuesta(datos: dict):
    r = mock.Mock()
    r.json.return_value = datos
    r.raise_for_status.return_value = None
    return r


# --- Contenido ---------------------------------------------------------------
def test_importar_un_curso_dos_veces_no_duplica_ni_pisa_lo_que_se_define_en_el_panel(curso):
    assert (curso.modulos.count(), curso.lecciones().count(), curso.evaluaciones.count()) == (16, 97, 2)
    antes = Pregunta.objects.count()
    otra_vez = importar(PAQUETE)
    assert otra_vez.pk == curso.pk and (otra_vez.precio, otra_vez.publicado) == (50000, True)
    assert (Curso.objects.count(), Leccion.objects.count(), Pregunta.objects.count()) == (1, 97, antes)
    m8 = curso.modulos.get(clave="m08")
    assert "CRM" in m8.titulo and m8.grupo == "Conectar Claude con tus sistemas" and m8.lecciones.count() == 7


def test_el_catalogo_muestra_solo_lo_publicado(client, curso):
    assert curso.titulo in client.get(reverse("catalogo")).content.decode()
    pagina = client.get(reverse("curso", args=[curso.slug])).content.decode()
    assert "50.000" in pagina and "CRM + Claude" in pagina and reverse("comprar", args=[curso.slug]) in pagina
    curso.publicado = False
    curso.save()
    assert curso.titulo not in client.get(reverse("catalogo")).content.decode()
    assert client.get(reverse("curso", args=[curso.slug])).status_code == 404


# --- Cuentas -----------------------------------------------------------------
def test_se_crea_la_cuenta_con_correo_y_contrasena_y_se_entra_con_ella(client, django_user_model):
    r = client.post(reverse("account_signup"), dict(nombre="Luis Rojas", email="luis@example.com", password1=CLAVE, password2=CLAVE))
    assert r.status_code == 302 and r.url == reverse("mi_aprendizaje")
    luis = django_user_model.objects.get(email="luis@example.com")
    assert luis.first_name == "Luis Rojas" and luis.check_password(CLAVE)
    client.post(reverse("account_logout"))
    assert client.get(reverse("mi_aprendizaje")).status_code == 302                     # sin sesión, pide entrar
    assert client.post(reverse("account_login"), dict(login="luis@example.com", password="otra")).status_code == 200
    assert client.post(reverse("account_login"), dict(login="luis@example.com", password=CLAVE)).status_code == 302
    assert client.get(reverse("mi_aprendizaje")).status_code == 200


def test_google_aparece_solo_cuando_estan_sus_credenciales(client, settings):
    sin = client.get(reverse("account_login")).content.decode()
    assert "Entra a la academia" in sin and "con Google" not in sin and "/cuentas/google/" not in sin
    settings.CON_GOOGLE = True
    settings.SOCIALACCOUNT_PROVIDERS = {"google": {"APPS": [{"client_id": "id", "secret": "secreto", "key": ""}]}}
    pagina = client.get(reverse("account_login")).content.decode()
    assert "Entrar con Google" in pagina and "/cuentas/google/login/" in pagina


# --- Acceso ------------------------------------------------------------------
def test_las_lecciones_son_para_los_inscritos_salvo_las_de_vista_previa(client, curso, alumno):
    x = curso.lecciones().first()
    x.video = SimpleUploadedFile("leccion.mp4", b"0123456789" * 100, content_type="video/mp4")
    x.save()
    url, video = reverse("leccion", args=[curso.slug, x.pk]), reverse("video", args=[x.pk])
    assert client.get(url).status_code == 302 and "/cuentas/login/" in client.get(url).url          # sin sesión
    assert client.get(video).status_code == 404
    client.force_login(alumno)
    assert client.get(url).url == reverse("curso", args=[curso.slug]) and client.get(video).status_code == 404   # con sesión, sin inscripción
    assert client.get(reverse("aprender", args=[curso.slug])).status_code == 302
    Inscripcion.objects.create(usuario=alumno, curso=curso)
    assert client.get(url).status_code == 200 and client.get(reverse("aprender", args=[curso.slug])).status_code == 200
    tramo = client.get(video, HTTP_RANGE="bytes=10-19")
    assert tramo.status_code == 206 and b"".join(tramo.streaming_content) == b"0123456789" and tramo["Content-Range"] == "bytes 10-19/1000"
    client.logout()
    x.muestra = True
    x.save()
    assert client.get(url).status_code == 200 and client.get(video).status_code == 200              # vista previa: cualquiera


def test_el_avance_se_guarda_en_el_servidor(client, curso, alumno):
    x = curso.lecciones().first()
    x.video = SimpleUploadedFile("leccion.mp4", b"x" * 50)
    x.save()
    Inscripcion.objects.create(usuario=alumno, curso=curso)
    client.force_login(alumno)
    url = reverse("guardar_avance", args=[x.pk])
    assert client.post(url, json.dumps(dict(segundos=73)), content_type="application/json").json() == dict(completada=False)
    assert client.post(url, json.dumps(dict(segundos=410, completada=True)), content_type="application/json").json() == dict(completada=True)
    a = Avance.objects.get(usuario=alumno, leccion=x)
    assert (a.segundos, a.completada) == (410, True)
    assert acceso.avance_del_curso(alumno, curso)["porcentaje"] == 100 and "Mi aprendizaje" in client.get(reverse("mi_aprendizaje")).content.decode()


# --- Compra ------------------------------------------------------------------
def test_comprar_con_la_pasarela_de_prueba_inscribe_una_sola_vez(client, curso, alumno, django_capture_on_commit_callbacks):
    comprar = reverse("comprar", args=[curso.slug])
    assert client.get(comprar).status_code == 302                                                   # hay que tener cuenta
    client.force_login(alumno)
    assert "Pago de prueba" in client.get(comprar).content.decode()
    r = client.post(comprar, dict(pasarela="prueba"))
    orden = Orden.objects.get()
    assert (orden.estado, orden.monto, orden.moneda) == ("pendiente", 50000, "CLP") and r.url == reverse("pago_de_prueba", args=[orden.codigo])
    assert not Inscripcion.objects.exists()
    with django_capture_on_commit_callbacks(execute=True):
        r = client.post(reverse("resolver_prueba"), dict(orden=orden.codigo, resultado="aprobar"))
    orden.refresh_from_db()
    assert orden.estado == "pagada" and Inscripcion.objects.filter(usuario=alumno, curso=curso, origen="compra").count() == 1
    assert len(mail.outbox) == 1 and curso.titulo in mail.outbox[0].subject and mail.outbox[0].to == ["ana@example.com"]
    assert orden.pagar() is False and Inscripcion.objects.count() == 1                              # un segundo aviso no hace nada
    assert "ya estás inscrito" in client.get(r.url).content.decode()
    assert client.get(comprar).url == reverse("aprender", args=[curso.slug])                        # ya inscrito: no se compra dos veces


def test_un_pago_rechazado_no_inscribe_y_una_orden_ajena_no_se_ve(client, curso, alumno, django_user_model):
    client.force_login(alumno)
    client.post(reverse("comprar", args=[curso.slug]), dict(pasarela="prueba"))
    orden = Orden.objects.get()
    client.post(reverse("resolver_prueba"), dict(orden=orden.codigo, resultado="rechazar"))
    orden.refresh_from_db()
    assert orden.estado == "rechazada" and not Inscripcion.objects.exists()
    client.force_login(django_user_model.objects.create_user(username="otro", email="otro@example.com", password=CLAVE))
    assert client.get(reverse("orden", args=[orden.codigo])).status_code == 404


def test_un_curso_gratuito_inscribe_sin_pasar_por_el_pago(client, curso, alumno):
    curso.precio = 0
    curso.save()
    client.force_login(alumno)
    assert client.post(reverse("comprar", args=[curso.slug])).url == reverse("aprender", args=[curso.slug])
    assert Inscripcion.objects.get().origen == "gratis" and not Orden.objects.exists()


def test_flow_firma_cada_llamada_y_solo_da_por_pagado_lo_que_flow_confirma(client, curso, alumno, settings):
    settings.PASARELAS, settings.FLOW_API_KEY, settings.FLOW_SECRET_KEY = ["flow"], "llave", "secreto"
    settings.FLOW_URL, settings.SITIO_URL = "https://sandbox.flow.cl/api", "https://academia.example"
    assert flow.firmar(dict(b="2", a="1"), "secreto") == hmac.new(b"secreto", b"a1b2", hashlib.sha256).hexdigest()
    client.force_login(alumno)
    with mock.patch("academia.pagos.pasarelas.flow.requests.post", return_value=_respuesta(dict(url="https://sandbox.flow.cl/app/web/pay.php", token="TOK", flowOrder=77))) as post:
        r = client.post(reverse("comprar", args=[curso.slug]), dict(pasarela="flow"))
    orden = Orden.objects.get()
    enviado = post.call_args.kwargs["data"]
    assert r.url == "https://sandbox.flow.cl/app/web/pay.php?token=TOK" and post.call_args.args[0] == "https://sandbox.flow.cl/api/payment/create"
    assert (enviado["commerceOrder"], enviado["amount"], enviado["currency"], enviado["email"]) == (orden.codigo, 50000, "CLP", "ana@example.com")
    assert enviado["urlConfirmation"] == "https://academia.example/pagos/flow/confirmacion/" and enviado["urlReturn"].endswith("/pagos/flow/retorno/")
    assert enviado["s"] == flow.firmar({k: v for k, v in enviado.items() if k != "s"}, "secreto")
    client.logout()                                                                                 # el aviso llega del servidor de Flow, sin sesión

    def estado(status, monto=50000):
        return mock.patch("academia.pagos.pasarelas.flow.requests.get", return_value=_respuesta(dict(status=status, commerceOrder=orden.codigo, amount=monto, flowOrder=77)))
    with estado(1):
        assert client.post(reverse("pago_confirmacion", args=["flow"]), dict(token="TOK")).status_code == 200
    orden.refresh_from_db()
    assert orden.estado == "pendiente"
    with estado(2, monto=1):                                                                        # pagado, pero no por el monto de la orden
        client.post(reverse("pago_confirmacion", args=["flow"]), dict(token="TOK"))
    orden.refresh_from_db()
    assert orden.estado == "pendiente" and not Inscripcion.objects.exists()
    with estado(2) as get:
        client.post(reverse("pago_confirmacion", args=["flow"]), dict(token="TOK"))
    orden.refresh_from_db()
    assert orden.estado == "pagada" and orden.referencia == "77" and Inscripcion.objects.filter(usuario=alumno, curso=curso).exists()
    assert get.call_args.kwargs["params"]["token"] == "TOK" and get.call_args.args[0].endswith("/payment/getStatus")


def test_mercado_pago_crea_la_preferencia_y_valida_el_aviso(client, curso, alumno, settings):
    settings.PASARELAS, settings.MERCADOPAGO_TOKEN, settings.MERCADOPAGO_SECRETO = ["mercadopago"], "TOKEN", ""
    settings.SITIO_URL = "https://academia.example"
    client.force_login(alumno)
    with mock.patch("academia.pagos.pasarelas.mercadopago.requests.post", return_value=_respuesta(dict(id="pref-1", init_point="https://www.mercadopago.cl/checkout/v1/redirect?pref_id=pref-1"))) as post:
        r = client.post(reverse("comprar", args=[curso.slug]), dict(pasarela="mercadopago"))
    orden = Orden.objects.get()
    cuerpo = post.call_args.kwargs["json"]
    assert r.url.startswith("https://www.mercadopago.cl/checkout") and post.call_args.kwargs["headers"]["Authorization"] == "Bearer TOKEN"
    assert cuerpo["external_reference"] == orden.codigo and cuerpo["notification_url"] == "https://academia.example/pagos/mercadopago/confirmacion/"
    assert cuerpo["items"] == [dict(title=curso.titulo, quantity=1, unit_price=50000, currency_id="CLP")]
    client.logout()
    aviso = reverse("pago_confirmacion", args=["mercadopago"]) + "?type=payment&data.id=123"
    pago = dict(id=123, status="approved", external_reference=orden.codigo, transaction_amount=50000, currency_id="CLP")

    # con clave secreta configurada, un aviso sin firma válida no se procesa
    settings.MERCADOPAGO_SECRETO = "clave-del-aviso"
    with mock.patch("academia.pagos.pasarelas.mercadopago.requests.get", return_value=_respuesta(pago)) as get:
        client.post(aviso, HTTP_X_SIGNATURE="ts=1700000000,v1=falsa", HTTP_X_REQUEST_ID="req-1")
        assert not get.called
        firma = hmac.new(b"clave-del-aviso", b"id:123;request-id:req-1;ts:1700000000;", hashlib.sha256).hexdigest()
        assert client.post(aviso, HTTP_X_SIGNATURE=f"ts=1700000000,v1={firma}", HTTP_X_REQUEST_ID="req-1").status_code == 200
        assert get.call_args.args[0] == "https://api.mercadopago.com/v1/payments/123"
    orden.refresh_from_db()
    assert orden.estado == "pagada" and orden.referencia == "123" and Inscripcion.objects.filter(usuario=alumno, curso=curso).count() == 1


# --- Pruebas -----------------------------------------------------------------
def test_la_prueba_se_corrige_en_el_servidor_y_no_muestra_las_respuestas_antes(client, curso, alumno):
    e = curso.evaluaciones.get(modulo__clave="m01")
    url = reverse("prueba", args=[curso.slug, e.pk])
    client.force_login(alumno)
    assert client.get(url).url == reverse("curso", args=[curso.slug])                               # la prueba de módulo es para inscritos
    Inscripcion.objects.create(usuario=alumno, curso=curso)
    assert "banco de 15" in client.get(url).content.decode()
    r = client.post(url)
    intento = Intento.objects.get()
    assert r.url == reverse("intento", args=[intento.pk]) and len(intento.preguntas) == 10
    pagina = client.get(r.url).content.decode()
    elegidas = Pregunta.objects.in_bulk([x["id"] for x in intento.preguntas])
    assert all(q.explicacion not in pagina for q in elegidas.values())                              # las explicaciones llegan al entregar

    def buena(q):
        if q.tipo == "multiple":
            return [str(k) for k in q.respuesta]
        if q.tipo == "orden":
            return ",".join(str(k) for k in range(len(q.opciones)))
        return str(q.respuesta).replace(".", ",") + " %" if q.tipo == "numero" else str(q.respuesta)
    datos = {f'p{x["id"]}': buena(elegidas[x["id"]]) for x in intento.preguntas}
    assert client.post(r.url, datos).status_code == 302
    intento.refresh_from_db()
    assert (intento.buenas, intento.nota, intento.aprobado) == (10, 100, True)
    assert client.post(r.url, {}).status_code == 302                                               # entregar de nuevo no cambia nada
    intento.refresh_from_db()
    assert intento.nota == 100 and "Aprobada" in client.get(r.url).content.decode()
    # un intento en blanco reprueba
    otro = pruebas.corregir(pruebas.armar(e, alumno), QueryDict(""))
    assert (otro.buenas, otro.aprobado) == (0, False)
    assert pruebas.numero("18,0 %") == 18 and pruebas.numero("1.234,5") == 1234.5 and pruebas.numero("11.800") == 11800 and pruebas.numero("x") is None


def test_el_diagnostico_lo_rinde_cualquiera_con_cuenta_y_recomienda_por_donde_partir(client, curso, alumno):
    d = curso.evaluaciones.get(tipo="diagnostico")
    client.force_login(alumno)
    r = client.post(reverse("prueba", args=[curso.slug, d.pk]))
    intento = Intento.objects.get()
    assert len(intento.preguntas) == 10
    client.post(r.url, {})                                                                          # sin responder
    pagina = client.get(r.url).content.decode()
    intento.refresh_from_db()
    assert not intento.aprobado and "Parte por el módulo 1" in pagina and "Para reforzar" in pagina


# --- Certificado -------------------------------------------------------------
def test_el_certificado_se_emite_al_cumplir_los_requisitos_y_se_puede_verificar(client, curso, alumno):
    Inscripcion.objects.create(usuario=alumno, curso=curso)
    client.force_login(alumno)
    pagina = client.get(reverse("certificado", args=[curso.slug])).content.decode()
    assert "Prueba del módulo 1" in pagina and "Examen final" in pagina and not Certificado.objects.exists()
    # un curso que certifica por lecciones: se completa la única lección y sale el certificado
    curso.certifica = "lecciones"
    curso.save()
    Leccion.objects.filter(modulo__curso=curso).exclude(clave="m01-l01").delete()
    x = Leccion.objects.get()
    x.video = SimpleUploadedFile("leccion.mp4", b"x" * 20)
    x.save()
    Avance.objects.create(usuario=alumno, leccion=x, completada=True)
    pagina = client.get(reverse("certificado", args=[curso.slug])).content.decode()
    cert = Certificado.objects.get()
    assert cert.nombre == "Ana Soto" and cert.codigo in pagina
    assert client.get(reverse("verificar", args=[cert.codigo.lower()])).status_code == 200
    assert client.get(reverse("verificar", args=["NOEXISTE"])).status_code == 404
    client.get(reverse("certificado", args=[curso.slug]))
    assert Certificado.objects.count() == 1                                                         # no se emite dos veces


def test_el_panel_de_administracion_abre_el_curso_con_sus_modulos(client, curso, django_user_model):
    client.force_login(django_user_model.objects.create_superuser(username="admin", email="admin@example.com", password=CLAVE))
    assert client.get(reverse("admin:cursos_curso_change", args=[curso.pk])).status_code == 200
    assert client.get(reverse("admin:cursos_modulo_change", args=[curso.modulos.first().pk])).status_code == 200
    assert client.get(reverse("admin:cursos_evaluacion_change", args=[curso.evaluaciones.first().pk])).status_code == 200
    assert client.get(reverse("admin:pagos_orden_changelist")).status_code == 200
    assert client.get(reverse("aprender", args=[curso.slug])).status_code == 200                    # el equipo ve los cursos sin comprarlos


# --- Varios cursos y carga a mano --------------------------------------------
def test_un_segundo_curso_creado_a_mano_convive_con_el_primero(client, curso, alumno):
    """La academia es un catálogo: un curso nuevo, creado en el panel, se vende y se cursa por separado."""
    from academia.cursos.models import Modulo
    nuevo = Curso.objects.create(slug="excel-para-el-cierre", titulo="Excel para el cierre mensual", bajada="Plantillas y controles", precio=30000,
                                 publicado=True, instructor=curso.instructor, certifica="lecciones")
    m = Modulo.objects.create(curso=nuevo, numero=1, titulo="Conciliaciones")
    x = Leccion.objects.create(modulo=m, numero=1, titulo="La plantilla", minutos=7, video=SimpleUploadedFile("x.mp4", b"x" * 30))
    catalogo = client.get(reverse("catalogo")).content.decode()
    assert "2 cursos disponibles" in catalogo and nuevo.titulo in catalogo and curso.titulo in catalogo
    client.force_login(alumno)
    client.post(reverse("comprar", args=[nuevo.slug]), dict(pasarela="prueba"))
    client.post(reverse("resolver_prueba"), dict(orden=Orden.objects.get().codigo, resultado="aprobar"))
    assert client.get(reverse("leccion", args=[nuevo.slug, x.pk])).status_code == 200              # inscrito en el nuevo…
    assert client.get(reverse("aprender", args=[curso.slug])).status_code == 302                   # …y no en el otro
    mio = client.get(reverse("mi_aprendizaje")).content.decode()
    assert nuevo.titulo in mio and curso.titulo not in mio


def test_los_formularios_del_panel_aceptan_listas_por_linea_y_respuestas_contadas_desde_uno(curso):
    from academia.cursos.formularios import CursoForm, PreguntaForm
    f = CursoForm(instance=curso, data=dict(
        slug=curso.slug, titulo=curso.titulo, bajada=curso.bajada, instructor=curso.instructor_id, precio=45000, orden=0, certifica="evaluaciones",
        logros="Leer un balance\n\nProyectar la caja\n", preguntas="¿Sirve para pymes? | Sí, el caso es una pyme."))
    assert f.is_valid(), f.errors
    guardado = f.save()
    assert guardado.logros == ["Leer un balance", "Proyectar la caja"] and guardado.preguntas == [["¿Sirve para pymes?", "Sí, el caso es una pyme."]]
    assert not CursoForm(instance=curso, data=dict(f.data, preguntas="una línea sin barra")).is_valid()

    e = curso.evaluaciones.get(modulo__clave="m01")
    base = dict(evaluacion=e.pk, enunciado="¿Cuál es?", explicacion="Porque sí.", tolerancia=0, activa="on")

    def pregunta(**datos):
        formulario = PreguntaForm(data={**base, **datos})
        return formulario.save().respuesta if formulario.is_valid() else None
    assert pregunta(tipo="unica", opciones="A\nB\nC", correcta="2") == 1
    assert pregunta(tipo="multiple", opciones="A\nB\nC", correcta="3, 1") == [0, 2]
    assert pregunta(tipo="numero", correcta="18,5") == 18.5
    assert pregunta(tipo="orden", opciones="Primero\nSegundo\nTercero") is None and Pregunta.objects.filter(tipo="orden", enunciado="¿Cuál es?").exists()
    assert not PreguntaForm(data={**base, "tipo": "unica", "opciones": "A\nB", "correcta": "5"}).is_valid()   # no existe la alternativa 5
    unica = Pregunta.objects.filter(evaluacion=e, tipo="unica").first()
    assert PreguntaForm(instance=unica).fields["correcta"].initial == str(unica.respuesta + 1)
