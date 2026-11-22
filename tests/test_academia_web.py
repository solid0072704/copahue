"""La academia como aplicación: cuentas, catálogo, precios por país, compra, aula, pruebas y certificado."""

import hashlib
import hmac
import json
from datetime import timedelta
from decimal import Decimal
from unittest import mock

import pytest

pytest.importorskip("django")
pytest.importorskip("allauth")

from django.core import mail  # noqa: E402
from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402
from django.http import QueryDict  # noqa: E402
from django.urls import reverse  # noqa: E402
from django.utils import timezone  # noqa: E402

from academia.cursos import acceso, pruebas  # noqa: E402
from academia.cursos.importar import importar  # noqa: E402
from academia.cursos.models import Avance, Certificado, Curso, Inscripcion, Intento, Leccion, Perfil, Pregunta  # noqa: E402
from academia.pagos import cambio, precios  # noqa: E402
from academia.pagos.models import Moneda, Orden  # noqa: E402
from academia.pagos.pasarelas import flow, mercadopago  # noqa: E402

pytestmark = pytest.mark.django_db
PAQUETE = "academia.contenido.claude_finanzas"
CLAVE = "una-clave-larga-y-rara-2026"
_ = "\xa0"            # el espacio que no corta la línea entre el símbolo, el número y el código de un precio


@pytest.fixture(autouse=True)
def sin_internet(settings):
    """Las pruebas no salen a buscar tipos de cambio: los pone cada una con `dolar_a`."""
    settings.CAMBIO_AUTOMATICO = False


def dolar_a(**valores) -> None:
    """Deja el tipo de cambio de esas monedas como recién llegado de la fuente: `dolar_a(CLP=1000, MXN=20)`."""
    for codigo, valor in valores.items():
        assert Moneda.objects.filter(codigo=codigo).update(por_dolar=valor, actualizada=timezone.now() if valor else None, fija=False) == 1


@pytest.fixture
def curso(settings, tmp_path) -> Curso:
    """El primer curso, publicado a 50 dólares, con el dólar a 1.000 pesos chilenos."""
    settings.MEDIA_ROOT = tmp_path
    settings.PASARELAS = ["prueba"]
    dolar_a(CLP=1000)
    c = importar(PAQUETE)
    c.precio_usd, c.publicado = 50, True
    c.save()
    return c


def persona(django_user_model, nombre: str, pais: str = ""):
    u = django_user_model.objects.create_user(username=nombre, email=f"{nombre}@example.com", password=CLAVE, first_name=nombre.title())
    if pais:
        Perfil.guardar_pais(u, pais)
    return u


@pytest.fixture
def alumno(django_user_model):
    """Ana, que al registrarse dijo que vive en Chile."""
    ana = django_user_model.objects.create_user(username="ana", email="ana@example.com", password=CLAVE, first_name="Ana Soto")
    Perfil.guardar_pais(ana, "CL")
    return ana


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
    assert otra_vez.pk == curso.pk and (otra_vez.precio_usd, otra_vez.publicado) == (50, True)
    assert (Curso.objects.count(), Leccion.objects.count(), Pregunta.objects.count()) == (1, 97, antes)
    m8 = curso.modulos.get(clave="m08")
    assert "CRM" in m8.titulo and m8.grupo == "Conectar Claude con tus sistemas" and m8.lecciones.count() == 7


def test_el_catalogo_muestra_solo_lo_publicado(client, curso):
    assert curso.titulo in client.get(reverse("catalogo")).content.decode()
    pagina = client.get(reverse("curso", args=[curso.slug])).content.decode()
    assert f"Comprar por US${_}50" in pagina and "CRM + Claude" in pagina and reverse("comprar", args=[curso.slug]) in pagina
    curso.publicado = False
    curso.save()
    assert curso.titulo not in client.get(reverse("catalogo")).content.decode()
    assert client.get(reverse("curso", args=[curso.slug])).status_code == 404


# --- Cuentas -----------------------------------------------------------------
def test_se_crea_la_cuenta_con_correo_y_contrasena_y_se_entra_con_ella(client, django_user_model):
    datos = dict(nombre="Luis Rojas", email="luis@example.com", password1=CLAVE, password2=CLAVE)
    sin_pais = client.post(reverse("account_signup"), datos)
    assert sin_pais.status_code == 200 and "Elige tu país." in sin_pais.content.decode() and not django_user_model.objects.exists()
    r = client.post(reverse("account_signup"), dict(datos, pais="PE"))
    assert r.status_code == 302 and r.url == reverse("mi_aprendizaje")
    luis = django_user_model.objects.get(email="luis@example.com")
    assert luis.first_name == "Luis Rojas" and luis.check_password(CLAVE) and Perfil.objects.get(usuario=luis).pais == "PE"
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
    assert (orden.precio_usd, orden.cambio) == (50, 1000)                                            # queda escrito con qué dólar se calculó
    assert not Inscripcion.objects.exists()
    with django_capture_on_commit_callbacks(execute=True):
        r = client.post(reverse("resolver_prueba"), dict(orden=orden.codigo, resultado="aprobar"))
    orden.refresh_from_db()
    assert orden.estado == "pagada" and Inscripcion.objects.filter(usuario=alumno, curso=curso, origen="compra").count() == 1
    assert len(mail.outbox) == 1 and curso.titulo in mail.outbox[0].subject and mail.outbox[0].to == ["ana@example.com"]
    assert f"Recibimos tu pago de $50.000{_}CLP (US${_}50)" in mail.outbox[0].body
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
    curso.precio_usd = 0
    curso.save()
    dolar_a(CLP=None)                                                                               # y no necesita tipo de cambio
    assert "Inscribirme gratis" in client.get(reverse("curso", args=[curso.slug])).content.decode()
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


# --- Precios por país --------------------------------------------------------
def test_cada_quien_ve_el_precio_en_dolares_y_en_la_moneda_de_su_pais(client, curso, django_user_model):
    dolar_a(MXN=20, PEN="3.4")
    url = reverse("curso", args=[curso.slug])

    def pagina(nombre: str = "", pais: str = "") -> str:
        if nombre:
            client.force_login(persona(django_user_model, nombre, pais))
        return client.get(url).content.decode()

    sin_cuenta = pagina()                                                                           # en dólares, y cómo verlo en su moneda
    assert f"US${_}50" in sin_cuenta and "CLP" not in sin_cuenta and "En tu moneda" not in sin_cuenta
    assert "Crea tu cuenta</a> y los verás también en la moneda de tu país" in sin_cuenta

    chile = pagina("carla", "CL")                                                                   # la moneda en que se cobra: el valor exacto
    assert f"Comprar por US${_}50" in chile and f"En tu moneda: <b>$50.000{_}CLP</b>" in chile and "≈" not in chile
    assert "se calcula con el valor del dólar del " in chile and '<a href="https://www.exchangerate-api.com"' in chile and "Rates By Exchange Rate API" in chile
    assert f"US${_}50<small>$50.000{_}CLP</small>" in client.get(reverse("catalogo")).content.decode()

    mexico = pagina("memo", "MX")                                                                   # otra moneda: una referencia
    assert f"En tu moneda: <b>≈{_}$1.000{_}MXN</b>" in mexico and "CLP" not in mexico
    assert "El precio en pesos mexicanos es una referencia" in mexico and "el pago se procesa en pesos chilenos" in mexico

    assert f"≈{_}S/{_}170" in pagina("pepe", "PE")                                                  # un símbolo propio no lleva código

    for nombre, pais in (("eva", "EC"), ("otto", "XX"), ("tito", "AR")):                            # el dólar, otro país, o una moneda sin tipo de cambio
        solo_dolares = pagina(nombre, pais)
        assert f"US${_}50" in solo_dolares and "En tu moneda" not in solo_dolares and "nota-precio" not in solo_dolares


def test_quien_no_ha_dicho_su_pais_lo_indica_en_su_cuenta_y_vuelve_a_donde_estaba(client, curso, django_user_model):
    sofia = persona(django_user_model, "sofia")                                                     # sin país: una cuenta creada en el panel
    client.force_login(sofia)
    url, cuenta = reverse("curso", args=[curso.slug]), reverse("cuenta")
    pagina = client.get(url).content.decode()
    assert "Indica tu país</a> para verlos también en tu moneda" in pagina and "En tu moneda" not in pagina and f'href="{cuenta}?siguiente=' in pagina
    formulario = client.get(cuenta, dict(siguiente=url)).content.decode()
    assert 'name="pais"' in formulario and 'value="Sofia"' in formulario and f'name="siguiente" value="{url}"' in formulario
    assert client.post(cuenta, dict(nombre="Sofía Vera", pais="ZZ")).status_code == 200            # un país que no está en la lista
    r = client.post(cuenta, dict(nombre="Sofía Vera", pais="CL", siguiente=url))
    assert r.status_code == 302 and r.url == url
    sofia.refresh_from_db()
    assert (sofia.first_name, sofia.last_name, Perfil.objects.get(usuario=sofia).pais) == ("Sofía Vera", "", "CL")
    assert f"$50.000{_}CLP" in client.get(url).content.decode()
    assert client.post(cuenta, dict(nombre="Sofía Vera", pais="MX", siguiente="https://otro.example/")).url == cuenta   # no lleva a otro sitio
    assert 'value="MX" selected' in client.get(cuenta).content.decode() and "Mi nombre y mi país" in client.get(reverse("mi_aprendizaje")).content.decode()
    client.logout()
    assert client.get(cuenta).status_code == 302


def test_quien_llega_por_primera_vez_con_google_tambien_dice_su_pais(client, curso, settings, django_user_model):
    from allauth.socialaccount.adapter import get_adapter
    from allauth.socialaccount.models import SocialAccount
    from django.test import RequestFactory
    settings.CON_GOOGLE = True
    settings.SOCIALACCOUNT_PROVIDERS = {"google": {"APPS": [{"client_id": "id", "secret": "secreto", "key": ""}]}}
    assert settings.SOCIALACCOUNT_AUTO_SIGNUP is False
    # lo que queda en la sesión cuando una persona nueva vuelve de Google: su nombre y su correo, ya verificado
    pedido = RequestFactory().get("/")
    de_google = get_adapter().get_provider(pedido, "google").sociallogin_from_response(pedido, dict(
        sub="g-123", email="gabi@example.com", email_verified=True, given_name="Gabriela", family_name="Mora"))
    sesion = client.session
    sesion["socialaccount_sociallogin"] = de_google.serialize()
    sesion.save()
    url = reverse("socialaccount_signup")
    pagina = client.get(url).content.decode()
    assert "Un último paso" in pagina and 'value="Gabriela Mora"' in pagina and 'name="pais"' in pagina and 'type="password"' not in pagina
    datos = dict(nombre="Gabriela Mora", email="gabi@example.com")
    assert client.post(url, datos).status_code == 200 and not django_user_model.objects.exists()   # sin país no se crea la cuenta
    assert client.post(url, dict(datos, pais="MX")).status_code == 302
    gabi = django_user_model.objects.get(email="gabi@example.com")
    assert (gabi.first_name, gabi.last_name, Perfil.objects.get(usuario=gabi).pais) == ("Gabriela Mora", "", "MX")
    assert SocialAccount.objects.filter(user=gabi, provider="google", uid="g-123").exists() and not gabi.has_usable_password()
    assert client.get(reverse("mi_aprendizaje")).status_code == 200                                 # y queda con la sesión iniciada


def test_se_cobra_en_la_moneda_de_la_pasarela_con_el_dolar_del_dia(client, curso, django_user_model):
    dolar_a(CLP="1014.084257", MXN="19.123957")
    curso.precio_usd = Decimal("49.90")
    curso.save()
    client.force_login(persona(django_user_model, "memo", "MX"))
    comprar = reverse("comprar", args=[curso.slug])
    pagina = client.get(comprar).content.decode()
    # 49,90 dólares a 1.014,084257 son 50.602,80 pesos: se cobran 50.603; en pesos mexicanos, cerca de 954
    assert f"US${_}49,90" in pagina and f"Pagar $50.603{_}CLP" in pagina and 'name="monto" value="50603"' in pagina
    assert f"<dt>Se cobra en pesos chilenos</dt><dd>$50.603{_}CLP</dd>" in pagina and f"<dt>En pesos mexicanos, cerca de</dt><dd>$954{_}MXN</dd>" in pagina
    assert f"Tipo de cambio: US${_}1 = $1.014,08{_}CLP" in pagina and "Rates By Exchange Rate API" in pagina and "la hace tu banco o tu tarjeta" in pagina
    client.post(comprar, dict(pasarela="prueba", monto="50603"))
    orden = Orden.objects.get()
    assert (orden.monto, orden.moneda, orden.precio_usd, orden.cambio) == (50603, "CLP", Decimal("49.90"), Decimal("1014.084257"))
    assert f"Simular el pago de $50.603{_}CLP" in client.get(reverse("pago_de_prueba", args=[orden.codigo])).content.decode()
    assert f"$50.603{_}CLP (US${_}49,90)" in client.get(reverse("orden", args=[orden.codigo])).content.decode()
    client.force_login(persona(django_user_model, "carla", "CL"))                                   # a quien paga en su moneda no se le habla de conversión
    chile = client.get(comprar).content.decode()
    assert f"Pagar $50.603{_}CLP" in chile and "cerca de" not in chile and "tu banco" not in chile


def test_si_el_dolar_cambia_entre_ver_y_pagar_no_se_cobra_un_monto_que_no_se_mostro(client, curso, alumno):
    client.force_login(alumno)
    comprar = reverse("comprar", args=[curso.slug])
    assert 'name="monto" value="50000"' in client.get(comprar).content.decode()
    dolar_a(CLP=1020)
    r = client.post(comprar, dict(pasarela="prueba", monto="50000"), follow=True)
    assert not Orden.objects.exists() and f"el curso queda en $51.000{_}CLP" in r.content.decode()
    client.post(comprar, dict(pasarela="prueba", monto="51000"))
    assert Orden.objects.get().monto == 51000


def test_sin_tipo_de_cambio_de_la_moneda_de_cobro_no_se_puede_pagar(client, curso, alumno):
    dolar_a(CLP=None)
    client.force_login(alumno)
    comprar = reverse("comprar", args=[curso.slug])
    pagina = client.get(comprar).content.decode()
    assert "no podemos calcular el precio en pesos chilenos" in pagina and "Pagar" not in pagina
    r = client.post(comprar, dict(pasarela="prueba"), follow=True)
    assert not Orden.objects.exists() and "No se hizo ningún cobro" in r.content.decode()
    assert f"Comprar por US${_}50" in client.get(reverse("curso", args=[curso.slug])).content.decode()   # el precio en dólares se sigue viendo


def test_si_las_pasarelas_cobran_en_dolares_el_monto_es_el_precio(client, curso, alumno, settings):
    settings.MONEDA = "USD"
    curso.precio_usd = Decimal("49.90")
    curso.save()
    client.force_login(alumno)
    comprar = reverse("comprar", args=[curso.slug])
    pagina = client.get(comprar).content.decode()
    assert f"Pagar US${_}49,90" in pagina and 'name="monto" value="49.90"' in pagina and "Tipo de cambio" not in pagina
    assert f"<dt>En pesos chilenos, cerca de</dt><dd>$49.900{_}CLP</dd>" in pagina                  # para Ana, en Chile, ahora es una referencia
    client.post(comprar, dict(pasarela="prueba", monto="49.90"))
    orden = Orden.objects.get()
    assert (orden.monto, orden.moneda, orden.cambio, orden.en_dolares) == (Decimal("49.90"), "USD", 1, "")
    assert precios.para_pasarela(orden.monto) == 49.9 and precios.para_pasarela(Decimal("50000.00")) == 50000


def test_los_montos_se_escriben_como_se_leen_en_espanol():
    def moneda(codigo: str, simbolo: str, decimales: int) -> Moneda:
        return Moneda(codigo=codigo, nombre=codigo, simbolo=simbolo, decimales=decimales)
    dolar, mexicano = moneda("USD", "US$", 2), moneda("MXN", "$", 2)
    assert precios.formato(Decimal("100394.4"), moneda("CLP", "$", 0)) == f"$100.394{_}CLP"
    assert precios.formato(99, dolar) == f"US${_}99" and precios.formato(Decimal("49.9"), dolar) == f"US${_}49,90"
    assert precios.formato(Decimal("1893.2717"), mexicano) == f"$1.893,27{_}MXN"
    assert precios.formato(Decimal("1893.2717"), mexicano, aproximado=True) == f"$1.893{_}MXN"       # una referencia no lleva centavos…
    assert precios.formato(Decimal("9.016"), moneda("EUR", "€", 2), aproximado=True) == f"€{_}9,02"  # …salvo en montos chicos
    assert precios.formato(Decimal("1234567.5"), moneda("PYG", "₲", 0)) == f"₲{_}1.234.568"
    assert precios.mismo_monto("50000.00", Decimal("50000")) and precios.mismo_monto(49.9, Decimal("49.90"))
    assert not precios.mismo_monto("50000.01", Decimal("50000")) and not precios.mismo_monto(None, Decimal("1")) and not precios.mismo_monto("x", Decimal("1"))


# --- Tipos de cambio ---------------------------------------------------------
def _de_la_fuente(hace_horas: float = 1) -> dict:
    """Una respuesta como las del servicio de tipos de cambio, con los valores del 22 de noviembre de 2026."""
    return dict(result="success", provider="https://www.exchangerate-api.com", base_code="USD",
                time_last_update_unix=int((timezone.now() - timedelta(hours=hace_horas)).timestamp()),
                rates=dict(USD=1, CLP=1014.084257, MXN=19.123957, PEN=3.441274, EUR=0.911256))


def test_los_tipos_de_cambio_llegan_de_la_fuente_sin_tocar_los_fijos_ni_aplicar_saltos_raros(settings, capsys):
    from django.core.management import CommandError, call_command
    dolar_a(CLP=1000, EUR="0.5", PEN="3.4")
    Moneda.objects.filter(codigo="PEN").update(fija=True)
    datos = _de_la_fuente()
    with mock.patch("academia.pagos.cambio.requests.get", return_value=_respuesta(datos)) as get:
        resultado = cambio.actualizar()
    assert get.call_args.args[0] == settings.CAMBIO_URL == "https://open.er-api.com/v6/latest/USD"
    m = {x.codigo: x for x in Moneda.objects.all()}
    assert (m["CLP"].por_dolar, m["MXN"].por_dolar) == (Decimal("1014.084257"), Decimal("19.123957"))   # la que tenía valor y la que no
    assert int(m["CLP"].actualizada.timestamp()) == datos["time_last_update_unix"] and m["CLP"].automatica  # con la fecha que dice la fuente
    assert (m["PEN"].por_dolar, m["USD"].por_dolar) == (Decimal("3.4"), 1) and not m["PEN"].automatica  # las fijas no se tocan
    assert m["EUR"].por_dolar == Decimal("0.5") and "0.911256" in m["EUR"].observacion                  # 82 % de diferencia: no se aplica sola
    assert m["ARS"].por_dolar is None                                                                   # la fuente no la trajo
    assert (set(resultado["actualizadas"]), resultado["rechazadas"]) == ({"CLP", "MXN"}, ["EUR"]) and "ARS" in resultado["sin_dato"]

    with mock.patch("academia.pagos.cambio.requests.get", return_value=_respuesta(dict(result="error"))):   # una respuesta inesperada no cambia nada
        with pytest.raises(ValueError):
            cambio.actualizar()
        with pytest.raises(CommandError):
            call_command("actualizar_monedas")
    assert Moneda.objects.get(codigo="CLP").por_dolar == Decimal("1014.084257")
    with mock.patch("academia.pagos.cambio.requests.get", return_value=_respuesta(datos)):
        call_command("actualizar_monedas")
    salida = capsys.readouterr().out
    assert "2 monedas actualizadas" in salida and "EUR" in salida and "fijo" in salida


def test_los_precios_van_a_buscar_el_dolar_cuando_el_guardado_ya_es_de_ayer(client, curso, alumno, settings, monkeypatch):
    import requests
    settings.CAMBIO_AUTOMATICO = True
    monkeypatch.setattr(cambio, "_ultimo_intento", None)
    client.force_login(alumno)
    url = reverse("curso", args=[curso.slug])

    def envejecer() -> None:
        Moneda.objects.filter(fija=False, actualizada__isnull=False).update(actualizada=timezone.now() - timedelta(hours=30))
    with mock.patch("academia.pagos.cambio.requests.get", return_value=_respuesta(_de_la_fuente())) as get:
        assert f"$50.000{_}CLP" in client.get(url).content.decode() and not get.called              # el guardado es de hoy: no se consulta
        envejecer()
        assert f"$50.704{_}CLP" in client.get(url).content.decode() and get.call_count == 1         # 50 × 1.014,084257
        client.get(url)
        assert get.call_count == 1                                                                  # ya está al día
        envejecer()
        client.get(url)
        assert get.call_count == 1                                                                  # y un mismo proceso no insiste antes de una hora
    monkeypatch.setattr(cambio, "_ultimo_intento", None)
    with mock.patch("academia.pagos.cambio.requests.get", side_effect=requests.ConnectionError("sin conexión")) as get:
        assert f"$50.704{_}CLP" in client.get(url).content.decode() and get.call_count == 1         # si la fuente falla, se sigue con lo guardado


def test_en_el_panel_se_fija_un_tipo_de_cambio_a_mano_y_se_pide_la_actualizacion(client, curso, django_user_model):
    client.force_login(django_user_model.objects.create_superuser(username="admin", email="admin@example.com", password=CLAVE))
    lista, clp = reverse("admin:pagos_moneda_changelist"), Moneda.objects.get(codigo="CLP")
    assert "pesos chilenos" in client.get(lista).content.decode()
    r = client.post(reverse("admin:pagos_moneda_change", args=[clp.pk]),
                    dict(codigo="CLP", nombre="pesos chilenos", simbolo="$", decimales=0, por_dolar="1010,5", fija="on"))
    clp.refresh_from_db()
    assert r.status_code == 302 and (clp.por_dolar, clp.fija, clp.automatica) == (Decimal("1010.5"), True, False)
    with mock.patch("academia.pagos.cambio.requests.get", return_value=_respuesta(_de_la_fuente())):
        r = client.post(lista, dict(action="actualizar_ahora", _selected_action=[clp.pk]), follow=True)
    clp.refresh_from_db()
    assert "3 monedas actualizadas" in r.content.decode() and clp.por_dolar == Decimal("1010.5")    # llegaron las otras tres; la fija no cambió
    assert Moneda.objects.get(codigo="MXN").por_dolar == Decimal("19.123957")
    # en la lista se corrige un valor a mano, con coma: deja de ser un valor de la fuente, y los demás conservan su fecha
    monedas = list(Moneda.objects.all())
    filas = {"form-TOTAL_FORMS": len(monedas), "form-INITIAL_FORMS": len(monedas), "_save": "Guardar"}
    for i, m in enumerate(monedas):
        filas.update({f"form-{i}-id": m.pk, f"form-{i}-por_dolar": "" if m.por_dolar is None else str(m.por_dolar).replace(".", ",")})
        filas.update({f"form-{i}-fija": "on"} if m.fija else {})
        if m.codigo == "MXN":
            filas[f"form-{i}-por_dolar"] = "20,5"
    r = client.post(lista, filas)
    mxn, pen = Moneda.objects.get(codigo="MXN"), Moneda.objects.get(codigo="PEN")
    assert r.status_code == 302 and (mxn.por_dolar, mxn.actualizada, mxn.automatica) == (Decimal("20.5"), None, False)
    assert pen.por_dolar == Decimal("3.441274") and pen.automatica
    client.force_login(persona(django_user_model, "carla", "CL"))                                   # y los precios usan el valor fijo, sin nombrar la fuente
    pagina = client.get(reverse("curso", args=[curso.slug])).content.decode()
    assert f"$50.525{_}CLP" in pagina and "Rates By Exchange Rate API" not in pagina and "valor del dólar." in pagina


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
    # el precio en dólares se cambia desde la lista de cursos, escrito con coma
    lista = reverse("admin:cursos_curso_changelist")
    assert 'name="form-0-precio_usd" value="50,00"' in client.get(lista).content.decode()
    datos = {"form-TOTAL_FORMS": 1, "form-INITIAL_FORMS": 1, "form-0-id": curso.pk, "form-0-precio_usd": "59,90", "form-0-publicado": "on", "_save": "Guardar"}
    assert client.post(lista, datos).status_code == 302
    curso.refresh_from_db()
    assert curso.precio_usd == Decimal("59.90") and curso.publicado and not curso.destacado


# --- Varios cursos y carga a mano --------------------------------------------
def test_un_segundo_curso_creado_a_mano_convive_con_el_primero(client, curso, alumno):
    """La academia es un catálogo: un curso nuevo, creado en el panel, se vende y se cursa por separado."""
    from academia.cursos.models import Modulo
    nuevo = Curso.objects.create(slug="excel-para-el-cierre", titulo="Excel para el cierre mensual", bajada="Plantillas y controles", precio_usd=30,
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
        slug=curso.slug, titulo=curso.titulo, bajada=curso.bajada, instructor=curso.instructor_id, precio_usd="49,90", orden=0, certifica="evaluaciones",
        logros="Leer un balance\n\nProyectar la caja\n", preguntas="¿Sirve para pymes? | Sí, el caso es una pyme."))
    assert f.is_valid(), f.errors
    guardado = f.save()
    assert guardado.logros == ["Leer un balance", "Proyectar la caja"] and guardado.preguntas == [["¿Sirve para pymes?", "Sí, el caso es una pyme."]]
    assert guardado.precio_usd == Decimal("49.90")                                                  # el precio en dólares admite centavos
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
