"""Pruebas del prospector de LinkedIn (sin conexión a Gmail, a LinkedIn ni a internet)."""

import dataclasses
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from agentes_ebook.config import Config
from agentes_ebook.db import iso
from agentes_ebook.gmail import encabezado_desde_bytes
from prospector_linkedin import calificador, fuentes, notificador, secuencia, sincronizar
from prospector_linkedin.config import ConfigProspector
from prospector_linkedin.db import BaseProspector
from prospector_linkedin.mensajes import nota_invitacion, redactar
from prospector_linkedin.orquestador import ejecutar_ciclo, preparar_demo

AHORA = datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc)  # lunes, mediodía en Chile

CONNECTIONS_CSV = """Notes:
"When exporting your connection data, you may notice that some of the email addresses are missing."

First Name,Last Name,URL,Email Address,Company,Position,Connected On
Ana,Pérez Soto,https://www.linkedin.com/in/ana-perez-demo,,Constructora Ejemplo,Gerente General,12 Oct 2024
Luis,Gómez,https://www.linkedin.com/in/luis-gomez-demo,luis@empresa.example,"Servicios, Ltda.",Analista Contable,03 Jan 2023
""".encode("utf-8")


@pytest.fixture
def cfgp(tmp_path):
    base = dataclasses.replace(
        Config.desde_entorno(),
        gmail_usuario="",
        gmail_app_password="",
        notificar_a="",
        anthropic_api_key="",
        link_agenda="",
        mercado_objetivo=["Chile"],
        zona_horaria="America/Santiago",
        ruta_db=str(tmp_path / "test.db"),
    )
    return ConfigProspector(base=base, recurso_link="https://sitio.example/ebook")


@pytest.fixture
def db(cfgp):
    base = BaseProspector(cfgp.base.ruta_db)
    yield base
    base.cerrar()


def agregar(db, cfgp, nombre="Ana Pérez Soto", cargo="Gerente General", empresa="Constructora Ejemplo", grado=2, **extra):
    reg = fuentes.normalizar_registro(
        {"nombre": nombre, "cargo": cargo, "empresa": empresa, "ubicacion": "Santiago, Chile", **extra}, "prueba", grado
    )
    pid, _ = db.guardar_prospecto(reg)
    calificador.calificar(db, cfgp)
    return pid


def aviso(asunto, de_nombre="LinkedIn", de_correo="invitations@linkedin.com", fecha=AHORA):
    return {"asunto": asunto, "de_nombre": de_nombre, "de_correo": de_correo, "fecha": fecha}


# ----------------------------------------------------------------------------- fuentes
def test_importa_la_exportacion_de_contactos_de_linkedin(db, cfgp):
    nuevos, existentes = fuentes.desde_archivo(db, "Connections.csv", CONNECTIONS_CSV)
    assert (nuevos, existentes) == (2, 0)
    ana = next(p for p in db.prospectos() if p["nombre"] == "Ana Pérez Soto")
    assert ana["grado"] == 1 and ana["etapa"] == "Conectado" and ana["fuente"] == "contactos"
    assert ana["empresa"] == "Constructora Ejemplo" and ana["cargo"] == "Gerente General"
    assert ana["conectado_en"].startswith("2024-10-12")
    assert fuentes.desde_archivo(db, "Connections.csv", CONNECTIONS_CSV) == (0, 2)  # no duplica


def test_importa_lista_generica_y_separa_el_titular(db, cfgp):
    df = pd.DataFrame(
        [
            {"Full Name": "Eva Lima", "Headline": "CEO en Logística Lima", "Profile URL": "linkedin.com/in/eva-lima/", "Location": "Santiago"},
            {"Full Name": "", "Headline": "Sin nombre"},
            {"Full Name": "Eva Lima", "Headline": "CEO", "Profile URL": "https://www.linkedin.com/in/Eva-Lima?trk=x"},
        ]
    )
    assert fuentes.desde_tabla(db, df) == (1, 1)
    eva = db.prospectos()[0]
    assert (eva["cargo"], eva["empresa"], eva["grado"], eva["etapa"]) == ("CEO", "Logística Lima", 2, "Por invitar")
    assert eva["url"] == "https://linkedin.com/in/eva-lima/"
    assert fuentes.link_perfil({"nombre": "Eva Lima", "empresa": "Acme"}).endswith("keywords=Eva%20Lima%20Acme")


def test_suma_leads_del_ebook(db, cfgp):
    reg = {
        "email": "ana@constructora.example", "nombre": "Ana Pérez", "cargo": "Gerente General", "empresa": "Constructora",
        "mensaje_id": "<1@x>", "fecha": iso(AHORA), "fuente": "gmail",
    }
    lead_id, _ = db.registrar_descarga(reg)
    db.actualizar_lead(lead_id, segmento="Caliente", pais="Chile")
    assert fuentes.desde_leads_ebook(db) == (1, 0)
    p = db.prospectos()[0]
    assert p["lead_id"] == lead_id and p["fuente"] == "ebook"
    calificador.calificar(db, cfgp)
    assert "ebook" in redactar("invitacion", db.prospectos()[0], cfgp)


# ----------------------------------------------------------------------------- calificador
def test_califica_segun_el_cliente_ideal(db, cfgp):
    decisor = db.prospecto(agregar(db, cfgp))
    assert decisor["prioridad"] == "A" and decisor["nivel"] == "Decisor" and decisor["score"] >= 70
    finanzas = db.prospecto(agregar(db, cfgp, "Luis Gómez", "Jefe de Finanzas", "Servicios Gómez"))
    assert finanzas["nivel"] == "Finanzas" and finanzas["prioridad"] == "B"
    analista = db.prospecto(agregar(db, cfgp, "Eva Lima", "Analista de marketing", ""))
    assert analista["prioridad"] == "C"
    for cargo in ("Estudiante de Ingeniería", "Fractional CFO", "Gerente General | Open to work"):
        excluido = db.prospecto(agregar(db, cfgp, f"Persona {cargo}", cargo, "Empresa"))
        assert (excluido["score"], excluido["nivel"]) == (0, "Excluido")


def test_contacto_de_primer_grado_suma_puntos(cfgp):
    base = {"cargo": "Socio", "empresa": "Agrícola", "ubicacion": ""}
    assert calificador.puntuar({**base, "grado": 1}, cfgp)[0] == calificador.puntuar({**base, "grado": 2}, cfgp)[0] + 10


# ----------------------------------------------------------------------------- mensajes
def test_la_nota_de_invitacion_respeta_el_limite(cfgp):
    largo = {"id": 1, "nombre": "Maximiliano Fernández", "empresa": "Sociedad Constructora e Inmobiliaria del Pacífico Sur y Compañía Limitada", "nivel": "Decisor"}
    for limite in (300, 200, 120, 60):
        nota = nota_invitacion(largo, dataclasses.replace(cfgp, nota_max=limite))
        assert 0 < len(nota) <= limite and nota.startswith("Hola Maximiliano")
    assert "Acme" in nota_invitacion({"id": 2, "nombre": "ana", "empresa": "Acme", "nivel": "Decisor"}, cfgp)


def test_mensajes_usan_tu_oferta_y_tus_enlaces(cfgp):
    p = {"id": 1, "nombre": "Ana Pérez", "empresa": "Acme", "grado": 2, "invitado_en": iso(AHORA)}
    assert "gracias por conectar" in redactar("mensaje_1", p, cfgp) and "Acme" in redactar("mensaje_1", p, cfgp)
    assert "tanto tiempo" in redactar("mensaje_1", {**p, "grado": 1, "invitado_en": None}, cfgp)
    assert cfgp.recurso_link in redactar("mensaje_2", p, cfgp)
    con_agenda = dataclasses.replace(cfgp, base=dataclasses.replace(cfgp.base, link_agenda="https://agenda.example/yo"))
    assert "https://agenda.example/yo" in redactar("mensaje_3", p, con_agenda)
    assert "https://agenda.example/yo" in redactar("reunion", p, con_agenda)


# ----------------------------------------------------------------------------- secuencia
def test_secuencia_completa_de_un_prospecto(db, cfgp):
    pid = agregar(db, cfgp)
    secuencia.planificar(db, cfgp, AHORA)
    (invitacion,) = secuencia.cola_de_hoy(db, cfgp, AHORA)
    assert invitacion["tipo"] == "invitacion" and len(invitacion["texto"]) <= cfgp.nota_max

    assert secuencia.marcar_hecho(db, invitacion["id"], AHORA)
    assert not secuencia.marcar_hecho(db, invitacion["id"], AHORA)  # no se cuenta dos veces
    assert db.prospecto(pid)["etapa"] == "Invitación enviada"
    secuencia.planificar(db, cfgp, AHORA)
    assert secuencia.cola_de_hoy(db, cfgp, AHORA) == []  # espera a que acepte

    dia2 = AHORA + timedelta(days=2)
    assert secuencia.marcar_conectado(db, pid, dia2)
    secuencia.planificar(db, cfgp, dia2)
    (m1,) = secuencia.cola_de_hoy(db, cfgp, dia2)
    assert m1["tipo"] == "mensaje_1" and "gracias por conectar" in m1["texto"]
    secuencia.marcar_hecho(db, m1["id"], dia2)
    assert db.prospecto(pid)["etapa"] == "En conversación"

    secuencia.planificar(db, cfgp, dia2)
    assert secuencia.cola_de_hoy(db, cfgp, dia2 + timedelta(days=3)) == []  # el mensaje 2 es a los 4 días
    (m2,) = secuencia.cola_de_hoy(db, cfgp, dia2 + timedelta(days=4))
    assert m2["tipo"] == "mensaje_2"
    secuencia.marcar_hecho(db, m2["id"], dia2 + timedelta(days=4))
    secuencia.planificar(db, cfgp, dia2 + timedelta(days=4))
    (m3,) = secuencia.cola_de_hoy(db, cfgp, dia2 + timedelta(days=12))
    assert m3["tipo"] == "mensaje_3"
    secuencia.marcar_hecho(db, m3["id"], dia2 + timedelta(days=12))

    secuencia.planificar(db, cfgp, dia2 + timedelta(days=13))
    assert db.prospecto(pid)["etapa"] == "En conversación"
    secuencia.planificar(db, cfgp, dia2 + timedelta(days=23))
    assert db.prospecto(pid)["etapa"] == "Sin respuesta"
    assert len(db.acciones(prospecto_id=pid)) == 4  # invitación + 3 mensajes, nada más


def test_respuesta_detiene_la_secuencia(db, cfgp):
    pid = agregar(db, cfgp, grado=1)
    secuencia.planificar(db, cfgp, AHORA)
    (m1,) = secuencia.cola_de_hoy(db, cfgp, AHORA)
    assert m1["tipo"] == "mensaje_1" and "tanto tiempo" in m1["texto"]  # contacto antiguo: sin invitación
    secuencia.marcar_hecho(db, m1["id"], AHORA)
    secuencia.planificar(db, cfgp, AHORA)
    assert secuencia.marcar_respondio(db, pid, AHORA + timedelta(days=1))
    secuencia.planificar(db, cfgp, AHORA + timedelta(days=30))
    assert db.prospecto(pid)["etapa"] == "Respondió"
    assert secuencia.cola_de_hoy(db, cfgp, AHORA + timedelta(days=30)) == []
    assert not db.acciones(prospecto_id=pid, estados=["pendiente"])


def test_limites_diarios_y_semanales(db, cfgp):
    cfgp = dataclasses.replace(cfgp, invitaciones_dia=3, invitaciones_semana=5, mensajes_dia=2)
    for i in range(8):
        agregar(db, cfgp, f"Decisor Número{i}", "Gerente General", f"Empresa {i}")
    for i in range(4):
        agregar(db, cfgp, f"Contacto Número{i}", "Socio", f"Sociedad {i}", grado=1)
    secuencia.planificar(db, cfgp, AHORA)
    cola = secuencia.cola_de_hoy(db, cfgp, AHORA)
    assert [a["tipo"] for a in cola] == ["mensaje_1"] * 2 + ["invitacion"] * 3  # mensajes primero
    for a in cola:
        secuencia.marcar_hecho(db, a["id"], AHORA)
    assert secuencia.cola_de_hoy(db, cfgp, AHORA + timedelta(hours=2)) == []  # cupo del día agotado
    manana = AHORA + timedelta(days=1)
    cola = secuencia.cola_de_hoy(db, cfgp, manana)
    assert [a["tipo"] for a in cola] == ["mensaje_1"] * 2 + ["invitacion"] * 2  # tope semanal: 5
    for a in cola:
        secuencia.marcar_hecho(db, a["id"], manana)
    assert not [a for a in secuencia.cola_de_hoy(db, cfgp, AHORA + timedelta(days=3)) if a["tipo"] == "invitacion"]
    assert len([a for a in secuencia.cola_de_hoy(db, cfgp, AHORA + timedelta(days=8)) if a["tipo"] == "invitacion"]) == 3


def test_prioridad_baja_y_excluidos_no_entran_a_la_cola(db, cfgp):
    agregar(db, cfgp, "Eva Lima", "Analista de marketing", "")
    agregar(db, cfgp, "Rut Paz", "Estudiante", "Universidad")
    secuencia.planificar(db, cfgp, AHORA)
    assert secuencia.cola_de_hoy(db, cfgp, AHORA) == []
    todas = dataclasses.replace(cfgp, prioridad_minima="C", excluir=[])
    calificador.calificar(db, todas)
    secuencia.planificar(db, todas, AHORA)
    assert len(secuencia.cola_de_hoy(db, todas, AHORA)) == 2


def test_invitacion_sin_aceptar_se_cierra_y_omitir_descarta(db, cfgp):
    pid, otro = agregar(db, cfgp), agregar(db, cfgp, "Luis Gómez", "CEO", "Gómez SpA")
    secuencia.planificar(db, cfgp, AHORA)
    acciones = {a["prospecto_id"]: a for a in secuencia.cola_de_hoy(db, cfgp, AHORA)}
    secuencia.marcar_hecho(db, acciones[pid]["id"], AHORA)
    secuencia.omitir(db, acciones[otro]["id"], AHORA)
    assert db.prospecto(otro)["etapa"] == "Descartado"
    secuencia.planificar(db, cfgp, AHORA + timedelta(days=22))
    p = db.prospecto(pid)
    assert p["etapa"] == "Sin respuesta" and "retira la invitación" in p["notas"]


def test_texto_editado_se_conserva(db, cfgp):
    agregar(db, cfgp)
    secuencia.planificar(db, cfgp, AHORA)
    (a,) = secuencia.cola_de_hoy(db, cfgp, AHORA)
    secuencia.editar_texto(db, a["id"], "  Hola Ana, mi versión.  ")

    class IA:
        def personalizar(self, *args):
            raise AssertionError("no debe reescribir un texto editado a mano")

    assert secuencia.personalizar_cola(db, cfgp, IA(), AHORA) == 0
    assert secuencia.cola_de_hoy(db, cfgp, AHORA)[0]["texto"] == "Hola Ana, mi versión."


def test_ia_personaliza_solo_la_cola_y_respeta_el_limite(db, cfgp):
    agregar(db, cfgp)
    agregar(db, cfgp, "Luis Gómez", "CEO", "Gómez SpA")
    secuencia.planificar(db, cfgp, AHORA)

    class IA:
        def personalizar(self, p, tipo, texto, limite):
            assert limite == cfgp.nota_max
            return f"Hola {p['nombre'].split()[0]}, texto único." if p["nombre"].startswith("Ana") else None

    assert secuencia.personalizar_cola(db, cfgp, IA(), AHORA) == 1
    assert secuencia.personalizar_cola(db, cfgp, IA(), AHORA) == 0  # no reintenta
    textos = {a["nombre"]: a for a in secuencia.cola_de_hoy(db, cfgp, AHORA)}
    assert textos["Ana Pérez Soto"]["texto"] == "Hola Ana, texto único." and textos["Ana Pérez Soto"]["personalizado_ia"] == 1
    assert textos["Luis Gómez"]["texto"].startswith("Hola Luis")


# ----------------------------------------------------------------------------- sincronización con Gmail
@pytest.mark.parametrize(
    "encabezado, esperado",
    [
        (aviso("Ana Pérez Soto aceptó tu invitación"), ("acepto", "ana perez soto")),
        (aviso("Ana Pérez ha aceptado tu invitación. Descubre su red"), ("acepto", "ana perez")),
        (aviso("Ana Perez accepted your invitation"), ("acepto", "ana perez")),
        (aviso("Rodrigo, comienza una conversación con tu nuevo contacto, Ana Pérez"), ("acepto", "ana perez")),
        (aviso("Ana Pérez te ha enviado un mensaje"), ("mensaje", "ana perez")),
        (aviso("Ana Pérez just messaged you"), ("mensaje", "ana perez")),
        (aviso("Sobre tu propuesta", "Ana Pérez via LinkedIn", "messaging-digest-noreply@linkedin.com"), ("mensaje", "ana perez")),
        (aviso("Ana Pérez te ha invitado a conectar", "Ana Pérez via LinkedIn"), (None, "")),
        (aviso("Tienes 5 nuevas visitas a tu perfil"), (None, "")),
    ],
)
def test_clasifica_avisos_de_linkedin(encabezado, esperado):
    assert sincronizar.clasificar(encabezado) == esperado


class GmailFalso:
    def __init__(self, encabezados):
        self.encabezados = encabezados

    def encabezados_de(self, remitente, desde, limite=400):
        assert remitente == "linkedin.com"
        return [e for e in self.encabezados if e["fecha"] >= desde]


def test_sincroniza_aceptaciones_y_respuestas(db, cfgp):
    ana, luis = agregar(db, cfgp), agregar(db, cfgp, "Luis Gómez", "CEO", "Gómez SpA")
    eva = agregar(db, cfgp, "Eva Lima", "Socia", "Lima y Cía", grado=1)
    secuencia.planificar(db, cfgp, AHORA)
    for a in secuencia.cola_de_hoy(db, cfgp, AHORA):
        secuencia.marcar_hecho(db, a["id"], AHORA)
    despues = AHORA + timedelta(days=1)
    gmail = GmailFalso(
        [
            aviso("Ana Pérez aceptó tu invitación", fecha=despues),
            aviso("Eva Lima te envió un mensaje", de_correo="messages-noreply@linkedin.com", fecha=despues),
            aviso("Eva Lima te envió un mensaje", de_correo="messages-noreply@linkedin.com", fecha=AHORA - timedelta(days=3)),
            aviso("Pedro Desconocido aceptó tu invitación", fecha=despues),
        ]
    )
    aceptaron, respondieron = sincronizar.sincronizar(db, cfgp, gmail, despues + timedelta(hours=1))
    assert (aceptaron, respondieron) == ([ana], [eva])
    assert db.prospecto(ana)["etapa"] == "Conectado" and db.prospecto(ana)["grado"] == 1
    assert db.prospecto(eva)["etapa"] == "Respondió"
    assert db.prospecto(luis)["etapa"] == "Invitación enviada"
    assert sincronizar.sincronizar(db, cfgp, gmail, despues + timedelta(hours=2)) == ([], [])  # idempotente


def test_no_adivina_con_nombres_ambiguos(db, cfgp):
    a = agregar(db, cfgp, "Ana Pérez Soto", empresa="Uno")
    b = agregar(db, cfgp, "Ana Pérez Lagos", empresa="Dos")
    for pid in (a, b):
        db.actualizar_prospecto(pid, etapa="Invitación enviada", invitado_en=iso(AHORA))
    prospectos = db.prospectos()
    assert sincronizar.detectar([aviso("Ana Pérez aceptó tu invitación")], prospectos) == ({}, {})
    assert sincronizar.detectar([aviso("Ana aceptó tu invitación")], prospectos) == ({}, {})
    assert list(sincronizar.detectar([aviso("Ana Pérez Lagos aceptó tu invitación")], prospectos)[0]) == [b]


def test_lee_encabezados_de_un_correo():
    crudo = (
        "From: =?UTF-8?Q?Ana_P=C3=A9rez_via_LinkedIn?= <messaging-digest-noreply@linkedin.com>\r\n"
        "Subject: =?UTF-8?Q?Ana_P=C3=A9rez_te_envi=C3=B3_un_mensaje?=\r\n"
        "Date: Mon, 16 Nov 2026 12:00:00 -0300\r\n\r\n"
    ).encode()
    e = encabezado_desde_bytes(crudo)
    assert e["de_nombre"] == "Ana Pérez via LinkedIn" and e["de_correo"] == "messaging-digest-noreply@linkedin.com"
    assert e["asunto"] == "Ana Pérez te envió un mensaje" and e["fecha"] == AHORA
    assert sincronizar.clasificar(e) == ("mensaje", "ana perez")
    assert encabezado_desde_bytes(b"Subject: sin fecha\r\n\r\n") is None


# ----------------------------------------------------------------------------- avisos y ciclo completo
def test_correo_de_la_cola_y_novedades(db, cfgp, monkeypatch):
    pid = agregar(db, cfgp, nombre="Ana <b>Pérez</b> Soto")
    secuencia.planificar(db, cfgp, AHORA)
    cola = secuencia.cola_de_hoy(db, cfgp, AHORA)
    titulo, html, texto = notificador.construir_cola(db, cfgp, cola)
    assert "0 mensajes y 1 invitaciones" in titulo
    assert "<b>Pérez</b>" not in html and "&lt;b&gt;" in html  # los datos del perfil se escapan
    assert cola[0]["texto"] in texto and "linkedin.com" in texto

    assert notificador.enviar_cola(db, cfgp, forzar=True, ahora=AHORA) == "sin_configurar"
    enviados = []
    monkeypatch.setattr(notificador, "enviar_correo", lambda cfg, para, asunto, texto, html: enviados.append((para, asunto)) or "ok")
    con_gmail = dataclasses.replace(
        cfgp, base=dataclasses.replace(cfgp.base, gmail_usuario="yo@x.example", gmail_app_password="x", notificar_a="yo@x.example", resumen_hora=8)
    )
    assert notificador.enviar_cola(db, con_gmail, ahora=AHORA) == "enviado"
    assert notificador.enviar_cola(db, con_gmail, ahora=AHORA + timedelta(hours=1)) is None  # una vez al día
    assert notificador.enviar_novedades(db, con_gmail, [pid], []) == "enviado"
    assert len(enviados) == 2 and "aceptadas" in enviados[1][1]


def test_demo_y_ciclo_completo(db, cfgp):
    r = preparar_demo(db, cfgp)
    assert not r.errores and r.cola > 0 and r.conteo["A"] > 0
    etapas = {p["etapa"] for p in db.prospectos()}
    assert {"Por invitar", "Invitación enviada", "En conversación", "Respondió"} <= etapas
    assert all(p["fuente"] == "demo" for p in db.prospectos())
    assert notificador.enviar_cola(db, cfgp, forzar=True) is None  # a la demo nunca se le avisa
    otra = ejecutar_ciclo(db, cfgp, leer_gmail=False, notificar=False, usar_ia=False)
    assert otra.creadas == 0  # repetir el ciclo no duplica acciones
    assert db.borrar_fuente_li("demo") == 14 and db.prospectos() == [] and db.acciones() == []
