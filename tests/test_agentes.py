"""Pruebas de los agentes del ebook (sin conexión a Gmail ni a internet)."""

import dataclasses
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import pandas as pd
import pytest

from agentes_ebook import analista, captura, notificador, reuniones, seguimiento
from agentes_ebook.config import Config
from agentes_ebook.db import BaseDatos, iso
from agentes_ebook.gmail import enviar_correo, es_destino_simulado, parsear_formsubmit, registro_desde_mensaje
from agentes_ebook.orquestador import ejecutar_ciclo, preparar_demo
from agentes_ebook.plantillas import CADENCIAS, redactar, telefono_internacional

# Misma estructura que la notificación real de FormSubmit, con datos ficticios.
HTML_FORMSUBMIT = """<!doctype html><html><body>
<p>Someone just submitted your form on https://www.hyperionadvisory.cl/.</p>
<table>
  <tr><th>Name</th><th>Value</th></tr>
  <tr><td><strong>name</strong></td><td><pre style="margin: 0">Ana Pérez Soto</pre></td></tr>
  <tr><td><strong>email</strong></td><td><pre>ana.perez@constructora.example</pre></td></tr>
  <tr><td><strong>Teléfono</strong></td><td><pre>+56 9 5550 1234</pre></td></tr>
  <tr><td><strong>Cargo</strong></td><td><pre>Gerente de Finanzas</pre></td></tr>
  <tr><td><strong>Empresa</strong></td><td><pre>Constructora Ejemplo</pre></td></tr>
  <tr><td><strong>Motivo_de_descarga</strong></td><td><pre>Mejorar el análisis financiero de mi empresa</pre></td></tr>
</table>
<p style="text-align: center">Submitted at Sun, Oct 11, 2026 3:28 PM (UTC)</p>
</body></html>"""

TEXTO_FORMSUBMIT = """Someone just submitted your form.
| Name | Value |
|---|---|
| name | Luis Gómez |
| email | luis@gmail.example |
| Teléfono | 955500001 |
| Cargo | Estudiante |
| Empresa | - |
| Motivo_de_descarga | Otro |
Submitted at Mon, Oct 12, 2026 9:05 AM (UTC)"""


@pytest.fixture
def cfg(tmp_path):
    base = Config.desde_entorno()
    return dataclasses.replace(
        base,
        gmail_usuario="",
        gmail_app_password="",
        notificar_a="",
        anthropic_api_key="",
        link_agenda="",
        seguimiento_modo="manual",
        excluir_correos=["yo@miempresa.example"],
        ruta_db=str(tmp_path / "test.db"),
    )


@pytest.fixture
def db(cfg):
    base = BaseDatos(cfg.ruta_db)
    yield base
    base.cerrar()


def correo_notificacion(html=HTML_FORMSUBMIT, asunto="Nueva descarga del ebook — Hyperion", msg_id="<abc@formsubmit>"):
    msg = EmailMessage()
    msg["From"] = "FormSubmit <submissions@formsubmit.co>"
    msg["To"] = "yo@miempresa.example"
    msg["Subject"] = asunto
    msg["Message-ID"] = msg_id
    msg["Date"] = "Sun, 11 Oct 2026 15:28:51 +0000"
    msg.set_content("ver versión html")
    msg.add_alternative(html, subtype="html")
    return msg.as_bytes()


# ----------------------------------------------------------------------------- captura
def test_parsea_html_de_formsubmit():
    datos = parsear_formsubmit(html=HTML_FORMSUBMIT)
    assert datos["nombre"] == "Ana Pérez Soto"
    assert datos["email"] == "ana.perez@constructora.example"
    assert datos["telefono"] == "+56 9 5550 1234"
    assert datos["cargo"] == "Gerente de Finanzas"
    assert datos["empresa"] == "Constructora Ejemplo"
    assert datos["motivo"] == "Mejorar el análisis financiero de mi empresa"
    assert datos["fecha_formulario"] == datetime(2026, 10, 11, 15, 28, tzinfo=timezone.utc)


def test_parsea_version_texto():
    datos = parsear_formsubmit(texto=TEXTO_FORMSUBMIT)
    assert datos["nombre"] == "Luis Gómez"
    assert datos["email"] == "luis@gmail.example"
    assert datos["motivo"] == "Otro"


def test_registro_desde_correo(cfg):
    reg = registro_desde_mensaje(correo_notificacion(), cfg)
    assert reg["email"] == "ana.perez@constructora.example"
    assert reg["mensaje_id"] == "<abc@formsubmit>"
    assert reg["fecha"] == "2026-10-11T15:28:00+00:00"
    assert reg["fuente"] == "gmail"


def test_ignora_otros_asuntos_y_correos_propios(cfg):
    assert registro_desde_mensaje(correo_notificacion(asunto="Contacto web"), cfg) is None
    propio = HTML_FORMSUBMIT.replace("ana.perez@constructora.example", "yo@miempresa.example")
    assert registro_desde_mensaje(correo_notificacion(html=propio), cfg) is None


def test_descargas_repetidas_actualizan_el_lead(db, cfg):
    reg = registro_desde_mensaje(correo_notificacion(), cfg)
    lead_id, nuevo = db.registrar_descarga(reg)
    assert nuevo
    assert db.registrar_descarga(reg) == (None, False)  # mismo correo: no se duplica
    otra = dict(reg, mensaje_id="<otra@formsubmit>", fecha="2026-10-13T10:00:00+00:00", cargo="CFO")
    assert db.registrar_descarga(otra) == (lead_id, False)
    lead = db.lead(lead_id)
    assert lead["num_descargas"] == 2
    assert lead["cargo"] == "CFO"
    assert lead["ultima_descarga"] == "2026-10-13T10:00:00+00:00"


def test_importa_csv(db, cfg):
    df = pd.DataFrame(
        [
            {"Nombre": "Eva Lima", "Correo": "eva@empresa.example", "Cargo": "Socia", "Fecha": "2026-10-01"},
            {"Nombre": "Sin correo", "Correo": "no-es-correo", "Cargo": "", "Fecha": "2026-10-01"},
        ]
    )
    nuevas, nuevos = captura.desde_csv(db, df, cfg)
    assert nuevas == 1 and len(nuevos) == 1
    assert db.leads()[0]["cargo"] == "Socia"


# ----------------------------------------------------------------------------- analista
def test_puntaje_y_segmentos(cfg):
    ahora = datetime(2026, 10, 11, 18, tzinfo=timezone.utc)
    decisor = {
        "email": "ana@constructora.example", "cargo": "Fundador", "empresa": "Constructora",
        "motivo": "Mejorar el análisis financiero de mi empresa", "telefono": "+56 9 5550 1234",
        "num_descargas": 1, "ultima_descarga": iso(ahora),
    }
    score, segmento, nivel, pais, detalle = analista.puntuar(decisor, cfg, ahora)
    assert segmento == "Caliente" and nivel == "Decisor" and pais == "Chile"
    assert score == 100

    estudiante = {
        "email": "luis@gmail.example", "cargo": "Alumna en práctica de finanzas", "empresa": "-",
        "motivo": "Otro", "telefono": "", "num_descargas": 1, "ultima_descarga": iso(ahora - timedelta(days=9)),
    }
    score, segmento, nivel, *_ = analista.puntuar(estudiante, cfg, ahora)
    assert nivel == "Estudiante" and segmento == "Frío"


@pytest.mark.parametrize(
    "telefono,correo,esperado",
    [
        ("+56955500002", "a@b.example", "Chile"),
        ("33 5550 0003", "a@empresa.mx", "México"),
        ("+51 987 654 321", "a@gmail.com", "Perú"),
        ("955500001", "a@gmail.com", "Sin identificar"),
    ],
)
def test_pais(telefono, correo, esperado):
    assert analista.pais_de(telefono, correo) == esperado


def test_correo_corporativo():
    assert not analista.es_correo_corporativo("x@gmail.com")
    assert not analista.es_correo_corporativo("x@hotmail.es")
    assert analista.es_correo_corporativo("x@consultora-ejemplo.cl")


# ----------------------------------------------------------------------------- seguimiento
def _lead(segmento="Tibio", dias=0, **extra):
    ahora = datetime(2026, 10, 11, 18, tzinfo=timezone.utc)
    return {
        "id": 1, "email": "a@b.example", "nombre": "Ana", "segmento": segmento, "etapa": "En seguimiento",
        "primera_descarga": iso(ahora - timedelta(days=dias)), "respondio_en": None, **extra,
    }, ahora


def test_primer_paso_y_reactivacion():
    lead, ahora = _lead("Caliente", dias=0)
    assert seguimiento.siguiente_paso(lead, [], ahora)[1] == "bienvenida_caliente"
    lead, ahora = _lead("Tibio", dias=20)
    assert seguimiento.siguiente_paso(lead, [], ahora)[1] == "reactivacion"
    lead, ahora = _lead("Tibio", dias=90)
    assert seguimiento.siguiente_paso(lead, [], ahora) is None


def test_cadencia_respeta_tiempos_y_pendientes():
    lead, ahora = _lead("Tibio")
    paso1 = {"paso": 1, "estado": "enviado", "creado_en": iso(ahora - timedelta(days=4)), "actualizado_en": iso(ahora - timedelta(days=4))}
    assert seguimiento.siguiente_paso(lead, [paso1], ahora)[:2] == (2, "caso")
    # Si el paso anterior sigue esperando aprobación, no se acumula otro correo.
    pendiente = dict(paso1, estado="pendiente")
    assert seguimiento.siguiente_paso(lead, [pendiente], ahora) is None
    # Aún no toca el paso 2 (día 3) si el primero fue ayer.
    reciente = dict(paso1, creado_en=iso(ahora - timedelta(days=1)), actualizado_en=iso(ahora - timedelta(days=1)))
    assert seguimiento.siguiente_paso(lead, [reciente], ahora) is None


def test_se_detiene_si_respondio():
    lead, ahora = _lead("Caliente", respondio_en="2026-10-10T10:00:00+00:00")
    assert seguimiento.siguiente_paso(lead, [], ahora) is None


def test_plantillas_incluyen_horarios_o_link(cfg):
    lead = {"nombre": "ana pérez", "empresa": "Constructora", "motivo": "Otro"}
    horarios = reuniones.proximos_horarios(cfg, desde=datetime(2026, 10, 9, 15, tzinfo=timezone.utc))
    for _, tipo, _ in CADENCIAS["Caliente"] + [(0, "reactivacion", "")]:
        asunto, cuerpo = redactar(tipo, lead, cfg, horarios)
        assert asunto and "Hola Ana:" in cuerpo and cfg.firma in cuerpo
    asunto, cuerpo = redactar("invitacion", lead, cfg, horarios)
    assert "lunes 12 de octubre" in cuerpo
    con_link = dataclasses.replace(cfg, link_agenda="https://calendly.com/mi-agenda/30min")
    assert "https://calendly.com/mi-agenda/30min" in redactar("invitacion", lead, con_link, horarios)[1]


# ----------------------------------------------------------------------------- reuniones
def test_horarios_habiles_sin_agendados(cfg):
    viernes = datetime(2026, 10, 9, 15, tzinfo=timezone.utc)
    horarios = reuniones.proximos_horarios(cfg, desde=viernes)
    assert len(horarios) == 3
    assert all(h.weekday() < 5 for h in horarios)
    assert horarios[0].date().isoformat() == "2026-10-12"  # salta el fin de semana
    ocupado = {iso(horarios[0])}
    assert iso(horarios[0]) not in {iso(h) for h in reuniones.proximos_horarios(cfg, ocupado, desde=viernes)}


def test_invitaciones(cfg):
    lead = {"id": 7, "email": "ana@constructora.example", "nombre": "Ana", "cargo": "CFO", "empresa": "Constructora"}
    inicio = datetime(2026, 10, 12, 10, tzinfo=cfg.tz)
    link = reuniones.link_google_calendar(lead, cfg, inicio)
    assert link.startswith("https://calendar.google.com/calendar/render?action=TEMPLATE")
    assert "add=ana%40constructora.example" in link
    ics = reuniones.archivo_ics(lead, cfg, inicio)
    assert "BEGIN:VEVENT" in ics and "ATTENDEE" in ics and "DTSTART:20261012T130000Z" in ics


def test_telefono_internacional():
    assert telefono_internacional("+56 9 5550 1234", "Chile") == "56955501234"
    assert telefono_internacional("955501234", "Chile") == "56955501234"
    assert telefono_internacional("33 5550 0003", "México") == "523355500003"
    assert telefono_internacional("955500001", "Sin identificar") is None


# ----------------------------------------------------------------------------- seguridad
def test_nunca_envia_a_dominios_de_ejemplo(cfg):
    assert es_destino_simulado("ana@gmail.example")
    assert es_destino_simulado("ana@example.com")
    assert not es_destino_simulado("ana@gmail.com")
    assert enviar_correo(cfg, "ana@empresa.example", "x", "y") == "simulado"
    with pytest.raises(RuntimeError):
        enviar_correo(cfg, "ana@empresa.com", "x", "y")  # sin credenciales no intenta enviar


# ----------------------------------------------------------------------------- ciclo completo
class GmailFalso:
    """Simula Gmail: entrega notificaciones y respuestas sin red."""

    def __init__(self, cfg, mensajes, respuestas=None, enviados=None):
        self.cfg, self.mensajes = cfg, mensajes
        self.respuestas, self.enviados = respuestas or {}, enviados or {}
        self.borradores = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def leer_descargas(self, dias=None):
        return [r for r in (registro_desde_mensaje(m, self.cfg) for m in self.mensajes) if r]

    def ultima_respuesta(self, correos_desde):
        return {c: f for c, f in self.respuestas.items() if c in correos_desde}

    def enviados_a(self, correos_desde):
        return {c: f for c, f in self.enviados.items() if c in correos_desde}

    def crear_borrador(self, para, asunto, texto, html=None):
        self.borradores.append((para, asunto))
        return "ok"


def test_ciclo_completo_con_gmail_simulado(monkeypatch, db, cfg):
    cfg = dataclasses.replace(cfg, gmail_usuario="yo@miempresa.example", gmail_app_password="x", notificar_a="yo@miempresa.example", seguimiento_modo="borradores")
    falso = GmailFalso(cfg, [correo_notificacion()])
    monkeypatch.setattr("agentes_ebook.orquestador.ClienteGmail", lambda c: falso)
    enviados = []
    monkeypatch.setattr("agentes_ebook.notificador.enviar_correo", lambda *a, **k: enviados.append(a[2]) or "ok")

    r = ejecutar_ciclo(db, cfg, usar_ia=False)
    assert r.nuevas_descargas == 1 and len(r.nuevos_leads) == 1
    lead = db.leads()[0]
    assert lead["segmento"] == "Caliente" and lead["etapa"] == "Reunión propuesta"
    assert falso.borradores and falso.borradores[0][0] == "ana.perez@constructora.example"
    assert db.seguimientos()[0]["estado"] == "borrador"
    assert r.alerta == "enviado" and any("oportunidad" in a for a in enviados)

    # El lead responde: se detiene la secuencia y se avisa de nuevo.
    falso.respuestas = {"ana.perez@constructora.example": datetime.now(timezone.utc)}
    r2 = ejecutar_ciclo(db, cfg, usar_ia=False)
    assert r2.respondieron == [lead["id"]]
    assert db.lead(lead["id"])["respondio_en"]
    assert all(s["estado"] != "borrador" for s in db.seguimientos())
    assert r2.alerta == "enviado"


def test_demo_no_notifica_ni_envia(db, cfg):
    r = preparar_demo(db, cfg)
    assert len(db.leads()) >= 30
    assert r.alerta is None
    assert notificador.enviar_alerta(db, cfg) is None  # los leads demo nunca generan alertas reales
    titulo, html, _ = notificador.construir_resumen(db, cfg)
    assert "Resumen del ebook" in titulo and "<html>" in html


# ----------------------------------------------------------------------------- cliente IMAP
class IMAPFalso:
    """Respuestas con el formato real de imaplib frente a Gmail (carpetas en español)."""

    def __init__(self, host):
        self.seleccionada, self.apendidos = None, []

    def login(self, usuario, clave):
        return "OK", [b"logged in"]

    def logout(self):
        return "BYE", []

    def list(self):
        return "OK", [
            b'(\\HasNoChildren) "/" "INBOX"',
            b'(\\All \\HasNoChildren) "/" "[Gmail]/Todos"',
            b'(\\HasNoChildren \\Sent) "/" "[Gmail]/Enviados"',
            b'(\\Drafts \\HasNoChildren) "/" "[Gmail]/Borradores"',
        ]

    def select(self, nombre, readonly=False):
        self.seleccionada = nombre
        return "OK", [b"2"]

    def uid(self, comando, *args):
        if comando == "SEARCH":
            if "FROM" in args and '"submissions@formsubmit.co"' in args:
                return "OK", [b"1 2"]
            if "FROM" in args and '"ana.perez@constructora.example"' in args:
                return "OK", [b"9"]
            return "OK", [b""]
        if comando == "FETCH" and args[1] == "(BODY.PEEK[])":
            otro = correo_notificacion(asunto="Otro asunto", msg_id="<2@x>")
            return "OK", [(b"1 (UID 1 BODY[] {100}", correo_notificacion()), b")", (b"2 (UID 2 BODY[] {100}", otro), b")"]
        if comando == "FETCH" and args[1] == "(INTERNALDATE)":
            return "OK", [b'9 (UID 9 INTERNALDATE "12-Oct-2026 11:00:00 +0000")']
        raise AssertionError(comando)

    def append(self, carpeta, flags, fecha, mensaje):
        self.apendidos.append((carpeta, flags, mensaje))
        return "OK", [b"APPEND completed"]


def test_cliente_gmail_con_imap_simulado(monkeypatch, cfg):
    from agentes_ebook.gmail import ClienteGmail

    cfg = dataclasses.replace(cfg, gmail_usuario="yo@miempresa.example", gmail_app_password="x")
    falso = IMAPFalso("imap.gmail.com")
    monkeypatch.setattr("agentes_ebook.gmail.imaplib.IMAP4_SSL", lambda host: falso)
    with ClienteGmail(cfg) as g:
        registros = g.leer_descargas()
        assert falso.seleccionada == '"[Gmail]/Todos"'
        assert [r["email"] for r in registros] == ["ana.perez@constructora.example"]
        desde = datetime(2026, 10, 11, tzinfo=timezone.utc)
        respuesta = g.ultima_respuesta({"ana.perez@constructora.example": desde, "otro@x.example": desde})
        assert respuesta == {"ana.perez@constructora.example": datetime(2026, 10, 12, 11, tzinfo=timezone.utc)}
        assert g.crear_borrador("ana.perez@constructora.com", "Hola", "Texto") == "ok"
        assert g.crear_borrador("demo@gmail.example", "Hola", "Texto") == "simulado"
    carpeta, flags, mensaje = falso.apendidos[0]
    assert carpeta == '"[Gmail]/Borradores"' and flags == "(\\Draft)"
    assert b"To: ana.perez@constructora.com" in mensaje


# ----------------------------------------------------------------------------- IA opcional
class _Bloque:
    def __init__(self, texto):
        self.type, self.text = "text", texto


class _Respuesta:
    def __init__(self, texto, stop_reason="end_turn"):
        self.content, self.stop_reason = [_Bloque(texto)], stop_reason


def test_asistente_ia_con_cliente_simulado(cfg):
    pytest.importorskip("anthropic")
    from agentes_ebook.llm import AsistenteIA

    cfg = dataclasses.replace(cfg, anthropic_api_key="sk-prueba", link_agenda="https://agenda.example/30")
    ia = AsistenteIA(cfg)
    llamadas = []

    def crear(**kwargs):
        llamadas.append(kwargs)
        return _Respuesta('{"lectura": "Decisor con necesidad clara.", "angulo": "Ofrecer diagnóstico."}')

    ia.cliente.beta.messages.create = crear
    lead = {"nombre": "Ana", "cargo": "CFO", "empresa": "Constructora", "motivo": "Otro", "score": 80}
    assert ia.analizar_lead(lead, [("Cargo: Decisor", 30)])["angulo"] == "Ofrecer diagnóstico."
    assert llamadas[0]["model"] == cfg.llm_modelo and llamadas[0]["fallbacks"] == "default"

    # Si la IA pierde el link de agenda, se conserva la plantilla original.
    ia.cliente.beta.messages.create = lambda **k: _Respuesta('{"cuerpo": "Hola Ana, conversemos."}')
    assert ia.personalizar_correo(lead, "invitacion", "Asunto", "Agenda aquí: https://agenda.example/30") is None
    ia.cliente.beta.messages.create = lambda **k: _Respuesta("", stop_reason="refusal")
    assert ia.analizar_lead(lead, []) is None
