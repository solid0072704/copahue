"""Pruebas de las señales de mercado (sin conexión a internet ni a Gmail)."""

import json
from datetime import date

import senales_mercado as sm
from senales_mercado import DEFENSIVO, FAVORABLE, NEUTRAL


def _ind(puntaje, **extra):
    return {"nombre": "Indicador", "puntaje": puntaje, "zona": "Neutral", **extra}


def test_clasificar_sin_zona_previa_usa_los_limites():
    assert sm.clasificar(60, (40, 60)) == FAVORABLE
    assert sm.clasificar(59.9, (40, 60)) == NEUTRAL
    assert sm.clasificar(40, (40, 60)) == NEUTRAL
    assert sm.clasificar(39.9, (40, 60)) == DEFENSIVO


def test_clasificar_exige_margen_para_cambiar():
    # Sube de NEUTRAL a FAVORABLE solo desde 62.
    assert sm.clasificar(61, (40, 60), NEUTRAL) == NEUTRAL
    assert sm.clasificar(62, (40, 60), NEUTRAL) == FAVORABLE
    # Baja de FAVORABLE a NEUTRAL solo bajo 58.
    assert sm.clasificar(58.5, (40, 60), FAVORABLE) == FAVORABLE
    assert sm.clasificar(57.9, (40, 60), FAVORABLE) == NEUTRAL
    # Baja de NEUTRAL a DEFENSIVO solo bajo 38.
    assert sm.clasificar(39, (40, 60), NEUTRAL) == NEUTRAL
    assert sm.clasificar(37.9, (40, 60), NEUTRAL) == DEFENSIVO
    # Sube de DEFENSIVO a NEUTRAL solo desde 42.
    assert sm.clasificar(41, (40, 60), DEFENSIVO) == DEFENSIVO
    assert sm.clasificar(42, (40, 60), DEFENSIVO) == NEUTRAL


def test_clasificar_salto_de_dos_zonas():
    assert sm.clasificar(61, (40, 60), DEFENSIVO) == NEUTRAL
    assert sm.clasificar(75, (40, 60), DEFENSIVO) == FAVORABLE
    assert sm.clasificar(10, (40, 60), FAVORABLE) == DEFENSIVO


def test_senal_eeuu_promedia_los_dos_indicadores():
    senal = sm.senal_eeuu(_ind(50.8), _ind(34.1))
    assert senal["puntaje"] == 42.5
    assert senal["zona"] == NEUTRAL
    assert "nota" not in senal


def test_senal_eeuu_con_un_solo_indicador_lo_advierte():
    senal = sm.senal_eeuu(_ind(70), None)
    assert senal["zona"] == FAVORABLE
    assert "un solo indicador" in senal["nota"]


def test_senal_eeuu_con_datos_viejos_mantiene_la_anterior():
    viejo = _ind(10, desactualizado=True)
    senal = sm.senal_eeuu(viejo, viejo, {"zona": FAVORABLE})
    assert senal["zona"] == FAVORABLE
    assert senal["puntaje"] is None
    assert sm.senal_eeuu(None, None) is None


def test_senal_cripto_usa_los_cortes_de_la_skill():
    assert sm.senal_cripto(_ind(51.4))["zona"] == NEUTRAL
    assert sm.senal_cripto(_ind(85))["zona"] == FAVORABLE
    assert sm.senal_cripto(_ind(20))["zona"] == DEFENSIVO
    assert sm.senal_cripto(None) is None


def test_detectar_cambios_primera_vez_y_sin_cambios():
    senales = {"eeuu": {"zona": NEUTRAL, "puntaje": 45}, "cripto": None}
    assert sm.detectar_cambios(senales, {}) == [("eeuu", None, NEUTRAL)]
    assert sm.detectar_cambios(senales, {"eeuu": {"zona": NEUTRAL}}) == []
    assert sm.detectar_cambios(senales, {"eeuu": {"zona": FAVORABLE}}) == [
        ("eeuu", FAVORABLE, NEUTRAL)
    ]


def test_actualizar_estado_conserva_desde_si_no_cambia():
    estado = {"eeuu": {"zona": NEUTRAL, "puntaje": 44, "desde": "2026-11-01"}}
    senales = {"eeuu": {"zona": NEUTRAL, "puntaje": 46}, "cripto": {"zona": DEFENSIVO, "puntaje": 30}}
    nuevo = sm.actualizar_estado(estado, senales, "2026-11-13", resumen_enviado=True)
    assert nuevo["eeuu"] == {"zona": NEUTRAL, "puntaje": 46, "desde": "2026-11-01"}
    assert nuevo["cripto"]["desde"] == "2026-11-13"
    assert nuevo["ultimo_resumen"] == "2026-11-13"
    assert nuevo["historial"] == [{"fecha": "2026-11-13", "eeuu": 46, "cripto": 30}]
    # Una señal sin datos no borra la anterior.
    sin_datos = sm.actualizar_estado(nuevo, {"eeuu": None, "cripto": None}, "2026-11-14", False)
    assert sin_datos["eeuu"]["zona"] == NEUTRAL


def test_mensaje_de_cambio_y_de_resumen():
    senales = {
        "eeuu": sm.senal_eeuu(
            _ind(70, exposicion="Normal (80-100%)", fecha_datos="2026-11-12"), _ind(66)
        ),
        "cripto": None,
    }
    cambios = [("eeuu", NEUTRAL, FAVORABLE)]
    asunto = sm.asunto_mensaje(cambios, senales, False)
    assert "Mercado de EE.UU. pasa a FAVORABLE" in asunto
    texto = sm.mensaje_texto(sm.secciones_mensaje(cambios, senales, {}, False))
    assert "cambió de NEUTRAL a FAVORABLE" in texto
    assert "exposición sugerida a acciones 80-100%" in texto
    assert "Sin datos en esta ejecución" in texto  # cripto
    assert sm.AVISO_LEGAL in texto

    estado = {"historial": [{"fecha": "2026-11-06", "eeuu": 61.0}]}
    asunto = sm.asunto_mensaje([], senales, True)
    assert asunto.startswith("Resumen semanal de señales: Mercado de EE.UU. FAVORABLE")
    texto = sm.mensaje_texto(sm.secciones_mensaje([], senales, estado, True))
    assert "No hubo cambios de señal" in texto
    assert "Puntaje el 2026-11-06: 61,0; hoy: 68,0." in texto
    assert "<li>" in sm.mensaje_html(sm.secciones_mensaje([], senales, estado, True))


def test_leer_informes_de_las_skills(tmp_path):
    (tmp_path / "market_breadth_history.json").write_text("[]")
    (tmp_path / "market_breadth_2026-11-13_161801.json").write_text(json.dumps({
        "composite": {"composite_score": 50.8, "zone": "Neutral", "exposure_guidance": "60-75%"},
        "metadata": {"data_freshness": {"latest_date": "2026-11-12"}},
    }))
    (tmp_path / "uptrend_analysis_2026-11-13_161802.json").write_text(json.dumps({
        "composite": {"composite_score": 34.1, "zone": "Cautious", "exposure_guidance": "Defensive (30-60%)"},
        "metadata": {"latest_data_date": "2026-11-01"},
    }))
    (tmp_path / "crypto_regime.json").write_text(json.dumps({
        "composite": {"score": 51.4, "zone": "NEUTRAL", "components_available": 4, "components_total": 6},
        "metadata": {"as_of": "2026-11-13T16:24:21+00:00"},
    }))
    hoy = date(2026, 11, 13)
    amplitud = sm.leer_amplitud(tmp_path, hoy)
    assert (amplitud["puntaje"], amplitud["desactualizado"]) == (50.8, False)
    tendencia = sm.leer_tendencia(tmp_path, hoy)
    assert (tendencia["puntaje"], tendencia["desactualizado"]) == (34.1, True)
    cripto = sm.leer_cripto(tmp_path)
    assert (cripto["puntaje"], cripto["componentes"]) == (51.4, "4 de 6")
    assert sm.leer_amplitud(tmp_path / "vacio", hoy) is None


def test_main_avisa_solo_cuando_cambia(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sm, "DIR_DATOS", tmp_path)
    for variable in ("GMAIL_USER", "GMAIL_APP_PASSWORD", "GITHUB_TOKEN"):
        monkeypatch.delenv(variable, raising=False)
    puntajes = {"amplitud": 50.0, "tendencia": 40.0}
    monkeypatch.setattr(
        sm, "recolectar",
        lambda hoy, con_cripto=True: (_ind(puntajes["amplitud"]), _ind(puntajes["tendencia"]), None),
    )
    avisos = []
    monkeypatch.setattr(sm, "notificar", lambda asunto, secciones: avisos.append(asunto) or "prueba")

    assert sm.main(["--sin-cripto"]) == 0  # primera vez: avisa
    assert len(avisos) == 1 and "inicia en NEUTRAL" in avisos[0]
    assert sm.main(["--sin-cripto"]) == 0  # sin cambios: no avisa
    assert len(avisos) == 1
    puntajes.update(amplitud=80.0, tendencia=70.0)
    assert sm.main(["--sin-cripto", "--sin-enviar"]) == 0  # no envía ni guarda
    assert len(avisos) == 1
    assert sm.main(["--sin-cripto"]) == 0  # cambió: avisa
    assert len(avisos) == 2 and "pasa a FAVORABLE" in avisos[1]
    assert sm.main(["--sin-cripto", "--resumen"]) == 0  # resumen semanal
    assert len(avisos) == 3 and avisos[2].startswith("Resumen semanal")
    estado = json.loads((tmp_path / "estado.json").read_text())
    assert estado["eeuu"]["zona"] == FAVORABLE and "ultimo_resumen" in estado

    monkeypatch.setattr(sm, "recolectar", lambda hoy, con_cripto=True: (None, None, None))
    assert sm.main(["--sin-cripto"]) == 1  # sin datos: falla, sin avisar
    assert len(avisos) == 3


def test_notificar_por_gmail_y_respaldo_por_issue(monkeypatch):
    secciones = [("Qué cambió", ["Mercado de EE.UU.: cambió de NEUTRAL a FAVORABLE."])]
    enviados, publicados = [], []

    class SMTPFalso:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def login(self, usuario, clave):
            if clave == "mala":
                raise sm.smtplib.SMTPAuthenticationError(535, b"clave incorrecta")

        def send_message(self, msg):
            enviados.append(msg)

    class RespuestaFalsa:
        def raise_for_status(self):
            pass

    def post_falso(url, headers, json, timeout):
        publicados.append((url, json))
        return RespuestaFalsa()

    import requests

    monkeypatch.setattr(sm.smtplib, "SMTP_SSL", SMTPFalso)
    monkeypatch.setattr(requests, "post", post_falso)
    for variable in ("NOTIFY_EMAIL", "GITHUB_TOKEN", "GITHUB_REPOSITORY", "GITHUB_REPOSITORY_OWNER"):
        monkeypatch.delenv(variable, raising=False)

    # Con Gmail configurado: correo a uno mismo, clave sin espacios.
    monkeypatch.setenv("GMAIL_USER", "yo@correo.example")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcd efgh")
    assert sm.notificar("Asunto", secciones) == "gmail"
    assert enviados[0]["To"] == "yo@correo.example" and enviados[0]["Subject"] == "Asunto"

    # Gmail falla y hay token de GitHub: se publica un issue.
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "mala")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_REPOSITORY", "duenio/repo")
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "duenio")
    assert sm.notificar("Asunto", secciones) == "issue"
    url, cuerpo = publicados[0]
    assert url == "https://api.github.com/repos/duenio/repo/issues"
    assert cuerpo["title"] == "Asunto" and "@duenio" in cuerpo["body"]

    # Sin Gmail ni GitHub: solo pantalla.
    for variable in ("GMAIL_USER", "GMAIL_APP_PASSWORD", "GITHUB_TOKEN"):
        monkeypatch.delenv(variable)
    assert sm.notificar("Asunto", secciones) == "pantalla"
