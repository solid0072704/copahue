"""Pruebas del agente de ayuno (sin internet: canales, reloj e IA simulados)."""

import io
import json
import urllib.error
from datetime import date, datetime, time, timedelta, timezone
from types import SimpleNamespace

import pytest

from ayuno import canales, mensajes, plan
from ayuno.agente import ejecutar
from ayuno.config import AVISOS, Config, ErrorConfig, leer_avisos, leer_dias, leer_hora
from ayuno.estado import Estado
from ayuno.mensajes import duracion, mensaje_valido, redactar, texto_base

UTC = timezone.utc


@pytest.fixture
def cfg(tmp_path):
    return Config(ruta_estado=str(tmp_path / "estado.json")).validar()


def local(cfg, y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=cfg.tz).astimezone(UTC)


class Reloj:
    def __init__(self, t):
        self.t = t
        self.dormido = 0.0

    def __call__(self):
        return self.t

    def dormir(self, segundos):
        self.dormido += segundos
        self.t += timedelta(seconds=segundos)


class CanalFalso:
    nombre = "falso"

    def __init__(self, funciona=True):
        self.funciona = funciona
        self.textos = []

    def enviar(self, texto):
        if not self.funciona:
            raise canales.ErrorCanal("caído")
        self.textos.append(texto)


def correr(cfg, reloj, canal, **kwargs):
    return ejecutar(cfg, usar_ia=False, canales=[canal], reloj=reloj, dormir=reloj.dormir, **kwargs)


def tipos_del_dia(cfg, y, m, d):
    desde = local(cfg, y, m, d, 0)
    return [a.tipo for a in plan.calendario(cfg, desde, desde + timedelta(hours=23, minutes=59))]


# ------------------------------------------------------------------ calendario


def test_calendario_16_8_en_verano(cfg):
    desde = local(cfg, 2026, 11, 13, 0)
    avisos = plan.calendario(cfg, desde, desde + timedelta(days=1) - timedelta(minutes=1))
    assert [(a.tipo, a.cuando.astimezone(cfg.tz).strftime("%H:%M")) for a in avisos] == [
        ("manana", "08:00"),
        ("apertura_previa", "11:30"),
        ("apertura", "12:00"),
        ("cierre_previo", "19:30"),
        ("inicio_ayuno", "20:00"),
    ]
    # Noviembre en Chile es UTC-3: las 12:00 son las 15:00 UTC.
    assert avisos[2].cuando == datetime(2026, 11, 13, 15, 0, tzinfo=UTC)
    assert cfg.protocolo == "16:8"


def test_respeta_horario_de_invierno(cfg):
    apertura = [a for a in plan.calendario(cfg, local(cfg, 2026, 7, 1, 0), local(cfg, 2026, 7, 1, 23)) if a.tipo == "apertura"]
    assert apertura[0].cuando == datetime(2026, 7, 1, 16, 0, tzinfo=UTC)  # julio: UTC-4


def test_dias_de_ayuno_lunes_a_viernes(cfg):
    cfg.dias = leer_dias("lunes a viernes")
    assert tipos_del_dia(cfg, 2026, 11, 13) == ["manana", "apertura_previa", "apertura"]  # viernes: sin ayuno el sábado
    assert tipos_del_dia(cfg, 2026, 11, 14) == []  # sábado libre
    assert tipos_del_dia(cfg, 2026, 11, 15) == ["cierre_previo", "inicio_ayuno"]  # domingo: empieza el del lunes
    assert tipos_del_dia(cfg, 2026, 11, 16) == list(AVISOS)  # lunes completo


def test_ventana_que_cruza_la_medianoche(cfg):
    cfg.ventana_inicio, cfg.ventana_fin = time(18, 0), time(2, 0)
    cfg.validar()
    avisos = plan.calendario(cfg, local(cfg, 2026, 11, 13, 0), local(cfg, 2026, 11, 14, 3))
    inicio = [a for a in avisos if a.tipo == "inicio_ayuno" and a.dia == date(2026, 11, 13)][0]
    assert inicio.cuando.astimezone(cfg.tz).strftime("%d %H:%M") == "14 02:00"
    assert inicio.ayuno_hasta.astimezone(cfg.tz).strftime("%d %H:%M") == "14 18:00"
    assert cfg.protocolo == "16:8"


def test_buenos_dias_fuera_del_ayuno_se_omite(cfg):
    cfg.ventana_inicio, cfg.ventana_fin = time(7, 0), time(15, 0)
    assert "manana" not in tipos_del_dia(cfg, 2026, 11, 13)


def test_solo_los_avisos_elegidos(cfg):
    cfg.avisos = leer_avisos("apertura, inicio_ayuno")
    assert tipos_del_dia(cfg, 2026, 11, 13) == ["apertura", "inicio_ayuno"]


# ------------------------------------------------------------------ agente


def test_envia_cada_aviso_una_sola_vez(cfg):
    reloj, canal = Reloj(local(cfg, 2026, 11, 13, 12, 5)), CanalFalso()
    resultado, _ = correr(cfg, reloj, canal)
    assert resultado.enviados == ["apertura"]
    assert "Completaste 16 h de ayuno" in canal.textos[0]
    resultado, _ = correr(cfg, reloj, canal)
    assert resultado.enviados == []
    assert len(canal.textos) == 1


def test_aviso_atrasado_se_ajusta_y_luego_caduca(cfg):
    canal = CanalFalso()
    correr(cfg, Reloj(local(cfg, 2026, 11, 13, 11, 50)), canal)
    assert canal.textos == [
        "⏳ En 10 min abres tu ventana de comida (12:00 a 20:00). "
        "Rompe el ayuno con proteína y verduras, y deja lo dulce para después."
    ]
    # Pasada la apertura, el "falta poco" ya no se envía; la apertura sí.
    otro = CanalFalso()
    cfg.ruta_estado += ".2"
    resultado, _ = correr(cfg, Reloj(local(cfg, 2026, 11, 13, 12, 1)), otro)
    assert resultado.enviados == ["apertura"]
    # 70 minutos tarde (tolerancia 60): nada.
    cfg.ruta_estado += ".3"
    resultado, _ = correr(cfg, Reloj(local(cfg, 2026, 11, 13, 13, 10)), CanalFalso())
    assert resultado.enviados == []


def test_espera_la_hora_exacta_del_proximo_aviso(cfg):
    reloj, canal = Reloj(local(cfg, 2026, 11, 13, 11, 52)), CanalFalso()
    resultado, _ = correr(cfg, reloj, canal, esperar_max_min=14)
    assert resultado.enviados == ["apertura_previa", "apertura"]
    assert canal.textos[0].startswith("⏳ En 8 min")
    assert local(cfg, 2026, 11, 13, 12, 0) <= reloj.t < local(cfg, 2026, 11, 13, 12, 1)
    assert resultado.proximo.tipo == "cierre_previo"


def test_no_espera_si_el_proximo_aviso_esta_lejos(cfg):
    reloj, canal = Reloj(local(cfg, 2026, 11, 13, 9, 0)), CanalFalso()
    resultado, _ = correr(cfg, reloj, canal, esperar_max_min=14)
    assert resultado.enviados == [] and reloj.dormido == 0


def test_si_el_canal_falla_se_reintenta_despues(cfg):
    ahora = local(cfg, 2026, 11, 13, 20, 2)
    resultado, estado = correr(cfg, Reloj(ahora), CanalFalso(funciona=False))
    assert resultado.fallidos == ["inicio_ayuno"] and not estado.cambio
    canal = CanalFalso()
    resultado, _ = correr(cfg, Reloj(ahora + timedelta(minutes=15)), canal)
    assert resultado.enviados == ["inicio_ayuno"]
    assert "próxima comida es mañana a las 12:00" in canal.textos[0]


def test_el_estado_persiste_y_olvida_lo_antiguo(tmp_path):
    ruta = str(tmp_path / "sub" / "estado.json")
    ahora = datetime(2026, 11, 13, 15, 0, tzinfo=UTC)
    estado = Estado(ruta)
    estado.marcar("2026-10-01:apertura", ahora - timedelta(days=40))
    estado.marcar("2026-11-13:apertura", ahora)
    assert estado.guardar(ahora)
    nuevo = Estado(ruta)
    assert nuevo.ya_enviado("2026-11-13:apertura")
    assert not nuevo.ya_enviado("2026-10-01:apertura")
    assert not nuevo.guardar(ahora)  # sin cambios no reescribe


def test_estado_danado_parte_de_cero(tmp_path):
    ruta = tmp_path / "estado.json"
    ruta.write_text("{no es json", encoding="utf-8")
    assert Estado(str(ruta)).enviados == {}


# ------------------------------------------------------------------ mensajes


def test_duracion():
    assert duracion(timedelta(hours=16)) == "16 h"
    assert duracion(timedelta(hours=3, minutes=20)) == "3 h 20 min"
    assert duracion(timedelta(minutes=25)) == "25 min"
    assert duracion(timedelta(minutes=-5)) == "0 min"


def test_mensaje_de_buenos_dias_con_nombre(cfg):
    cfg.nombre = "Ana"
    aviso = plan.pendientes(cfg, local(cfg, 2026, 11, 13, 8, 0))[0]
    texto = texto_base(aviso, cfg, aviso.cuando)
    assert texto.startswith("☀️ Buenos días, Ana. Llevas 12 h de ayuno y te faltan 4 h")
    assert "12:00" in texto


def test_proxima_comida_en_otro_dia(cfg):
    cfg.dias = leer_dias("lun,mie")
    aviso = [a for a in plan.calendario(cfg, local(cfg, 2026, 11, 15, 0), local(cfg, 2026, 11, 15, 23)) if a.tipo == "inicio_ayuno"][0]
    assert "mañana a las 12:00" in texto_base(aviso, cfg, aviso.cuando)  # domingo -> lunes


class RedactorFalso:
    def __init__(self, respuesta=None, error=None):
        self.respuesta, self.error = respuesta, error

    def reescribir(self, tipo, datos, base):
        if self.error:
            raise self.error
        return self.respuesta


def test_la_ia_no_puede_cambiar_las_horas(cfg):
    ahora = local(cfg, 2026, 11, 13, 12, 0)
    aviso = plan.pendientes(cfg, ahora)[0]
    base = texto_base(aviso, cfg, ahora)
    assert mensaje_valido("🍽️ ¡Bien hecho! Come con calma hasta las 20:00.", base)
    assert not mensaje_valido("🍽️ ¡Bien hecho! Come hasta las 21:00.", base)
    assert not mensaje_valido("", base)
    assert redactar(aviso, cfg, ahora, RedactorFalso("🍽️ Ventana abierta hasta las 20:00, ¡a disfrutar!")) == (
        "🍽️ Ventana abierta hasta las 20:00, ¡a disfrutar!"
    )
    assert redactar(aviso, cfg, ahora, RedactorFalso(None)) == base
    assert redactar(aviso, cfg, ahora, RedactorFalso(error=RuntimeError("caída"))) == base


def test_redactor_llama_a_claude_con_respaldo(cfg, monkeypatch):
    pytest.importorskip("anthropic")
    cfg.anthropic_api_key = "sk-test"
    redactor = mensajes.RedactorIA(cfg)
    llamadas = []

    def crear(**kwargs):
        llamadas.append(kwargs)
        bloque = SimpleNamespace(type="text", text="🍽️ ¡Ventana abierta! Disfruta hasta las 20:00.")
        return SimpleNamespace(stop_reason="end_turn", content=[bloque])

    redactor.cliente = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=crear)))
    ahora = local(cfg, 2026, 11, 13, 12, 0)
    aviso = plan.pendientes(cfg, ahora)[0]
    assert redactar(aviso, cfg, ahora, redactor) == "🍽️ ¡Ventana abierta! Disfruta hasta las 20:00."
    assert llamadas[0]["model"] == "claude-opus-5-5"
    assert llamadas[0]["fallbacks"] == "default"
    assert llamadas[0]["output_config"] == {"effort": "low"}


# ------------------------------------------------------------------ configuración


def test_leer_dias():
    assert leer_dias("todos") == set(range(7))
    assert leer_dias("lunes a viernes") == {0, 1, 2, 3, 4}
    assert leer_dias("lun, mié, sáb") == {0, 2, 5}
    assert leer_dias("vie-lun") == {4, 5, 6, 0}
    with pytest.raises(ErrorConfig):
        leer_dias("feriados")


def test_leer_hora():
    assert leer_hora("8:30") == time(8, 30)
    assert leer_hora("20") == time(20, 0)
    assert leer_hora("12h") == time(12, 0)
    with pytest.raises(ErrorConfig):
        leer_hora("25:00")


def test_config_desde_entorno(monkeypatch):
    monkeypatch.setenv("AYUNO_VENTANA_INICIO", "13:00")
    monkeypatch.setenv("AYUNO_VENTANA_FIN", "19:00")
    monkeypatch.setenv("AYUNO_AVISOS", "apertura,inicio_ayuno")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setenv("NTFY_TEMA", "ayuno-secreto")
    cfg = Config.desde_entorno()
    assert cfg.protocolo == "18:6"
    assert cfg.avisos == ["apertura", "inicio_ayuno"]
    assert cfg.canales_configurados == ["telegram", "ntfy"]


def test_config_invalida():
    with pytest.raises(ErrorConfig):
        Config(ventana_inicio=time(12, 0), ventana_fin=time(12, 0)).validar()
    with pytest.raises(ErrorConfig):
        Config(zona_horaria="Marte/Olympus").validar()
    with pytest.raises(ErrorConfig):
        Config(ventana_inicio=time(12, 0), ventana_fin=time(12, 20), aviso_previo_min=30).validar()


# ------------------------------------------------------------------ canales


class Respuesta(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def http(monkeypatch):
    pedidos = []
    respuesta = {"cuerpo": b'{"ok": true}'}

    def urlopen(pedido, timeout=None):
        pedidos.append(pedido)
        return Respuesta(respuesta["cuerpo"])

    monkeypatch.setattr(canales.urllib.request, "urlopen", urlopen)
    return SimpleNamespace(pedidos=pedidos, respuesta=respuesta)


def test_telegram(http):
    canales.Telegram("123:abc", "42").enviar("Hola ☀️")
    pedido = http.pedidos[0]
    assert pedido.full_url == "https://api.telegram.org/bot123:abc/sendMessage"
    assert json.loads(pedido.data) == {"chat_id": "42", "text": "Hola ☀️"}


def test_whatsapp_callmebot(http):
    http.respuesta["cuerpo"] = b"Message queued. You will receive it in a few seconds."
    canales.WhatsAppCallMeBot("+56912345678", "999").enviar("Ventana abierta")
    url = http.pedidos[0].full_url
    assert url.startswith("https://api.callmebot.com/whatsapp.php?")
    assert "phone=%2B56912345678" in url and "text=Ventana+abierta" in url and "apikey=999" in url
    http.respuesta["cuerpo"] = b"APIKey is invalid"
    with pytest.raises(canales.ErrorCanal):
        canales.WhatsAppCallMeBot("+56912345678", "000").enviar("x")


def test_ntfy(http):
    canales.Ntfy("https://ntfy.sh", "ayuno-secreto").enviar("Comienza tu ayuno 🌙")
    pedido = http.pedidos[0]
    assert pedido.full_url == "https://ntfy.sh/ayuno-secreto"
    assert pedido.data.decode("utf-8") == "Comienza tu ayuno 🌙"
    assert pedido.get_header("Title") == "Ayuno intermitente"


def test_error_http_no_muestra_el_token(monkeypatch):
    def urlopen(pedido, timeout=None):
        raise urllib.error.HTTPError(pedido.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"ok":false}'))

    monkeypatch.setattr(canales.urllib.request, "urlopen", urlopen)
    with pytest.raises(canales.ErrorCanal) as error:
        canales.Telegram("123:TOKEN-SECRETO", "42").enviar("hola")
    assert "TOKEN-SECRETO" not in str(error.value)
    assert "401" in str(error.value)


def test_enviar_a_todos_sigue_si_un_canal_falla():
    bueno, malo = CanalFalso(), CanalFalso(funciona=False)
    assert canales.enviar_a_todos([malo, bueno], "hola") == ["falso"]
    assert bueno.textos == ["hola"]


def test_chats_de_telegram(http):
    http.respuesta["cuerpo"] = json.dumps(
        {"ok": True, "result": [{"message": {"chat": {"id": 42, "first_name": "Ana"}}}]}
    ).encode()
    assert canales.chats_de_telegram("123:abc") == {42: "Ana"}
