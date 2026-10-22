"""Pruebas de los agentes de trading (sin internet: precios sintéticos y broker simulado)."""

import dataclasses
from datetime import timedelta

import pytest

from agentes_trading import ejecutor, estratega, guardia, notificador
from agentes_trading.brokers import BrokerAlpaca, BrokerSimulado, ErrorBroker
from agentes_trading.config import ACTIVO_REFUGIO, Config, parsear_cartera
from agentes_trading.db import BaseDatos, ahora_utc, iso
from agentes_trading.mercado import MercadoSintetico
from agentes_trading.orquestador import ejecutar_ciclo, preparar_demo


@pytest.fixture
def cfg():
    return dataclasses.replace(Config(), filtro_tendencia=False, comision_simulada_pct=0.0)


@pytest.fixture
def entorno(cfg):
    db = BaseDatos(":memory:")
    mercado = MercadoSintetico()
    broker = BrokerSimulado(db, mercado, cfg.comision_simulada_pct)
    return db, mercado, broker


def ciclo(db, cfg, mercado, broker, tg=False, **kw):
    return ejecutar_ciclo(db, cfg, mercado=mercado, broker=broker, tg=tg, usar_ia=False, **kw)


def aprobar_todo(db):
    for p in db.propuestas(["pendiente"]):
        assert ejecutor.decidir(db, p["id"], True, "prueba")[0]


# ---------------- estrategia ----------------
def test_deposito_genera_compras_dentro_del_efectivo(cfg, entorno):
    db, mercado, broker = entorno
    broker.depositar(10_000)
    r = ciclo(db, cfg, mercado, broker)
    assert r.nuevas and all(p["lado"] == "compra" and p["estado"] == "pendiente" for p in r.nuevas)
    gastado = sum(p["monto"] for p in r.nuevas)
    assert gastado <= 10_000 * (1 - cfg.reserva_efectivo_pct / 100) + 0.01
    assert {p["simbolo"] for p in r.nuevas} <= set(cfg.universo)
    assert all(p["monto"] / 10_000 * 100 <= cfg.max_posicion_pct for p in r.nuevas)


def test_filtro_de_tendencia_mueve_el_peso_al_refugio(cfg):
    cfg = dataclasses.replace(cfg, filtro_tendencia=True)
    mercado = MercadoSintetico(tendencias={"VTI": -0.004})
    inds = {s: estratega.indicadores(mercado, s) for s in cfg.universo}
    pesos, notas = estratega.pesos_tacticos(cfg, inds)
    assert pesos["VTI"] == 0 and "VTI" in notas
    assert pesos[ACTIVO_REFUGIO] >= cfg.pesos_objetivo["VTI"]


def test_cartera_personalizada_respeta_el_maximo_por_posicion():
    cfg = Config(cartera_objetivo=parsear_cartera("VTI:80, agg:20, malo"), max_posicion_pct=50)
    assert cfg.pesos_objetivo == {"VTI": 50, "AGG": 20}
    assert cfg.universo == ["AGG", "SHY", "VTI"]


def test_cartera_en_objetivo_no_propone_nada(cfg, entorno):
    db, mercado, broker = entorno
    broker.depositar(10_000)
    ciclo(db, cfg, mercado, broker)
    aprobar_todo(db)
    ciclo(db, cfg, mercado, broker)
    r = ciclo(db, cfg, mercado, broker, forzar_plan=True)
    assert r.nuevas == []


# ---------------- aprobación humana ----------------
def test_nada_se_ejecuta_sin_tu_aprobacion(cfg, entorno):
    db, mercado, broker = entorno
    broker.depositar(5_000)
    for _ in range(3):
        ciclo(db, cfg, mercado, broker)
    assert broker.cuenta()["posiciones"] == {}
    assert all(p["estado"] == "pendiente" for p in db.propuestas())


def test_aprobar_ejecuta_y_rechazar_no(cfg, entorno):
    db, mercado, broker = entorno
    broker.depositar(10_000)
    props = ciclo(db, cfg, mercado, broker).nuevas
    si, no = props[0], props[1]
    assert ejecutor.decidir(db, si["id"], True, "prueba")[0]
    assert ejecutor.decidir(db, no["id"], False, "prueba")[0]
    ciclo(db, cfg, mercado, broker)
    pos = broker.cuenta()["posiciones"]
    assert db.propuesta(si["id"])["estado"] == "ejecutada"
    assert pos[si["simbolo"]]["cantidad"] == pytest.approx(si["cantidad"])
    assert no["simbolo"] not in pos
    # Una decisión no se puede cambiar ni repetir
    assert not ejecutor.decidir(db, no["id"], True, "prueba")[0]


def test_una_orden_no_se_ejecuta_dos_veces(cfg, entorno):
    db, mercado, broker = entorno
    broker.depositar(10_000)
    p = ciclo(db, cfg, mercado, broker).nuevas[0]
    ejecutor.decidir(db, p["id"], True, "prueba")
    ejecutor.ejecutar(db, cfg, broker, mercado, p["id"])
    ejecutor.ejecutar(db, cfg, broker, mercado, p["id"])
    assert broker.cuenta()["posiciones"][p["simbolo"]]["cantidad"] == pytest.approx(p["cantidad"])


def test_propuestas_sin_respuesta_vencen(cfg, entorno):
    db, mercado, broker = entorno
    broker.depositar(10_000)
    props = ciclo(db, cfg, mercado, broker).nuevas
    pasado = iso(ahora_utc() - timedelta(minutes=1))
    for p in props:
        db.actualizar_propuesta(p["id"], vence_en=pasado)
    ok, _ = ejecutor.decidir(db, props[0]["id"], True, "prueba")
    assert not ok
    ciclo(db, cfg, mercado, broker)
    assert all(db.propuesta(p["id"])["estado"] == "vencida" for p in props)
    assert broker.cuenta()["posiciones"] == {}


def test_movimiento_brusco_de_precio_bloquea_la_ejecucion(cfg, entorno):
    db, mercado, broker = entorno
    broker.depositar(10_000)
    p = ciclo(db, cfg, mercado, broker).nuevas[0]
    ejecutor.decidir(db, p["id"], True, "prueba")
    mercado.fijar_precio(p["simbolo"], p["precio_ref"] * 1.05)
    res = ejecutor.ejecutar(db, cfg, broker, mercado, p["id"])
    assert res["estado"] == "bloqueada" and "precio se movió" in res["error"]
    assert broker.cuenta()["posiciones"] == {}


# ---------------- guardia de riesgo ----------------
def _valor(broker, mercado):
    return estratega.valorizar(broker.cuenta(), mercado)


def test_guardia_bloquea_simbolos_fuera_del_universo_y_apalancamiento(cfg, entorno):
    db, mercado, broker = entorno
    broker.depositar(1_000)
    valor = _valor(broker, mercado)
    fuera = {"simbolo": "GME", "lado": "compra", "cantidad": 1, "precio_ref": 20}
    assert any("permitidos" in m for m in guardia.evaluar(fuera, valor, cfg, db))
    caro = {"simbolo": "VTI", "lado": "compra", "cantidad": 100, "precio_ref": 100}
    assert any("efectivo" in m for m in guardia.evaluar(caro, valor, cfg, db))
    corto = {"simbolo": "VTI", "lado": "venta", "cantidad": 1, "precio_ref": 100}
    assert any("corto" in m for m in guardia.evaluar(corto, valor, cfg, db))


def test_guardia_limita_ordenes_por_dia(cfg, entorno):
    db, mercado, broker = entorno
    broker.depositar(1_000)
    orden = {"simbolo": "VTI", "lado": "compra", "cantidad": 0.1, "precio_ref": mercado.precio("VTI")}
    motivos = guardia.evaluar(orden, _valor(broker, mercado), cfg, db, ordenes_hoy=cfg.max_ordenes_dia)
    assert any("órdenes hoy" in m for m in motivos)


def test_freno_de_perdidas_pausa_compras(cfg, entorno):
    db, mercado, broker = entorno
    broker.depositar(10_000)
    ciclo(db, cfg, mercado, broker)
    aprobar_todo(db)
    ciclo(db, cfg, mercado, broker)
    for s in broker.cuenta()["posiciones"]:
        mercado.fijar_precio(s, mercado.precio(s) * 0.7)
    valor = _valor(broker, mercado)
    assert guardia.freno_activo(db, cfg, valor)
    compra = {"simbolo": "VTI", "lado": "compra", "cantidad": 0.1, "precio_ref": mercado.precio("VTI")}
    assert any("Freno" in m for m in guardia.evaluar(compra, valor, cfg, db))


def test_stop_de_perdida_propone_vender(cfg, entorno):
    db, mercado, broker = entorno
    broker.depositar(10_000)
    ciclo(db, cfg, mercado, broker)
    aprobar_todo(db)
    ciclo(db, cfg, mercado, broker)
    simbolo = next(iter(broker.cuenta()["posiciones"]))
    mercado.fijar_precio(simbolo, mercado.precio(simbolo) * 0.8)
    r = ciclo(db, cfg, mercado, broker)
    stop = [p for p in r.nuevas if p["tipo"] == "stop"]
    assert stop and stop[0]["simbolo"] == simbolo and stop[0]["lado"] == "venta"
    assert broker.cuenta()["posiciones"][simbolo]["cantidad"] > 0  # aún espera tu aprobación


# ---------------- Telegram ----------------
class TelegramFalso(notificador.Telegram):
    def __init__(self, actualizaciones):
        self.chat_id = "111"
        self._act = actualizaciones
        self.enviados = []

    def _api(self, metodo, **datos):
        if metodo == "getUpdates":
            act, self._act = self._act, []
            return act
        if metodo == "sendMessage":
            self.enviados.append(datos)
            return {"message_id": len(self.enviados)}
        return True


def _boton(update_id, chat, usuario, data):
    return {"update_id": update_id, "callback_query": {
        "id": f"cb{update_id}", "from": {"id": usuario}, "data": data, "message": {"chat": {"id": chat}},
    }}


def test_telegram_consulta_y_solo_acepta_tu_chat(cfg, entorno):
    db, mercado, broker = entorno
    broker.depositar(10_000)
    tg = TelegramFalso([])
    props = ciclo(db, cfg, mercado, broker, tg=tg).nuevas
    botones = [d for d in tg.enviados if "reply_markup" in d]
    assert len(botones) == len(props)

    intruso, mio = props[0]["id"], props[1]["id"]
    tg._act = [_boton(5, 999, 999, f"si:{intruso}"), _boton(6, 111, 111, f"si:{mio}")]
    ciclo(db, cfg, mercado, broker, tg=tg)
    assert db.propuesta(intruso)["estado"] == "pendiente"
    assert db.propuesta(mio)["estado"] == "ejecutada"
    assert db.get_meta("telegram_offset") == "7"


# ---------------- configuración y broker ----------------
def test_dinero_real_requiere_confirmacion_explicita():
    cfg = Config(broker="alpaca", alpaca_api_key="k", alpaca_secret_key="s", alpaca_paper=False)
    assert not cfg.broker_listo[0]
    with pytest.raises(ErrorBroker):
        BrokerAlpaca(cfg)
    assert dataclasses.replace(cfg, trading_real_confirmado=True).broker_listo[0]
    assert Config(broker="alpaca", alpaca_api_key="k", alpaca_secret_key="s").broker_listo[0]


def test_alpaca_envia_orden_idempotente():
    class Resp:
        def __init__(self, codigo, datos):
            self.status_code, self._d, self.text = codigo, datos, ""

        def json(self):
            return self._d

    class Sesion:
        headers = {}

        def __init__(self):
            self.pedidos = []

        def request(self, metodo, url, **kw):
            self.pedidos.append((metodo, url, kw))
            if metodo == "POST":
                return Resp(422, {"message": "client_order_id must be unique"})
            return Resp(200, {"id": "abc", "status": "filled", "filled_qty": "2", "filled_avg_price": "100.5"})

    sesion = Sesion()
    broker = BrokerAlpaca(Config(broker="alpaca", alpaca_api_key="k", alpaca_secret_key="s"), sesion=sesion)
    prop = {"id": 3, "plan_id": "p1", "simbolo": "VTI", "lado": "compra", "cantidad": 2.0, "estado": "aprobada"}
    res = broker.ejecutar(prop)
    assert res == {"estado": "ejecutada", "precio": 100.5, "cantidad": 2.0, "id_broker": "abc"}
    metodo, url, kw = sesion.pedidos[0]
    assert url.startswith(BrokerAlpaca.URL_PAPER) and kw["json"]["client_order_id"] == "copahue-p1-3"
    assert kw["json"]["qty"] == "2"


def test_demo_funciona_sin_internet(cfg):
    db = BaseDatos(":memory:")
    r = preparar_demo(db, cfg)
    assert r.nuevas and db.total_aportes() == 10_000


def test_yahoo_lee_cierres_ajustados_y_precio_actual():
    from agentes_trading.mercado import ErrorMercado, MercadoYahoo

    class Resp:
        def __init__(self, datos):
            self._d = datos

        def raise_for_status(self):
            pass

        def json(self):
            return self._d

    class Sesion:
        headers = {}

        def get(self, url, params=None, timeout=None):
            if "MALO" in url:
                return Resp({"chart": {"result": None, "error": {"code": "Not Found"}}})
            return Resp({"chart": {"result": [{
                "meta": {"regularMarketPrice": 101.5},
                "timestamp": [1760000000, 1760086400, 1760172800],
                "indicators": {"quote": [{"close": [100, None, 102]}], "adjclose": [{"adjclose": [99.0, None, 101.0]}]},
            }]}})

    m = MercadoYahoo(sesion=Sesion())
    assert [c for _, c in m.historial("VTI")] == [99.0, 101.0]
    assert m.precio("VTI") == 101.5
    with pytest.raises(ErrorMercado):
        m.precio("MALO")


def test_stop_rechazado_no_se_repite_y_lo_vendido_no_se_recompra(cfg, entorno):
    db, mercado, broker = entorno
    broker.depositar(10_000)
    ciclo(db, cfg, mercado, broker)
    aprobar_todo(db)
    ciclo(db, cfg, mercado, broker)
    simbolo = next(iter(broker.cuenta()["posiciones"]))
    mercado.fijar_precio(simbolo, mercado.precio(simbolo) * 0.8)
    stop = ciclo(db, cfg, mercado, broker).nuevas[0]
    ejecutor.decidir(db, stop["id"], False, "prueba")
    assert ciclo(db, cfg, mercado, broker).nuevas == []

    db.actualizar_propuesta(stop["id"], estado="ejecutada")  # como si lo hubieras vendido
    broker.db.sim_guardar_posicion(simbolo, 0, 0)
    plan = ciclo(db, cfg, mercado, broker, forzar_plan=True).nuevas
    assert simbolo not in {p["simbolo"] for p in plan if p["lado"] == "compra"}
