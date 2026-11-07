"""Pruebas del agente de trading (sin internet ni broker real)."""

import dataclasses
import json

import pandas as pd
import pytest

from agentes_trading import ConfigTrading, ejecutar_ciclo
from agentes_trading.backtest import ejecutar_backtest, metricas, probabilidad_de_meta
from agentes_trading.broker import URL_PAPEL, URL_REAL, BrokerAlpaca
from agentes_trading.cartera import Cartera
from agentes_trading.datos import cache_actualizada, obtener_precios, precios_sinteticos, tabla
from agentes_trading.estrategia import caida_excesiva, revisar_stops, señal_objetivo
from agentes_trading.informe import AVISO, construir_informe
from agentes_trading.motor import decidir, ejecutar_pendiente, paso
from agentes_trading.proyeccion import aporte_necesario, capital_para_renta, escenarios, meses_hasta, patrimonio_futuro, texto_escenarios
from agentes_trading.datos import tipo_cambio_usdclp

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture
def cfg(tmp_path):
    return ConfigTrading(universo=TICKERS, referencia="AAA", ruta_datos=str(tmp_path / "trading"), max_posiciones=2)


@pytest.fixture
def precios():
    return precios_sinteticos(TICKERS, dias=700, semilla=3)


# ---------------------------------------------------------------- configuración
def test_config_modo_invalido_cae_en_simulado(monkeypatch):
    monkeypatch.setenv("TRADING_MODO", "lo-que-sea")
    assert ConfigTrading.desde_entorno().modo == "simulado"


def test_dinero_real_requiere_confirmacion():
    cfg = ConfigTrading(modo="alpaca_real", alpaca_key="k", alpaca_secret="s")
    assert not cfg.dinero_real
    assert any("CONFIRMO_DINERO_REAL" in p for p in cfg.validar())
    assert dataclasses.replace(cfg, confirmo_dinero_real=True).dinero_real


def test_validar_detecta_faltantes():
    problemas = ConfigTrading(modo="alpaca_papel", universo=["SPY"], referencia="QQQ").validar()
    assert any("ALPACA" in p for p in problemas) and any("referencia" in p.lower() for p in problemas)


# ---------------------------------------------------------------- datos
def test_precios_sinteticos_y_tabla(precios):
    assert set(precios) == set(TICKERS)
    for df in precios.values():
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"] and (df["Close"] > 0).all()
    cierres = tabla(precios, "Close")
    assert cierres.shape == (700, 4)


def test_cache_actualizada_cuenta_fines_de_semana():
    df = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2026-11-06"]))  # viernes
    assert cache_actualizada(df, hoy=pd.Timestamp("2026-11-08").date())      # domingo
    assert not cache_actualizada(df, hoy=pd.Timestamp("2026-11-09").date())  # lunes: hay que bajar el cierre del lunes
    assert not cache_actualizada(df, hoy=pd.Timestamp("2026-11-11").date())  # miércoles
    assert not cache_actualizada(None)


def test_obtener_precios_usa_cache_si_no_hay_red(tmp_path, precios, monkeypatch):
    import requests

    def sin_red(*a, **k):
        raise requests.ConnectionError("sin red")

    monkeypatch.setattr(requests, "get", sin_red)
    ruta = str(tmp_path / "precios")
    with pytest.raises(RuntimeError):
        obtener_precios("AAA", ruta)
    import os

    os.makedirs(ruta, exist_ok=True)
    precios["AAA"].to_csv(os.path.join(ruta, "AAA.csv"))
    df = obtener_precios("AAA", ruta, hoy=pd.Timestamp("2030-01-01").date())  # caché vieja pero sin red
    assert len(df) == 700


# ---------------------------------------------------------------- estrategia
def test_señal_elige_mejor_momentum_en_tendencia(cfg):
    fechas = pd.bdate_range("2020-01-01", periods=260)
    sube = pd.Series(range(100, 360), index=fechas, dtype=float)             # tendencia alcista
    baja = pd.Series(range(360, 100, -1), index=fechas, dtype=float)         # bajista
    plano = pd.Series([200.0] * 260, index=fechas)
    sube2 = sube * 1.5
    cierres = pd.DataFrame({"AAA": sube, "BBB": baja, "CCC": plano, "DDD": sube2})
    s = señal_objetivo(cierres, cfg)
    assert set(s.objetivo) == {"AAA", "DDD"}
    assert abs(sum(s.objetivo.values()) - (1 - cfg.reserva_efectivo_pct)) < 0.01
    assert s.ranking[0][0] in ("AAA", "DDD") and s.ranking[-1][0] == "BBB"


def test_señal_sin_candidatos_va_a_efectivo(cfg):
    fechas = pd.bdate_range("2020-01-01", periods=260)
    baja = pd.Series(range(360, 100, -1), index=fechas, dtype=float)
    cierres = pd.DataFrame({t: baja for t in TICKERS})
    s = señal_objetivo(cierres, cfg)
    assert s.objetivo == {} and "efectivo" in s.motivo


def test_señal_con_pocos_datos(cfg):
    cierres = pd.DataFrame({"AAA": [1.0, 2.0]}, index=pd.bdate_range("2020-01-01", periods=2))
    assert señal_objetivo(cierres, cfg).objetivo == {}


def test_stops_y_caida(cfg):
    posiciones = {"AAA": {"cantidad": 1, "precio_promedio": 100}, "BBB": {"cantidad": 1, "precio_promedio": 100}}
    assert revisar_stops(posiciones, {"AAA": 91.0, "BBB": 95.0}, cfg) == ["AAA"]
    assert caida_excesiva(8_400, 10_000, cfg) and not caida_excesiva(9_000, 10_000, cfg)


# ---------------------------------------------------------------- cartera
def test_cartera_compra_vende_y_persiste(tmp_path):
    c = Cartera(capital_inicial=1000, efectivo=1000, maximo=1000)
    op = c.comprar("AAA", 500, 50.0, "2024-01-02", comision_pct=0.01)
    assert op["lado"] == "compra" and c.efectivo == 500 and c.posiciones["AAA"]["cantidad"] == pytest.approx(9.9)
    assert c.patrimonio({"AAA": 60.0}) == pytest.approx(500 + 9.9 * 60)
    venta = c.vender("AAA", 5, 60.0, "2024-02-02")
    assert venta["resultado"] == pytest.approx(50.0, abs=0.1)
    assert c.vender_todo("AAA", 60.0, "2024-02-03") and "AAA" not in c.posiciones
    assert c.comprar("AAA", 0.5, 50.0, "2024-02-04") is None
    ruta = tmp_path / "c.json"
    c.guardar(str(ruta))
    c2 = Cartera.cargar(str(ruta))
    assert c2.efectivo == pytest.approx(c.efectivo) and len(c2.operaciones) == 3
    assert json.loads(ruta.read_text())["capital_inicial"] == 1000


# ---------------------------------------------------------------- motor
def _mercado(subida_final=1.0):
    fechas = pd.bdate_range("2020-01-01", periods=240)
    base = pd.Series(range(100, 340), index=fechas, dtype=float)
    cierres = pd.DataFrame({"AAA": base, "BBB": base * 0.5, "CCC": base * 2, "DDD": base * 0.8})
    cierres.iloc[-1] *= subida_final
    aperturas = cierres * 1.001
    return aperturas, cierres


def test_pendiente_se_ejecuta_en_la_apertura_siguiente(cfg):
    cfg = dataclasses.replace(cfg, ventana_momentum=100, ventana_tendencia=150)
    aperturas, cierres = _mercado()
    c = Cartera(capital_inicial=10_000, efectivo=10_000, maximo=10_000)
    pendiente = decidir(c, cierres, cierres.index[-2], cfg)
    assert pendiente["objetivo"] and not c.posiciones
    hechas = ejecutar_pendiente(c, aperturas, cierres.index[-1], cfg)
    assert hechas and all(op["fecha"] == str(cierres.index[-1].date()) for op in hechas)
    assert set(c.posiciones) == set(pendiente["objetivo"]) and c.pendiente == {}
    for op in hechas:  # se compró al precio de apertura, no al de cierre
        assert op["precio"] == pytest.approx(aperturas.loc[cierres.index[-1], op["ticker"]], rel=1e-6)


def test_stop_loss_vende_y_no_recompra_el_mismo_dia(cfg):
    aperturas, cierres = _mercado()
    c = Cartera(capital_inicial=10_000, efectivo=2_000, maximo=5_000, ultimo_rebalanceo=str(cierres.index[-2].date()))
    c.posiciones = {"AAA": {"cantidad": 10, "precio_promedio": 500.0, "fecha": "2020-01-02"}}  # muy por encima del precio actual
    pendiente = decidir(c, cierres, cierres.index[-1], cfg)
    assert pendiente["vender"] == ["AAA"] and "stop-loss" in pendiente["motivo_venta"]


def test_caida_de_cartera_manda_todo_a_efectivo(cfg):
    aperturas, cierres = _mercado()
    c = Cartera(capital_inicial=10_000, efectivo=0, maximo=20_000, ultimo_rebalanceo=str(cierres.index[-2].date()))
    precio = cierres.iloc[-1]["AAA"]
    c.posiciones = {"AAA": {"cantidad": 16_000 / precio, "precio_promedio": precio, "fecha": ""}}
    pendiente = decidir(c, cierres, cierres.index[-1], cfg)
    assert pendiente["vender"] == ["AAA"] and c.refugio_hasta and "máximo" in pendiente["motivo_venta"]
    assert c.maximo == pytest.approx(16_000)


def test_paso_completo_no_falla_sin_pendiente(cfg):
    aperturas, cierres = _mercado()
    c = Cartera()
    hechas, pendiente = paso(c, aperturas, cierres, cierres.index[-1], cfg)
    assert hechas == [] and c.ultima_fecha == str(cierres.index[-1].date())


# ---------------------------------------------------------------- backtest
def test_backtest_produce_metricas_coherentes(cfg, precios):
    r = ejecutar_backtest(precios, cfg)
    assert len(r.curva) > 300 and r.curva.iloc[0] == pytest.approx(cfg.capital_inicial, rel=0.05)
    assert -1 < r.metricas["caida_maxima"] <= 0 and r.metricas["operaciones"] == len(r.operaciones) > 0
    assert 0 <= r.probabilidad_meta <= 1 and "Estrategia" in r.resumen()
    assert len(r.referencia) == len(r.curva)


def test_metricas_y_probabilidad():
    fechas = pd.bdate_range("2020-01-01", periods=504)
    curva = pd.Series([100 * 1.0005**i for i in range(504)], index=fechas)
    m = metricas(curva)
    assert m["cagr"] > 0.1 and m["caida_maxima"] == 0 and m["meses_positivos"] == 1.0
    assert probabilidad_de_meta(curva, 10_000, 1_000) == 1.0
    assert probabilidad_de_meta(curva, 10_000, 5_000) == 0.0
    assert metricas(pd.Series([1.0]))["cagr"] == 0.0


# ---------------------------------------------------------------- agente
def test_ciclo_simulado_arranca_hoy_y_ejecuta_al_dia_siguiente(cfg, precios):
    cierres = tabla(precios, "Close")
    penultimo = cierres.index[-2]
    r1 = ejecutar_ciclo(cfg, precios=precios, notificar=False, hoy=penultimo.date())
    assert not r1.errores and r1.operaciones == [] and r1.fecha == str(penultimo.date())
    c = Cartera.cargar(cfg.ruta_cartera)
    assert c.efectivo == cfg.capital_inicial and len(c.curva) == 1
    r2 = ejecutar_ciclo(cfg, precios=precios, notificar=False)
    assert not r2.errores and r2.fecha == str(cierres.index[-1].date())
    if r1.pendiente.get("objetivo"):
        assert r2.operaciones and Cartera.cargar(cfg.ruta_cartera).posiciones
    assert AVISO in r2.informe
    # tercera vez el mismo día: no repite nada
    r3 = ejecutar_ciclo(cfg, precios=precios, notificar=False)
    assert r3.operaciones == []


def test_ciclo_sin_datos_informa_error(cfg):
    r = ejecutar_ciclo(cfg, precios={}, notificar=False)
    assert r.errores


def test_informe_incluye_posiciones_y_aviso(cfg):
    c = Cartera(capital_inicial=1000, efectivo=400, maximo=1000, ultima_fecha="2026-01-02")
    c.posiciones = {"AAA": {"cantidad": 10, "precio_promedio": 60.0, "fecha": ""}}
    texto = construir_informe(c, {"AAA": 66.0}, cfg, [], {"objetivo": {}, "motivo": "todo a efectivo"})
    assert "AAA" in texto and "+10.0%" in texto and "100% efectivo" in texto and AVISO in texto
    assert "Progreso 6%" in texto


# ---------------------------------------------------------------- broker Alpaca (sesión falsa)
class _Respuesta:
    def __init__(self, datos, status=200):
        self._datos, self.status_code, self.text = datos, status, json.dumps(datos)

    def json(self):
        return self._datos

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _SesionFalsa:
    def __init__(self):
        self.headers, self.llamadas = {}, []
        self.cuenta = {"cash": "5000", "equity": "9000"}
        self.posiciones = [{"symbol": "AAA", "qty": "10", "avg_entry_price": "400", "current_price": "400"}]

    def get(self, url, params=None, timeout=None):
        self.llamadas.append(("GET", url, params))
        if url.endswith("/v2/account"):
            return _Respuesta(self.cuenta)
        if url.endswith("/v2/positions"):
            return _Respuesta(self.posiciones)
        return _Respuesta({})

    def post(self, url, json=None, timeout=None):
        self.llamadas.append(("POST", url, json))
        return _Respuesta({"id": "orden", **json})

    def delete(self, url, timeout=None):
        self.llamadas.append(("DELETE", url, None))
        return _Respuesta({})


def test_broker_usa_cuenta_de_practica_salvo_confirmacion():
    cfg = ConfigTrading(modo="alpaca_real", alpaca_key="k", alpaca_secret="s")
    assert BrokerAlpaca(cfg, _SesionFalsa()).base == URL_PAPEL
    assert BrokerAlpaca(dataclasses.replace(cfg, confirmo_dinero_real=True), _SesionFalsa()).base == URL_REAL
    with pytest.raises(RuntimeError):
        BrokerAlpaca(ConfigTrading(modo="alpaca_papel"))


def test_broker_sincroniza_y_envia_ordenes():
    cfg = ConfigTrading(modo="alpaca_papel", alpaca_key="k", alpaca_secret="s")
    sesion = _SesionFalsa()
    broker = BrokerAlpaca(cfg, sesion)
    assert sesion.headers["APCA-API-KEY-ID"] == "k"
    c = Cartera()
    assert broker.sincronizar(c) == 9000
    assert c.efectivo == 5000 and c.posiciones["AAA"]["cantidad"] == 10 and c.capital_inicial == 9000
    precios = {"AAA": 400.0, "BBB": 50.0}
    pendiente = {"fecha": "2026-01-05", "vender": [], "objetivo": {"BBB": 0.5}, "motivo": "rebalanceo"}
    enviadas = broker.enviar(c, pendiente, precios)
    lados = [(op["lado"], op["ticker"]) for op in enviadas]
    assert ("venta", "AAA") in lados and ("compra", "BBB") in lados
    metodos = [(m, u.split("/v2/")[-1]) for m, u, _ in sesion.llamadas]
    assert ("DELETE", "positions/AAA") in metodos
    compra = next(j for m, u, j in sesion.llamadas if m == "POST")
    assert compra["symbol"] == "BBB" and compra["side"] == "buy" and float(compra["notional"]) == pytest.approx(4500, rel=0.01)


def test_ciclo_alpaca_decide_y_envia(cfg, precios, monkeypatch):
    import agentes_trading.agente as agente

    sesion = _SesionFalsa()
    sesion.posiciones = []
    cfg = dataclasses.replace(cfg, modo="alpaca_papel", alpaca_key="k", alpaca_secret="s")
    monkeypatch.setattr(agente, "_broker", lambda cfg: BrokerAlpaca(cfg, sesion))
    r = ejecutar_ciclo(cfg, precios=precios, notificar=False)
    assert not r.errores and r.pendiente == {}
    c = Cartera.cargar(cfg.ruta_cartera)
    assert c.modo == "alpaca_papel" and c.efectivo == 5000
    if c.ultima_señal.get("objetivo"):
        assert any(m == "POST" for m, _, _ in sesion.llamadas) and r.operaciones


# ---------------------------------------------------------------- aportes mensuales en pesos
def test_cartera_aporta_y_distingue_ganancia():
    c = Cartera(capital_inicial=0, efectivo=0, maximo=0)
    assert c.capital_aportado == 0
    a = c.aportar(15.79, "2026-01-02", clp=15000, usdclp=950, detalle="aporte mensual")
    assert a["clp"] == 15000 and c.efectivo == pytest.approx(15.79) and c.capital_aportado == pytest.approx(15.79)
    assert c.ganancia({}) == pytest.approx(0)
    assert not c.aporte_del_mes_pendiente("2026-01-20") and c.aporte_del_mes_pendiente("2026-02-02")
    assert c.aportar(0, "2026-02-02") is None
    c.anotar_curva("2026-01-02", {})
    assert c.curva[-1]["aporte"] == pytest.approx(15.79)


def test_cartera_antigua_sin_campo_aportado_sigue_funcionando(tmp_path):
    ruta = tmp_path / "vieja.json"
    ruta.write_text(json.dumps({"capital_inicial": 500, "efectivo": 500, "maximo": 500}))
    c = Cartera.cargar(str(ruta))
    assert c.capital_aportado == 500 and c.ganancia({}) == 0


def test_paso_suma_aporte_el_primer_dia_habil_del_mes(cfg):
    aperturas, cierres = _mercado()
    c = Cartera(capital_inicial=0, efectivo=0, maximo=0)
    fechas = cierres.index
    paso(c, aperturas, cierres, fechas[0], cfg, aporte_usd=10, clp=9500, usdclp=950)
    paso(c, aperturas, cierres, fechas[1], cfg, aporte_usd=10, clp=9500, usdclp=950)
    assert len(c.aportes) == 1 and c.capital_aportado == 10
    primera_del_mes_siguiente = next(f for f in fechas if f.month != fechas[0].month)
    paso(c, aperturas, cierres, primera_del_mes_siguiente, cfg, aporte_usd=10, clp=9500, usdclp=950)
    assert len(c.aportes) == 2 and c.aportes[-1]["fecha"] == str(primera_del_mes_siguiente.date())


def test_backtest_con_aportes_mide_solo_la_estrategia(cfg, precios):
    cfg0 = dataclasses.replace(cfg, capital_inicial=0.0)
    r = ejecutar_backtest(precios, cfg0, aporte_usd=20)
    assert r.aportado_total > 0 and r.patrimonio_final > 0 and r.aportado_total == pytest.approx(20 * len({p["fecha"][:7] for p in r.cartera.curva}))
    assert r.curva.iloc[0] == pytest.approx(100) and r.referencia.iloc[0] == pytest.approx(100)
    assert -1 < r.metricas["caida_maxima"] <= 0
    # sin aportes ni capital no pasa nada y no falla
    r0 = ejecutar_backtest(precios, cfg0)
    assert r0.metricas["operaciones"] == 0


def test_ciclo_con_aporte_en_pesos(cfg, precios):
    cfg = dataclasses.replace(cfg, capital_inicial=0.0, aporte_mensual_clp=15000, meta_mensual_clp=500000)
    r = ejecutar_ciclo(cfg, precios=precios, notificar=False, usdclp=950)
    assert not r.errores and r.aporte and r.aporte["clp"] == 15000 and r.aporte["importe"] == pytest.approx(15000 / 950, abs=0.01)
    assert "150.000.000 CLP" in r.informe and "15.000 CLP" in r.informe and "Tipo de cambio: 950" in r.informe
    assert r.texto().startswith("datos al")
    r2 = ejecutar_ciclo(cfg, precios=precios, notificar=False, usdclp=950)
    assert not r2.aporte  # mismo mes: no se repite el aporte


def test_tipo_de_cambio_usa_cache_o_defecto(tmp_path, monkeypatch):
    import requests

    def sin_red(*a, **k):
        raise requests.ConnectionError("sin red")

    monkeypatch.setattr(requests, "get", sin_red)
    ruta = str(tmp_path / "usdclp.json")
    assert tipo_cambio_usdclp(ruta, 950.0) == (950.0, "valor por defecto")
    (tmp_path / "usdclp.json").write_text(json.dumps({"valor": 930.5, "fecha": "2026-11-05", "consultado": "2026-11-05"}))
    assert tipo_cambio_usdclp(ruta, 950.0, hoy="2026-11-06") == (930.5, "2026-11-05")
    assert tipo_cambio_usdclp(ruta, 950.0, hoy="2026-11-05") == (930.5, "2026-11-05")


def test_proyeccion_aritmetica():
    assert patrimonio_futuro(100, 0.0, 12) == pytest.approx(1200)
    assert patrimonio_futuro(0, 0.10, 12, capital_inicial=1000) == pytest.approx(1100)
    assert patrimonio_futuro(15000, 0.10, 240) == pytest.approx(10_773_888, rel=1e-3)
    assert capital_para_renta(500_000, 0.04) == 150_000_000
    assert meses_hasta(1000, 100, 0.0) == 10 and meses_hasta(10**12, 1, 0.0) is None and meses_hasta(5, 1, 0.0, capital_inicial=10) == 0
    assert aporte_necesario(500_000, 0.10, 20) == pytest.approx(208_838, rel=1e-3)
    e = escenarios(15000, 500_000)
    assert e["capital_meta"] == 150_000_000 and e["filas"][2]["años_para_meta"] == pytest.approx(46, abs=1)
    texto = texto_escenarios(15000, 500_000, cagr_historico=0.07)
    assert "150.000.000 CLP" in texto and "(hist.)" in texto and "Aporte mensual que haría falta" in texto
