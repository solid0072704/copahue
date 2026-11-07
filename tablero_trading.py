"""Tablero del agente de trading: cartera, operaciones, señal y prueba histórica.

Ejecutar:  streamlit run tablero_trading.py
"""

import dataclasses
import os

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Agente de trading", page_icon="📈", layout="wide")

from agentes_ebook.config import aplicar_secrets_streamlit  # noqa: E402

aplicar_secrets_streamlit()

from agentes_trading import ConfigTrading, ejecutar_ciclo  # noqa: E402
from agentes_trading.backtest import ejecutar_backtest  # noqa: E402
from agentes_trading.cartera import Cartera  # noqa: E402
from agentes_trading.datos import cargar_universo, precios_sinteticos, tabla  # noqa: E402
from agentes_trading.informe import AVISO  # noqa: E402

cfg = ConfigTrading.desde_entorno()
NOMBRE_MODO = {"simulado": "Cartera simulada (dinero ficticio)", "alpaca_papel": "Alpaca · cuenta de práctica", "alpaca_real": "Alpaca · dinero REAL"}


@st.cache_data(ttl=3600, show_spinner="Descargando precios…")
def precios_reales(tickers, ruta, años):
    return cargar_universo(list(tickers), ruta, años)


def cargar_precios(demo):
    if demo:
        return precios_sinteticos(cfg.universo), []
    return precios_reales(tuple(cfg.universo), cfg.ruta_precios, cfg.años_historia)


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.title("📈 Agente de trading")
    modo = cfg.modo if cfg.dinero_real or cfg.modo != "alpaca_real" else "alpaca_papel"
    st.caption(NOMBRE_MODO[modo])
    demo = st.toggle("🧪 Precios ficticios (sin internet)", value=False)
    for problema in cfg.validar():
        st.warning(problema)
    if st.button("▶️ Ejecutar el agente ahora", type="primary", width="stretch", disabled=demo):
        with st.spinner("Descargando precios, revisando la cartera y decidiendo…"):
            resultado = ejecutar_ciclo(cfg, notificar=False)
        st.session_state.resultado = resultado
        precios_reales.clear()
        st.rerun()
    if "resultado" in st.session_state:
        r = st.session_state.resultado
        (st.error if r.errores else st.success)(r.texto())
    st.divider()
    st.caption(f"Universo: {', '.join(cfg.universo)}")
    st.caption(f"Máx. {cfg.max_posiciones} posiciones · stop {cfg.stop_loss_pct:.0%} · refugio con caída de {cfg.max_caida_cartera_pct:.0%}")
    st.caption("Corre solo cada día hábil desde GitHub Actions si activas la variable TRADING_ACTIVO.")

st.warning(AVISO, icon="⚠️")

precios, fallidos = cargar_precios(demo)
if fallidos:
    st.error("Sin precios para: " + ", ".join(fallidos))
if not precios:
    st.stop()
cierres = tabla(precios, "Close")
ultimos = {t: float(v) for t, v in cierres.iloc[-1].dropna().items()}

# ---------------------------------------------------------------- cartera
cartera = Cartera.abrir(cfg) if not demo else Cartera(capital_inicial=cfg.capital_inicial, efectivo=cfg.capital_inicial)
patrimonio = cartera.patrimonio(ultimos)
ganancia = patrimonio - cartera.capital_inicial

st.subheader("Cartera")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Patrimonio", f"{patrimonio:,.2f} USD", f"{ganancia:+,.2f} USD")
c2.metric("Meta", f"{cfg.meta_usd:,.0f} USD", f"{ganancia / cfg.meta_usd:.0%} logrado" if cfg.meta_usd else "")
c3.metric("Efectivo", f"{cartera.efectivo:,.2f} USD")
c4.metric("Datos al", cartera.ultima_fecha or str(cierres.index[-1].date()))

if cartera.posiciones:
    filas = []
    for t, pos in sorted(cartera.posiciones.items()):
        precio = ultimos.get(t, pos["precio_promedio"])
        filas.append({"Activo": t, "Unidades": round(pos["cantidad"], 4), "Precio compra": pos["precio_promedio"], "Precio actual": round(precio, 2),
                      "Valor": round(pos["cantidad"] * precio, 2), "Resultado": f"{precio / pos['precio_promedio'] - 1:+.1%}" if pos["precio_promedio"] else ""})
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
else:
    st.info("Sin posiciones: la cartera está en efectivo." if not demo else "En modo ficticio no hay cartera guardada.")

if cartera.pendiente:
    p = cartera.pendiente
    detalle = []
    if p.get("vender"):
        detalle.append("vender " + ", ".join(p["vender"]))
    if p.get("objetivo") is not None:
        detalle.append("objetivo " + (", ".join(f"{t} {w:.0%}" for t, w in p["objetivo"].items()) or "100% efectivo"))
    st.info(f"Órdenes para la próxima apertura: {'; '.join(detalle)}. {p.get('motivo', '')}")

if cartera.curva:
    curva = pd.DataFrame(cartera.curva)
    curva["fecha"] = pd.to_datetime(curva["fecha"])
    graf = alt.Chart(curva).mark_line().encode(x=alt.X("fecha:T", title=""), y=alt.Y("patrimonio:Q", title="USD", scale=alt.Scale(zero=False)))
    st.altair_chart(graf, width="stretch")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Señal del agente")
    s = cartera.ultima_señal
    if s.get("ranking"):
        st.caption(f"{s.get('fecha', '')} · {s.get('motivo', '')}")
        rk = pd.DataFrame(s["ranking"])
        rk["momentum"] = rk["momentum"].map(lambda v: f"{v:+.1%}")
        rk["tendencia"] = rk["tendencia"].map(lambda v: "✅ alcista" if v else "❌ bajo su media")
        rk["peso objetivo"] = rk["ticker"].map(lambda t: f"{s['objetivo'].get(t, 0):.0%}")
        st.dataframe(rk, hide_index=True, width="stretch")
    else:
        st.caption("Todavía no hay señal: ejecuta el agente.")
with col_b:
    st.subheader("Operaciones")
    if cartera.operaciones:
        ops = pd.DataFrame(cartera.operaciones[::-1])
        st.dataframe(ops, hide_index=True, width="stretch", height=320)
    else:
        st.caption("Todavía no hay operaciones.")

# ---------------------------------------------------------------- backtest
st.subheader("Prueba histórica de la estrategia")
st.caption("Misma lógica que usa el agente en vivo, aplicada a los precios pasados. No garantiza resultados futuros.")
col1, col2 = st.columns([1, 3])
with col1:
    capital = st.number_input("Capital (USD)", 100.0, 10_000_000.0, float(cfg.capital_inicial), step=500.0)
    desde = st.date_input("Desde", value=cierres.index[0].date(), min_value=cierres.index[0].date(), max_value=cierres.index[-1].date())
    correr = st.button("Calcular", width="stretch")
if correr or "backtest" in st.session_state:
    if correr:
        with st.spinner("Simulando rueda a rueda…"):
            st.session_state.backtest = ejecutar_backtest(precios, dataclasses.replace(cfg, capital_inicial=capital), desde=str(desde))
    r = st.session_state.backtest
    m, ref = r.metricas, r.metricas_referencia
    with col2:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rentabilidad anual", f"{m['cagr']:+.1%}", f"{cfg.referencia}: {ref['cagr']:+.1%}", delta_color="off")
        k2.metric("Caída máxima", f"{m['caida_maxima']:.1%}", f"{cfg.referencia}: {ref['caida_maxima']:.1%}", delta_color="off")
        k3.metric("Sharpe", f"{m['sharpe']:.2f}", f"{cfg.referencia}: {ref['sharpe']:.2f}", delta_color="off")
        k4.metric(f"Meta de {cfg.meta_usd:,.0f} USD en 12 meses", f"{r.probabilidad_meta:.0%} de los periodos")
        datos = pd.concat([r.curva, r.referencia], axis=1).reset_index().melt("index", var_name="serie", value_name="USD")
        graf = alt.Chart(datos).mark_line().encode(x=alt.X("index:T", title=""), y=alt.Y("USD:Q", scale=alt.Scale(zero=False)), color=alt.Color("serie:N", title=""))
        st.altair_chart(graf, width="stretch")
        if r.meses_para_meta:
            st.caption(f"Si se repitiera la rentabilidad histórica, la meta tardaría ~{r.meses_para_meta:.0f} meses. {m['operaciones']} operaciones en {m['años']:.1f} años.")
