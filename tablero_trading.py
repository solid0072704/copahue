"""Tablero del agente de inversiones: depositas, el agente propone y tú decides.

Ejecutar:  streamlit run tablero_trading.py
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Agente de inversiones", page_icon="🤖", layout="wide")

from agentes_trading.config import Config, aplicar_secrets_streamlit  # noqa: E402

aplicar_secrets_streamlit()

from agentes_trading import ejecutor, estratega, guardia  # noqa: E402
from agentes_trading.brokers import ErrorBroker  # noqa: E402
from agentes_trading.db import ESTADOS, desde_iso  # noqa: E402
from agentes_trading.llm import ETIQUETA_OPINION  # noqa: E402
from agentes_trading.mercado import ErrorMercado  # noqa: E402
from agentes_trading.orquestador import abrir_db, ejecutar_ciclo, preparar, preparar_demo  # noqa: E402

cfg = Config.desde_entorno()
db = abrir_db(cfg)


@st.cache_resource(ttl=300, show_spinner=False)
def _mercado(demo):
    # Se guarda 5 minutos para no descargar los precios en cada clic.
    return preparar(cfg, db, tg=False)[0]


try:
    mercado, broker, _ = preparar(cfg, db, mercado=_mercado(db.get_meta("demo")), tg=False)
except ErrorBroker as e:
    st.error(f"No se pudo conectar con el broker: {e}")
    st.stop()


def usd(x):
    return f"US$ {x:,.2f}"


def local(texto_iso, formato="%d/%m %H:%M"):
    dt = desde_iso(texto_iso)
    return dt.astimezone(cfg.tz).strftime(formato) if dt else ""


def ciclo(**kw):
    with st.spinner("Los agentes están trabajando…"):
        r = ejecutar_ciclo(db, cfg, mercado=mercado, broker=broker, **kw)
    (st.warning if r.errores else st.toast)(r.texto())


# ---------------------------------------------------------------------------
# Barra lateral: dinero, acciones y conexiones
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🤖 Tu agente")
    listo, estado_broker = cfg.broker_listo
    st.caption(("🟢 " if listo else "🔴 ") + estado_broker)

    if st.button("▶️ Ejecutar agentes ahora", type="primary", width="stretch"):
        ciclo()
    if st.button("🧭 Pedir un plan nuevo", width="stretch"):
        ciclo(forzar_plan=True)

    st.subheader("💵 Tu dinero")
    if broker.acepta_depositos:
        monto = st.number_input("Monto (US$)", min_value=0.0, step=100.0, value=1000.0)
        c1, c2 = st.columns(2)
        if c1.button("Depositar", width="stretch") and monto > 0:
            broker.depositar(monto)
            ciclo()
            st.rerun()
        if c2.button("Retirar", width="stretch") and monto > 0:
            try:
                broker.retirar(monto)
                st.rerun()
            except ErrorBroker as e:
                st.error(str(e))
    else:
        st.caption("Deposita desde la app de Alpaca: el agente verá tu saldo y propondrá cómo invertirlo.")
        with st.expander("Registrar un depósito (para calcular tu rentabilidad)"):
            monto = st.number_input("Monto depositado (US$)", min_value=0.0, step=100.0)
            if st.button("Registrar") and monto > 0:
                db.registrar_deposito(monto, "Depósito en Alpaca")
                st.rerun()

    st.subheader("🔌 Conexiones")
    st.caption(
        "\n".join([
            f"{'🟢' if cfg.telegram_configurado else '⚪'} Telegram (te consulta cada orden)",
            f"{'🟢' if cfg.correo_configurado else '⚪'} Correo",
            f"{'🟢' if cfg.ia_configurada else '⚪'} Segunda opinión con IA (Claude)",
        ])
    )
    st.caption(
        f"Perfil **{cfg.perfil}** · máx. {cfg.max_posicion_pct:g}% por activo · stop {cfg.stop_perdida_pct:g}% · "
        f"freno {cfg.freno_perdida_pct:g}% · reserva {cfg.reserva_efectivo_pct:g}%"
    )
    if db.total_aportes() == 0:
        with st.expander("🧪 Probar con datos de demostración"):
            st.caption("Deposita US$ 10.000 ficticios con precios sintéticos. No envía nada.")
            if st.button("Cargar demo"):
                preparar_demo(db, cfg)
                st.cache_resource.clear()
                st.rerun()

# ---------------------------------------------------------------------------
# Encabezado y cifras
# ---------------------------------------------------------------------------
st.title("Agente de inversiones")
if db.get_meta("demo") == "1":
    st.info("🧪 Modo demostración: dinero y precios ficticios.")
elif cfg.dinero_real:
    st.error("💵 Operando con DINERO REAL en Alpaca. Cada orden igual espera tu aprobación.")
else:
    st.caption("🧪 Modo práctica: dinero ficticio con precios reales. Cuando estés conforme, conecta tu broker.")

try:
    valor = estratega.valorizar(broker.cuenta(), mercado)
except (ErrorBroker, ErrorMercado) as e:
    st.error(f"No se pudo leer tu cartera: {e}")
    st.stop()

aportes = db.total_aportes()
ganancia = valor["total"] - aportes
k = st.columns(4)
k[0].metric("Valor de tu cartera", usd(valor["total"]))
k[1].metric("Depositado", usd(aportes))
k[2].metric("Resultado", usd(ganancia), f"{(ganancia / aportes * 100) if aportes else 0:+.2f}%")
k[3].metric("Efectivo", usd(valor["efectivo"]))
if guardia.freno_activo(db, cfg, valor):
    st.warning("🛑 Freno de pérdidas activo: las compras están en pausa.")

# ---------------------------------------------------------------------------
# Propuestas que esperan tu decisión
# ---------------------------------------------------------------------------
pendientes = db.propuestas(["pendiente"])
st.subheader(f"🕒 Esperan tu decisión ({len(pendientes)})")
if not pendientes:
    st.caption("No hay órdenes por aprobar. El agente te avisará cuando proponga algo.")
else:
    if st.button(f"✅ Aprobar y ejecutar las {len(pendientes)}", type="primary"):
        for p in pendientes:
            ejecutor.decidir(db, p["id"], True, "tablero")
        for p in ejecutor.ejecutar_aprobadas(db, cfg, broker, mercado):
            st.toast(f"{p['simbolo']}: {ESTADOS[p['estado']]}")
        st.rerun()
for p in pendientes:
    with st.container(border=True):
        izq, der = st.columns([4, 1])
        lado = "🟢 Comprar" if p["lado"] == "compra" else "🔴 Vender"
        izq.markdown(
            f"**{lado} {p['cantidad']:.4f} {p['simbolo']}** a ~{usd(p['precio_ref'])} = **{usd(p['monto'])}** "
            f"· #{p['id']} · vence {local(p['vence_en'])}"
        )
        izq.write(p["motivo"])
        if p["opinion_ia"]:
            izq.caption(f"Segunda opinión IA — {ETIQUETA_OPINION.get(p['opinion_ia'], '')}: {p['explicacion_ia']}")
        if der.button("✅ Ejecutar", key=f"si-{p['id']}", type="primary", width="stretch"):
            ok, mensaje = ejecutor.decidir(db, p["id"], True, "tablero")
            if ok:
                r = ejecutor.ejecutar(db, cfg, broker, mercado, p["id"])
                mensaje = f"{ESTADOS[r['estado']]} {r['error'] or ''}"
            st.toast(mensaje)
            st.rerun()
        if der.button("❌ Rechazar", key=f"no-{p['id']}", width="stretch"):
            st.toast(ejecutor.decidir(db, p["id"], False, "tablero")[1])
            st.rerun()

# ---------------------------------------------------------------------------
# Detalle
# ---------------------------------------------------------------------------
tab_cartera, tab_evolucion, tab_historial, tab_log, tab_ayuda = st.tabs(
    ["📊 Cartera", "📈 Evolución", "🧾 Órdenes", "📜 Bitácora", "❓ Cómo funciona"]
)

with tab_cartera:
    objetivo = {s: p * (1 - cfg.reserva_efectivo_pct / 100) for s, p in cfg.pesos_objetivo.items()}
    filas = []
    for s in sorted(set(objetivo) | set(valor["posiciones"])):
        pos = valor["posiciones"].get(s, {})
        filas.append({
            "Activo": s,
            "Cantidad": pos.get("cantidad", 0.0),
            "Precio": pos.get("precio"),
            "Valor US$": pos.get("valor", 0.0),
            "Peso %": pos.get("peso", 0.0),
            "Objetivo %": objetivo.get(s, 0.0),
            "Ganancia %": pos.get("ganancia_pct"),
        })
    st.dataframe(
        pd.DataFrame(filas), hide_index=True, width="stretch",
        column_config={
            "Cantidad": st.column_config.NumberColumn(format="%.4f"),
            "Precio": st.column_config.NumberColumn(format="$%.2f"),
            "Valor US$": st.column_config.NumberColumn(format="$%.2f"),
            "Peso %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
            "Objetivo %": st.column_config.NumberColumn(format="%.1f%%"),
            "Ganancia %": st.column_config.NumberColumn(format="%+.2f%%"),
        },
    )
    st.caption("El objetivo puede cambiar temporalmente: el filtro de tendencia mueve a SHY el peso de los activos en caída.")

with tab_evolucion:
    vals = db.valorizaciones()
    if len(vals) < 2:
        st.caption("Aún no hay suficiente historial. Cada ciclo de los agentes agrega un punto.")
    else:
        df = pd.DataFrame(vals)
        df["fecha"] = pd.to_datetime(df["fecha"]).dt.tz_convert(cfg.zona_horaria)
        st.line_chart(df.set_index("fecha")[["valor_total", "aportes"]].rename(
            columns={"valor_total": "Valor de la cartera", "aportes": "Depositado"}
        ))

with tab_historial:
    props = db.propuestas()
    if props:
        df = pd.DataFrame(props)
        df["estado"] = df["estado"].map(ESTADOS)
        df["creada_en"] = df["creada_en"].map(local)
        st.dataframe(
            df[["id", "creada_en", "estado", "lado", "simbolo", "cantidad", "precio_ref", "monto",
                "precio_ejecucion", "decidida_por", "motivo", "riesgo", "error"]],
            hide_index=True, width="stretch",
        )
    else:
        st.caption("Todavía no hay órdenes.")

with tab_log:
    st.dataframe(pd.DataFrame(db.bitacora(300)), hide_index=True, width="stretch")

with tab_ayuda:
    st.markdown(
        """
1. **Depositas** tu dinero (en el broker simulado aquí, o en tu cuenta de Alpaca).
2. El **Estratega** arma un plan para llevar tu cartera al objetivo de tu perfil de riesgo,
   con filtro de tendencia, rebalanceo y stop de pérdida.
3. La **Guardia de riesgo** revisa cada orden: activos permitidos, sin apalancamiento ni ventas en corto,
   máximo por activo, órdenes por día y freno de pérdidas. Lo que no cumple, se bloquea.
4. Opcional: **Claude** da una segunda opinión y te explica cada orden en simple.
5. El **Notificador** te consulta por Telegram (botones ✅ / ❌), por correo o aquí.
6. El **Ejecutor** solo ejecuta lo que apruebas, y antes vuelve a revisar que el precio no se haya movido
   más del límite. Si no respondes a tiempo, la propuesta vence y no se ejecuta.

⚠️ Invertir tiene riesgo: puedes perder dinero. Este sistema no garantiza resultados ni es asesoría financiera.
"""
    )
