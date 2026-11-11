"""Plataforma de gestión de clientes para estudios contables.

Ejecutar:  streamlit run gestion_clientes.py
"""

import hmac
import os
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Estudio contable · Clientes", page_icon="🗂️", layout="wide")

from estudio.config import Config, aplicar_secrets_streamlit  # noqa: E402

aplicar_secrets_streamlit()

from estudio import calendario, cobranza, demo, intercambio  # noqa: E402
from estudio import rut as rutlib  # noqa: E402
from estudio.db import (  # noqa: E402
    ESTADOS_CERRADOS,
    ESTADOS_CLIENTE,
    ESTADOS_OBLIGACION,
    MEDIOS_PAGO,
    MONEDAS,
    REGIMENES,
    TIPOS_CLIENTE,
    TIPOS_INTERACCION,
    BaseDatos,
    ErrorValidacion,
    estado_cobro,
)

try:
    OSCURO = st.context.theme.type == "dark"
except Exception:
    OSCURO = False
AZUL = "#3987e5" if OSCURO else "#2a78d6"
NARANJA = "#d95926" if OSCURO else "#eb6834"
TINTA = "#c3c2b7" if OSCURO else "#52514e"
ICONO_ESTADO_CLIENTE = {"Activo": "🟢", "Prospecto": "🔵", "Suspendido": "🟠", "De baja": "⚪"}
FECHA = "DD/MM/YYYY"

cfg = Config.desde_entorno()

# ---------------------------------------------------------------------------
# Acceso
# ---------------------------------------------------------------------------
if cfg.clave and not st.session_state.get("autenticado"):
    st.title(f"🗂️ {cfg.estudio_nombre}")
    with st.form("form-acceso"):
        clave = st.text_input("Clave de acceso", type="password")
        if st.form_submit_button("Entrar", type="primary"):
            if hmac.compare_digest(clave.encode(), cfg.clave.encode()):
                st.session_state.autenticado = True
                st.rerun()
            st.error("Clave incorrecta.")
    st.stop()


@st.cache_resource
def abrir_db(ruta):
    return BaseDatos(ruta)


db = abrir_db(cfg.ruta_db)
hoy = cfg.hoy()

for clave, valor in (("avisos", []), ("ver", 0), ("cliente_id", None), ("cobro_id", None), ("ids_clientes", []), ("ids_cobros", [])):
    st.session_state.setdefault(clave, valor)


def avisar(texto, icono="✅"):
    """Deja un aviso para mostrarlo después del próximo refresco de la página."""
    st.session_state.avisos.append((texto, icono))


def nueva_version():
    """Reinicia formularios, tablas editables y selecciones (sus claves llevan el número de versión)."""
    st.session_state.ver += 1
    st.session_state.cobro_id = None


def refrescar():
    """Vuelve a dibujar la página con los datos recién guardados."""
    nueva_version()
    st.rerun()


for texto, icono in st.session_state.avisos:
    st.toast(texto, icon=icono)
st.session_state.avisos = []


def indice(opciones, valor, defecto=0):
    return opciones.index(valor) if valor in opciones else defecto


def a_fecha(texto):
    return date.fromisoformat(texto) if texto else None


def fecha_corta(texto):
    return a_fecha(texto[:10]).strftime("%d/%m/%Y") if texto else ""


def etiqueta_periodo(periodo):
    return calendario.nombre_periodo(periodo).capitalize()


def periodos_mensuales(cuantos=14):
    """Períodos para elegir: desde el mes actual hacia atrás."""
    return [
        "{:04d}-{:02d}".format(*calendario.mes_siguiente(hoy.year, hoy.month, -i)) for i in range(cuantos)
    ]


# ---------------------------------------------------------------------------
# Datos base
# ---------------------------------------------------------------------------
clientes = db.clientes()
activos = [c for c in clientes if c["estado"] == "Activo"]
equipo = db.get_json("equipo") or cfg.equipo
responsables = sorted(set(equipo) | {c["responsable"] for c in clientes if c["responsable"]})
uf = cobranza.valor_uf(db, cfg)
nombre_cliente = {c["id"]: c["razon_social"] for c in clientes}


def selector_cliente(etiqueta, clave, solo=None, contenedor=st):
    lista = solo if solo is not None else clientes
    return contenedor.selectbox(
        etiqueta, [c["id"] for c in lista], format_func=lambda i: nombre_cliente.get(i, "—"), index=None,
        placeholder="Elige un cliente", key=clave,
    )


def selector_responsable(etiqueta, valor, clave, contenedor=st):
    opciones = [""] + sorted(set(responsables) | ({valor} if valor else set()))
    return contenedor.selectbox(
        etiqueta, opciones, index=indice(opciones, valor or ""), format_func=lambda r: r or "Sin asignar", key=clave
    )


def _exportar():
    """Arma el Excel al hacer clic. Corre en otro hilo, por eso abre su propia conexión."""
    otra = BaseDatos(cfg.ruta_db)
    try:
        return intercambio.exportar_excel(otra, hoy, cfg.dias_alerta)
    finally:
        otra.cerrar()


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🗂️ Estudio contable")
    st.caption(f"{cfg.estudio_nombre} · {hoy.strftime('%d/%m/%Y')}")

    yo = st.selectbox(
        "Ver el trabajo de", ["Todo el equipo"] + responsables,
        help="Filtra el panel y los vencimientos. Tus notas en la bitácora quedan firmadas con este nombre.",
    )
    yo = "" if yo == "Todo el equipo" else yo

    st.divider()
    st.download_button(
        "⬇️ Exportar todo a Excel",
        data=_exportar,
        file_name=f"estudio_{hoy:%Y%m%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
        disabled=not clientes,
        help="Clientes, vencimientos, honorarios y pagos. Sirve también como respaldo.",
    )
    with st.expander("🧪 Datos de demostración"):
        st.caption("Un estudio ficticio (RUT y correos inventados) para recorrer la plataforma.")
        d1, d2 = st.columns(2)
        if d1.button("Cargar demo", width="stretch"):
            creados = demo.cargar(db, cfg, hoy)
            avisar(f"{creados} clientes de demostración cargados" if creados else "La demo ya estaba cargada", "🧪")
            refrescar()
        if d2.button("Borrar demo", width="stretch"):
            if st.session_state.cliente_id and (db.cliente(st.session_state.cliente_id) or {}).get("fuente") == demo.FUENTE:
                st.session_state.cliente_id = None
            avisar(f"{demo.borrar(db)} clientes de demostración eliminados", "🧹")
            refrescar()
    if not cfg.clave:
        st.caption("🔓 Sin clave de acceso. Define `ESTUDIO_CLAVE` antes de publicar la app en internet.")

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("🗂️ Gestión de clientes")
if not clientes:
    st.info(
        "Todavía no hay clientes. Créalos en la pestaña **👥 Clientes**, impórtalos desde una planilla en "
        "**⚙️ Configuración** o carga los **datos de demostración** desde la barra lateral."
    )
elif any(c["fuente"] == demo.FUENTE for c in clientes):
    st.warning("Estás viendo datos de demostración (ficticios). Bórralos desde la barra lateral cuando cargues tus clientes.")

tab_panel, tab_clientes, tab_venc, tab_hon, tab_bit, tab_conf = st.tabs(
    ["📊 Panel", "👥 Clientes", "📅 Vencimientos", "💰 Honorarios", "🗒️ Bitácora", "⚙️ Configuración"]
)


# ---------------------------------------------------------------------------
# Piezas compartidas
# ---------------------------------------------------------------------------
def tabla_obligaciones(filas):
    """Obligaciones como DataFrame listo para mostrar o editar."""
    return pd.DataFrame(
        [
            {
                "Situación": f"{calendario.ICONO_SEMAFORO[s]} " + ("Cerrada" if s == "Cerrada" else calendario.texto_plazo(o["vence"], hoy).capitalize()),
                "Vence": a_fecha(o["vence"]),
                "Cliente": o["razon_social"],
                "Obligación": o["nombre"],
                "Período": etiqueta_periodo(o["periodo"]) if o["periodo"] else "",
                "Estado": o["estado"],
                "Responsable": o["responsable"] or None,
                "Notas": o["notas"],
            }
            for o in filas
            for s in [calendario.semaforo(o, hoy, cfg.dias_alerta)]
        ],
        columns=["Situación", "Vence", "Cliente", "Obligación", "Período", "Estado", "Responsable", "Notas"],
    )


def _guardar_obligaciones(clave, ids):
    """Callback del editor: guarda en la base las celdas modificadas."""
    campos_por_columna = {"Estado": "estado", "Vence": "vence", "Responsable": "responsable", "Notas": "notas"}
    for fila, valores in st.session_state[clave]["edited_rows"].items():
        campos = {campos_por_columna[c]: v for c, v in valores.items() if c in campos_por_columna}
        try:
            db.actualizar_obligacion(ids[int(fila)], **campos)
        except ErrorValidacion as e:
            avisar(str(e), "⚠️")
    nueva_version()


def editor_obligaciones(filas, clave, ocultar=()):
    """Tabla editable de vencimientos: estado, fecha, responsable y notas se guardan al cambiar la celda."""
    clave = f"{clave}-{st.session_state.ver}"
    opciones_resp = sorted(set(responsables) | {o["responsable"] for o in filas if o["responsable"]})
    st.data_editor(
        tabla_obligaciones(filas).drop(columns=list(ocultar)),
        key=clave,
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        disabled=["Situación", "Cliente", "Obligación", "Período"],
        on_change=_guardar_obligaciones,
        args=(clave, [o["id"] for o in filas]),
        column_config={
            "Vence": st.column_config.DateColumn("Vence", format=FECHA, required=True),
            "Estado": st.column_config.SelectboxColumn("Estado", options=ESTADOS_OBLIGACION, required=True),
            "Responsable": st.column_config.SelectboxColumn("Responsable", options=opciones_resp),
            "Notas": st.column_config.TextColumn("Notas", width="large"),
        },
    )


def formulario_tarea(clave, cliente_id=None):
    """Tarea o vencimiento puntual fuera del calendario (trámites, requerimientos del SII, etc.)."""
    with st.form(f"{clave}-{st.session_state.ver}", border=False):
        c1, c2, c3, c4 = st.columns([2, 3, 2, 2])
        if cliente_id is None:
            cliente_id = selector_cliente("Cliente", f"{clave}-cliente", contenedor=c1)
        else:
            c1.text_input("Cliente", nombre_cliente.get(cliente_id, ""), disabled=True)
        nombre = c2.text_input("Tarea", placeholder="Ej.: responder requerimiento del SII")
        vence = c3.date_input("Vence", hoy + timedelta(days=7), format=FECHA)
        por_defecto = (db.cliente(cliente_id) or {}).get("responsable", "") if cliente_id else yo
        responsable = selector_responsable("Responsable", por_defecto, f"{clave}-resp-{cliente_id}-{yo}", contenedor=c4)
        if st.form_submit_button("Agregar tarea"):
            if not cliente_id:
                st.error("Elige un cliente.")
            else:
                try:
                    db.crear_obligacion(cliente_id, "OTRO", nombre, vence, responsable=responsable)
                    avisar("Tarea agregada")
                    refrescar()
                except ErrorValidacion as e:
                    st.error(str(e))


def botones_mensaje(cliente, asunto, cuerpo, clave, tipo="Correo"):
    """Botones para enviar un mensaje ya redactado por correo o WhatsApp."""
    b1, b2 = st.columns(2)
    if cliente.get("email"):
        b1.link_button("✉️ Abrir en mi correo", cobranza.link_mailto(cliente["email"], asunto, cuerpo), width="stretch")
    else:
        b1.caption("Sin correo registrado.")
    wa = cobranza.link_whatsapp(cliente.get("telefono"), cuerpo)
    if wa:
        b2.link_button("💬 Enviar por WhatsApp", wa, width="stretch")
    else:
        b2.caption("Sin celular válido para WhatsApp.")
    with st.expander("Ver el mensaje"):
        st.text(f"Asunto: {asunto}\n\n{cuerpo}")
    if st.button("Anotar en la bitácora que se envió", key=f"{clave}-{st.session_state.ver}"):
        db.registrar_interaccion(cliente["id"], f"Se envió: {asunto.split(' · ')[0].lower()}.", tipo, yo)
        avisar("Anotado en la bitácora")
        refrescar()


def barras_horizontales(df, categoria, valor, orden=None, etiqueta=None):
    alto = max(120, 34 * len(df))
    base = alt.Chart(df).encode(
        y=alt.Y(f"{categoria}:N", sort=orden or "-x", title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X(f"{valor}:Q", title=None, axis=None),
        tooltip=[alt.Tooltip(f"{categoria}:N"), alt.Tooltip(f"{etiqueta or valor}:{'N' if etiqueta else 'Q'}", title=valor)],
    )
    barras = base.mark_bar(color=AZUL, cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=18)
    etiquetas = base.mark_text(align="left", dx=6, color=TINTA).encode(text=f"{etiqueta or valor}:{'N' if etiqueta else 'Q'}")
    return (barras + etiquetas).properties(height=alto)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
with tab_panel:
    abiertas = [o for o in db.obligaciones(abiertas=True) if not yo or o["responsable"] == yo]
    situacion = [calendario.semaforo(o, hoy, cfg.dias_alerta) for o in abiertas]
    res = cobranza.resumen(db, hoy)

    k = st.columns(6)
    k[0].metric("Clientes activos", len([c for c in activos if not yo or c["responsable"] == yo]))
    k[1].metric("🔴 Atrasados", situacion.count("Atrasada"), help="Vencimientos abiertos cuya fecha ya pasó.")
    k[2].metric(f"🟡 Vencen en {cfg.dias_alerta} días", situacion.count("Próxima") + situacion.count("Vence hoy"))
    k[3].metric("📨 Esperando al cliente", sum(o["estado"] == "Esperando cliente" for o in abiertas))
    k[4].metric("Por cobrar", cobranza.formato_corto(res["por_cobrar"]), help=cobranza.formato_clp(res["por_cobrar"]))
    k[5].metric(
        "Honorarios vencidos", cobranza.formato_corto(res["vencido"]),
        help=f"{cobranza.formato_clp(res['vencido'])} en {res['clientes_morosos']} clientes con al menos un cobro vencido.",
    )

    st.subheader("Lo que viene" + (f" · {yo}" if yo else ""))
    limite = (hoy + timedelta(days=14)).isoformat()
    urgentes = [o for o in abiertas if o["vence"] <= limite]
    if urgentes:
        st.caption("Vencimientos atrasados y de los próximos 14 días. Se gestionan en la pestaña 📅 Vencimientos.")
        st.dataframe(
            tabla_obligaciones(urgentes).drop(columns=["Notas"]),
            hide_index=True, width="stretch", height=min(390, 38 + 35 * len(urgentes)),
            column_config={"Vence": st.column_config.DateColumn("Vence", format=FECHA)},
        )
    elif abiertas:
        st.success("Nada atrasado ni por vencer en los próximos 14 días.")
    else:
        st.caption("No hay vencimientos abiertos. Genera los del mes en la pestaña 📅 Vencimientos.")

    g1, g2, g3 = st.columns([2, 2, 3])
    with g1:
        st.markdown("**Vencimientos abiertos por responsable**")
        if abiertas:
            carga = (
                pd.Series([o["responsable"] or "Sin asignar" for o in db.obligaciones(abiertas=True)])
                .value_counts().rename_axis("Responsable").reset_index(name="Abiertos")
            )
            st.altair_chart(barras_horizontales(carga, "Responsable", "Abiertos"), width="stretch")
        else:
            st.caption("Sin vencimientos abiertos.")
    with g2:
        st.markdown("**Honorarios por cobrar según antigüedad**")
        tramos = cobranza.por_tramo(cobranza.cartera(db, hoy))
        if res["por_cobrar"]:
            df_tramos = pd.DataFrame(
                [{"Tramo": t, "Saldo": m, "Monto": cobranza.formato_clp(m)} for t, m in tramos.items() if m]
            )
            st.altair_chart(
                barras_horizontales(df_tramos, "Tramo", "Saldo", orden=cobranza.TRAMOS_MORA, etiqueta="Monto"),
                width="stretch",
            )
        else:
            st.caption("No hay honorarios pendientes de pago.")
    with g3:
        st.markdown("**Honorarios emitidos y cobrados por mes**")
        serie = cobranza.serie_mensual(db, hoy)
        if any(m["emitido"] or m["cobrado"] for m in serie):
            df_serie = pd.DataFrame(
                [
                    {
                        "Mes": f"{calendario.MESES[int(m['mes'][5:]) - 1][:3]} {m['mes'][2:4]}", "orden": m["mes"],
                        "Serie": serie_nombre, "Millones": m[campo] / 1e6, "Monto": cobranza.formato_clp(m[campo]),
                    }
                    for m in serie
                    for serie_nombre, campo in (("Emitido", "emitido"), ("Cobrado", "cobrado"))
                ]
            )
            st.altair_chart(
                alt.Chart(df_serie)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Mes:N", sort=alt.SortField("orden"), title=None, axis=alt.Axis(labelAngle=0), scale=alt.Scale(paddingInner=0.35)),
                    xOffset=alt.XOffset("Serie:N", sort=["Emitido", "Cobrado"], scale=alt.Scale(paddingInner=0.1)),
                    y=alt.Y("Millones:Q", title="Millones de $", axis=alt.Axis(tickCount=5, labelExpr="replace(format(datum.value, '~g'), '.', ',')")),
                    color=alt.Color(
                        "Serie:N", sort=["Emitido", "Cobrado"], scale=alt.Scale(domain=["Emitido", "Cobrado"], range=[AZUL, NARANJA]),
                        legend=alt.Legend(orient="top", title=None),
                    ),
                    tooltip=["Mes:N", "Serie:N", "Monto:N"],
                )
                .properties(height=230),
                width="stretch",
            )
            st.caption("Emitido según la fecha de emisión del cobro; cobrado según la fecha de cada pago.")
        else:
            st.caption("Aún no hay honorarios emitidos. Emítelos en la pestaña 💰 Honorarios.")


# ---------------------------------------------------------------------------
# Clientes
# ---------------------------------------------------------------------------
def formulario_cliente(cliente=None):
    c = cliente or {}
    k = f"cli-{c.get('id', 'nuevo')}-{st.session_state.ver}"
    with st.form(k, border=False):
        a1, a2, a3 = st.columns([1, 2, 2])
        rut = a1.text_input("RUT", rutlib.formatear(c["rut"]) if c.get("rut") else "", placeholder="76.123.456-7", key=f"{k}-rut")
        razon = a2.text_input("Razón social o nombre *", c.get("razon_social", ""), key=f"{k}-razon")
        fantasia = a3.text_input("Nombre de fantasía", c.get("nombre_fantasia", ""), key=f"{k}-fantasia")

        b1, b2, b3 = st.columns([1, 2, 2])
        tipo = b1.selectbox("Tipo", TIPOS_CLIENTE, index=indice(TIPOS_CLIENTE, c.get("tipo")), key=f"{k}-tipo")
        regimenes = [""] + REGIMENES
        regimen = b2.selectbox(
            "Régimen tributario", regimenes, index=indice(regimenes, c.get("regimen")),
            format_func=lambda r: r or "Sin definir", key=f"{k}-regimen",
        )
        giro = b3.text_input("Giro", c.get("giro", ""), key=f"{k}-giro")

        c1, c2, c3 = st.columns([1, 2, 2])
        estado = c1.selectbox(
            "Estado", ESTADOS_CLIENTE, index=indice(ESTADOS_CLIENTE, c.get("estado")),
            format_func=lambda e: f"{ICONO_ESTADO_CLIENTE[e]} {e}", key=f"{k}-estado",
        )
        responsable = selector_responsable("Responsable en el estudio", c.get("responsable", yo), f"{k}-resp", contenedor=c2)
        inicio = c3.date_input("Cliente desde", a_fecha(c.get("fecha_inicio")), format=FECHA, key=f"{k}-inicio")

        d1, d2, d3 = st.columns([2, 2, 1])
        contacto = d1.text_input("Persona de contacto", c.get("contacto", ""), key=f"{k}-contacto")
        email = d2.text_input("Correo", c.get("email", ""), key=f"{k}-email")
        telefono = d3.text_input("Teléfono", c.get("telefono", ""), placeholder="+56 9 …", key=f"{k}-fono")

        e1, e2, e3 = st.columns([2, 2, 1])
        representante = e1.text_input("Representante legal", c.get("representante", ""), key=f"{k}-rep")
        direccion = e2.text_input("Dirección", c.get("direccion", ""), key=f"{k}-dir")
        comuna = e3.text_input("Comuna", c.get("comuna", ""), key=f"{k}-comuna")

        servicios = st.multiselect(
            "Servicios contratados", calendario.SERVICIOS,
            default=[s for s in c.get("servicios", []) if s in calendario.SERVICIOS], key=f"{k}-servicios",
            help="De aquí salen los vencimientos que se generan cada mes y cada año para este cliente.",
        )
        f1, f2, f3 = st.columns(3)
        honorario = f1.number_input(
            "Honorario mensual", min_value=0.0, value=float(c.get("honorario") or 0), step=1.0, format="%g", key=f"{k}-hon"
        )
        moneda = f2.selectbox("Moneda", MONEDAS, index=indice(MONEDAS, c.get("moneda")), key=f"{k}-moneda")
        trabajadores = f3.number_input("Trabajadores", min_value=0, value=int(c.get("trabajadores") or 0), step=1, key=f"{k}-trab")
        notas = st.text_area("Notas", c.get("notas", ""), key=f"{k}-notas")
        enviado = st.form_submit_button("Guardar cambios" if cliente else "Crear cliente", type="primary")
    if enviado:
        campos = dict(
            rut=rut, razon_social=razon, nombre_fantasia=fantasia, tipo=tipo, regimen=regimen, giro=giro, estado=estado,
            responsable=responsable, fecha_inicio=inicio, contacto=contacto, email=email, telefono=telefono,
            representante=representante, direccion=direccion, comuna=comuna, servicios=servicios,
            honorario=honorario, moneda=moneda, trabajadores=trabajadores, notas=notas,
        )
        try:
            if cliente:
                db.actualizar_cliente(c["id"], **campos)
                avisar("Cliente actualizado")
            else:
                st.session_state.cliente_id = db.crear_cliente(**campos)
                avisar(f"Cliente creado: {razon}")
            refrescar()
        except ErrorValidacion as e:
            st.error(str(e))


def _abrir_cliente():
    filas = st.session_state[f"tabla-clientes-{st.session_state.ver}"].selection.rows
    if filas:
        st.session_state.cliente_id = st.session_state.ids_clientes[filas[0]]


def ficha_cliente(c):
    cabecera, cerrar = st.columns([6, 1])
    cabecera.subheader(f"{ICONO_ESTADO_CLIENTE.get(c['estado'], '')} {c['razon_social']}")
    cabecera.caption(
        " · ".join(
            x for x in (
                rutlib.formatear(c["rut"]) if c["rut"] else "Sin RUT", c["regimen"], c["giro"],
                f"Responsable: {c['responsable']}" if c["responsable"] else "Sin responsable",
                f"Honorario: {cobranza.formato_honorario(c)}",
            ) if x
        )
    )
    if cerrar.button("✖ Cerrar ficha", width="stretch"):
        st.session_state.cliente_id = None
        refrescar()

    obligaciones = db.obligaciones(cliente_id=c["id"])
    abiertas_c = [o for o in obligaciones if o["estado"] not in ESTADOS_CERRADOS]
    pendientes = cobranza.cartera(db, hoy, c["id"])
    t_venc, t_hon, t_bit, t_datos = st.tabs(
        [f"📅 Vencimientos ({len(abiertas_c)})", f"💰 Honorarios ({len(pendientes)})", "🗒️ Bitácora", "📋 Datos"]
    )

    with t_venc:
        ver_cerradas = st.toggle("Mostrar también los ya presentados", key=f"cerradas-{c['id']}")
        visibles = obligaciones if ver_cerradas else abiertas_c
        if visibles:
            editor_obligaciones(visibles, f"obl-cliente-{c['id']}", ocultar=["Cliente"])
        else:
            st.caption("Sin vencimientos abiertos para este cliente.")
        esperando = [o for o in abiertas_c if o["estado"] == "Esperando cliente"]
        if esperando:
            st.markdown(f"**📨 Pedir la información pendiente** ({len(esperando)} en espera)")
            botones_mensaje(c, *cobranza.mensaje_documentos(c, esperando, cfg), clave=f"msg-doc-{c['id']}")
        with st.expander("➕ Agregar una tarea para este cliente"):
            formulario_tarea(f"tarea-cliente-{c['id']}", c["id"])

    with t_hon:
        todos = db.cobros(cliente_id=c["id"], incluir_anulados=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Saldo pendiente", cobranza.formato_clp(sum(p["saldo"] for p in pendientes)))
        m2.metric("Vencido", cobranza.formato_clp(sum(p["saldo"] for p in pendientes if p["dias_mora"] > 0)))
        m3.metric("Pagado históricamente", cobranza.formato_clp(sum(x["pagado"] for x in todos)))
        if todos:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Concepto": x["concepto"], "Período": etiqueta_periodo(x["periodo"]) if x["periodo"] else "",
                            "Vence": a_fecha(x["vence"]), "Monto": x["monto"], "Pagado": x["pagado"],
                            "Saldo": 0 if x["anulado"] else x["saldo"], "Estado": estado_cobro(x),
                        }
                        for x in reversed(todos)
                    ]
                ),
                hide_index=True, width="stretch",
                column_config={
                    "Vence": st.column_config.DateColumn(format=FECHA),
                    **{col: st.column_config.NumberColumn(f"{col} ($)", format="localized") for col in ("Monto", "Pagado", "Saldo")},
                },
            )
            st.caption("Los pagos se registran en la pestaña 💰 Honorarios.")
        else:
            st.caption("Aún no se le han emitido cobros.")
        if pendientes:
            st.markdown("**🔔 Recordatorio de pago**")
            botones_mensaje(c, *cobranza.mensaje_cobranza(c, pendientes, cfg), clave=f"msg-cob-{c['id']}", tipo="Cobranza")

    with t_bit:
        with st.form(f"nota-{c['id']}-{st.session_state.ver}", border=False):
            n1, n2 = st.columns([1, 4])
            tipo = n1.selectbox("Tipo", TIPOS_INTERACCION)
            resumen = n2.text_input("¿Qué pasó?", placeholder="Ej.: llamó para pedir certificado de renta")
            if st.form_submit_button("Anotar"):
                try:
                    db.registrar_interaccion(c["id"], resumen, tipo, yo)
                    refrescar()
                except ErrorValidacion as e:
                    st.error(str(e))
        notas = db.interacciones(cliente_id=c["id"])
        for n in notas:
            st.markdown(f"**{fecha_corta(n['fecha'])} · {n['tipo']}**" + (f" · {n['autor']}" if n["autor"] else "") + f"  \n{n['resumen']}")
        if not notas:
            st.caption("Sin anotaciones todavía.")

    with t_datos:
        formulario_cliente(c)
        with st.expander("🗑️ Eliminar cliente"):
            st.caption(
                "Borra al cliente con todos sus vencimientos, cobros, pagos y notas. No se puede deshacer. "
                "Si solo dejaste de atenderlo, cambia su estado a «De baja» para conservar el historial."
            )
            seguro = st.checkbox(f"Sí, eliminar a {c['razon_social']} y todo su historial", key=f"seguro-{c['id']}")
            if st.button("Eliminar definitivamente", disabled=not seguro, key=f"eliminar-{c['id']}"):
                db.eliminar_cliente(c["id"])
                st.session_state.cliente_id = None
                avisar(f"{c['razon_social']} eliminado", "🗑️")
                refrescar()


with tab_clientes:
    f1, f2, f3 = st.columns([3, 3, 2])
    buscar = f1.text_input("Buscar", placeholder="Razón social, RUT, contacto o giro")
    estados = f2.multiselect("Estado", ESTADOS_CLIENTE, default=["Activo", "Prospecto"])
    resp_filtro = f3.selectbox("Responsable", ["Todos"] + responsables, index=indice(["Todos"] + responsables, yo))

    saldos = {}
    for p in cobranza.cartera(db, hoy):
        saldos[p["cliente_id"]] = saldos.get(p["cliente_id"], 0) + p["saldo"]
    abiertas_por_cliente = {}
    for o in db.obligaciones(abiertas=True):
        abiertas_por_cliente[o["cliente_id"]] = abiertas_por_cliente.get(o["cliente_id"], 0) + 1

    aguja = "".join(ch for ch in buscar.lower() if ch not in ".")
    vista = [
        c for c in clientes
        if c["estado"] in estados
        and (resp_filtro == "Todos" or c["responsable"] == resp_filtro)
        and (
            not aguja
            or aguja in " ".join(str(c[x] or "") for x in ("razon_social", "nombre_fantasia", "rut", "contacto", "giro", "email")).lower()
        )
    ]
    st.session_state.ids_clientes = [c["id"] for c in vista]
    st.caption(f"{len(vista)} clientes · haz clic en una fila para abrir su ficha")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Cliente": c["razon_social"], "RUT": rutlib.formatear(c["rut"]) if c["rut"] else "",
                    "Estado": f"{ICONO_ESTADO_CLIENTE[c['estado']]} {c['estado']}", "Responsable": c["responsable"],
                    "Contacto": c["contacto"], "Honorario": cobranza.formato_honorario(c),
                    "Abiertos": abiertas_por_cliente.get(c["id"], 0), "Saldo": saldos.get(c["id"], 0),
                }
                for c in vista
            ],
            columns=["Cliente", "RUT", "Estado", "Responsable", "Contacto", "Honorario", "Abiertos", "Saldo"],
        ),
        key=f"tabla-clientes-{st.session_state.ver}",
        hide_index=True, width="stretch", on_select=_abrir_cliente, selection_mode="single-row",
        height=min(458, 38 + 35 * max(1, len(vista))),
        column_config={
            "Abiertos": st.column_config.NumberColumn("Venc. abiertos", help="Vencimientos y tareas sin cerrar."),
            "Saldo": st.column_config.NumberColumn("Por cobrar ($)", format="localized"),
        },
    )
    with st.expander("➕ Nuevo cliente", expanded=not clientes):
        formulario_cliente()

    seleccionado = db.cliente(st.session_state.cliente_id) if st.session_state.cliente_id else None
    if seleccionado:
        st.divider()
        ficha_cliente(seleccionado)

# ---------------------------------------------------------------------------
# Vencimientos
# ---------------------------------------------------------------------------
with tab_venc:
    periodo_actual = calendario.periodo_anterior(hoy)
    con_servicios = [c for c in activos if calendario.tipos_para(c)]
    if con_servicios and not db.obligaciones(periodo=periodo_actual):
        st.info(
            f"Aún no generas los vencimientos de **{etiqueta_periodo(periodo_actual)}**, que se declaran este mes "
            f"para {len(con_servicios)} clientes."
        )
        if st.button(f"Generar vencimientos de {calendario.nombre_periodo(periodo_actual)}", type="primary"):
            avisar(f"{calendario.generar(db, cfg, periodo_actual)} vencimientos creados", "📅")
            refrescar()

    todas = db.obligaciones()
    v1, v2, v3, v4 = st.columns([3, 2, 2, 3])
    ver = v1.radio("Mostrar", ["Abiertos", "Cerrados", "Todos"], horizontal=True)
    resp_v = v2.selectbox("Responsable", ["Todos"] + responsables, index=indice(["Todos"] + responsables, yo), key=f"resp-venc-{yo}")
    periodos = sorted({o["periodo"] for o in todas if o["periodo"]}, reverse=True)
    periodo_v = v3.selectbox("Período", ["Todos"] + periodos, format_func=lambda p: p if p == "Todos" else etiqueta_periodo(p))
    buscar_v = v4.text_input("Buscar", placeholder="Cliente u obligación", key="buscar-venc")

    filtradas = [
        o for o in todas
        if (ver == "Todos" or (o["estado"] in ESTADOS_CERRADOS) == (ver == "Cerrados"))
        and (resp_v == "Todos" or o["responsable"] == resp_v)
        and (periodo_v == "Todos" or o["periodo"] == periodo_v)
        and (not buscar_v or buscar_v.lower() in f"{o['razon_social']} {o['nombre']}".lower())
    ]
    if ver == "Cerrados":
        filtradas.reverse()
    conteo = pd.Series([calendario.semaforo(o, hoy, cfg.dias_alerta) for o in filtradas]).value_counts()
    plural = {"Atrasada": "atrasados", "Vence hoy": "vencen hoy", "Próxima": "próximos", "Al día": "al día", "Cerrada": "cerrados"}
    st.caption(
        f"{len(filtradas)} vencimientos"
        + "".join(f" · {calendario.ICONO_SEMAFORO[s]} {conteo[s]} {plural[s]}" for s in calendario.SEMAFOROS if s in conteo)
        + ". Cambia el estado, la fecha, el responsable o las notas directamente en la tabla."
    )
    if filtradas:
        editor_obligaciones(filtradas, "obl-general")
    elif todas:
        st.success("No hay vencimientos con estos filtros.")

    g1, g2 = st.columns(2)
    with g1.expander("📅 Generar vencimientos de un período"):
        st.caption(
            "Crea, para cada cliente activo, las obligaciones que corresponden a sus servicios contratados. "
            "Se puede repetir sin riesgo: no duplica las que ya existen."
        )
        p1, p2 = st.columns([3, 2], vertical_alignment="bottom")
        per_m = p1.selectbox("Período mensual", periodos_mensuales(), index=1, format_func=etiqueta_periodo)
        if p2.button("Generar mes", width="stretch"):
            avisar(f"{calendario.generar(db, cfg, per_m)} vencimientos creados para {calendario.nombre_periodo(per_m)}", "📅")
            refrescar()
        a1, a2 = st.columns([3, 2], vertical_alignment="bottom")
        anios = [str(hoy.year - i) for i in range(3)]
        per_a = a1.selectbox(
            "Año comercial (renta, declaraciones juradas, patente)", anios, index=1,
            help="El año comercial 2025 se declara en la Operación Renta 2026.",
        )
        if a2.button("Generar año", width="stretch"):
            avisar(f"{calendario.generar(db, cfg, per_a)} vencimientos anuales creados para el año comercial {per_a}", "📅")
            refrescar()
    with g2.expander("➕ Agregar una tarea puntual"):
        formulario_tarea("tarea-general")

# ---------------------------------------------------------------------------
# Honorarios
# ---------------------------------------------------------------------------
def _elegir_cobro():
    filas = st.session_state[f"tabla-cobros-{st.session_state.ver}"].selection.rows
    st.session_state.cobro_id = st.session_state.ids_cobros[filas[0]] if filas else None


with tab_hon:
    res = cobranza.resumen(db, hoy)
    k = st.columns(4)
    k[0].metric("Por cobrar", cobranza.formato_clp(res["por_cobrar"]))
    k[1].metric("Vencido", cobranza.formato_clp(res["vencido"]))
    k[2].metric("Clientes con mora", res["clientes_morosos"])
    k[3].metric(f"Cobrado en {calendario.MESES[hoy.month - 1]}", cobranza.formato_clp(res["cobrado_mes"]))

    with st.expander("🧾 Emitir los honorarios del mes", expanded=not db.cobros()):
        h1, h2 = st.columns([2, 3], vertical_alignment="bottom")
        per_h = h1.selectbox("Mes a cobrar", periodos_mensuales(), index=1, format_func=etiqueta_periodo, key="per-hon")
        if h2.button(f"Emitir honorarios de {calendario.nombre_periodo(per_h)}", type="primary"):
            creados, avisos = cobranza.emitir_honorarios(db, cfg, per_h, hoy)
            avisar(f"{creados} cobros emitidos" if creados else "No había cobros nuevos que emitir", "🧾")
            for a in avisos:
                avisar(a, "⚠️")
            refrescar()
        st.caption(
            "Crea un cobro por cada cliente activo con honorario mensual, con vencimiento el día "
            f"{cfg.honorarios_dia_vence} del mes siguiente. No duplica los ya emitidos. "
            + (f"Los honorarios en UF se convierten a {cobranza.formato_clp(uf)} por UF." if uf else "Define el valor de la UF en ⚙️ Configuración para los honorarios pactados en UF.")
        )

    st.subheader("Por cobrar")
    pendientes = cobranza.cartera(db, hoy)
    pendientes.sort(key=lambda p: (-p["dias_mora"], p["razon_social"]))
    st.session_state.ids_cobros = [p["id"] for p in pendientes]
    if pendientes:
        st.caption("Haz clic en un cobro para registrar su pago o enviar un recordatorio.")
        izq, der = st.columns([3, 2])
        with izq:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Mora": (
                                f"🔴 {p['dias_mora']} días" if p["dias_mora"] > 30
                                else f"🟠 {p['dias_mora']} días" if p["dias_mora"] > 0 else "🟢 Por vencer"
                            ),
                            "Cliente": p["razon_social"],
                            "Concepto": (
                                f"Honorario {calendario.nombre_periodo(p['periodo'])}"
                                if p["concepto"] == cobranza.CONCEPTO_HONORARIO and p["periodo"]
                                else p["concepto"] + (f" · {calendario.nombre_periodo(p['periodo'])}" if p["periodo"] else "")
                            ),
                            "Saldo": p["saldo"], "Vence": a_fecha(p["vence"]),
                        }
                        for p in pendientes
                    ]
                ),
                key=f"tabla-cobros-{st.session_state.ver}",
                hide_index=True, width="stretch", on_select=_elegir_cobro, selection_mode="single-row",
                column_config={
                    "Vence": st.column_config.DateColumn(format=FECHA),
                    "Saldo": st.column_config.NumberColumn("Saldo ($)", format="localized"),
                },
            )
        with der:
            cobro = db.cobro(st.session_state.cobro_id) if st.session_state.cobro_id else None
            if not cobro or cobro["saldo"] <= 0 or cobro["anulado"]:
                st.info("Selecciona un cobro de la tabla.")
            else:
                cliente = db.cliente(cobro["cliente_id"])
                st.markdown(
                    f"**{cobro['razon_social']}**  \n{cobro['concepto']}"
                    + (f" · {calendario.nombre_periodo(cobro['periodo'])}" if cobro["periodo"] else "")
                    + f" · saldo {cobranza.formato_clp(cobro['saldo'])}"
                    + (f" (abonado {cobranza.formato_clp(cobro['pagado'])})" if cobro["pagado"] else "")
                )
                with st.form(f"pago-{cobro['id']}-{st.session_state.ver}"):
                    q1, q2 = st.columns(2)
                    monto = q1.number_input("Monto pagado ($)", min_value=1, max_value=int(cobro["saldo"]), value=int(cobro["saldo"]), step=1000)
                    fecha_pago = q2.date_input("Fecha del pago", hoy, max_value=hoy, format=FECHA)
                    q3, q4 = st.columns(2)
                    medio = q3.selectbox("Medio", MEDIOS_PAGO)
                    nota = q4.text_input("Nota", placeholder="N° de operación, boleta…")
                    if st.form_submit_button("Registrar pago", type="primary"):
                        try:
                            db.registrar_pago(cobro["id"], monto, fecha_pago, medio, nota)
                            avisar(f"Pago de {cobranza.formato_clp(monto)} registrado para {cobro['razon_social']}", "💰")
                            refrescar()
                        except ErrorValidacion as e:
                            st.error(str(e))
                del_cliente = [p for p in pendientes if p["cliente_id"] == cobro["cliente_id"]]
                st.markdown(f"**🔔 Recordatorio** por sus {len(del_cliente)} cobros pendientes" if len(del_cliente) > 1 else "**🔔 Recordatorio de pago**")
                botones_mensaje(cliente, *cobranza.mensaje_cobranza(cliente, del_cliente, cfg), clave=f"msg-cobro-{cobro['id']}", tipo="Cobranza")
                if not cobro["pagado"]:
                    with st.popover("Anular este cobro"):
                        st.caption("Úsalo si se emitió por error. El cobro deja de contar como deuda.")
                        if st.button("Anular", key=f"anular-{cobro['id']}"):
                            db.actualizar_cobro(cobro["id"], anulado=True)
                            avisar("Cobro anulado", "🚫")
                            refrescar()
    else:
        st.success("No hay honorarios pendientes de pago." if db.cobros() else "Aún no hay cobros emitidos.")

    x1, x2 = st.columns(2)
    with x1.expander("➕ Cobro extraordinario"):
        st.caption("Trabajos fuera del honorario mensual: constitución de sociedad, regularizaciones, certificados…")
        with st.form(f"cobro-extra-{st.session_state.ver}", border=False):
            cliente_extra = selector_cliente("Cliente", "extra-cliente")
            concepto = st.text_input("Concepto", placeholder="Ej.: Constitución de sociedad")
            y1, y2 = st.columns(2)
            monto_extra = y1.number_input("Monto ($)", min_value=0, step=10000)
            vence_extra = y2.date_input("Vence", hoy + timedelta(days=10), format=FECHA)
            if st.form_submit_button("Crear cobro"):
                if not cliente_extra:
                    st.error("Elige un cliente.")
                else:
                    try:
                        db.crear_cobro(cliente_extra, concepto, monto_extra, hoy, vence_extra)
                        avisar("Cobro creado", "🧾")
                        refrescar()
                    except ErrorValidacion as e:
                        st.error(str(e))
    with x2.expander("↩️ Pagos recibidos"):
        pagos = db.pagos()[:200]
        if pagos:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Fecha": a_fecha(p["fecha"]), "Cliente": p["razon_social"],
                            "Concepto": p["concepto"] + (f" · {p['periodo']}" if p["periodo"] else ""),
                            "Monto": p["monto"], "Medio": p["medio"], "Nota": p["nota"],
                        }
                        for p in pagos
                    ]
                ),
                hide_index=True, width="stretch", height=240,
                column_config={
                    "Fecha": st.column_config.DateColumn(format=FECHA),
                    "Monto": st.column_config.NumberColumn("Monto ($)", format="localized"),
                },
            )
            borrar = st.selectbox(
                "¿Registraste un pago por error?", [p["id"] for p in pagos[:50]], index=None, placeholder="Elige el pago a eliminar",
                format_func=lambda i: next(
                    f"{fecha_corta(p['fecha'])} · {p['razon_social']} · {cobranza.formato_clp(p['monto'])}" for p in pagos if p["id"] == i
                ),
            )
            if borrar and st.button("Eliminar ese pago"):
                db.eliminar_pago(borrar)
                avisar("Pago eliminado: el cobro vuelve a quedar pendiente", "↩️")
                refrescar()
        else:
            st.caption("Aún no se registran pagos.")

# ---------------------------------------------------------------------------
# Bitácora
# ---------------------------------------------------------------------------
with tab_bit:
    st.caption("Llamadas, correos, reuniones y acuerdos con cada cliente, para que todo el equipo sepa en qué va.")
    with st.form(f"nota-general-{st.session_state.ver}", border=False):
        n1, n2, n3 = st.columns([2, 1, 4])
        cliente_nota = selector_cliente("Cliente", "nota-cliente", contenedor=n1)
        tipo_nota = n2.selectbox("Tipo", TIPOS_INTERACCION, key="nota-tipo")
        resumen_nota = n3.text_input("¿Qué pasó?", placeholder="Ej.: pidió certificado de renta para el banco")
        if st.form_submit_button("Anotar"):
            if not cliente_nota:
                st.error("Elige un cliente.")
            else:
                try:
                    db.registrar_interaccion(cliente_nota, resumen_nota, tipo_nota, yo)
                    avisar("Anotado")
                    refrescar()
                except ErrorValidacion as e:
                    st.error(str(e))
    notas = db.interacciones()
    if notas:
        st.dataframe(
            pd.DataFrame(
                [
                    {"Fecha": a_fecha(n["fecha"][:10]), "Cliente": n["razon_social"], "Tipo": n["tipo"], "Resumen": n["resumen"], "Autor": n["autor"]}
                    for n in notas
                ]
            ),
            hide_index=True, width="stretch",
            column_config={"Fecha": st.column_config.DateColumn(format=FECHA), "Resumen": st.column_config.TextColumn(width="large")},
        )
    else:
        st.caption("Sin anotaciones todavía.")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
with tab_conf:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("👩‍💼 Equipo del estudio")
        with st.form("equipo", border=False):
            texto_equipo = st.text_input(
                "Nombres separados por coma", ", ".join(equipo), placeholder="Carolina, Matías, Fernanda",
                help="Aparecen como responsables de clientes y vencimientos.",
            )
            if st.form_submit_button("Guardar equipo"):
                db.set_json("equipo", [n.strip() for n in texto_equipo.split(",") if n.strip()])
                avisar("Equipo actualizado")
                refrescar()
    with c2:
        st.subheader("💱 Valor de la UF")
        with st.form("uf", border=False):
            nueva_uf = st.number_input(
                "Pesos por UF", min_value=0.0, value=float(uf), step=10.0, format="%.2f",
                help="Se usa al emitir los honorarios pactados en UF. Actualízalo antes de emitir cada mes.",
            )
            if st.form_submit_button("Guardar UF"):
                db.set_meta(cobranza.META_UF, str(nueva_uf))
                avisar("Valor de la UF actualizado")
                refrescar()

    st.subheader("📅 Calendario de vencimientos")
    st.caption(
        "Fechas **referenciales**: confírmalas cada año con los calendarios del SII, la Dirección del Trabajo y Previred, "
        "y ajusta aquí el día (y el mes, en las anuales). Si la fecha cae en fin de semana o feriado se corre al día hábil "
        "siguiente. Los cambios aplican a los vencimientos que generes desde ahora; los ya creados se editan en su tabla."
    )
    tipos = calendario.catalogo(db.get_json(calendario.META_AJUSTES, {}))
    meses_cap = [m.capitalize() for m in calendario.MESES]
    MES_SIGUIENTE = "Mes siguiente"
    df_cat = pd.DataFrame(
        [
            {
                "Obligación": t.nombre, "Frecuencia": t.frecuencia.capitalize(), "Día": t.dia,
                "Mes": calendario.MESES[t.mes - 1].capitalize() if t.mes else MES_SIGUIENTE, "Servicio que la activa": t.servicio, "Nota": t.ayuda,
            }
            for t in tipos.values()
        ]
    )
    editado = st.data_editor(
        df_cat, hide_index=True, width="stretch", num_rows="fixed", key=f"catalogo-{st.session_state.ver}",
        disabled=["Obligación", "Frecuencia", "Servicio que la activa", "Nota"],
        column_config={
            "Día": st.column_config.NumberColumn(min_value=1, max_value=31, step=1, required=True),
            "Mes": st.column_config.SelectboxColumn(
                options=[MES_SIGUIENTE] + meses_cap, required=True,
                help="Las mensuales vencen siempre el mes siguiente al período; en las anuales elige el mes.",
            ),
            "Nota": st.column_config.TextColumn(width="large"),
        },
    )
    z1, z2, _ = st.columns([1, 1, 3])
    if z1.button("Guardar calendario", type="primary", width="stretch"):
        ajustes = {}
        for (codigo, base), (_, fila) in zip(calendario.CATALOGO.items(), editado.iterrows()):
            mes = meses_cap.index(fila["Mes"]) + 1 if base.frecuencia == "anual" and fila["Mes"] in meses_cap else base.mes
            if int(fila["Día"]) != base.dia or mes != base.mes:
                ajustes[codigo] = {"dia": int(fila["Día"]), "mes": mes}
        db.set_json(calendario.META_AJUSTES, ajustes)
        avisar("Calendario guardado")
        refrescar()
    if z2.button("Restablecer", width="stretch"):
        db.set_json(calendario.META_AJUSTES, {})
        avisar("Calendario restablecido a las fechas de referencia")
        refrescar()

    st.subheader("📂 Importar clientes")
    i1, i2 = st.columns([3, 2])
    with i1:
        archivo = st.file_uploader(
            "Planilla CSV o Excel con columnas rut, razon_social, contacto, email, servicios, honorario…", type=["csv", "xlsx"],
            key=f"importar-{st.session_state.ver}",
        )
        if archivo and st.button("Importar clientes", type="primary"):
            try:
                df_imp = (
                    pd.read_csv(archivo, dtype=str, sep=None, engine="python") if archivo.name.lower().endswith(".csv")
                    else pd.read_excel(archivo, dtype=str)
                )
                creados, existentes, errores = intercambio.importar_clientes(db, df_imp)
                st.session_state.resultado_importacion = (creados, existentes, errores)
                refrescar()
            except Exception as e:
                st.error(f"No se pudo importar: {e}")
        if st.session_state.get("resultado_importacion"):
            creados, existentes, errores = st.session_state.pop("resultado_importacion")
            st.success(f"{creados} clientes creados" + (f" · {existentes} ya existían y no se tocaron" if existentes else "") + ".")
            if errores:
                st.warning("Filas que no se pudieron importar:\n\n" + "\n".join(f"- {e}" for e in errores[:30]))
    with i2:
        st.download_button("⬇️ Descargar planilla de ejemplo", intercambio.plantilla_importacion(), "clientes_ejemplo.csv", "text/csv")
        st.caption("Los RUT que ya existen se omiten: importar dos veces no duplica ni sobrescribe.")
        if os.path.exists(cfg.ruta_db_ebook):
            if st.button("📘 Traer clientes ganados en el tablero del ebook"):
                n = intercambio.importar_leads_ebook(db, cfg.ruta_db_ebook)
                avisar(f"{n} prospectos creados desde el ebook" if n else "No hay leads nuevos en la etapa «Cliente»", "📘")
                refrescar()
            st.caption("Crea como prospectos los leads que marcaste en la etapa «Cliente» del tablero del ebook.")

    st.subheader("🔐 Datos y seguridad")
    st.markdown(
        f"- Los datos se guardan solo en `{cfg.ruta_db}` (ignorado por git). Respáldalo, o usa **Exportar todo a Excel**.\n"
        "- La plataforma **no guarda claves tributarias** ni de Previred, y no se conecta al SII: no las anotes en las notas.\n"
        f"- Acceso: {'protegido con clave (`ESTUDIO_CLAVE`).' if cfg.clave else 'sin clave. Define `ESTUDIO_CLAVE` si la app queda accesible desde internet.'}"
    )
