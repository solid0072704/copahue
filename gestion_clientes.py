"""Plataforma de gestión de clientes para estudios contables.

Ejecutar:  streamlit run gestion_clientes.py

La interfaz está pensada primero para el celular: navegación al alcance del pulgar,
listas de tarjetas que se tocan para abrir y formularios en ventanas propias.
"""

import hmac
import os
from datetime import date, datetime, timedelta

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Estudio contable", page_icon="🗂️", layout="wide", initial_sidebar_state="collapsed")

from estudio.config import Config, aplicar_secrets_streamlit  # noqa: E402

aplicar_secrets_streamlit()

from estudio import calendario, cobranza, demo, indicadores, intercambio, tema  # noqa: E402
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

FECHA = "DD/MM/YYYY"
POR_PAGINA = 30
TODO_EL_EQUIPO = "Todo el equipo"

cfg = Config.desde_entorno()
st.html(tema.CSS)

# ---------------------------------------------------------------------------
# Acceso
# ---------------------------------------------------------------------------
if cfg.clave and not st.session_state.get("autenticado"):
    _, centro, _ = st.columns([1, 1.2, 1])
    with centro:
        st.html(f'<div class="saludo" style="margin-top:12vh">{tema.e(cfg.estudio_nombre)}<small>Gestión de clientes</small></div>')
        with st.form("form-acceso", border=False):
            clave = st.text_input("Clave de acceso", type="password")
            if st.form_submit_button("Entrar", type="primary", width="stretch"):
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

for _clave, _valor in (
    ("avisos", []), ("ver", 0), ("pagina", "inicio"), ("cliente_id", None), ("yo", ""),
    ("filtro_venc", "Urgentes"), ("limite", POR_PAGINA),
):
    st.session_state.setdefault(_clave, _valor)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def avisar(texto, icono="✅"):
    """Deja un aviso para mostrarlo después del próximo refresco de la página."""
    st.session_state.avisos.append((texto, icono))


def refrescar():
    """Reinicia formularios y tablas editables, y vuelve a dibujar con los datos guardados."""
    st.session_state.ver += 1
    st.rerun()


def ir(pagina, cliente_id=None, filtro_venc=None):
    """Cambia de pantalla (se usa como callback de los botones de navegación)."""
    st.session_state.pagina = pagina
    st.session_state.limite = POR_PAGINA
    if cliente_id is not None:
        st.session_state.cliente_id = cliente_id
    if filtro_venc:
        st.session_state.filtro_venc = filtro_venc


def indice(opciones, valor, defecto=0):
    return opciones.index(valor) if valor in opciones else defecto


def a_fecha(texto):
    return date.fromisoformat(texto[:10]) if texto else None


def fecha_corta(texto):
    return a_fecha(texto).strftime("%d/%m/%Y") if texto else ""


def etiqueta_periodo(periodo):
    return calendario.nombre_periodo(periodo).capitalize()


def periodos_mensuales(cuantos=14):
    return ["{:04d}-{:02d}".format(*calendario.mes_siguiente(hoy.year, hoy.month, -i)) for i in range(cuantos)]


def plural(n, singular, plural_=None):
    return f"{n} {singular if n == 1 else (plural_ or singular + 's')}"


for _texto, _icono in st.session_state.avisos:
    st.toast(_texto, icon=_icono)
st.session_state.avisos = []

# La UF se actualiza sola una vez al día (si no hay conexión, se conserva el último valor).
if cfg.uf_automatica and indicadores.actualizar_uf(db, cfg, hoy) == "actualizada":
    st.toast(f"UF actualizada: {cobranza.formato_clp(cobranza.valor_uf(db, cfg))}", icon="💱")

# ---------------------------------------------------------------------------
# Datos base
# ---------------------------------------------------------------------------
clientes = db.clientes()
activos = [c for c in clientes if c["estado"] == "Activo"]
equipo = db.get_json("equipo") or cfg.equipo
responsables = sorted(set(equipo) | {c["responsable"] for c in clientes if c["responsable"]})
uf = cobranza.valor_uf(db, cfg)
nombre_cliente = {c["id"]: c["razon_social"] for c in clientes}
hay_demo = any(c["fuente"] == demo.FUENTE for c in clientes)
if st.session_state.yo not in responsables:
    st.session_state.yo = ""


def selector_cliente(etiqueta, clave, contenedor=st):
    return contenedor.selectbox(
        etiqueta, [c["id"] for c in clientes], format_func=lambda i: nombre_cliente.get(i, "—"), index=None,
        placeholder="Elige un cliente", key=clave,
    )


def selector_responsable(etiqueta, valor, clave, contenedor=st):
    opciones = [""] + sorted(set(responsables) | ({valor} if valor else set()))
    return contenedor.selectbox(
        etiqueta, opciones, index=indice(opciones, valor or ""), format_func=lambda r: r or "Sin asignar", key=clave
    )


def selector_persona(clave):
    """Filtro «de quién es el trabajo», compartido entre pantallas. Devuelve '' para todo el equipo."""
    if not responsables:
        return ""
    actual = st.session_state.yo or TODO_EL_EQUIPO
    elegido = st.pills(
        "Ver el trabajo de", [TODO_EL_EQUIPO] + responsables, default=actual, key=f"persona-{clave}",
        label_visibility="collapsed",
    )
    st.session_state.yo = "" if elegido in (None, TODO_EL_EQUIPO) else elegido
    return st.session_state.yo


def tarjeta(clave, contenido, abrir=None, acciones=()):
    """Tarjeta de lista. Devuelve 'abrir' si se tocó la tarjeta, el nombre de la acción pulsada, o None.

    `abrir` es el texto accesible del botón invisible que cubre la tarjeta; `acciones` son
    tuplas (nombre, texto, icono, tipo) de botones visibles a la derecha.
    """
    pulsado = None
    with st.container(key=f"fila-{clave}", horizontal=True, vertical_alignment="center"):
        st.html(contenido)
        for nombre, texto, icono, tipo in acciones:
            if st.button(texto, key=f"accion-{nombre}-{clave}", icon=icono, type=tipo):
                pulsado = nombre
        if abrir and st.button(abrir, key=f"abrir-{clave}"):
            pulsado = "abrir"
    return pulsado


def mostrar_mas(total):
    """Botón de paginación para listas largas."""
    if total > st.session_state.limite:
        faltan = total - st.session_state.limite
        if st.button(f"Mostrar más ({faltan} restantes)", key=f"mas-{st.session_state.pagina}", width="stretch"):
            st.session_state.limite += POR_PAGINA
            st.rerun()


def botones_mensaje(cliente, asunto, cuerpo, clave, tipo="Correo"):
    """Mensaje ya redactado: se abre en el correo o en WhatsApp de quien lo envía."""
    b1, b2 = st.columns(2)
    if cliente.get("email"):
        b1.link_button("Abrir en mi correo", cobranza.link_mailto(cliente["email"], asunto, cuerpo), icon=":material/mail:", width="stretch")
    else:
        b1.caption("Sin correo registrado.")
    wa = cobranza.link_whatsapp(cliente.get("telefono"), cuerpo)
    if wa:
        b2.link_button("Enviar por WhatsApp", wa, icon=":material/chat:", width="stretch")
    else:
        b2.caption("Sin celular válido para WhatsApp.")
    with st.expander("Ver el mensaje"):
        st.html(f'<div class="mensaje"><b>{tema.e(asunto)}</b>\n\n{tema.e(cuerpo)}</div>')
    if st.button("Anotar en la bitácora que se envió", key=f"{clave}-{st.session_state.ver}", type="tertiary"):
        db.registrar_interaccion(cliente["id"], f"Se envió: {asunto.split(' · ')[0].lower()}.", tipo, st.session_state.yo)
        avisar("Anotado en la bitácora")
        refrescar()


def _exportar():
    """Arma el Excel al hacer clic. Corre en otro hilo, por eso abre su propia conexión."""
    otra = BaseDatos(cfg.ruta_db)
    try:
        return intercambio.exportar_excel(otra, hoy, cfg.dias_alerta)
    finally:
        otra.cerrar()


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def _estilo(grafico):
    return (
        grafico.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor=tema.TINTA, titleColor=tema.TINTA, gridColor=tema.REJILLA, domainColor=tema.REJILLA, tickColor=tema.REJILLA)
        .configure_legend(labelColor=tema.TINTA, titleColor=tema.TINTA)
    )


def barras_horizontales(df, categoria, valor, orden=None, etiqueta=None):
    tipo_etiqueta = "N" if etiqueta else "Q"
    base = alt.Chart(df).encode(
        y=alt.Y(f"{categoria}:N", sort=orden or "-x", title=None, axis=alt.Axis(labelLimit=200, domain=False, ticks=False)),
        x=alt.X(f"{valor}:Q", title=None, axis=None),
        tooltip=[alt.Tooltip(f"{categoria}:N"), alt.Tooltip(f"{etiqueta or valor}:{tipo_etiqueta}", title=valor)],
    )
    barras = base.mark_bar(color=tema.SERIE_1, cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=18)
    textos = base.mark_text(align="left", dx=6, color=tema.TINTA).encode(text=f"{etiqueta or valor}:{tipo_etiqueta}")
    return _estilo((barras + textos).properties(height=max(110, 36 * len(df))))


def grafico_emitido_cobrado(serie):
    df = pd.DataFrame(
        [
            {
                "Mes": f"{calendario.MESES[int(m['mes'][5:]) - 1][:3]} {m['mes'][2:4]}", "orden": m["mes"],
                "Serie": nombre, "Millones": m[campo] / 1e6, "Monto": cobranza.formato_clp(m[campo]),
            }
            for m in serie
            for nombre, campo in (("Emitido", "emitido"), ("Cobrado", "cobrado"))
        ]
    )
    return _estilo(
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Mes:N", sort=alt.SortField("orden"), title=None, axis=alt.Axis(labelAngle=0, domain=False, ticks=False), scale=alt.Scale(paddingInner=0.35)),
            xOffset=alt.XOffset("Serie:N", sort=["Emitido", "Cobrado"], scale=alt.Scale(paddingInner=0.1)),
            y=alt.Y("Millones:Q", title="Millones de $", axis=alt.Axis(tickCount=4, domain=False, labelExpr="replace(format(datum.value, '~g'), '.', ',')")),
            color=alt.Color(
                "Serie:N", sort=["Emitido", "Cobrado"],
                scale=alt.Scale(domain=["Emitido", "Cobrado"], range=[tema.SERIE_1, tema.SERIE_2]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["Mes:N", "Serie:N", "Monto:N"],
        )
        .properties(height=230)
    )


# ---------------------------------------------------------------------------
# Vencimientos: tarjetas, tabla y ventanas
# ---------------------------------------------------------------------------
def tarjeta_vencimiento(o, clave, con_cliente=True):
    """Tarjeta de un vencimiento: se toca para editar; «Listo» lo marca como presentado."""
    situacion = calendario.semaforo(o, hoy, cfg.dias_alerta)
    color = tema.COLOR_SEMAFORO[situacion]
    cerrada = situacion == "Cerrada"
    sub = " · ".join(
        x for x in (
            tema.e(o["razon_social"]) if con_cliente else "",
            tema.e(etiqueta_periodo(o["periodo"])) if o["periodo"] else "",
            tema.e(o["responsable"]) if o["responsable"] and not st.session_state.yo else "",
        ) if x
    )
    if o["estado"] not in ("Pendiente", "Presentada"):
        sub = tema.chip(o["estado"], tema.AZUL if o["estado"] == "Esperando cliente" else None) + sub
    contenido = tema.fila(
        o["nombre"], sub, icono=tema.punto(color),
        lado=o["estado"] if cerrada else calendario.texto_plazo(o["vence"], hoy).capitalize(),
        lado_sub=fecha_corta(o["vence"]), color_lado=None if situacion in ("Al día", "Cerrada") else color,
    )
    acciones = [] if cerrada else [("listo", "Listo", ":material/check:", "secondary")]
    pulsado = tarjeta(f"{clave}-{o['id']}", contenido, abrir=f"Editar {o['nombre']} de {o['razon_social']}", acciones=acciones)
    if pulsado == "listo":
        db.actualizar_obligacion(o["id"], estado="Presentada")
        avisar(f"{o['nombre']} · {o['razon_social']}: presentado")
        refrescar()
    elif pulsado == "abrir":
        dlg_vencimiento(o["id"])


@st.dialog("Vencimiento")
def dlg_vencimiento(obligacion_id):
    o = db.obligacion(obligacion_id)
    if not o:
        st.info("Este vencimiento ya no existe.")
        return
    k = f"venc-{o['id']}-{st.session_state.ver}"
    st.html(
        tema.fila(
            o["nombre"], tema.e(" · ".join(x for x in (nombre_cliente.get(o["cliente_id"], ""), etiqueta_periodo(o["periodo"]) if o["periodo"] else "") if x)),
            icono=tema.punto(tema.COLOR_SEMAFORO[calendario.semaforo(o, hoy, cfg.dias_alerta)]),
        )
    )
    estado = st.pills("Estado", ESTADOS_OBLIGACION, default=o["estado"], key=f"{k}-estado") or o["estado"]
    c1, c2 = st.columns(2)
    vence = c1.date_input("Vence", a_fecha(o["vence"]), format=FECHA, key=f"{k}-vence")
    responsable = selector_responsable("Responsable", o["responsable"], f"{k}-resp", contenedor=c2)
    notas = st.text_area("Notas", o["notas"], key=f"{k}-notas", placeholder="Ej.: faltan las facturas de compra")
    if st.button("Guardar", type="primary", width="stretch", key=f"{k}-guardar"):
        try:
            db.actualizar_obligacion(o["id"], estado=estado, vence=vence, responsable=responsable, notas=notas)
            avisar("Vencimiento actualizado")
            refrescar()
        except ErrorValidacion as error:
            st.error(str(error))
    if o["tipo"] == "OTRO" and st.button("Eliminar esta tarea", type="tertiary", icon=":material/delete:", key=f"{k}-eliminar"):
        db.eliminar_obligacion(o["id"])
        avisar("Tarea eliminada", "🗑️")
        refrescar()


@st.dialog("Nueva tarea")
def dlg_tarea(cliente_id=None):
    k = f"tarea-{cliente_id}-{st.session_state.ver}"
    st.caption("Para lo que no está en el calendario: requerimientos del SII, trámites, certificados…")
    if cliente_id is None:
        cliente_id = selector_cliente("Cliente", f"{k}-cliente")
    nombre = st.text_input("¿Qué hay que hacer?", placeholder="Ej.: responder requerimiento del SII", key=f"{k}-nombre")
    c1, c2 = st.columns(2)
    vence = c1.date_input("Vence", hoy + timedelta(days=7), format=FECHA, key=f"{k}-vence")
    por_defecto = (db.cliente(cliente_id) or {}).get("responsable", "") if cliente_id else st.session_state.yo
    responsable = selector_responsable("Responsable", por_defecto, f"{k}-resp-{cliente_id}", contenedor=c2)
    if st.button("Agregar tarea", type="primary", width="stretch", key=f"{k}-crear"):
        if not cliente_id:
            st.error("Elige un cliente.")
            return
        try:
            db.crear_obligacion(cliente_id, "OTRO", nombre, vence, responsable=responsable)
            avisar("Tarea agregada")
            refrescar()
        except ErrorValidacion as error:
            st.error(str(error))


@st.dialog("Generar vencimientos")
def dlg_generar():
    st.caption(
        "Crea, para cada cliente activo, las obligaciones que corresponden a sus servicios contratados. "
        "Se puede repetir sin riesgo: no duplica las que ya existen."
    )
    per_m = st.selectbox("Mes", periodos_mensuales(), index=1, format_func=etiqueta_periodo, help="El período que se declara. Octubre se declara en noviembre.")
    if st.button(f"Generar {calendario.nombre_periodo(per_m)}", type="primary", width="stretch"):
        avisar(f"{calendario.generar(db, cfg, per_m)} vencimientos creados para {calendario.nombre_periodo(per_m)}", "📅")
        refrescar()
    st.divider()
    per_a = st.selectbox(
        "Año comercial (renta, declaraciones juradas, patente)", [str(hoy.year - i) for i in range(3)], index=1,
        help="El año comercial 2025 se declara en la Operación Renta 2026.",
    )
    if st.button(f"Generar año {per_a}", width="stretch"):
        avisar(f"{calendario.generar(db, cfg, per_a)} vencimientos anuales creados para el año comercial {per_a}", "📅")
        refrescar()


@st.dialog("Pedir información al cliente")
def dlg_pedir_info(cliente_id):
    c = db.cliente(cliente_id)
    esperando = [o for o in db.obligaciones(cliente_id=cliente_id, abiertas=True) if o["estado"] == "Esperando cliente"]
    if not esperando:
        st.info("No hay vencimientos en «Esperando cliente».")
        return
    st.caption(f"{plural(len(esperando), 'vencimiento')} esperando información de {c['razon_social']}.")
    botones_mensaje(c, *cobranza.mensaje_documentos(c, esperando, cfg), clave=f"msg-doc-{c['id']}")


def _guardar_obligaciones(clave, ids):
    """Callback de la tabla editable: guarda en la base las celdas modificadas."""
    campos_por_columna = {"Estado": "estado", "Vence": "vence", "Responsable": "responsable", "Notas": "notas"}
    for fila, valores in st.session_state[clave]["edited_rows"].items():
        campos = {campos_por_columna[c]: v for c, v in valores.items() if c in campos_por_columna}
        try:
            db.actualizar_obligacion(ids[int(fila)], **campos)
        except ErrorValidacion as error:
            avisar(str(error), "⚠️")
    st.session_state.ver += 1


def tabla_vencimientos(filas, clave):
    """Vista de tabla (cómoda en PC): estado, fecha, responsable y notas se guardan al cambiar la celda."""
    clave = f"{clave}-{st.session_state.ver}"
    df = pd.DataFrame(
        [
            {
                "Situación": f"{calendario.ICONO_SEMAFORO[s]} " + ("Cerrada" if s == "Cerrada" else calendario.texto_plazo(o["vence"], hoy).capitalize()),
                "Vence": a_fecha(o["vence"]), "Cliente": o["razon_social"], "Obligación": o["nombre"],
                "Período": etiqueta_periodo(o["periodo"]) if o["periodo"] else "", "Estado": o["estado"],
                "Responsable": o["responsable"] or None, "Notas": o["notas"],
            }
            for o in filas
            for s in [calendario.semaforo(o, hoy, cfg.dias_alerta)]
        ],
        columns=["Situación", "Vence", "Cliente", "Obligación", "Período", "Estado", "Responsable", "Notas"],
    )
    st.data_editor(
        df, key=clave, hide_index=True, width="stretch", num_rows="fixed",
        disabled=["Situación", "Cliente", "Obligación", "Período"],
        on_change=_guardar_obligaciones, args=(clave, [o["id"] for o in filas]),
        column_config={
            "Vence": st.column_config.DateColumn("Vence", format=FECHA, required=True),
            "Estado": st.column_config.SelectboxColumn("Estado", options=ESTADOS_OBLIGACION, required=True),
            "Responsable": st.column_config.SelectboxColumn(
                "Responsable", options=sorted(set(responsables) | {o["responsable"] for o in filas if o["responsable"]})
            ),
            "Notas": st.column_config.TextColumn("Notas", width="large"),
        },
    )


# ---------------------------------------------------------------------------
# Cobranza: ventanas
# ---------------------------------------------------------------------------
def texto_cobro(c):
    if c["concepto"] == cobranza.CONCEPTO_HONORARIO and c["periodo"]:
        return f"Honorario {calendario.nombre_periodo(c['periodo'])}"
    return c["concepto"] + (f" · {calendario.nombre_periodo(c['periodo'])}" if c["periodo"] else "")


def color_mora(dias):
    return tema.ROJO if dias > 30 else tema.AMBAR if dias > 0 else tema.VERDE


def texto_mora(dias):
    return f"{plural(dias, 'día')} de atraso" if dias > 0 else "Al día"


@st.dialog("Cobranza")
def dlg_cobranza(cliente_id):
    c = db.cliente(cliente_id)
    pendientes = cobranza.cartera(db, hoy, cliente_id)
    if not c or not pendientes:
        st.success("Este cliente no tiene cobros pendientes.")
        return
    k = f"cob-{cliente_id}-{st.session_state.ver}"
    total = sum(p["saldo"] for p in pendientes)
    st.html(tema.fila(c["razon_social"], tema.e(plural(len(pendientes), "cobro pendiente", "cobros pendientes")), icono=tema.avatar(c["razon_social"]), lado=cobranza.formato_clp(total)))

    st.markdown("**Registrar un pago**")
    por_id = {p["id"]: p for p in pendientes}
    cobro_id = st.selectbox(
        "¿Qué pagó?", list(por_id), key=f"{k}-cobro",
        format_func=lambda i: f"{texto_cobro(por_id[i])} · {cobranza.formato_clp(por_id[i]['saldo'])}",
    )
    cobro = por_id[cobro_id]
    q1, q2 = st.columns(2)
    monto = q1.number_input("Monto ($)", min_value=1, max_value=int(cobro["saldo"]), value=int(cobro["saldo"]), step=1000, key=f"{k}-monto-{cobro_id}")
    fecha_pago = q2.date_input("Fecha", hoy, max_value=hoy, format=FECHA, key=f"{k}-fecha")
    q3, q4 = st.columns(2)
    medio = q3.selectbox("Medio", MEDIOS_PAGO, key=f"{k}-medio")
    nota = q4.text_input("Nota", placeholder="N° de operación…", key=f"{k}-nota")
    if st.button("Registrar pago", type="primary", width="stretch", key=f"{k}-pagar"):
        try:
            db.registrar_pago(cobro["id"], monto, fecha_pago, medio, nota)
            avisar(f"Pago de {cobranza.formato_clp(monto)} registrado para {c['razon_social']}", "💰")
            refrescar()
        except ErrorValidacion as error:
            st.error(str(error))
    if not cobro["pagado"] and st.button("Este cobro fue un error: anularlo", type="tertiary", key=f"{k}-anular-{cobro_id}"):
        db.actualizar_cobro(cobro["id"], anulado=True)
        avisar("Cobro anulado", "🚫")
        refrescar()

    st.divider()
    st.markdown("**Recordarle el pago**")
    botones_mensaje(c, *cobranza.mensaje_cobranza(c, pendientes, cfg), clave=f"msg-cob-{cliente_id}", tipo="Cobranza")


@st.dialog("Cobro extraordinario")
def dlg_cobro_extra(cliente_id=None):
    k = f"extra-{cliente_id}-{st.session_state.ver}"
    st.caption("Trabajos fuera del honorario mensual: constitución de sociedad, regularizaciones, certificados…")
    if cliente_id is None:
        cliente_id = selector_cliente("Cliente", f"{k}-cliente")
    concepto = st.text_input("Concepto", placeholder="Ej.: Constitución de sociedad", key=f"{k}-concepto")
    y1, y2 = st.columns(2)
    monto = y1.number_input("Monto ($)", min_value=0, step=10000, key=f"{k}-monto")
    vence = y2.date_input("Vence", hoy + timedelta(days=10), format=FECHA, key=f"{k}-vence")
    if st.button("Crear cobro", type="primary", width="stretch", key=f"{k}-crear"):
        if not cliente_id:
            st.error("Elige un cliente.")
            return
        try:
            db.crear_cobro(cliente_id, concepto, monto, hoy, vence)
            avisar("Cobro creado", "🧾")
            refrescar()
        except ErrorValidacion as error:
            st.error(str(error))


# ---------------------------------------------------------------------------
# Clientes: formulario
# ---------------------------------------------------------------------------
def formulario_cliente(cliente=None):
    c = cliente or {}
    k = f"cli-{c.get('id', 'nuevo')}-{st.session_state.ver}"
    with st.form(k, border=False):
        a1, a2 = st.columns([1, 2])
        rut = a1.text_input("RUT", rutlib.formatear(c["rut"]) if c.get("rut") else "", placeholder="76.123.456-7", key=f"{k}-rut")
        razon = a2.text_input("Razón social o nombre *", c.get("razon_social", ""), key=f"{k}-razon")
        servicios = st.multiselect(
            "Servicios contratados", calendario.SERVICIOS,
            default=[s for s in c.get("servicios", []) if s in calendario.SERVICIOS], key=f"{k}-servicios",
            placeholder="Elige los servicios",
            help="De aquí salen los vencimientos que se generan cada mes y cada año para este cliente.",
        )
        f1, f2, f3 = st.columns(3)
        honorario = f1.number_input("Honorario mensual", min_value=0.0, value=float(c.get("honorario") or 0), step=1.0, format="%g", key=f"{k}-hon")
        moneda = f2.selectbox("Moneda", MONEDAS, index=indice(MONEDAS, c.get("moneda")), key=f"{k}-moneda")
        responsable = selector_responsable("Responsable en el estudio", c.get("responsable", st.session_state.yo), f"{k}-resp", contenedor=f3)
        d1, d2, d3 = st.columns(3)
        contacto = d1.text_input("Persona de contacto", c.get("contacto", ""), key=f"{k}-contacto")
        email = d2.text_input("Correo", c.get("email", ""), key=f"{k}-email")
        telefono = d3.text_input("Teléfono", c.get("telefono", ""), placeholder="+56 9 …", key=f"{k}-fono")

        with st.expander("Más datos (régimen, giro, dirección…)"):
            b1, b2, b3 = st.columns(3)
            estado = b1.selectbox("Estado", ESTADOS_CLIENTE, index=indice(ESTADOS_CLIENTE, c.get("estado")), key=f"{k}-estado")
            tipo = b2.selectbox("Tipo", TIPOS_CLIENTE, index=indice(TIPOS_CLIENTE, c.get("tipo")), key=f"{k}-tipo")
            regimenes = [""] + REGIMENES
            regimen = b3.selectbox(
                "Régimen tributario", regimenes, index=indice(regimenes, c.get("regimen")),
                format_func=lambda r: r or "Sin definir", key=f"{k}-regimen",
            )
            g1, g2 = st.columns(2)
            fantasia = g1.text_input("Nombre de fantasía", c.get("nombre_fantasia", ""), key=f"{k}-fantasia")
            giro = g2.text_input("Giro", c.get("giro", ""), key=f"{k}-giro")
            e1, e2, e3 = st.columns([2, 2, 1])
            representante = e1.text_input("Representante legal", c.get("representante", ""), key=f"{k}-rep")
            direccion = e2.text_input("Dirección", c.get("direccion", ""), key=f"{k}-dir")
            comuna = e3.text_input("Comuna", c.get("comuna", ""), key=f"{k}-comuna")
            h1, h2 = st.columns(2)
            inicio = h1.date_input("Cliente desde", a_fecha(c.get("fecha_inicio")), format=FECHA, key=f"{k}-inicio")
            trabajadores = h2.number_input("Trabajadores", min_value=0, value=int(c.get("trabajadores") or 0), step=1, key=f"{k}-trab")
            notas = st.text_area("Notas", c.get("notas", ""), key=f"{k}-notas")
        enviado = st.form_submit_button("Guardar cambios" if cliente else "Crear cliente", type="primary", width="stretch")
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
                nuevo = db.crear_cliente(**campos)
                ir("cliente", nuevo)
                avisar(f"Cliente creado: {razon}")
            refrescar()
        except ErrorValidacion as error:
            st.error(str(error))


@st.dialog("Nuevo cliente", width="large")
def dlg_cliente_nuevo():
    st.caption("Con el RUT, el nombre y los servicios basta para empezar. Lo demás se completa después.")
    formulario_cliente()


@st.dialog("Editar cliente", width="large")
def dlg_cliente_editar(cliente_id):
    formulario_cliente(db.cliente(cliente_id))


# ---------------------------------------------------------------------------
# Navegación
# ---------------------------------------------------------------------------
PAGINAS = [
    ("inicio", "Inicio", ":material/home:"),
    ("clientes", "Clientes", ":material/groups:"),
    ("vencimientos", "Vencimientos", ":material/event_available:"),
    ("cobros", "Cobros", ":material/payments:"),
    ("mas", "Más", ":material/menu:"),
]
pagina = st.session_state.pagina
if pagina == "cliente" and not (st.session_state.cliente_id and db.cliente(st.session_state.cliente_id)):
    pagina = st.session_state.pagina = "clientes"

with st.container(key="nav"):
    st.html(f'<div class="marca">Estudio contable<small>{tema.e(cfg.estudio_nombre)}</small></div>')
    for _id, _titulo, _icono in PAGINAS:
        activa = pagina == _id or (pagina == "cliente" and _id == "clientes")
        st.button(_titulo, key=f"nav-{_id}", icon=_icono, type="primary" if activa else "tertiary", on_click=ir, args=(_id,), width="stretch")


# ---------------------------------------------------------------------------
# Inicio
# ---------------------------------------------------------------------------
def pagina_inicio():
    hora = datetime.now(cfg.tz).hour
    saludo = "Buenos días" if hora < 12 else "Buenas tardes" if hora < 20 else "Buenas noches"
    dia = f"{['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'][hoy.weekday()]} {hoy.day} de {calendario.MESES[hoy.month - 1]}"
    st.html(
        f'<div class="saludo">{saludo}<small>{dia} · {tema.e(cfg.estudio_nombre)}</small></div>'
        + ('<div style="margin-top:.5rem"><span class="aviso-demo">Datos de demostración · se borran en Más</span></div>' if hay_demo else "")
    )

    if not clientes:
        st.html(tema.vacio("Empecemos por tus clientes", "Crea el primero, importa tu planilla o mira cómo funciona con datos de ejemplo."))
        b1, b2, b3 = st.columns(3)
        if b1.button("Crear mi primer cliente", type="primary", icon=":material/add:", width="stretch"):
            dlg_cliente_nuevo()
        b2.button("Importar una planilla", icon=":material/upload_file:", width="stretch", on_click=ir, args=("mas",))
        if b3.button("Ver con datos de ejemplo", icon=":material/science:", width="stretch"):
            demo.cargar(db, cfg, hoy)
            avisar("Datos de demostración cargados", "🧪")
            refrescar()
        return

    yo = selector_persona("inicio")
    abiertas = [o for o in db.obligaciones(abiertas=True) if not yo or o["responsable"] == yo]
    situacion = [calendario.semaforo(o, hoy, cfg.dias_alerta) for o in abiertas]
    res = cobranza.resumen(db, hoy)
    atrasados = situacion.count("Atrasada")
    proximos = situacion.count("Próxima") + situacion.count("Vence hoy")
    esperando = sum(o["estado"] == "Esperando cliente" for o in abiertas)

    with st.container(key="kpis"):
        for clave, html_kpi, destino, filtro, texto in (
            ("atrasados", tema.kpi(atrasados, "Atrasados", "vencimientos pasados de fecha", tema.ROJO if atrasados else None), "vencimientos", "Atrasados", "Ver vencimientos atrasados"),
            ("proximos", tema.kpi(proximos, f"Vencen en {cfg.dias_alerta} días", "incluye los de hoy", tema.AMARILLO if proximos else None), "vencimientos", "Urgentes", "Ver vencimientos próximos"),
            ("esperando", tema.kpi(esperando, "Esperando al cliente", "falta que envíen información"), "vencimientos", "Esperando cliente", "Ver vencimientos en espera"),
            ("cobrar", tema.kpi(cobranza.formato_corto(res["por_cobrar"]), "Por cobrar", f"{cobranza.formato_corto(res['vencido'])} vencido" if res["vencido"] else "nada vencido"), "cobros", None, "Ver cobros"),
        ):
            with st.container(key=f"fila-kpi-{clave}"):
                st.html(html_kpi)
                st.button(texto, key=f"abrir-kpi-{clave}", on_click=ir, args=(destino, None, filtro))

    st.subheader("Lo que viene")
    limite = (hoy + timedelta(days=14)).isoformat()
    urgentes = [o for o in abiertas if o["vence"] <= limite]
    if urgentes:
        for o in urgentes[:6]:
            tarjeta_vencimiento(o, "ini")
        if len(urgentes) > 6:
            st.button(f"Ver los {len(urgentes)} vencimientos", key="ver-venc", on_click=ir, args=("vencimientos", None, "Urgentes"), width="stretch")
    elif abiertas:
        st.html(tema.vacio("Todo al día", "Nada atrasado ni por vencer en los próximos 14 días."))
    else:
        st.html(tema.vacio("Sin vencimientos abiertos", "Genera los del mes en la pantalla Vencimientos."))

    deudores = _deudores(cobranza.cartera(db, hoy))
    if deudores:
        st.subheader("Quién debe")
        for d in deudores[:4]:
            tarjeta_deudor(d, "ini")
        if len(deudores) > 4:
            st.button(f"Ver los {len(deudores)} clientes con saldo", key="ver-cobros", on_click=ir, args=("cobros",), width="stretch")

    st.subheader("Resumen")
    g1, g2, g3 = st.columns([2, 2, 3])
    with g1:
        st.markdown("**Vencimientos abiertos por responsable**")
        todas = db.obligaciones(abiertas=True)
        if todas:
            carga = pd.Series([o["responsable"] or "Sin asignar" for o in todas]).value_counts().rename_axis("Responsable").reset_index(name="Abiertos")
            st.altair_chart(barras_horizontales(carga, "Responsable", "Abiertos"), width="stretch")
        else:
            st.caption("Sin vencimientos abiertos.")
    with g2:
        st.markdown("**Por cobrar según antigüedad**")
        if res["por_cobrar"]:
            tramos = cobranza.por_tramo(cobranza.cartera(db, hoy))
            df_tramos = pd.DataFrame([{"Tramo": t, "Saldo": m, "Monto": cobranza.formato_clp(m)} for t, m in tramos.items() if m])
            st.altair_chart(barras_horizontales(df_tramos, "Tramo", "Saldo", orden=cobranza.TRAMOS_MORA, etiqueta="Monto"), width="stretch")
        else:
            st.caption("No hay honorarios pendientes de pago.")
    with g3:
        st.markdown("**Honorarios emitidos y cobrados por mes**")
        serie = cobranza.serie_mensual(db, hoy)
        if any(m["emitido"] or m["cobrado"] for m in serie):
            st.altair_chart(grafico_emitido_cobrado(serie), width="stretch")
        else:
            st.caption("Aún no hay honorarios emitidos.")


# ---------------------------------------------------------------------------
# Clientes
# ---------------------------------------------------------------------------
def pagina_clientes():
    t1, t2 = st.columns([3, 1], vertical_alignment="bottom")
    t1.title("Clientes")
    if t2.button("Nuevo cliente", type="primary", icon=":material/add:", width="stretch"):
        dlg_cliente_nuevo()
    if not clientes:
        st.html(tema.vacio("Aún no hay clientes", "Crea el primero con el botón «Nuevo cliente» o importa tu planilla en Más."))
        return

    buscar = st.text_input("Buscar", placeholder="Buscar por nombre, RUT o contacto", label_visibility="collapsed")
    cuenta = {e: sum(c["estado"] == e for c in clientes) for e in ESTADOS_CLIENTE}
    grupos = {"Activos": ["Activo"], "Prospectos": ["Prospecto"], "Inactivos": ["Suspendido", "De baja"], "Todos": ESTADOS_CLIENTE}
    visibles = [g for g, estados in grupos.items() if g in ("Activos", "Todos") or sum(cuenta[e] for e in estados)]
    grupo = st.pills(
        "Estado", visibles, default="Activos", label_visibility="collapsed",
        format_func=lambda g: f"{g} · {sum(cuenta[e] for e in grupos[g])}",
    ) or "Todos"

    saldos, abiertas_por_cliente = {}, {}
    for p in cobranza.cartera(db, hoy):
        saldos[p["cliente_id"]] = saldos.get(p["cliente_id"], 0) + p["saldo"]
    for o in db.obligaciones(abiertas=True):
        abiertas_por_cliente[o["cliente_id"]] = abiertas_por_cliente.get(o["cliente_id"], 0) + 1

    aguja = buscar.lower().replace(".", "").strip()
    vista = [
        c for c in clientes
        if (aguja or c["estado"] in grupos[grupo])
        and (not aguja or aguja in " ".join(str(c[x] or "") for x in ("razon_social", "nombre_fantasia", "rut", "contacto", "giro", "email")).lower())
    ]
    if not vista:
        st.html(tema.vacio("Sin resultados", "Prueba con otro nombre o cambia el filtro."))
        return
    for c in vista[: st.session_state.limite]:
        sub = " · ".join(tema.e(x) for x in (rutlib.formatear(c["rut"]) if c["rut"] else "Sin RUT", c["responsable"]) if x)
        if c["estado"] != "Activo":
            sub = tema.chip(c["estado"], tema.COLOR_ESTADO_CLIENTE[c["estado"]]) + sub
        saldo, n_abiertas = saldos.get(c["id"], 0), abiertas_por_cliente.get(c["id"], 0)
        contenido = tema.fila(
            c["razon_social"], sub, icono=tema.avatar(c["razon_social"]),
            lado=f"Debe {cobranza.formato_corto(saldo)}" if saldo else "", color_lado=tema.AMBAR,
            lado_sub=plural(n_abiertas, "vencimiento") if n_abiertas else "",
        )
        if tarjeta(f"cli-{c['id']}", contenido, abrir=f"Abrir {c['razon_social']}"):
            ir("cliente", c["id"])
            st.rerun()
    mostrar_mas(len(vista))


def pagina_cliente():
    c = db.cliente(st.session_state.cliente_id)
    st.button("Clientes", icon=":material/arrow_back:", type="tertiary", on_click=ir, args=("clientes",), key="volver")
    obligaciones = db.obligaciones(cliente_id=c["id"])
    abiertas_c = [o for o in obligaciones if o["estado"] not in ESTADOS_CERRADOS]
    pendientes = cobranza.cartera(db, hoy, c["id"])
    saldo = sum(p["saldo"] for p in pendientes)

    chips = (
        tema.chip(c["estado"], tema.COLOR_ESTADO_CLIENTE.get(c["estado"]))
        + (tema.chip(f"Debe {cobranza.formato_clp(saldo)}", tema.AMBAR) if saldo else "")
        + "".join(tema.chip(s) for s in c["servicios"])
    )
    detalle = " · ".join(
        tema.e(x) for x in (
            rutlib.formatear(c["rut"]) if c["rut"] else "Sin RUT",
            f"Responsable: {c['responsable']}" if c["responsable"] else "", f"Honorario {cobranza.formato_honorario(c)}" if c["honorario"] else "",
        ) if x
    )
    st.html(
        f'<div class="heroe">{tema.avatar(c["razon_social"], grande=True)}<div><div class="nombre">{tema.e(c["razon_social"])}</div>'
        f'<div class="sub">{detalle}</div></div></div><div style="margin-top:.6rem">{chips}</div>'
    )

    with st.container(horizontal=True, key="acciones-cliente"):
        if pendientes and st.button("Registrar pago", type="primary", icon=":material/payments:"):
            dlg_cobranza(c["id"])
        wa = cobranza.link_whatsapp(c["telefono"], f"Hola {c['contacto'].split(' ')[0]}," if c["contacto"] else "Hola,")
        if wa:
            st.link_button("WhatsApp", wa, icon=":material/chat:")
        if c["telefono"]:
            st.link_button("Llamar", "tel:" + "".join(ch for ch in c["telefono"] if ch.isdigit() or ch == "+"), icon=":material/call:")
        if c["email"]:
            st.link_button("Correo", f"mailto:{c['email']}", icon=":material/mail:")
        if st.button("Editar", icon=":material/edit:"):
            dlg_cliente_editar(c["id"])

    secciones = {"Vencimientos": f"Vencimientos ({len(abiertas_c)})", "Cobros": "Cobros", "Notas": "Notas", "Datos": "Datos"}
    seccion = st.segmented_control(
        "Sección", list(secciones), default="Vencimientos", format_func=secciones.get, key=f"sec-{c['id']}", label_visibility="collapsed",
    ) or "Vencimientos"

    if seccion == "Vencimientos":
        ver_cerradas = st.toggle("Mostrar también los ya presentados", key=f"cerradas-{c['id']}")
        visibles = obligaciones if ver_cerradas else abiertas_c
        for o in visibles[: st.session_state.limite]:
            tarjeta_vencimiento(o, "ficha", con_cliente=False)
        mostrar_mas(len(visibles))
        if not visibles:
            st.html(tema.vacio("Sin vencimientos abiertos", "Los vencimientos salen de los servicios contratados del cliente."))
        with st.container(horizontal=True):
            if st.button("Agregar tarea", icon=":material/add:", key="tarea-ficha"):
                dlg_tarea(c["id"])
            if any(o["estado"] == "Esperando cliente" for o in abiertas_c) and st.button("Pedir información", icon=":material/forward_to_inbox:"):
                dlg_pedir_info(c["id"])

    elif seccion == "Cobros":
        todos = db.cobros(cliente_id=c["id"], incluir_anulados=True)
        st.html(
            tema.kpis(
                tema.kpi(cobranza.formato_clp(saldo), "Saldo pendiente"),
                tema.kpi(cobranza.formato_clp(sum(p["saldo"] for p in pendientes if p["dias_mora"] > 0)), "Vencido"),
                tema.kpi(cobranza.formato_clp(sum(x["pagado"] for x in todos)), "Pagado históricamente"),
            )
        )
        for x in reversed(todos[-st.session_state.limite:]):
            estado = estado_cobro(x)
            color = {"Pagado": tema.VERDE, "Anulado": tema.GRIS}.get(estado, color_mora(max(0, (hoy - a_fecha(x["vence"])).days)))
            st.html(
                '<div class="fila" style="padding:.55rem .2rem;border-bottom:1px solid var(--borde)">'
                + tema.punto(color)
                + f'<div class="cuerpo"><div class="titulo">{tema.e(texto_cobro(x))}</div><div class="sub">Vence el {fecha_corta(x["vence"])}'
                + (f" · abonado {cobranza.formato_clp(x['pagado'])}" if 0 < x["pagado"] < x["monto"] else "")
                + f'</div></div><div class="lado"><b>{cobranza.formato_clp(x["monto"])}</b>{estado}</div></div>'
            )
        if not todos:
            st.html(tema.vacio("Aún no se le han emitido cobros", "Los honorarios del mes se emiten en la pantalla Cobros."))
        with st.container(horizontal=True):
            if pendientes and st.button("Registrar pago o recordar", icon=":material/payments:", key="pago-ficha"):
                dlg_cobranza(c["id"])
            if st.button("Cobro extraordinario", icon=":material/add:", key="extra-ficha"):
                dlg_cobro_extra(c["id"])

    elif seccion == "Notas":
        with st.form(f"nota-{c['id']}-{st.session_state.ver}", border=False):
            n1, n2 = st.columns([1, 3])
            tipo = n1.selectbox("Tipo", TIPOS_INTERACCION)
            resumen = n2.text_input("¿Qué pasó?", placeholder="Ej.: llamó para pedir certificado de renta")
            if st.form_submit_button("Anotar", type="primary"):
                try:
                    db.registrar_interaccion(c["id"], resumen, tipo, st.session_state.yo)
                    refrescar()
                except ErrorValidacion as error:
                    st.error(str(error))
        notas = db.interacciones(cliente_id=c["id"])
        st.html(
            "".join(
                f'<div class="nota"><span>{fecha_corta(n["fecha"])} · {tema.e(n["tipo"])}'
                + (f" · {tema.e(n['autor'])}" if n["autor"] else "")
                + f"</span><div>{tema.e(n['resumen'])}</div></div>"
                for n in notas
            )
            or tema.vacio("Sin anotaciones todavía", "Anota llamadas, acuerdos y pendientes para que todo el equipo sepa en qué va.")
        )

    else:
        st.html(
            tema.datos(
                [
                    ("RUT", rutlib.formatear(c["rut"]) if c["rut"] else ""), ("Nombre de fantasía", c["nombre_fantasia"]), ("Tipo", c["tipo"]),
                    ("Régimen tributario", c["regimen"]), ("Giro", c["giro"]), ("Estado", c["estado"]),
                    ("Responsable en el estudio", c["responsable"]), ("Cliente desde", fecha_corta(c["fecha_inicio"])),
                    ("Persona de contacto", c["contacto"]), ("Correo", c["email"]), ("Teléfono", c["telefono"]),
                    ("Representante legal", c["representante"]), ("Dirección", ", ".join(x for x in (c["direccion"], c["comuna"]) if x)),
                    ("Honorario mensual", cobranza.formato_honorario(c) if c["honorario"] else ""),
                    ("Trabajadores", c["trabajadores"] or ""), ("Notas", c["notas"]),
                ]
            )
        )
        if st.button("Editar datos", icon=":material/edit:", key="editar-datos"):
            dlg_cliente_editar(c["id"])
        with st.expander("Eliminar cliente"):
            st.caption(
                "Borra al cliente con todos sus vencimientos, cobros, pagos y notas. No se puede deshacer. "
                "Si solo dejaste de atenderlo, cambia su estado a «De baja» para conservar el historial."
            )
            seguro = st.checkbox(f"Sí, eliminar a {c['razon_social']} y todo su historial", key=f"seguro-{c['id']}")
            if st.button("Eliminar definitivamente", disabled=not seguro, key=f"eliminar-{c['id']}"):
                db.eliminar_cliente(c["id"])
                ir("clientes")
                st.session_state.cliente_id = None
                avisar(f"{c['razon_social']} eliminado", "🗑️")
                refrescar()


# ---------------------------------------------------------------------------
# Vencimientos
# ---------------------------------------------------------------------------
FILTROS_VENC = ["Urgentes", "Atrasados", "Esperando cliente", "Abiertos", "Cerrados"]


def pagina_vencimientos():
    t1, t2 = st.columns([3, 1], vertical_alignment="bottom")
    t1.title("Vencimientos")
    if t2.button("Agregar tarea", icon=":material/add:", width="stretch"):
        dlg_tarea()

    periodo_actual = calendario.periodo_anterior(hoy)
    con_servicios = [c for c in activos if calendario.tipos_para(c)]
    if con_servicios and not db.obligaciones(periodo=periodo_actual):
        st.html(tema.vacio(
            f"Faltan los vencimientos de {calendario.nombre_periodo(periodo_actual)}",
            f"Se declaran este mes para {plural(len(con_servicios), 'cliente')}.",
        ))
        if st.button(f"Generar vencimientos de {calendario.nombre_periodo(periodo_actual)}", type="primary", width="stretch"):
            avisar(f"{calendario.generar(db, cfg, periodo_actual)} vencimientos creados", "📅")
            refrescar()

    yo = selector_persona("venc")
    todas = [o for o in db.obligaciones() if not yo or o["responsable"] == yo]
    sem = {o["id"]: calendario.semaforo(o, hoy, cfg.dias_alerta) for o in todas}
    grupos = {
        "Urgentes": [o for o in todas if sem[o["id"]] in ("Atrasada", "Vence hoy", "Próxima")],
        "Atrasados": [o for o in todas if sem[o["id"]] == "Atrasada"],
        "Esperando cliente": [o for o in todas if o["estado"] == "Esperando cliente"],
        "Abiertos": [o for o in todas if sem[o["id"]] != "Cerrada"],
        "Cerrados": [o for o in reversed(todas) if sem[o["id"]] == "Cerrada"],
    }
    filtro = st.pills(
        "Mostrar", FILTROS_VENC, default=st.session_state.filtro_venc if st.session_state.filtro_venc in FILTROS_VENC else "Urgentes",
        format_func=lambda f: f"{f} · {len(grupos[f])}", label_visibility="collapsed", key="pills-venc",
    ) or "Abiertos"
    st.session_state.filtro_venc = filtro

    b1, b2 = st.columns([3, 1], vertical_alignment="bottom")
    buscar = b1.text_input("Buscar", placeholder="Buscar cliente u obligación", label_visibility="collapsed", key="buscar-venc")
    vista = b2.segmented_control("Vista", ["Tarjetas", "Tabla"], default="Tarjetas", label_visibility="collapsed", key="vista-venc") or "Tarjetas"
    filtradas = [o for o in grupos[filtro] if not buscar or buscar.lower() in f"{o['razon_social']} {o['nombre']}".lower()]

    if not filtradas:
        st.html(tema.vacio("Nada por aquí", "No hay vencimientos con este filtro." if todas else "Genera los vencimientos del mes para empezar."))
    elif vista == "Tabla":
        st.caption("Cambia el estado, la fecha, el responsable o las notas directamente en la tabla.")
        tabla_vencimientos(filtradas, "obl-general")
    else:
        for o in filtradas[: st.session_state.limite]:
            tarjeta_vencimiento(o, "venc")
        mostrar_mas(len(filtradas))

    if st.button("Generar vencimientos de otro período", icon=":material/calendar_add_on:", type="tertiary"):
        dlg_generar()


# ---------------------------------------------------------------------------
# Cobros
# ---------------------------------------------------------------------------
def _deudores(pendientes):
    """Agrupa los cobros pendientes por cliente, del más atrasado al menos."""
    por_cliente = {}
    for p in pendientes:
        d = por_cliente.setdefault(p["cliente_id"], {"cliente_id": p["cliente_id"], "razon_social": p["razon_social"], "saldo": 0, "cobros": 0, "mora": 0})
        d["saldo"] += p["saldo"]
        d["cobros"] += 1
        d["mora"] = max(d["mora"], p["dias_mora"])
    return sorted(por_cliente.values(), key=lambda d: (-d["mora"], -d["saldo"]))


def tarjeta_deudor(d, clave):
    contenido = tema.fila(
        d["razon_social"], tema.chip(texto_mora(d["mora"]), color_mora(d["mora"])) + tema.e(plural(d["cobros"], "cobro")),
        icono=tema.avatar(d["razon_social"]), lado=cobranza.formato_clp(d["saldo"]),
    )
    if tarjeta(f"{clave}-deu-{d['cliente_id']}", contenido, abrir=f"Cobrar a {d['razon_social']}", acciones=[("pago", "Pagó", ":material/check:", "secondary")]):
        dlg_cobranza(d["cliente_id"])


def pagina_cobros():
    t1, t2 = st.columns([3, 1], vertical_alignment="bottom")
    t1.title("Cobros")
    if t2.button("Cobro extraordinario", icon=":material/add:", width="stretch"):
        dlg_cobro_extra()

    res = cobranza.resumen(db, hoy)
    st.html(
        tema.kpis(
            tema.kpi(cobranza.formato_clp(res["por_cobrar"]), "Por cobrar"),
            tema.kpi(cobranza.formato_clp(res["vencido"]), "Vencido", plural(res["clientes_morosos"], "cliente con atraso", "clientes con atraso"), tema.ROJO if res["vencido"] else None),
            tema.kpi(cobranza.formato_clp(res["cobrado_mes"]), f"Cobrado en {calendario.MESES[hoy.month - 1]}", color=tema.VERDE if res["cobrado_mes"] else None),
        )
    )

    periodo = calendario.periodo_anterior(hoy)
    con_honorario = [c for c in activos if c["honorario"]]
    emitidos = {c["cliente_id"] for c in db.cobros(incluir_anulados=True) if c["periodo"] == periodo and c["concepto"] == cobranza.CONCEPTO_HONORARIO}
    faltan = [c for c in con_honorario if c["id"] not in emitidos]
    estado_uf = indicadores.estado_uf(db, cfg)
    if faltan:
        st.html(tema.vacio(
            f"Honorarios de {calendario.nombre_periodo(periodo)} por emitir",
            f"{plural(len(faltan), 'cliente')} con honorario mensual. Vencen el día {cfg.honorarios_dia_vence} del mes siguiente.",
        ))
        if st.button(f"Emitir honorarios de {calendario.nombre_periodo(periodo)}", type="primary", width="stretch"):
            creados, avisos = cobranza.emitir_honorarios(db, cfg, periodo, hoy)
            avisar(f"{plural(creados, 'cobro emitido', 'cobros emitidos')}", "🧾")
            for a in avisos:
                avisar(a, "⚠️")
            refrescar()
    if any(c["moneda"] == "UF" for c in con_honorario):
        st.caption(
            f"UF: {cobranza.formato_clp(uf)}" + (f" al {fecha_corta(estado_uf['fecha'])}" if estado_uf["fecha"] else "")
            if uf else "Falta el valor de la UF para los honorarios pactados en UF: defínelo en Más."
        )

    st.subheader("Por cobrar")
    deudores = _deudores(cobranza.cartera(db, hoy))
    if deudores:
        st.caption("Toca un cliente para registrar su pago o enviarle un recordatorio.")
        for d in deudores[: st.session_state.limite]:
            tarjeta_deudor(d, "cob")
        mostrar_mas(len(deudores))
    else:
        st.html(tema.vacio("Nadie debe", "No hay honorarios pendientes de pago.") if db.cobros() else tema.vacio("Aún no hay cobros", "Emite los honorarios del mes para empezar."))

    with st.expander("Emitir honorarios de otro mes"):
        h1, h2 = st.columns([2, 3], vertical_alignment="bottom")
        per_h = h1.selectbox("Mes a cobrar", periodos_mensuales(), index=1, format_func=etiqueta_periodo, key="per-hon")
        if h2.button(f"Emitir {calendario.nombre_periodo(per_h)}", width="stretch"):
            creados, avisos = cobranza.emitir_honorarios(db, cfg, per_h, hoy)
            avisar(f"{plural(creados, 'cobro emitido', 'cobros emitidos')}" if creados else "No había cobros nuevos que emitir", "🧾")
            for a in avisos:
                avisar(a, "⚠️")
            refrescar()
        st.caption("Crea un cobro por cada cliente activo con honorario mensual. No duplica los ya emitidos.")
    with st.expander("Pagos recibidos"):
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
                hide_index=True, width="stretch", height=260,
                column_config={"Fecha": st.column_config.DateColumn(format=FECHA), "Monto": st.column_config.NumberColumn("Monto ($)", format="localized")},
            )
            borrar = st.selectbox(
                "¿Registraste un pago por error?", [p["id"] for p in pagos[:50]], index=None, placeholder="Elige el pago a eliminar",
                format_func=lambda i: next(f"{fecha_corta(p['fecha'])} · {p['razon_social']} · {cobranza.formato_clp(p['monto'])}" for p in pagos if p["id"] == i),
            )
            if borrar and st.button("Eliminar ese pago"):
                db.eliminar_pago(borrar)
                avisar("Pago eliminado: el cobro vuelve a quedar pendiente", "↩️")
                refrescar()
        else:
            st.caption("Aún no se registran pagos.")


# ---------------------------------------------------------------------------
# Más: bitácora y configuración
# ---------------------------------------------------------------------------
def pagina_mas():
    st.title("Más")

    with st.expander("Bitácora del estudio", icon=":material/history:"):
        st.caption("Llamadas, correos, reuniones y acuerdos con cada cliente, para que todo el equipo sepa en qué va.")
        with st.form(f"nota-general-{st.session_state.ver}", border=False):
            n1, n2, n3 = st.columns([2, 1, 3])
            cliente_nota = selector_cliente("Cliente", "nota-cliente", contenedor=n1)
            tipo_nota = n2.selectbox("Tipo", TIPOS_INTERACCION, key="nota-tipo")
            resumen_nota = n3.text_input("¿Qué pasó?", placeholder="Ej.: pidió certificado de renta para el banco")
            if st.form_submit_button("Anotar", type="primary"):
                if not cliente_nota:
                    st.error("Elige un cliente.")
                else:
                    try:
                        db.registrar_interaccion(cliente_nota, resumen_nota, tipo_nota, st.session_state.yo)
                        avisar("Anotado")
                        refrescar()
                    except ErrorValidacion as error:
                        st.error(str(error))
        notas = db.interacciones(limite=40)
        st.html(
            "".join(
                f'<div class="nota"><span>{fecha_corta(n["fecha"])} · {tema.e(n["razon_social"])} · {tema.e(n["tipo"])}'
                + (f" · {tema.e(n['autor'])}" if n["autor"] else "")
                + f"</span><div>{tema.e(n['resumen'])}</div></div>"
                for n in notas
            )
            or '<div class="sub" style="color:var(--txt2)">Sin anotaciones todavía.</div>'
        )

    with st.expander("Valor de la UF", icon=":material/currency_exchange:"):
        estado_uf = indicadores.estado_uf(db, cfg)
        st.html(
            tema.kpis(
                tema.kpi(
                    cobranza.formato_clp(estado_uf["valor"]) if estado_uf["valor"] else "Sin definir", "UF",
                    " · ".join(x for x in (f"al {fecha_corta(estado_uf['fecha'])}" if estado_uf["fecha"] else "", estado_uf["fuente"]) if x),
                )
            )
        )
        if cfg.uf_automatica:
            st.caption("Se actualiza sola cada día desde mindicador.cl, que publica los datos del Banco Central de Chile.")
            if estado_uf["error"]:
                st.warning(f"El último intento falló: {estado_uf['error']} Se sigue usando el valor guardado.")
            if st.button("Actualizar ahora", icon=":material/sync:"):
                resultado = indicadores.actualizar_uf(db, cfg, hoy, forzar=True)
                avisar("UF actualizada" if resultado == "actualizada" else "No se pudo actualizar la UF", "💱" if resultado == "actualizada" else "⚠️")
                refrescar()
        else:
            st.caption("La actualización automática está desactivada (`ESTUDIO_UF_AUTOMATICA=no`).")
        with st.form("uf", border=False):
            u1, u2 = st.columns([2, 1], vertical_alignment="bottom")
            nueva_uf = u1.number_input("Ingresar el valor a mano (pesos por UF)", min_value=0.0, value=float(uf), step=10.0, format="%.2f")
            if u2.form_submit_button("Guardar UF", width="stretch"):
                indicadores.guardar_uf_manual(db, nueva_uf, hoy)
                avisar("Valor de la UF guardado")
                refrescar()

    with st.expander("Equipo del estudio", icon=":material/badge:"):
        with st.form("equipo", border=False):
            texto_equipo = st.text_input(
                "Nombres separados por coma", ", ".join(equipo), placeholder="Carolina, Matías, Fernanda",
                help="Aparecen como responsables de clientes y vencimientos.",
            )
            if st.form_submit_button("Guardar equipo"):
                db.set_json("equipo", [n.strip() for n in texto_equipo.split(",") if n.strip()])
                avisar("Equipo actualizado")
                refrescar()

    with st.expander("Calendario de vencimientos", icon=":material/calendar_month:"):
        st.caption(
            "Fechas **referenciales**: confírmalas cada año con los calendarios del SII, la Dirección del Trabajo y Previred, "
            "y ajusta aquí el día (y el mes, en las anuales). Si la fecha cae en fin de semana o feriado se corre al día hábil "
            "siguiente. Los cambios aplican a los vencimientos que generes desde ahora."
        )
        tipos = calendario.catalogo(db.get_json(calendario.META_AJUSTES, {}))
        meses_cap = [m.capitalize() for m in calendario.MESES]
        mes_siguiente = "Mes siguiente"
        editado = st.data_editor(
            pd.DataFrame(
                [
                    {
                        "Obligación": t.nombre, "Día": t.dia, "Mes": calendario.MESES[t.mes - 1].capitalize() if t.mes else mes_siguiente,
                        "Servicio que la activa": t.servicio, "Nota": t.ayuda,
                    }
                    for t in tipos.values()
                ]
            ),
            hide_index=True, width="stretch", num_rows="fixed", key=f"catalogo-{st.session_state.ver}",
            disabled=["Obligación", "Servicio que la activa", "Nota"],
            column_config={
                "Día": st.column_config.NumberColumn(min_value=1, max_value=31, step=1, required=True),
                "Mes": st.column_config.SelectboxColumn(
                    options=[mes_siguiente] + meses_cap, required=True,
                    help="Las mensuales vencen siempre el mes siguiente al período; en las anuales elige el mes.",
                ),
                "Nota": st.column_config.TextColumn(width="large"),
            },
        )
        z1, z2 = st.columns(2)
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

    with st.expander("Importar y exportar", icon=":material/swap_vert:", expanded=not clientes):
        archivo = st.file_uploader(
            "Importar clientes desde una planilla CSV o Excel (columnas rut, razon_social, contacto, email, servicios, honorario…)",
            type=["csv", "xlsx"], key=f"importar-{st.session_state.ver}",
        )
        if archivo and st.button("Importar clientes", type="primary"):
            try:
                df_imp = (
                    pd.read_csv(archivo, dtype=str, sep=None, engine="python") if archivo.name.lower().endswith(".csv")
                    else pd.read_excel(archivo, dtype=str)
                )
                st.session_state.resultado_importacion = intercambio.importar_clientes(db, df_imp)
                refrescar()
            except Exception as error:
                st.error(f"No se pudo importar: {error}")
        if st.session_state.get("resultado_importacion"):
            creados, existentes, errores = st.session_state.pop("resultado_importacion")
            st.success(f"{plural(creados, 'cliente creado', 'clientes creados')}" + (f" · {existentes} ya existían y no se tocaron" if existentes else "") + ".")
            if errores:
                st.warning("Filas que no se pudieron importar:\n\n" + "\n".join(f"- {x}" for x in errores[:30]))
        st.caption("Los clientes que ya existen se omiten: importar dos veces no duplica ni sobrescribe.")
        with st.container(horizontal=True):
            st.download_button("Planilla de ejemplo", intercambio.plantilla_importacion(), "clientes_ejemplo.csv", "text/csv", icon=":material/download:")
            st.download_button(
                "Exportar todo a Excel", data=_exportar, file_name=f"estudio_{hoy:%Y%m%d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", icon=":material/download:",
                disabled=not clientes, help="Clientes, vencimientos, honorarios y pagos. Sirve también como respaldo.",
            )
            if os.path.exists(cfg.ruta_db_ebook) and st.button("Traer clientes del tablero del ebook", icon=":material/menu_book:"):
                n = intercambio.importar_leads_ebook(db, cfg.ruta_db_ebook)
                avisar(f"{plural(n, 'prospecto creado', 'prospectos creados')} desde el ebook" if n else "No hay leads nuevos en la etapa «Cliente»", "📘")
                refrescar()

    with st.expander("Datos de demostración", icon=":material/science:"):
        st.caption("Un estudio ficticio (RUT y correos inventados) para recorrer la plataforma sin cargar clientes reales.")
        with st.container(horizontal=True):
            if st.button("Cargar demo"):
                creados = demo.cargar(db, cfg, hoy)
                avisar(f"{creados} clientes de demostración cargados" if creados else "La demo ya estaba cargada", "🧪")
                refrescar()
            if st.button("Borrar demo"):
                avisar(f"{demo.borrar(db)} clientes de demostración eliminados", "🧹")
                refrescar()

    with st.expander("Datos y seguridad", icon=":material/lock:"):
        st.markdown(
            f"- Los datos se guardan solo en `{cfg.ruta_db}` (ignorado por git). Respáldalo, o usa **Exportar todo a Excel**.\n"
            "- La plataforma **no guarda claves tributarias** ni de Previred, y no se conecta al SII: no las anotes en las notas.\n"
            "- Los mensajes a clientes se abren en tu correo o WhatsApp: la plataforma no envía nada por su cuenta.\n"
            f"- Acceso: {'protegido con clave (`ESTUDIO_CLAVE`).' if cfg.clave else 'sin clave. Define `ESTUDIO_CLAVE` si la app queda accesible desde internet.'}"
        )


{"inicio": pagina_inicio, "clientes": pagina_clientes, "cliente": pagina_cliente, "vencimientos": pagina_vencimientos, "cobros": pagina_cobros, "mas": pagina_mas}[pagina]()
