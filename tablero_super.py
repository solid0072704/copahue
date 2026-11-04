"""Agente de compras de supermercado: compara precios, optimiza y automatiza compras recurrentes.

Ejecutar:  streamlit run tablero_super.py
"""

from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Agente de compras", page_icon="🛒", layout="wide")

from agentes_super.config import ConfigSuper, aplicar_secrets_streamlit  # noqa: E402

aplicar_secrets_streamlit()

from agentes_super import conectores as cx  # noqa: E402
from agentes_super.agente import (  # noqa: E402
    Cotizador,
    abrir_db,
    cotizar_lista,
    detalle_plan,
    ejecutar_ciclo,
    marcar_comprada,
    preparar_demo,
)
from agentes_super.db import FRECUENCIAS  # noqa: E402
from agentes_super.emparejar import coincide  # noqa: E402
from agentes_super.exportar import excel_compra, tabla_compra, texto_whatsapp  # noqa: E402
from agentes_super.unidades import NOMBRE_BASE, pesos, precio_por_unidad, texto_formato  # noqa: E402

cfg = ConfigSuper.desde_entorno()


@st.cache_resource
def _db(ruta):
    return abrir_db(cfg)


db = _db(cfg.ruta_db)
def md(texto):
    """Escapa el signo $ (Streamlit lo toma como fórmula matemática)."""
    return str(texto).replace("$", "\\$")


ESTADOS = {"propuesta": "🕒 Por comprar", "comprada": "✅ Comprada", "descartada": "⏭️ Descartada"}


def aviso_ciclo(resultado):
    st.toast(f"Agente: {resultado.texto()}", icon="🛒")
    for e in resultado.errores:
        st.warning(e)


FORMATO_PESOS = "localized"  # $ 12.990 según el idioma del navegador


def mostrar_detalle(detalle, clave):
    """Muestra un plan de compra: una tabla por supermercado con su botón de carrito."""
    if not detalle.get("tiendas"):
        st.info("No se encontraron productos para esta lista en los supermercados activos.")
        return
    columnas = st.columns(len(detalle["tiendas"]))
    for col, t in zip(columnas, detalle["tiendas"]):
        with col:
            st.markdown(md(f"**{t['tienda']}** · {pesos(t['subtotal'])}"))
            st.caption(md(f"Despacho {pesos(t['despacho'])}") if t["despacho"] else "Despacho gratis / retiro")
            if t.get("carrito"):
                st.link_button(f"🛒 Abrir carrito en {t['tienda']}", t["carrito"], use_container_width=True)
    df = tabla_compra(detalle)
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Precio": st.column_config.NumberColumn(format=FORMATO_PESOS),
            "Total": st.column_config.NumberColumn(format=FORMATO_PESOS),
            "Precio por unidad": st.column_config.NumberColumn(format=FORMATO_PESOS),
            "Enlace": st.column_config.LinkColumn(display_text="ver"),
        },
    )
    if detalle.get("faltantes"):
        st.warning("No encontrados: " + ", ".join(detalle["faltantes"]) + ". Prueba otra palabra o quita filtros.")
    if detalle.get("errores"):
        with st.expander("Supermercados con problemas"):
            for e in detalle["errores"]:
                st.write("•", e)


def mostrar_comparacion(comparacion, total_plan=None):
    if not comparacion:
        return
    df = pd.DataFrame(comparacion).rename(
        columns={"tienda": "Supermercado", "encontrados": "Encontrados", "faltan": "Faltan",
                 "productos": "Productos", "despacho": "Despacho", "total": "Total"}
    )
    if total_plan:
        df["Pagas de más"] = df["Total"] - total_plan
    st.dataframe(
        df, hide_index=True, use_container_width=True,
        column_config={c: st.column_config.NumberColumn(format=FORMATO_PESOS)
                       for c in ("Productos", "Despacho", "Total", "Pagas de más")},
    )


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛒 Agente de compras")
    activas = db.tiendas(solo_activas=True)
    st.caption(f"{len(activas)} supermercados activos · máx. {cfg.max_tiendas} por compra")
    if st.button("▶️ Ejecutar agente ahora", type="primary", use_container_width=True, disabled=not activas):
        with st.spinner("Cotizando en los supermercados…"):
            aviso_ciclo(ejecutar_ciclo(db, cfg))
    st.caption(f"Última revisión: {db.get_meta('ultimo_ciclo', 'nunca')}")
    st.divider()
    st.markdown(
        f"{'✅' if cfg.gmail_configurado else '⚪'} Avisos por correo"
        f"{' a ' + cfg.notificar_a if cfg.gmail_configurado else ' (configura GMAIL_USER)'}  \n"
        f"{'✅' if cfg.ia_configurada else '⚪'} IA para descartar productos que no son"
    )
    st.divider()
    with st.expander("🧪 Datos de demostración"):
        st.caption("4 supermercados ficticios, 2 listas recurrentes y 8 semanas de historial de precios.")
        if st.button("Cargar demo", use_container_width=True):
            with st.spinner("Preparando demo…"):
                aviso_ciclo(preparar_demo(db, cfg))
            st.rerun()

if not db.tiendas(solo_activas=True):
    st.info(
        "👋 Para empezar, activa tus supermercados en la pestaña **🏪 Supermercados** "
        "o carga la **demo** desde la barra lateral."
    )

tab_compras, tab_listas, tab_buscar, tab_ofertas, tab_tiendas, tab_bitacora = st.tabs(
    ["🧾 Compras", "📝 Mis listas", "🔎 Comparar un producto", "🔥 Ofertas e historial", "🏪 Supermercados", "📜 Bitácora"]
)

# ---------------------------------------------------------------------------
# Compras preparadas por el agente
# ---------------------------------------------------------------------------
with tab_compras:
    compras = db.compras()
    pendientes = [c for c in compras if c["estado"] == "propuesta"]
    hechas = [c for c in compras if c["estado"] == "comprada"]
    listas = db.listas(solo_activas=True)
    proximas = sorted((l["proxima_fecha"], l["nombre"]) for l in listas if l["frecuencia"] != "unica")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Compras por hacer", len(pendientes), help="Preparadas por el agente, esperando que compres")
    m2.metric("Por pagar", pesos(sum(c["total"] for c in pendientes)))
    m3.metric("Ahorro acumulado", pesos(sum(c["ahorro"] for c in hechas)),
              help="En compras realizadas, frente al precio promedio entre supermercados")
    m4.metric("Próxima compra", date.fromisoformat(proximas[0][0]).strftime("%d/%m/%Y") if proximas else "—",
              help=f"Lista: {proximas[0][1]}" if proximas else None)

    if not compras:
        st.info("Aún no hay compras. Crea una lista en **📝 Mis listas** y presiona **💰 Cotizar ahora**, "
                "o deja una lista recurrente: el agente la prepara antes de cada fecha.")
    for c in compras[:30]:
        lista = db.lista(c["lista_id"]) or {"nombre": "(lista borrada)"}
        titulo = (f"{ESTADOS.get(c['estado'], c['estado'])} · **{lista['nombre']}** · {c['fecha_objetivo']} · "
                  f"{pesos(c['total'])} en {c['tiendas']} · ahorro {pesos(c['ahorro'])}")
        with st.expander(md(titulo), expanded=c["estado"] == "propuesta" and c is pendientes[0]):
            d = c["detalle"]
            a, b, cc, dd = st.columns(4)
            a.metric("Total", pesos(c["total"]))
            b.metric("Productos", pesos(d.get("productos")))
            cc.metric("Despacho", pesos(d.get("despacho")))
            dd.metric("Ahorro vs. promedio", pesos(c["ahorro"]),
                      help=md(f"Además, {pesos(d.get('ahorro_ofertas'))} en descuentos de los supermercados"))
            mostrar_detalle(d, f"c{c['id']}")
            with st.expander("Si compraras todo en un solo supermercado"):
                mostrar_comparacion(d.get("comparacion"), c["total"])
            botones = st.columns(5)
            if c["estado"] == "propuesta":
                if botones[0].button("✅ Ya la compré", key=f"ok{c['id']}"):
                    marcar_comprada(db, c["id"])
                    st.rerun()
                if botones[1].button("🔄 Recotizar", key=f"re{c['id']}"):
                    with st.spinner("Actualizando precios…"):
                        cotizar_lista(db, cfg, c["lista_id"], fecha_objetivo=c["fecha_objetivo"])
                    st.rerun()
                if botones[2].button("⏭️ Descartar", key=f"no{c['id']}"):
                    db.actualizar_compra(c["id"], estado="descartada")
                    st.rerun()
            botones[3].download_button("⬇️ Excel", excel_compra(d), f"compra_{c['fecha_objetivo']}.xlsx",
                                       key=f"xl{c['id']}")
            with botones[4].popover("💬 WhatsApp"):
                st.code(texto_whatsapp(lista["nombre"], d), language=None)

# ---------------------------------------------------------------------------
# Listas de compra
# ---------------------------------------------------------------------------
with tab_listas:
    todas = db.listas()
    opciones = {l["id"]: f"{l['nombre']} ({FRECUENCIAS.get(l['frecuencia'], '')})" for l in todas}
    col_sel, col_nueva = st.columns([3, 2])
    with col_nueva.popover("➕ Nueva lista", use_container_width=True):
        with st.form("nueva_lista", clear_on_submit=True):
            nombre = st.text_input("Nombre", placeholder="Despensa semanal")
            frecuencia = st.selectbox("Se repite", list(FRECUENCIAS), format_func=FRECUENCIAS.get, index=1)
            fecha = st.date_input("Primera compra", date.today() + timedelta(days=1))
            if st.form_submit_button("Crear", type="primary") and nombre.strip():
                st.session_state.lista_sel = db.crear_lista(nombre.strip(), frecuencia, fecha, cfg.max_tiendas)
                st.rerun()
    if not todas:
        st.info("Crea tu primera lista con **➕ Nueva lista**.")
    else:
        ids = list(opciones)
        actual = st.session_state.get("lista_sel")
        lista_id = col_sel.selectbox("Lista", ids, format_func=opciones.get,
                                     index=ids.index(actual) if actual in ids else 0, label_visibility="collapsed")
        st.session_state.lista_sel = lista_id
        lista = db.lista(lista_id)

        c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 1])
        nuevo_nombre = c1.text_input("Nombre", lista["nombre"])
        frecuencia = c2.selectbox("Se repite", list(FRECUENCIAS), format_func=FRECUENCIAS.get,
                                  index=list(FRECUENCIAS).index(lista["frecuencia"]))
        proxima = c3.date_input("Próxima compra", date.fromisoformat(lista["proxima_fecha"]))
        max_t = c4.number_input("Máx. supermercados", 1, 6, int(lista["max_tiendas"] or cfg.max_tiendas),
                                help="1 = todo en un solo lugar. 2 o más = el agente divide la compra si conviene "
                                     "(considerando el despacho de cada uno).")
        activa = c5.toggle("Activa", bool(lista["activa"]))

        st.markdown("**Productos** · escribe lo que buscas como lo buscarías en el súper. "
                    "Con *formato* (1 L, 500 g, 12 un) el agente compara por precio por litro/kilo/unidad "
                    "y puede elegir otro tamaño si sale más barato.")
        items = db.items(lista_id)
        df_items = pd.DataFrame(items or [{"consulta": "", "cantidad": 1.0, "formato": "", "marcas": "", "excluir": ""}])
        editado = st.data_editor(
            df_items[["consulta", "cantidad", "formato", "marcas", "excluir"]],
            num_rows="dynamic",
            use_container_width=True,
            key=f"items_{lista_id}",
            column_config={
                "consulta": st.column_config.TextColumn("Producto", required=True, help="Ej.: leche entera, arroz grado 2"),
                "cantidad": st.column_config.NumberColumn("Cantidad", min_value=0.1, step=1.0, default=1.0),
                "formato": st.column_config.TextColumn("Formato (opcional)", help="Tamaño de referencia: 1 L, 500 g, 12 un"),
                "marcas": st.column_config.TextColumn("Solo marcas", help="Separadas por coma. Vacío = cualquier marca"),
                "excluir": st.column_config.TextColumn("Excluir palabras", help="Ej.: chocolate, sin lactosa"),
            },
        )
        b1, b2, b3, _ = st.columns([2, 2, 2, 3])
        if b1.button("💾 Guardar lista", use_container_width=True):
            db.actualizar_lista(lista_id, nombre=nuevo_nombre.strip() or lista["nombre"], frecuencia=frecuencia,
                                proxima_fecha=proxima, max_tiendas=int(max_t), activa=int(activa))
            db.reemplazar_items(lista_id, editado.fillna("").to_dict("records"))
            st.toast("Lista guardada", icon="💾")
            st.rerun()
        cotizar = b2.button("💰 Cotizar ahora", type="primary", use_container_width=True,
                            disabled=not db.tiendas(solo_activas=True))
        if b3.button("🗑️ Borrar lista", use_container_width=True):
            db.borrar_lista(lista_id)
            st.session_state.pop("lista_sel", None)
            st.rerun()

        if cotizar:
            db.actualizar_lista(lista_id, max_tiendas=int(max_t))
            db.reemplazar_items(lista_id, editado.fillna("").to_dict("records"))
            with st.spinner("Buscando el mejor precio en cada supermercado…"):
                compra_id, cot = cotizar_lista(db, cfg, lista_id, fecha_objetivo=proxima.isoformat())
            st.session_state[f"cot_{lista_id}"] = cot
        cot = st.session_state.get(f"cot_{lista_id}")
        if cot:
            st.divider()
            if not cot.plan:
                st.error("No se encontró ningún producto. Revisa los nombres o los supermercados activos.")
            else:
                p = cot.plan
                st.subheader(md(f"Mejor opción: {pesos(p.total)} en {' + '.join(p.tiendas)}"))
                st.caption(md(f"Ahorras {pesos(p.ahorro_vs_promedio)} frente al precio promedio entre supermercados. "
                           "Quedó guardada en 🧾 Compras."))
                mostrar_detalle(detalle_plan(cot), f"l{lista_id}")
                st.markdown("**Si compraras todo en un solo supermercado**")
                mostrar_comparacion(cot.comparacion, p.total)
                with st.expander("🔍 Todas las opciones encontradas por producto"):
                    filas = []
                    for item in cot.items:
                        for o in cot.opciones.get(item["id"], []):
                            filas.append({
                                "Ítem": item["consulta"], "Supermercado": o.tienda, "Producto": o.producto.nombre,
                                "Envases": o.envases, "Pagas": o.costo, "Equivale a lo pedido": round(o.comparable),
                                "Elegido": "⭐" if p.asignacion.get(item["id"]) is o else "",
                            })
                    st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True,
                                 column_config={"Pagas": st.column_config.NumberColumn(format=FORMATO_PESOS),
                                                "Equivale a lo pedido": st.column_config.NumberColumn(format=FORMATO_PESOS)})

# ---------------------------------------------------------------------------
# Búsqueda puntual
# ---------------------------------------------------------------------------
with tab_buscar:
    with st.form("buscar"):
        q1, q2 = st.columns([4, 1])
        consulta = q1.text_input("¿Qué producto quieres comparar?", placeholder="aceite maravilla 1 L")
        enviar = q2.form_submit_button("🔎 Comparar", use_container_width=True)
    if enviar and consulta.strip():
        cotizador = Cotizador(db, cfg, usar_ia=False)
        filas = []
        with st.spinner("Buscando…"):
            for conector in cotizador.conectores:
                for p in cotizador.buscar(conector, consulta.strip()):
                    if not coincide(consulta, p.nombre) and not any(c.isdigit() for c in consulta):
                        continue
                    filas.append({
                        "Supermercado": p.tienda, "Producto": p.nombre, "Marca": p.marca, "Precio": p.precio,
                        "Normal": p.precio_normal, "Formato": texto_formato(p.cantidad, p.unidad),
                        "Precio por kg/L/un": precio_por_unidad(p.precio, p.cantidad, p.unidad),
                        "Unidad": NOMBRE_BASE.get(p.unidad, ""), "Enlace": p.url,
                    })
        for e in cotizador.errores:
            st.warning(e)
        if filas:
            df = pd.DataFrame(filas).sort_values(["Precio por kg/L/un", "Precio"], na_position="last")
            st.dataframe(df, hide_index=True, use_container_width=True, column_config={
                "Precio": st.column_config.NumberColumn(format=FORMATO_PESOS),
                "Normal": st.column_config.NumberColumn(format=FORMATO_PESOS),
                "Precio por kg/L/un": st.column_config.NumberColumn(format=FORMATO_PESOS),
                "Enlace": st.column_config.LinkColumn(display_text="ver"),
            })
        elif enviar:
            st.info("Sin resultados en los supermercados activos.")

# ---------------------------------------------------------------------------
# Ofertas e historial
# ---------------------------------------------------------------------------
with tab_ofertas:
    alertas = db.alertas(limite=50)
    st.subheader("🔥 Buen momento para abastecerse")
    st.caption(f"El agente avisa cuando el mejor precio de hoy está al menos {cfg.umbral_oferta:.0%} bajo "
               "la mediana de los últimos 30 días (se necesitan 3 días de historial).")
    if alertas:
        st.dataframe(
            pd.DataFrame([{
                "Fecha": a["fecha"], "Producto": a["consulta"], "Dónde": a["tienda"], "Oferta": a["nombre"],
                "Hoy": round(a["precio_unidad"]), "Habitual": round(a["mediana"]),
                "Descuento": f"{1 - a['precio_unidad'] / a['mediana']:.0%}", "Enlace": a["url"],
            } for a in alertas]),
            hide_index=True, use_container_width=True,
            column_config={"Hoy": st.column_config.NumberColumn(format=FORMATO_PESOS),
                           "Habitual": st.column_config.NumberColumn(format=FORMATO_PESOS),
                           "Enlace": st.column_config.LinkColumn(display_text="ver")},
        )
    else:
        st.info("Sin ofertas destacadas por ahora.")

    st.subheader("📈 Historial de precios")
    consultas = db.consultas_con_historial()
    if consultas:
        elegida = st.selectbox("Producto", consultas)
        hist = pd.DataFrame(db.historial(elegida))
        hist["precio_unidad"] = [precio_por_unidad(r.precio, r.cantidad, r.unidad) for r in hist.itertuples()]
        hist = hist.dropna(subset=["precio_unidad"])
        if hist.empty:
            st.info("Estos productos no informan su tamaño, así que no se pueden comparar por unidad.")
        else:
            unidad = NOMBRE_BASE.get(hist["unidad"].mode()[0], "un")
            hist = hist[hist["unidad"] == hist["unidad"].mode()[0]]
            diario = hist.groupby(["fecha", "tienda"], as_index=False)["precio_unidad"].min()
            grafico = (
                alt.Chart(diario)
                .mark_line(point=True)
                .encode(
                    x=alt.X("fecha:T", title=None),
                    y=alt.Y("precio_unidad:Q", title=f"Mejor precio por {unidad}", scale=alt.Scale(zero=False)),
                    color=alt.Color("tienda:N", title="Supermercado"),
                    tooltip=[alt.Tooltip("fecha:T", title="Fecha"), alt.Tooltip("tienda:N", title="Supermercado"),
                             alt.Tooltip("precio_unidad:Q", title=f"$ por {unidad}", format=",.0f")],
                )
                .properties(height=320)
            )
            st.altair_chart(grafico, use_container_width=True)
    else:
        st.info("El historial se arma solo cada vez que el agente cotiza.")

# ---------------------------------------------------------------------------
# Supermercados
# ---------------------------------------------------------------------------
with tab_tiendas:
    st.markdown(
        "Activa los supermercados donde compras e indica el **costo de despacho**, desde qué monto es "
        "**gratis** y el **mínimo de compra**: el optimizador los usa para decidir si conviene dividir la compra. "
        "Los conectores **VTEX** leen el catálogo público de la tienda y arman el carrito por ti."
    )
    tiendas = db.tiendas()
    if tiendas:
        df_t = pd.DataFrame(tiendas)[["nombre", "tipo", "url", "activa", "despacho", "gratis_desde", "minimo",
                                      "ultimo_estado"]]
        df_t["activa"] = df_t["activa"].astype(bool)
        editado_t = st.data_editor(
            df_t, hide_index=True, use_container_width=True, disabled=["nombre", "tipo", "ultimo_estado"],
            column_config={
                "nombre": "Supermercado", "tipo": "Conector", "url": st.column_config.TextColumn("Sitio"),
                "activa": st.column_config.CheckboxColumn("Activo"),
                "despacho": st.column_config.NumberColumn("Despacho", format=FORMATO_PESOS, min_value=0),
                "gratis_desde": st.column_config.NumberColumn("Gratis desde", format=FORMATO_PESOS, min_value=0,
                                                              help="0 = nunca gratis"),
                "minimo": st.column_config.NumberColumn("Compra mínima", format=FORMATO_PESOS, min_value=0),
                "ultimo_estado": "Última prueba",
            },
        )
        g1, g2, g3 = st.columns(3)
        if g1.button("💾 Guardar cambios", use_container_width=True):
            for fila in editado_t.to_dict("records"):
                db.actualizar_tienda(fila["nombre"], url=fila["url"] or "", activa=fila["activa"],
                                     despacho=fila["despacho"] or 0, gratis_desde=fila["gratis_desde"] or 0,
                                     minimo=fila["minimo"] or 0)
            st.toast("Supermercados guardados", icon="💾")
            st.rerun()
        if g2.button("🔌 Probar conexión de los activos", use_container_width=True):
            cotizador = Cotizador(db, cfg, usar_ia=False)
            for conector in cotizador.conectores:
                ok, mensaje = cx.probar_conector(conector)
                db.actualizar_tienda(conector.nombre, ultimo_estado=("✅ " if ok else "❌ ") + mensaje[:120])
                (st.success if ok else st.error)(f"{conector.nombre}: {mensaje}")
        borrar = g3.selectbox("Quitar supermercado", ["—"] + [t["nombre"] for t in tiendas])
        if borrar != "—" and g3.button(f"🗑️ Quitar {borrar}"):
            db.borrar_tienda(borrar)
            st.rerun()

    with st.expander("➕ Agregar un supermercado con tienda VTEX"):
        st.caption("Muchas tiendas en línea de Latinoamérica usan VTEX. Pega la dirección del sitio y prueba la "
                   "conexión: si responde, el agente puede comparar sus precios y armar tu carrito.")
        with st.form("nueva_tienda", clear_on_submit=True):
            n1, n2 = st.columns(2)
            nombre_t = n1.text_input("Nombre", placeholder="Mi súper")
            url_t = n2.text_input("Sitio", placeholder="https://www.misuper.cl")
            n3, n4, n5 = st.columns(3)
            desp = n3.number_input("Despacho", 0, 50000, 2990, step=500)
            gratis = n4.number_input("Gratis desde", 0, 500000, 0, step=5000)
            minimo = n5.number_input("Compra mínima", 0, 500000, 0, step=5000)
            if st.form_submit_button("Agregar y probar") and nombre_t.strip() and url_t.strip():
                db.guardar_tienda(nombre_t.strip(), "vtex", url_t.strip(), True, desp, gratis, minimo)
                ok, mensaje = cx.probar_conector(cx.ConectorVTEX(nombre_t.strip(), url_t.strip()))
                db.actualizar_tienda(nombre_t.strip(), ultimo_estado=("✅ " if ok else "❌ ") + mensaje[:120])
                (st.success if ok else st.error)(mensaje)

    with st.expander("📄 Cargar una lista de precios (CSV o Excel)"):
        st.caption("Para supermercados sin tienda en línea compatible: columnas **tienda, producto, precio** "
                   "(opcionales: marca, precio_normal, url). Cada tienda del archivo queda como supermercado.")
        archivo = st.file_uploader("Archivo", type=["csv", "xlsx", "xls"])
        if archivo:
            try:
                productos = cx.leer_lista_precios(archivo)
            except ValueError as e:
                st.error(str(e))
            else:
                ruta = cx.ruta_lista_precios()
                pd.DataFrame([{"tienda": p.tienda, "producto": p.nombre, "precio": p.precio, "marca": p.marca,
                               "precio_normal": p.precio_normal, "url": p.url, "sku": p.sku} for p in productos]
                             ).to_csv(ruta, index=False)
                for nombre_t in sorted({p.tienda for p in productos}):
                    db.guardar_tienda(nombre_t, "csv", activa=True)
                st.success(f"{len(productos)} precios cargados de {len({p.tienda for p in productos})} supermercados.")

# ---------------------------------------------------------------------------
# Bitácora
# ---------------------------------------------------------------------------
with tab_bitacora:
    registros = db.bitacora()
    if registros:
        st.dataframe(pd.DataFrame(registros)[["fecha", "agente", "accion", "detalle"]], hide_index=True,
                     use_container_width=True)
    else:
        st.info("Aquí verás lo que hace el agente.")
