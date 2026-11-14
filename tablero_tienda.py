"""Tablero de la tienda: qué encontró el Buscador, qué escribió el Posicionador y cómo va el SEO.

Ejecutar:  streamlit run tablero_tienda.py
"""

import os

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tablero de la tienda", page_icon="🛍️", layout="wide")

from agentes_tienda.config import Config, aplicar_secrets_streamlit  # noqa: E402

aplicar_secrets_streamlit()

from agentes_tienda import buscador, divisas, seo  # noqa: E402
from agentes_tienda.fuentes import archivo  # noqa: E402
from agentes_tienda.orquestador import abrir, ejecutar_ciclo, preparar_demo  # noqa: E402
from agentes_tienda.texto import precio_texto  # noqa: E402

ESTADOS = {"publicado": "🟢 Publicado", "candidato": "🟡 Candidato", "pausado": "⏸️ Pausado", "descartado": "⛔ Descartado"}

cfg = Config.desde_entorno()
almacen = abrir(cfg)


def recargar():
    st.rerun()


# ---------------------------------------------------------------------------
# Barra lateral: conexiones y acciones
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🛍️ " + cfg.nombre)
    st.caption(f"Modo **{cfg.modo}** · {cfg.url or 'sin URL pública (TIENDA_URL)'}")
    conexiones = {
        "AliExpress (afiliados)": cfg.aliexpress_configurado,
        "CJ Dropshipping": cfg.cj_configurado,
        "Claude (redacción IA)": cfg.ia_configurada,
        "Google Search Console": cfg.gsc_configurado,
        "IndexNow (Bing y otros)": bool(cfg.indexnow_clave),
    }
    if not cfg.es_afiliado:
        conexiones["Mercado Pago"] = bool(cfg.mercadopago_token)
        conexiones["Correo de pedidos"] = bool(cfg.pedidos_correo)
    for nombre, ok in conexiones.items():
        st.write(("✅ " if ok else "⚪ ") + nombre)

    st.divider()
    sin_ia = st.toggle("No usar IA en este ciclo", value=False)
    if st.button("▶️ Ejecutar agentes ahora", type="primary", width="stretch"):
        with st.spinner("Buscando productos, escribiendo textos y construyendo el sitio…"):
            r = ejecutar_ciclo(almacen, cfg, usar_ia=not sin_ia)
        st.session_state.aviso = r.texto()
        recargar()
    if st.button("🏗️ Solo reconstruir el sitio", width="stretch"):
        r = ejecutar_ciclo(almacen, cfg, buscar=False, posicionar=False)
        st.session_state.aviso = r.texto()
        recargar()

    st.divider()
    with st.expander("📥 Importar CSV / Excel"):
        subido = st.file_uploader("Archivo de productos", type=["csv", "xlsx", "xls"])
        nicho_imp = st.selectbox("Categoría", [n["nombre"] for n in cfg.nichos] + ["Otra…"])
        if nicho_imp == "Otra…":
            nicho_imp = st.text_input("Nombre de la categoría")
        moneda = st.radio("Los costos están en", ["USD", "CLP"], horizontal=True)
        if subido and nicho_imp and st.button("Importar"):
            tabla = archivo.leer(subido.getvalue(), subido.name)
            tc = divisas.tipo_cambio(almacen, cfg)
            crudos = archivo.productos_desde_tabla(tabla, moneda, tc, nicho_imp)
            nuevos = buscador.incorporar(almacen, cfg, crudos, tc, nicho_imp, agente="Importación")
            almacen.guardar()
            st.session_state.aviso = f"Importados {len(crudos)} productos ({len(nuevos)} nuevos). Ejecuta los agentes para publicarlos."
            recargar()
    with st.expander("🧪 Datos de demostración"):
        st.caption("Productos ficticios para ver la tienda sin claves. Mientras existan, el sitio no se indexa.")
        c1, c2 = st.columns(2)
        if c1.button("Cargar demo"):
            st.session_state.aviso = preparar_demo(almacen, cfg).texto()
            recargar()
        if c2.button("Borrar demo"):
            st.session_state.aviso = f"Eliminados {almacen.borrar_demo()} productos de ejemplo."
            almacen.guardar()
            recargar()

if st.session_state.get("aviso"):
    st.success(st.session_state.pop("aviso"))

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
productos = list(almacen.productos.values())
filas = [
    {
        "id": p["id"],
        "Foto": (p.get("imagenes") or [None])[0],
        "Producto": p.get("nombre"),
        "Categoría": p.get("nicho"),
        "Estado": ESTADOS.get(p.get("estado"), p.get("estado")),
        "Score": p.get("score", 0),
        "Precio": p.get("precio_clp") or 0,
        "Ganancia/venta": p.get("ganancia_clp") or 0,
        "Margen": p.get("margen"),
        "Pedidos": p.get("pedidos"),
        "Valoración %": p.get("valoracion"),
        "Envío (días)": p.get("dias_envio"),
        "Interés Google": ((p.get("seo") or {}).get("keywords") or {}).get("interes"),
        "Fuente": p.get("fuente"),
        "Manual": p.get("manual") or "",
        "Motivo": p.get("motivo") or "",
    }
    for p in productos
]
df = pd.DataFrame(filas)
publicados = [p for p in productos if p.get("estado") == "publicado"]
auditoria = almacen.estado.get("auditoria") or {}

st.title("Tablero de la tienda")
if almacen.hay_demo():
    st.warning("Hay productos de demostración: el sitio se publica con *noindex* hasta que los borres.")
if not productos:
    st.info("Aún no hay productos. Configura AliExpress o CJ (ver README), importa un CSV o carga la demo desde la barra lateral.")
    st.stop()

tab_resumen, tab_productos, tab_seo, tab_guias, tab_bitacora = st.tabs(["📊 Resumen", "📦 Productos", "🔎 SEO", "📘 Guías", "🗒️ Bitácora"])

with tab_resumen:
    c = st.columns(5)
    c[0].metric("En vitrina", len(publicados))
    c[1].metric("Candidatos", sum(p.get("estado") == "candidato" for p in productos))
    c[2].metric("Descartados", sum(p.get("estado") == "descartado" for p in productos))
    c[3].metric("Guías de compra", len(almacen.guias))
    c[4].metric("SEO promedio", f"{auditoria.get('promedio', 0)}/100")
    c = st.columns(5)
    ganancias = [p.get("ganancia_clp") or 0 for p in publicados]
    c[0].metric("Ganancia media por venta", precio_texto(sum(ganancias) / len(ganancias)) if ganancias else "—")
    c[1].metric("Score medio en vitrina", round(sum(p.get("score", 0) for p in publicados) / len(publicados), 1) if publicados else "—")
    c[2].metric("Dólar", precio_texto(almacen.estado.get("tipo_cambio") or cfg.tipo_cambio_defecto))
    ultimo = almacen.estado.get("ultimo_ciclo") or ""
    c[3].metric("Último ciclo (UTC)", f"{ultimo[8:10]}/{ultimo[5:7]} {ultimo[11:16]}" if ultimo else "—")
    c[4].metric("Páginas auditadas", auditoria.get("paginas", 0))

    if publicados:
        por_nicho = pd.DataFrame(
            [{"Categoría": p.get("nicho"), "Score": p.get("score", 0), "Ganancia": p.get("ganancia_clp") or 0} for p in publicados]
        )
        resumen = por_nicho.groupby("Categoría").agg(Productos=("Score", "size"), Score=("Score", "mean"), Ganancia=("Ganancia", "mean")).reset_index()
        st.subheader("Vitrina por categoría")
        grafico = (
            alt.Chart(resumen)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Productos:Q", title="Productos publicados", axis=alt.Axis(tickMinStep=1, format="d")),
                y=alt.Y("Categoría:N", sort="-x", title=None),
                tooltip=["Categoría", "Productos", alt.Tooltip("Score:Q", format=".0f"), alt.Tooltip("Ganancia:Q", format=",.0f", title="Ganancia media CLP")],
            )
            .properties(height=36 * len(resumen) + 20)
        )
        st.altair_chart(grafico, width="stretch")
        st.subheader("Top 10 por score")
        top = df[df["id"].isin([p["id"] for p in publicados])].sort_values("Score", ascending=False).head(10)
        st.dataframe(top[["Foto", "Producto", "Categoría", "Score", "Precio", "Ganancia/venta", "Pedidos", "Valoración %"]],
                     hide_index=True, width="stretch",
                     column_config={"Foto": st.column_config.ImageColumn(width="small"),
                                    "Precio": st.column_config.NumberColumn(format="$%d"),
                                    "Ganancia/venta": st.column_config.NumberColumn(format="$%d")})

with tab_productos:
    f1, f2, f3 = st.columns([1, 1, 2])
    estados = f1.multiselect("Estado", list(ESTADOS.values()), default=[ESTADOS["publicado"], ESTADOS["candidato"]])
    nichos = f2.multiselect("Categoría", sorted(df["Categoría"].dropna().unique()))
    texto = f3.text_input("Buscar", placeholder="nombre del producto")
    vista = df[df["Estado"].isin(estados)] if estados else df
    if nichos:
        vista = vista[vista["Categoría"].isin(nichos)]
    if texto:
        vista = vista[vista["Producto"].str.contains(texto, case=False, na=False)]
    vista = vista.sort_values("Score", ascending=False)
    seleccion = st.dataframe(
        vista.drop(columns=["id"]), hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        column_config={
            "Foto": st.column_config.ImageColumn(width="small"),
            "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
            "Precio": st.column_config.NumberColumn(format="$%d"),
            "Ganancia/venta": st.column_config.NumberColumn(format="$%d"),
            "Margen": st.column_config.NumberColumn(format="percent"),
        },
    )
    filas_sel = seleccion.selection.rows if seleccion and seleccion.selection else []
    if filas_sel:
        p = almacen.productos[vista.iloc[filas_sel[0]]["id"]]
        st.subheader(p.get("nombre"))
        izq, der = st.columns([1, 2])
        with izq:
            if p.get("imagenes"):
                st.image(p["imagenes"][0], width="stretch")
            if p.get("slug"):
                st.markdown(f"[Ver página en la tienda]({cfg.url_base}/p/{p['slug']}/)")
            if p.get("url_proveedor"):
                st.markdown(f"[Ver en el proveedor]({p['url_proveedor']})")
        with der:
            st.caption(f"Título original: {p.get('titulo_original')}")
            detalle = p.get("score_detalle") or {}
            st.write("**Score:** " + " · ".join(f"{k} {v}" for k, v in detalle.items()) + f" = **{p.get('score')}**")
            st.write(f"**Precio:** {precio_texto(p.get('precio_clp'))} · **Costo:** US$ {p.get('costo_usd')} · "
                     f"**Ganancia por venta:** {precio_texto(p.get('ganancia_clp'))}")
            if p.get("motivo"):
                st.write(f"**Motivo:** {p['motivo']}")
            s = p.get("seo") or {}
            if s:
                st.write(f"**Palabra clave:** {(s.get('keywords') or {}).get('principal')} · **Textos:** {s.get('origen')} (v{s.get('version')})")
                st.write(f"**Título SEO:** {s.get('titulo_seo')}")
                st.write(f"**Meta:** {s.get('meta_descripcion')}")
            if p.get("gsc"):
                g = p["gsc"]
                st.write(f"**Google (28 días):** {g['impresiones']} impresiones · {g['clics']} clics · CTR {g['ctr']:.1%} · posición {g['posicion']}")
            b = st.columns(4)
            accion = None
            if b[0].button("🟢 Publicar"):
                accion = "publicado"
            if b[1].button("⏸️ Pausar"):
                accion = "pausado"
            if b[2].button("⛔ Descartar"):
                accion = "descartado"
            if b[3].button("🤖 Decide el agente"):
                accion = "auto"
            if accion == "auto":
                p["manual"] = None
                if p.get("estado") != "publicado":
                    p["estado"], p["motivo"] = "candidato", ""
            elif accion:
                p["manual"], p["estado"], p["motivo"] = accion, accion, f"{accion} a mano"
                if accion == "publicado":
                    seo._asegurar_slug(almacen, p)
            if accion:
                almacen.log("Tablero", f"Cambio manual: {accion}", p.get("nombre"), p["id"])
                almacen.guardar()
                st.session_state.aviso = f"«{p.get('nombre')}»: {accion}. Reconstruye el sitio para verlo publicado."
                recargar()
            st.caption("Los cambios se ven en el sitio al reconstruirlo (botón de la barra lateral o el próximo ciclo automático).")

with tab_seo:
    st.write(f"Última auditoría: {(auditoria.get('fecha') or '—')[:16].replace('T', ' ')} · {auditoria.get('paginas', 0)} páginas · promedio **{auditoria.get('promedio', 0)}/100**")
    problemas = auditoria.get("con_problemas") or []
    if problemas:
        st.dataframe(
            pd.DataFrame([{"Página": x["ruta"], "Tipo": x["tipo"], "Puntaje": x["puntaje"], "Problemas": "; ".join(x["problemas"]), "Palabras": x.get("palabras")} for x in problemas]),
            hide_index=True, width="stretch",
        )
    else:
        st.success("Ninguna página indexable con problemas.")
    st.subheader("Palabras clave por producto")
    kw = [
        {"Producto": p.get("nombre"), "Palabra clave": ((p.get("seo") or {}).get("keywords") or {}).get("principal"),
         "Interés en Google": ((p.get("seo") or {}).get("keywords") or {}).get("interes"),
         "Relacionadas": ", ".join((((p.get("seo") or {}).get("keywords") or {}).get("secundarias") or [])[:5]),
         "Textos": (p.get("seo") or {}).get("origen")}
        for p in publicados
    ]
    st.dataframe(pd.DataFrame(kw), hide_index=True, width="stretch")
    con_gsc = [p for p in productos if p.get("gsc")]
    if con_gsc:
        st.subheader("Rendimiento en Google (Search Console)")
        st.dataframe(pd.DataFrame([{"Producto": p["nombre"], **p["gsc"]} for p in con_gsc]).sort_values("impresiones", ascending=False),
                     hide_index=True, width="stretch")
    elif not cfg.gsc_configurado:
        st.info("Conecta Google Search Console (GSC_CREDENCIALES) para que el agente aprenda de las búsquedas reales y reoptimice las páginas.")

with tab_guias:
    if not almacen.guias:
        st.info("Las guías se crean cuando una categoría tiene al menos 3 productos publicados.")
    for g in sorted(almacen.guias.values(), key=lambda g: g.get("nombre_nicho", "")):
        with st.expander(f"📘 {g['titulo']}"):
            st.write(f"[Abrir guía]({cfg.url_base}/guias/{g['slug']}/) · textos: {g.get('origen')} · actualizada {(g.get('actualizado') or '')[:10]}")
            st.write(g.get("meta_descripcion"))

with tab_bitacora:
    eventos = almacen.bitacora(500)
    if eventos:
        st.dataframe(pd.DataFrame(eventos)[["fecha", "agente", "accion", "detalle", "ref"]], hide_index=True, width="stretch")

if os.path.isdir(cfg.carpeta_sitio):
    st.caption(f"Sitio generado en `{cfg.carpeta_sitio}/`. Vista previa local: `python run_tienda.py --construir --servir`.")
