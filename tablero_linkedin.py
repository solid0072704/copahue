"""Prospector de LinkedIn: tu cola diaria de invitaciones y mensajes para ofrecer finanzas fraccionales.

Ejecutar:  streamlit run tablero_linkedin.py
"""

import json
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prospector de LinkedIn", page_icon="💼", layout="wide")

from agentes_ebook.config import aplicar_secrets_streamlit  # noqa: E402

aplicar_secrets_streamlit()

from agentes_ebook.db import ahora_utc, desde_iso  # noqa: E402
from prospector_linkedin import ConfigProspector, abrir_db, ejecutar_ciclo, preparar_demo  # noqa: E402
from prospector_linkedin import calificador, fuentes, secuencia  # noqa: E402
from prospector_linkedin.db import ETAPAS  # noqa: E402
from prospector_linkedin.llm import crear_asistente  # noqa: E402
from prospector_linkedin.mensajes import nombre_tipo, redactar  # noqa: E402

try:
    OSCURO = st.context.theme.type == "dark"
except Exception:
    OSCURO = False
AZUL = "#3987e5" if OSCURO else "#2a78d6"
TINTA = "#c3c2b7" if OSCURO else "#52514e"
AGENTES = ("Prospector", "Fuentes", "Calificador", "Secuencia", "Sincronización", "Avisos LinkedIn")

cfg = ConfigProspector.desde_entorno()
db = abrir_db(cfg)


def hace(texto_iso):
    if not texto_iso:
        return "nunca"
    minutos = int((ahora_utc() - desde_iso(texto_iso)).total_seconds() // 60)
    if minutos < 1:
        return "recién"
    if minutos < 60:
        return f"hace {minutos} min"
    if minutos < 60 * 24:
        return f"hace {minutos // 60} h"
    return f"hace {minutos // (60 * 24)} días"


def local(texto_iso, formato="%d/%m/%Y"):
    dt = desde_iso(texto_iso)
    return dt.astimezone(cfg.tz).strftime(formato) if dt else ""


def actualizar(**kwargs):
    with st.spinner("Calificando prospectos, revisando avisos de LinkedIn en tu Gmail y armando tu cola…"):
        resultado = ejecutar_ciclo(db, cfg, **kwargs)
    st.session_state.li_resultado = resultado.texto()
    st.session_state.li_errores = list(resultado.errores)


def replanificar():
    """Tras un cambio tuyo (marcar enviado, importar, etc.): recalcula sin leer Gmail ni enviar correos."""
    calificador.calificar(db, cfg)
    secuencia.planificar(db, cfg)
    st.rerun()


if "li_auto" not in st.session_state:
    st.session_state.li_auto = True
    ultima = db.get_meta("li_ultima_ejecucion")
    if db.prospectos() and (not ultima or ahora_utc() - desde_iso(ultima) > timedelta(minutes=15)):
        actualizar()

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("💼 Prospector de LinkedIn")
    st.caption(f"{cfg.servicio} · {cfg.base.empresa_nombre}")

    if st.button("🔄 Actualizar ahora", type="primary", width="stretch"):
        actualizar()
        st.rerun()
    st.caption(f"Última actualización: {hace(db.get_meta('li_ultima_ejecucion'))}")
    if st.session_state.get("li_resultado"):
        st.info(st.session_state.li_resultado)
    for error in st.session_state.pop("li_errores", []):
        st.error(error)

    st.divider()
    st.subheader("Tu ritmo de hoy")
    c = secuencia.cupos(db, cfg)
    st.progress(
        min(1.0, c["invitaciones_hoy"] / cfg.invitaciones_dia) if cfg.invitaciones_dia else 0.0,
        text=f"Invitaciones: {c['invitaciones_hoy']} de {cfg.invitaciones_dia} hoy · {c['invitaciones_semana']} de {cfg.invitaciones_semana} en 7 días",
    )
    st.progress(
        min(1.0, c["mensajes_hoy"] / cfg.mensajes_dia) if cfg.mensajes_dia else 0.0,
        text=f"Mensajes: {c['mensajes_hoy']} de {cfg.mensajes_dia} hoy",
    )
    st.caption("Límites conservadores para cuidar tu cuenta. Se cambian en `.env` (`LI_INVITACIONES_DIA`, etc.).")

    st.divider()
    st.subheader("Conexiones")
    gmail = cfg.base.gmail_configurado
    st.markdown(
        f"{'✅' if gmail else '➖'} **Gmail**: "
        + ("detecta quién aceptó y quién respondió" if gmail else "sin configurar (marcas a mano)")
    )
    st.markdown(f"{'✅' if cfg.base.ia_configurada else '➖'} **IA (Claude)**: {'personaliza los textos' if cfg.base.ia_configurada else 'opcional, usa plantillas'}")
    st.markdown(f"{'✅' if cfg.base.link_agenda else '➖'} **Link de agenda**: {'configurado' if cfg.base.link_agenda else 'sin configurar'}")

    with st.expander("ℹ️ ¿Qué es automático y qué no?"):
        st.markdown(
            "**Automático:** calificar prospectos, redactar cada invitación y mensaje, decidir a quién le toca hoy, "
            "respetar tus límites, detectar aceptaciones y respuestas (vía los avisos de LinkedIn en tu Gmail) "
            "y enviarte la cola diaria por correo.\n\n"
            "**Lo haces tú (un clic):** enviar la invitación o el mensaje en LinkedIn. LinkedIn prohíbe los "
            "bots que operan tu cuenta y restringe los perfiles que los usan; como es tu perfil personal, "
            "este prospector nunca inicia sesión ni actúa por ti allí."
        )
    with st.expander("🧪 Datos de demostración"):
        st.caption("Prospectos ficticios para conocer el tablero. Sus perfiles no existen.")
        d1, d2 = st.columns(2)
        if d1.button("Cargar demo"):
            preparar_demo(db, cfg)
            st.rerun()
        if d2.button("Borrar demo"):
            st.toast(f"{db.borrar_fuente_li('demo')} prospectos demo eliminados")
            st.rerun()

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
prospectos = db.prospectos()
ahora = ahora_utc()

st.title("💼 Prospector de LinkedIn")
if any(p["fuente"] == "demo" for p in prospectos):
    st.warning("Estás viendo datos de demostración (ficticios). Bórralos desde la barra lateral cuando cargues tus prospectos.")

tab_hoy, tab_nuevos, tab_embudo, tab_log = st.tabs(["📋 Hoy", "➕ Sumar prospectos", "🎯 Embudo", "📜 Bitácora"])


def linea_prospecto(p):
    return " · ".join(x for x in (p.get("cargo"), p.get("empresa")) if x)


def tarjeta_accion(a):
    es_invitacion = a["tipo"] == "invitacion"
    with st.container(border=True):
        izq, der = st.columns([4, 1])
        with izq:
            st.markdown(f"**{a['nombre']}** · {linea_prospecto(a) or 'sin cargo informado'}")
            detalle = f"{nombre_tipo(a['tipo'])} · prioridad {a['prioridad']} ({a['score']})"
            if a["personalizado_ia"] == 1:
                detalle += " · ✨ personalizado con IA"
            if es_invitacion:
                detalle += f" · {len(a['texto'])}/{cfg.nota_max} caracteres"
            st.caption(detalle)
            st.code(a["texto"], language=None, wrap_lines=True)
            with st.expander("✏️ Editar texto"):
                nuevo = st.text_area("Texto", a["texto"], key=f"txt-{a['id']}", height=160, label_visibility="collapsed")
                if st.button("Guardar texto", key=f"gtxt-{a['id']}"):
                    if es_invitacion and len(nuevo.strip()) > cfg.nota_max:
                        st.error(f"La nota supera los {cfg.nota_max} caracteres que permite LinkedIn.")
                    else:
                        secuencia.editar_texto(db, a["id"], nuevo)
                        st.rerun()
        with der:
            st.link_button("Abrir en LinkedIn ↗", fuentes.link_perfil(a), width="stretch")
            if st.button("✅ Enviado", key=f"ok-{a['id']}", type="primary", width="stretch"):
                secuencia.marcar_hecho(db, a["id"])
                replanificar()
            if st.button("⏭️ Omitir", key=f"om-{a['id']}", width="stretch"):
                secuencia.omitir(db, a["id"])
                replanificar()


# ---------------------------------------------------------------------------
# Hoy
# ---------------------------------------------------------------------------
with tab_hoy:
    if not prospectos:
        st.info(
            "Todavía no hay prospectos. Ve a **➕ Sumar prospectos** para importar tus contactos de LinkedIn "
            "o una lista, o carga los **datos de demostración** desde la barra lateral."
        )
    cola = secuencia.cola_de_hoy(db, cfg, ahora)
    mensajes = [a for a in cola if a["tipo"] != "invitacion"]
    invitaciones = [a for a in cola if a["tipo"] == "invitacion"]
    todos_msj, todas_inv = secuencia.pendientes(db, cfg, ahora)
    esperando = [p for p in prospectos if p["etapa"] == "Invitación enviada"]
    respondieron = [p for p in prospectos if p["etapa"] == "Respondió"]

    k = st.columns(4)
    k[0].metric("💬 Mensajes por enviar hoy", len(mensajes), help="A personas que ya son tus contactos")
    k[1].metric("🤝 Invitaciones por enviar hoy", len(invitaciones))
    k[2].metric("⏳ Invitaciones esperando", len(esperando))
    k[3].metric("🙌 Respondieron", len(respondieron), help="La secuencia se detiene: ahora conversas tú")

    if respondieron:
        st.subheader("🙌 Te respondieron: ahora conversas tú")
        for p in respondieron:
            with st.container(border=True):
                izq, der = st.columns([4, 1])
                with izq:
                    st.markdown(f"**{p['nombre']}** · {linea_prospecto(p)}")
                    st.caption(f"Respondió el {local(p['respondio_en'])}. Lee su mensaje en LinkedIn; si calza, propón la reunión:")
                    st.code(redactar("reunion", p, cfg), language=None, wrap_lines=True)
                with der:
                    st.link_button("Abrir en LinkedIn ↗", fuentes.link_perfil(p), width="stretch")
                    if st.button("📅 Reunión agendada", key=f"reu-{p['id']}", type="primary", width="stretch"):
                        secuencia.cambiar_etapa(db, p["id"], "Reunión agendada")
                        st.rerun()
                    if st.button("Descartar", key=f"desc-{p['id']}", width="stretch"):
                        secuencia.cambiar_etapa(db, p["id"], "Descartado")
                        st.rerun()

    if mensajes:
        st.subheader("💬 Mensajes a tus contactos")
        st.caption("Abre el perfil → **Mensaje** → pega el texto (botón de copiar arriba a la derecha de cada cuadro) → marca **Enviado**.")
        for a in mensajes:
            tarjeta_accion(a)
    if invitaciones:
        st.subheader("🤝 Invitaciones a conectar")
        st.caption(
            "Abre el perfil → **Conectar** → **Añadir una nota** → pega el texto → marca **Enviado**. "
            "Si tu cuenta gratuita ya agotó las notas del mes, envía la invitación sin nota: igual avanza la secuencia."
        )
        for a in invitaciones:
            tarjeta_accion(a)
    resto = len(todos_msj) + len(todas_inv) - len(cola)
    if prospectos and not cola:
        st.success("No tienes acciones pendientes por ahora. Suma más prospectos o vuelve mañana.")
    elif resto > 0:
        st.caption(f"Quedan {resto} acciones más en espera: aparecerán a medida que avances o mañana, para no superar tus límites diarios.")

    if esperando:
        with st.expander(f"⏳ Invitaciones enviadas que aún no aceptan ({len(esperando)})"):
            st.caption(
                "Con Gmail conectado, las aceptaciones se detectan solas por los avisos de LinkedIn. "
                "Si ves que alguien ya aceptó, márcalo aquí."
            )
            for p in esperando:
                c1, c2, c3 = st.columns([5, 1, 1])
                c1.markdown(f"**{p['nombre']}** · {linea_prospecto(p)} · invitado {hace(p['invitado_en'])}")
                c2.link_button("Perfil ↗", fuentes.link_perfil(p), width="stretch")
                if c3.button("🤝 Aceptó", key=f"acc-{p['id']}", width="stretch"):
                    secuencia.marcar_conectado(db, p["id"])
                    replanificar()
    en_conversacion = [
        p for p in prospectos
        if p["etapa"] in ("Conectado", "En conversación") and (p["ultimo_mensaje_en"] or p["invitado_en"])
    ]
    if en_conversacion:
        with st.expander(f"💬 En secuencia de mensajes ({len(en_conversacion)}) — ¿alguien te respondió?"):
            for p in en_conversacion[:60]:
                c1, c2, c3 = st.columns([5, 1, 1])
                c1.markdown(f"**{p['nombre']}** · {linea_prospecto(p)} · {p['etapa'].lower()}")
                c2.link_button("Perfil ↗", fuentes.link_perfil(p), width="stretch")
                if c3.button("🙌 Respondió", key=f"resp-{p['id']}", width="stretch"):
                    secuencia.marcar_respondio(db, p["id"])
                    st.rerun()

# ---------------------------------------------------------------------------
# Sumar prospectos
# ---------------------------------------------------------------------------
with tab_nuevos:
    if st.session_state.get("li_importe"):
        st.success(st.session_state.pop("li_importe"))

    def informar(nuevos, existentes, origen):
        st.session_state.li_importe = f"{nuevos} prospectos nuevos desde {origen} ({existentes} ya estaban en tu embudo)."
        replanificar()

    izq, der = st.columns(2)
    with izq:
        st.subheader("1 · Tus contactos de LinkedIn")
        st.markdown(
            "Tu red actual es tu mejor fuente: ya te conocen y puedes escribirles sin invitación.\n\n"
            "En LinkedIn: **Yo → Ajustes y privacidad → Privacidad de datos → Obtener una copia de tus datos → "
            "Contactos**. En unos minutos te llega `Connections.csv`; súbelo aquí."
        )
        st.subheader("2 · Una lista de prospectos")
        st.markdown(
            "Cualquier CSV o Excel (Sales Navigator, Apollo, una lista de una cámara o gremio, tu CRM). "
            "Columnas reconocidas: nombre, cargo, empresa, URL del perfil, ubicación, industria."
        )
        archivo = st.file_uploader("Archivo CSV o Excel", type=["csv", "xlsx"])
        ya_contactos = st.checkbox("Las personas de este archivo ya son mis contactos", value=False, help="El archivo Connections.csv de LinkedIn se reconoce solo.")
        if archivo and st.button("Importar archivo", type="primary"):
            try:
                n, e = fuentes.desde_archivo(db, archivo.name, archivo.getvalue(), grado=1 if ya_contactos else 2)
                informar(n, e, archivo.name)
            except Exception as error:
                st.error(f"No se pudo importar: {error}")

        st.subheader("3 · Quienes descargaron tu ebook")
        st.markdown("Suma a los leads calientes y tibios del tablero del ebook: ya conocen tu contenido.")
        if st.button("Traer leads del ebook"):
            n, e = fuentes.desde_leads_ebook(db)
            informar(n, e, "el ebook")

    with der:
        st.subheader("4 · Búsquedas listas en LinkedIn")
        st.markdown("Tus cargos objetivo en tu mercado, entre contactos de 2.º grado. Abre una y suma a quienes calcen:")
        columnas = st.columns(2)
        for i, (etiqueta, url) in enumerate(fuentes.busquedas_sugeridas(cfg)):
            columnas[i % 2].link_button(f"🔎 {etiqueta}", url, width="stretch")

        with st.form("alta-manual", clear_on_submit=True):
            st.markdown("**Agregar una persona**")
            f1, f2 = st.columns(2)
            nombre = f1.text_input("Nombre y apellido")
            cargo = f2.text_input("Cargo")
            empresa = f1.text_input("Empresa")
            ubicacion = f2.text_input("Ubicación", placeholder="Santiago, Chile")
            url = st.text_input("URL del perfil", placeholder="https://www.linkedin.com/in/…")
            es_contacto = st.checkbox("Ya es mi contacto")
            if st.form_submit_button("Agregar") and nombre.strip():
                n, e = fuentes.agregar_manual(db, nombre, cargo, empresa, url, ubicacion, grado=1 if es_contacto else 2)
                informar(n, e, "alta manual")

        st.markdown("**Pegar resultados de una búsqueda** (con IA)")
        if cfg.base.ia_configurada:
            pegado = st.text_area(
                "Texto copiado", height=120, label_visibility="collapsed",
                placeholder="En la página de resultados de LinkedIn: selecciona todo (Ctrl+A), copia (Ctrl+C) y pega aquí.",
            )
            if st.button("Extraer prospectos") and pegado.strip():
                try:
                    with st.spinner("Leyendo los resultados…"):
                        n, e = fuentes.desde_texto(db, pegado, crear_asistente(cfg))
                    informar(n, e, "la búsqueda pegada")
                except Exception as error:
                    st.error(f"No se pudo leer: {error}")
        else:
            st.caption("Disponible al configurar `ANTHROPIC_API_KEY`: copias la página de resultados y la IA extrae nombre, cargo y empresa.")

# ---------------------------------------------------------------------------
# Embudo
# ---------------------------------------------------------------------------
with tab_embudo:
    if not prospectos:
        st.info("Aún no hay prospectos.")
    else:
        df = pd.DataFrame(prospectos)
        invitados = df[df["invitado_en"].notna() & (df["fuente"] != "contactos")]
        aceptaron = invitados[invitados["conectado_en"].notna()]
        contactados = df[df["ultimo_mensaje_en"].notna()]
        con_respuesta = df[df["respondio_en"].notna()]
        k = st.columns(5)
        k[0].metric("Prospectos", len(df))
        k[1].metric("Prioridad A", int((df["prioridad"] == "A").sum()))
        k[2].metric("Tasa de aceptación", f"{len(aceptaron) / len(invitados):.0%}" if len(invitados) else "—", help="Invitaciones aceptadas / enviadas")
        k[3].metric("Tasa de respuesta", f"{len(con_respuesta) / max(len(contactados), len(con_respuesta)):.0%}" if len(con_respuesta) or len(contactados) else "—", help="Respondieron / personas a las que escribiste")
        k[4].metric("Reuniones y clientes", int(df["etapa"].isin(["Reunión agendada", "Cliente"]).sum()))

        e = df["etapa"].value_counts().reindex(ETAPAS, fill_value=0).rename_axis("Etapa").reset_index(name="Prospectos")
        base = alt.Chart(e).encode(
            y=alt.Y("Etapa:N", sort=ETAPAS, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("Prospectos:Q", title=None, axis=None),
            tooltip=["Etapa:N", "Prospectos:Q"],
        )
        st.altair_chart(
            (
                base.mark_bar(color=AZUL, cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=18)
                + base.mark_text(align="left", dx=6, color=TINTA).encode(text="Prospectos:Q")
            ).properties(height=34 * len(ETAPAS)),
            width="stretch",
        )

        f1, f2, f3 = st.columns([2, 2, 3])
        prioridades = f1.multiselect("Prioridad", ["A", "B", "C"], default=["A", "B", "C"])
        etapas = f2.multiselect("Etapa", ETAPAS, default=ETAPAS)
        buscar = f3.text_input("Buscar", placeholder="Nombre, empresa o cargo")
        vista = df[df["prioridad"].isin(prioridades) & df["etapa"].isin(etapas)].copy()
        if buscar:
            texto = vista[["nombre", "empresa", "cargo"]].astype(str).agg(" ".join, axis=1).str.lower()
            vista = vista[texto.str.contains(buscar.lower(), regex=False)]
        vista["Contacto"] = vista["grado"].map({1: "1.º", 2: "2.º+"})
        tabla = vista[["nombre", "cargo", "empresa", "ubicacion", "score", "prioridad", "etapa", "Contacto", "fuente"]]
        st.caption(f"{len(tabla)} prospectos · haz clic en una fila para ver el detalle")
        seleccion = st.dataframe(
            tabla, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
            column_config={
                "nombre": "Nombre", "cargo": "Cargo", "empresa": "Empresa", "ubicacion": "Ubicación",
                "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                "prioridad": "Prioridad", "etapa": "Etapa", "fuente": "Fuente",
            },
        )
        filas = seleccion.selection.rows if seleccion else []
        if filas:
            p = db.prospecto(int(vista.iloc[filas[0]]["id"]))
            st.divider()
            a, b = st.columns([3, 2])
            with a:
                st.subheader(p["nombre"])
                st.caption(" · ".join(x for x in (p["cargo"], p["empresa"], p["ubicacion"], p["email"]) if x))
                st.markdown(f"**Cómo abordarlo:** {p['angulo']}")
                detalle = pd.DataFrame(json.loads(p["score_detalle"] or "[]"), columns=["Factor", "Puntos"])
                st.dataframe(detalle, hide_index=True, width="stretch")
                st.markdown("**Historial**")
                historial = db.acciones(prospecto_id=p["id"])
                if not historial:
                    st.caption("Sin acciones todavía." + ("" if p["prioridad"] in cfg.prioridades_activas else f" Su prioridad ({p['prioridad']}) está bajo tu mínimo ({cfg.prioridad_minima})."))
                for h in sorted(historial, key=lambda x: x["paso"]):
                    estado = {"hecho": f"✅ enviado el {local(h['hecho_en'])}", "omitido": "⏭️ omitido", "pendiente": f"🕒 para el {local(h['programado_para'])}"}[h["estado"]]
                    with st.expander(f"{nombre_tipo(h['tipo'])} · {estado}"):
                        st.code(h["texto"], language=None, wrap_lines=True)
            with b:
                st.link_button("Abrir en LinkedIn ↗", fuentes.link_perfil(p), width="stretch")
                nueva = st.selectbox("Etapa", ETAPAS, index=ETAPAS.index(p["etapa"]) if p["etapa"] in ETAPAS else 0, key=f"et-{p['id']}")
                notas = st.text_area("Notas", p["notas"] or "", key=f"nt-{p['id']}")
                if st.button("Guardar", key=f"gd-{p['id']}"):
                    db.actualizar_prospecto(p["id"], notas=notas)
                    if nueva != p["etapa"]:
                        secuencia.cambiar_etapa(db, p["id"], nueva)
                    replanificar()

# ---------------------------------------------------------------------------
# Bitácora
# ---------------------------------------------------------------------------
with tab_log:
    registros = [r for r in db.bitacora(limite=2000) if r["agente"] in AGENTES][:400]
    if not registros:
        st.info("Aún no hay actividad.")
    else:
        log = pd.DataFrame(registros)
        log["Fecha"] = log["fecha"].map(lambda f: local(f, "%d/%m/%Y %H:%M"))
        st.dataframe(
            log[["Fecha", "agente", "accion", "detalle"]], hide_index=True, width="stretch",
            column_config={"agente": "Agente", "accion": "Acción", "detalle": "Detalle"},
        )
