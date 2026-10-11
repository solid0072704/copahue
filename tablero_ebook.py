"""Tablero de agentes para las descargas del ebook.

Ejecutar:  streamlit run tablero_ebook.py
"""

import dataclasses
import json
from datetime import datetime, time, timedelta

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tablero del ebook", page_icon="📘", layout="wide")

from agentes_ebook.config import MODOS_SEGUIMIENTO, Config, aplicar_secrets_streamlit  # noqa: E402

aplicar_secrets_streamlit()

from agentes_ebook import analista, captura, notificador, reuniones, seguimiento  # noqa: E402
from agentes_ebook.db import ETAPAS, ahora_utc, desde_iso  # noqa: E402
from agentes_ebook.gmail import ClienteGmail, enviar_correo, es_destino_simulado  # noqa: E402
from agentes_ebook.orquestador import abrir_db, ejecutar_ciclo, preparar_demo  # noqa: E402
from agentes_ebook.plantillas import fecha_larga, link_mailto, link_whatsapp  # noqa: E402

try:
    OSCURO = st.context.theme.type == "dark"
except Exception:
    OSCURO = False
AZUL = "#3987e5" if OSCURO else "#2a78d6"
TINTA = "#c3c2b7" if OSCURO else "#52514e"
ICONO_SEGMENTO = {"Caliente": "🔥", "Tibio": "🌤️", "Frío": "❄️"}
ICONO_ESTADO = {"pendiente": "🕒 Por aprobar", "borrador": "📝 Borrador en Gmail", "enviado": "✅ Enviado", "omitido": "⏭️ Omitido"}

# ---------------------------------------------------------------------------
# Configuración y datos
# ---------------------------------------------------------------------------
cfg_base = Config.desde_entorno()
if "modo" not in st.session_state:
    st.session_state.modo = cfg_base.seguimiento_modo
cfg = dataclasses.replace(cfg_base, seguimiento_modo=st.session_state.modo)
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


def local(texto_iso, formato="%d/%m/%Y %H:%M"):
    dt = desde_iso(texto_iso)
    return dt.astimezone(cfg.tz).strftime(formato) if dt else ""


def correr_agentes(**kwargs):
    with st.spinner("Los agentes están trabajando: leyendo Gmail, analizando y preparando seguimientos…"):
        resultado = ejecutar_ciclo(db, cfg, **kwargs)
    st.session_state.ultimo_resultado = resultado.texto()
    for error in resultado.errores:
        st.session_state.setdefault("errores", []).append(error)


# Ejecución automática al abrir el tablero (si pasaron más de 15 minutos).
if "auto" not in st.session_state:
    st.session_state.auto = True
    ultima = db.get_meta("ultima_ejecucion")
    if cfg.gmail_configurado and (not ultima or ahora_utc() - desde_iso(ultima) > timedelta(minutes=15)):
        correr_agentes()

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 Agentes del ebook")
    st.caption(cfg.empresa_nombre)

    if st.button("▶️ Ejecutar agentes ahora", type="primary", width="stretch"):
        correr_agentes()
        st.rerun()
    st.caption(f"Última ejecución: {hace(db.get_meta('ultima_ejecucion'))}")
    if st.session_state.get("ultimo_resultado"):
        st.info(st.session_state.ultimo_resultado)
    for error in st.session_state.pop("errores", []):
        st.error(error)

    st.divider()
    st.subheader("Conexiones")
    st.markdown(
        f"{'✅' if cfg.gmail_configurado else '⚠️'} **Gmail**: "
        + (cfg.gmail_usuario if cfg.gmail_configurado else "sin configurar")
    )
    st.markdown(f"{'✅' if cfg.notificar_a else '⚠️'} **Avisos a**: {cfg.notificar_a or 'sin configurar'}")
    st.markdown(f"{'✅' if cfg.ia_configurada else '➖'} **IA (Claude)**: {'activa' if cfg.ia_configurada else 'opcional, sin API key'}")
    st.markdown(f"{'✅' if cfg.link_agenda else '➖'} **Link de agenda**: {'configurado' if cfg.link_agenda else 'se proponen horarios'}")

    st.selectbox(
        "Modo de seguimiento",
        MODOS_SEGUIMIENTO,
        key="modo",
        format_func={
            "borradores": "📝 Borradores en Gmail (tú envías)",
            "automatico": "⚡ Envío automático",
            "manual": "✋ Manual (apruebo aquí)",
        }.get,
        help="Borradores: los agentes dejan cada correo listo en tu Gmail. Automático: los envían solos "
        f"(máx. {cfg.seguimiento_max_dia}/día). Manual: esperan tu aprobación en este tablero.",
    )
    if not cfg.gmail_configurado:
        with st.expander("¿Cómo conecto mi Gmail?"):
            st.markdown(
                "1. Activa la verificación en 2 pasos en tu cuenta de Google.\n"
                "2. Crea una **contraseña de aplicación** en "
                "[myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords).\n"
                "3. Copia `.env.example` a `.env` y completa `GMAIL_USER` y `GMAIL_APP_PASSWORD`.\n"
                "4. Reinicia el tablero."
            )

    st.divider()
    with st.expander("📂 Importar descargas desde CSV/Excel"):
        archivo = st.file_uploader("Archivo con columnas email, nombre, cargo…", type=["csv", "xlsx"])
        if archivo and st.button("Importar"):
            try:
                df_imp = pd.read_csv(archivo) if archivo.name.endswith(".csv") else pd.read_excel(archivo)
                nuevas, nuevos = captura.desde_csv(db, df_imp, cfg)
                analista.analizar(db, cfg)
                st.success(f"{nuevas} descargas importadas ({len(nuevos)} leads nuevos).")
            except Exception as e:
                st.error(f"No se pudo importar: {e}")
    with st.expander("🧪 Datos de demostración"):
        st.caption("Leads ficticios para probar el tablero. Nunca se les envían correos.")
        c1, c2 = st.columns(2)
        if c1.button("Cargar demo"):
            preparar_demo(db, cfg)
            st.rerun()
        if c2.button("Borrar demo"):
            st.toast(f"{db.borrar_fuente('demo')} leads demo eliminados")
            st.rerun()

# ---------------------------------------------------------------------------
# Datos para las vistas
# ---------------------------------------------------------------------------
leads = db.leads()
descargas = db.descargas()
df_leads = pd.DataFrame(leads)
ahora = ahora_utc()

st.title("📘 Tablero de descargas del ebook")
if not leads:
    st.info(
        "Todavía no hay descargas. Conecta tu Gmail y presiona **Ejecutar agentes ahora**, "
        "importa un CSV o carga los **datos de demostración** desde la barra lateral."
    )
    st.stop()
if any(l["fuente"] == "demo" for l in leads):
    st.warning("Estás viendo datos de demostración (ficticios). Bórralos desde la barra lateral cuando conectes tu Gmail.")

tab_resumen, tab_leads, tab_seg, tab_reu, tab_notif, tab_log = st.tabs(
    ["📊 Resumen", "🧠 Leads", "✉️ Seguimiento", "🤝 Reuniones", "🔔 Notificaciones", "📜 Bitácora"]
)


def barras_horizontales(df, categoria, valor, orden=None):
    alto = max(120, 34 * len(df))
    base = alt.Chart(df).encode(
        y=alt.Y(f"{categoria}:N", sort=orden or "-x", title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X(f"{valor}:Q", title=None, axis=None),
        tooltip=[alt.Tooltip(f"{categoria}:N"), alt.Tooltip(f"{valor}:Q")],
    )
    barras = base.mark_bar(color=AZUL, cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=18)
    etiquetas = base.mark_text(align="left", dx=6, color=TINTA).encode(text=f"{valor}:Q")
    return (barras + etiquetas).properties(height=alto)


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
with tab_resumen:
    fechas = pd.to_datetime(pd.Series([d["fecha"] for d in descargas]), utc=True)
    ult7 = int((fechas > ahora - timedelta(days=7)).sum())
    prev7 = int(((fechas <= ahora - timedelta(days=7)) & (fechas > ahora - timedelta(days=14))).sum())
    abiertos = db.seguimientos(estados=("pendiente", "borrador"))
    reus = db.reuniones()

    k = st.columns(6)
    k[0].metric("Descargas totales", len(descargas))
    k[1].metric("Últimos 7 días", ult7, delta=ult7 - prev7 if prev7 else None, help="Variación vs. los 7 días anteriores")
    k[2].metric("🔥 Leads calientes", int((df_leads["segmento"] == "Caliente").sum()))
    k[3].metric("✉️ Correos por aprobar", len(abiertos))
    k[4].metric("📅 Reuniones propuestas", sum(1 for r in reus if r["estado"] == "propuesta"))
    k[5].metric("🤝 Reuniones agendadas", sum(1 for r in reus if r["estado"] == "agendada"))

    izq, der = st.columns([3, 2])
    with izq:
        st.subheader("Descargas por día (últimos 30 días)")
        dias = pd.date_range((ahora - timedelta(days=29)).astimezone(cfg.tz).date(), ahora.astimezone(cfg.tz).date(), freq="D")
        por_dia = (
            fechas.dt.tz_convert(cfg.zona_horaria).dt.date.value_counts()
            .reindex(dias.date, fill_value=0)
            .rename_axis("Día").reset_index(name="Descargas")
        )
        por_dia["Día"] = pd.to_datetime(por_dia["Día"])
        st.altair_chart(
            alt.Chart(por_dia)
            .mark_bar(color=AZUL, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width=alt.RelativeBandSize(0.75))
            .encode(
                x=alt.X("yearmonthdate(Día):T", title=None, axis=alt.Axis(format="%d/%m", grid=False)),
                y=alt.Y("Descargas:Q", title=None, axis=alt.Axis(tickMinStep=1)),
                tooltip=[alt.Tooltip("yearmonthdate(Día):T", title="Día", format="%d/%m/%Y"), "Descargas:Q"],
            )
            .properties(height=260),
            width="stretch",
        )
    with der:
        st.subheader("🧠 Lo que ve el analista")
        for h in analista.insights(leads, descargas, ahora, cfg.tz):
            st.markdown(f"- {h}")
        s = df_leads["segmento"].value_counts()
        c = st.columns(3)
        for i, seg in enumerate(("Caliente", "Tibio", "Frío")):
            c[i].metric(f"{ICONO_SEGMENTO[seg]} {seg}", int(s.get(seg, 0)))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Motivo de descarga")
        m = df_leads["motivo"].replace("", "Sin informar").value_counts().rename_axis("Motivo").reset_index(name="Leads")
        st.altair_chart(barras_horizontales(m, "Motivo", "Leads"), width="stretch")
    with c2:
        st.subheader("Nivel de cargo")
        n = df_leads["nivel_cargo"].value_counts().rename_axis("Nivel").reset_index(name="Leads")
        st.altair_chart(barras_horizontales(n, "Nivel", "Leads"), width="stretch")
    with c3:
        st.subheader("País")
        p = df_leads["pais"].value_counts().rename_axis("País").reset_index(name="Leads")
        st.altair_chart(barras_horizontales(p, "País", "Leads"), width="stretch")

    st.subheader("Embudo comercial")
    e = df_leads["etapa"].value_counts().reindex(ETAPAS, fill_value=0).rename_axis("Etapa").reset_index(name="Leads")
    st.altair_chart(barras_horizontales(e, "Etapa", "Leads", orden=ETAPAS), width="stretch")

# ---------------------------------------------------------------------------
# Leads
# ---------------------------------------------------------------------------
with tab_leads:
    f1, f2, f3 = st.columns([2, 2, 3])
    segs = f1.multiselect("Segmento", ["Caliente", "Tibio", "Frío"], default=["Caliente", "Tibio", "Frío"])
    etapas = f2.multiselect("Etapa", ETAPAS, default=ETAPAS)
    buscar = f3.text_input("Buscar", placeholder="Nombre, empresa, correo o cargo")
    vista = df_leads[df_leads["segmento"].isin(segs) & df_leads["etapa"].isin(etapas)].copy()
    if buscar:
        texto = vista[["nombre", "empresa", "email", "cargo"]].astype(str).agg(" ".join, axis=1).str.lower()
        vista = vista[texto.str.contains(buscar.lower(), regex=False)]
    vista["Segmento"] = vista["segmento"].map(lambda s: f"{ICONO_SEGMENTO.get(s, '')} {s}")
    vista["Última descarga"] = vista["ultima_descarga"].map(local)
    tabla = vista[["nombre", "email", "empresa", "cargo", "pais", "score", "Segmento", "etapa", "num_descargas", "Última descarga", "motivo"]]
    st.caption(f"{len(tabla)} leads · haz clic en una fila para ver el detalle")
    seleccion = st.dataframe(
        tabla,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "nombre": "Nombre", "email": "Correo", "empresa": "Empresa", "cargo": "Cargo", "pais": "País",
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
            "etapa": "Etapa", "num_descargas": st.column_config.NumberColumn("Descargas", format="%d"),
            "motivo": "Motivo",
        },
    )
    filas = seleccion.selection.rows if seleccion else []
    if filas:
        lead = db.lead(int(vista.iloc[filas[0]]["id"]))
        st.divider()
        a, b = st.columns([3, 2])
        with a:
            st.subheader(f"{ICONO_SEGMENTO.get(lead['segmento'], '')} {lead['nombre'] or lead['email']}")
            st.caption(" · ".join(x for x in (lead["cargo"], lead["empresa"], lead["pais"], lead["email"], lead["telefono"]) if x))
            st.markdown(f"**Lectura del analista{' (IA)' if lead['lectura_ia'] else ''}:** {lead['lectura']}")
            st.markdown(f"**Cómo abordarlo:** {lead['angulo']}")
            st.markdown(f"**Motivo de descarga:** {lead['motivo'] or '—'}")
            detalle = pd.DataFrame(json.loads(lead["score_detalle"] or "[]"), columns=["Factor", "Puntos"])
            st.dataframe(detalle, hide_index=True, width="stretch")
        with b:
            nueva_etapa = st.selectbox("Etapa", ETAPAS, index=ETAPAS.index(lead["etapa"]) if lead["etapa"] in ETAPAS else 0, key=f"etapa-{lead['id']}")
            notas = st.text_area("Notas", lead["notas"] or "", key=f"notas-{lead['id']}")
            if st.button("Guardar", key=f"guardar-{lead['id']}"):
                db.actualizar_lead(lead["id"], etapa=nueva_etapa, notas=notas)
                db.log("Tablero", "Lead actualizado", f"{lead['email']} → {nueva_etapa}", lead["id"])
                st.rerun()
            wa = link_whatsapp(lead, cfg)
            if wa:
                st.link_button("💬 Escribir por WhatsApp", wa, width="stretch")
            st.markdown("**Historial de seguimiento**")
            for s in db.seguimientos(lead_id=lead["id"]):
                st.markdown(f"- Paso {s['paso']} · {s['nombre_paso']} · {ICONO_ESTADO.get(s['estado'], s['estado'])} · {local(s['actualizado_en'], '%d/%m')}")

# ---------------------------------------------------------------------------
# Seguimiento
# ---------------------------------------------------------------------------
with tab_seg:
    abiertos = db.seguimientos(estados=("pendiente", "borrador"))
    st.subheader(f"Correos listos para enviar ({len(abiertos)})")
    st.caption(
        "El agente de seguimiento redacta cada correo según el segmento del lead y la cadencia "
        "(bienvenida → valor → invitación → último contacto). Revisa, edita y envía."
    )
    if not abiertos:
        st.success("No hay correos pendientes. Los agentes prepararán los siguientes cuando corresponda.")
    for s in abiertos:
        demo = es_destino_simulado(s["email"])
        titulo = f"{ICONO_SEGMENTO.get(s['segmento'], '')} {s['nombre'] or s['email']} · Paso {s['paso']}: {s['nombre_paso']} · {ICONO_ESTADO[s['estado']]}"
        with st.expander(titulo, expanded=s is abiertos[0]):
            st.caption(f"{s['cargo']} · {s['empresa']} · {s['email']}" + (" · ✨ personalizado con IA" if s["personalizado_ia"] else ""))
            asunto = st.text_input("Asunto", s["asunto"], key=f"asunto-{s['id']}")
            cuerpo = st.text_area("Mensaje", s["cuerpo"], height=280, key=f"cuerpo-{s['id']}")
            if asunto != s["asunto"] or cuerpo != s["cuerpo"]:
                db.actualizar_seguimiento(s["id"], asunto=asunto, cuerpo=cuerpo)
            b1, b2, b3, b4 = st.columns(4)
            if b1.button("📤 Enviar ahora", key=f"enviar-{s['id']}", type="primary", disabled=not cfg.gmail_configurado and not demo):
                try:
                    r = enviar_correo(cfg, s["email"], asunto, cuerpo)
                    seguimiento.marcar(db, s["id"], "enviado")
                    st.toast("Enviado (simulado: dato demo)" if r == "simulado" else f"Correo enviado a {s['email']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"No se pudo enviar: {e}")
            if b2.button("📝 Crear borrador en Gmail", key=f"borrador-{s['id']}", disabled=not cfg.gmail_configurado or s["estado"] == "borrador"):
                try:
                    with ClienteGmail(cfg) as g:
                        g.crear_borrador(s["email"], asunto, cuerpo)
                    seguimiento.marcar(db, s["id"], "borrador")
                    st.rerun()
                except Exception as e:
                    st.error(f"No se pudo crear el borrador: {e}")
            if b3.button("✅ Ya lo envié", key=f"marcar-{s['id']}"):
                seguimiento.marcar(db, s["id"], "enviado")
                st.rerun()
            if b4.button("⏭️ Omitir", key=f"omitir-{s['id']}"):
                seguimiento.marcar(db, s["id"], "omitido")
                st.rerun()
            l1, l2 = st.columns(2)
            l1.link_button("✉️ Abrir en mi correo", link_mailto(s["email"], asunto, cuerpo), width="stretch")
            wa = link_whatsapp(db.lead(s["lead_id"]), cfg)
            if wa:
                l2.link_button("💬 Enviar versión corta por WhatsApp", wa, width="stretch")

    st.subheader("Historial")
    hist = pd.DataFrame(db.seguimientos(estados=("enviado", "omitido")))
    if not hist.empty:
        hist["Estado"] = hist["estado"].map(ICONO_ESTADO)
        hist["Fecha"] = hist["actualizado_en"].map(local)
        st.dataframe(
            hist.sort_values("actualizado_en", ascending=False)[["Fecha", "nombre", "email", "paso", "nombre_paso", "Estado", "asunto"]],
            hide_index=True,
            width="stretch",
            column_config={"nombre": "Nombre", "email": "Correo", "paso": "Paso", "nombre_paso": "Tipo", "asunto": "Asunto"},
        )

# ---------------------------------------------------------------------------
# Reuniones y contactos
# ---------------------------------------------------------------------------
with tab_reu:
    reus = db.reuniones()
    propuestas = [r for r in reus if r["estado"] == "propuesta"]
    agendadas = sorted((r for r in reus if r["estado"] == "agendada"), key=lambda r: r["fecha_agendada"] or "")

    st.subheader(f"📅 Próximas reuniones ({len(agendadas)})")
    if agendadas:
        for r in agendadas:
            c1, c2, c3 = st.columns([4, 1, 1])
            c1.markdown(f"**{fecha_larga(desde_iso(r['fecha_agendada']).astimezone(cfg.tz))}** · {r['nombre']} ({r['cargo']}, {r['empresa']}) · {r['email']}")
            if c2.button("🏆 Cliente", key=f"cliente-{r['lead_id']}"):
                db.actualizar_reunion(r["lead_id"], estado="realizada")
                db.actualizar_lead(r["lead_id"], etapa="Cliente")
                db.log("Reuniones", "Convertido en cliente", r["email"], r["lead_id"])
                st.rerun()
            if c3.button("✖️ No avanzó", key=f"noavanzo-{r['lead_id']}"):
                db.actualizar_reunion(r["lead_id"], estado="realizada")
                db.actualizar_lead(r["lead_id"], etapa="Descartado")
                db.log("Reuniones", "Reunión sin avance", r["email"], r["lead_id"])
                st.rerun()
    else:
        st.caption("Aún no hay reuniones agendadas.")

    st.subheader(f"🤝 Reuniones propuestas por el agente ({len(propuestas)})")
    st.caption(
        "El agente de reuniones elige a los leads calientes y a quienes respondieron, y les ofrece horarios hábiles. "
        "Crea la invitación en Google Calendar (con el lead como invitado) o descarga el .ics."
    )
    for r in propuestas:
        lead = db.lead(r["lead_id"])
        with st.expander(f"{ICONO_SEGMENTO.get(r['segmento'], '')} {r['nombre'] or r['email']} · {r['cargo']} · {r['empresa']} · score {r['score']}"):
            st.markdown(f"**Cómo abordarlo:** {lead['angulo']}")
            horarios = reuniones.horarios_de(db, r["lead_id"], cfg)
            for h in horarios:
                c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                c1.markdown(f"**{fecha_larga(h)}** ({cfg.reunion_duracion_min} min)")
                c2.link_button("Google Calendar", reuniones.link_google_calendar(lead, cfg, h), width="stretch")
                c3.download_button(
                    "Descargar .ics", reuniones.archivo_ics(lead, cfg, h), file_name=f"reunion-{r['lead_id']}-{h:%Y%m%d%H%M}.ics",
                    mime="text/calendar", key=f"ics-{r['lead_id']}-{h:%Y%m%d%H%M}", width="stretch",
                )
                if c4.button("✅ Confirmada", key=f"agendar-{r['lead_id']}-{h:%Y%m%d%H%M}", width="stretch"):
                    reuniones.agendar(db, r["lead_id"], h)
                    st.rerun()
            st.markdown("**¿Acordaron otro horario?**")
            d1, d2, d3 = st.columns([2, 2, 2])
            dia = d1.date_input("Día", ahora.astimezone(cfg.tz).date() + timedelta(days=1), key=f"dia-{r['lead_id']}")
            hora = d2.time_input("Hora", time(10, 0), key=f"hora-{r['lead_id']}")
            if d3.button("Agendar", key=f"otra-{r['lead_id']}"):
                reuniones.agendar(db, r["lead_id"], datetime.combine(dia, hora, tzinfo=cfg.tz))
                st.rerun()
            wa = link_whatsapp(lead, cfg)
            if wa:
                st.link_button("💬 Coordinar por WhatsApp", wa)

    st.subheader("📇 Contactos para tu CRM")
    st.caption("CSV con columnas compatibles con HubSpot (y fáciles de mapear en Pipedrive o Zoho).")
    solo_calientes = st.checkbox("Solo leads calientes y en conversación", value=True)
    exportar = [l for l in leads if not solo_calientes or l["segmento"] == "Caliente" or l["etapa"] in ("Respondió", "Reunión propuesta", "Reunión agendada", "Cliente")]
    crm = reuniones.exportar_crm(exportar)
    st.dataframe(crm, hide_index=True, width="stretch")
    st.download_button("⬇️ Descargar contactos (CSV)", crm.to_csv(index=False).encode("utf-8-sig"), "contactos_ebook.csv", "text/csv")

# ---------------------------------------------------------------------------
# Notificaciones
# ---------------------------------------------------------------------------
with tab_notif:
    st.subheader("🔔 Avisos por correo")
    st.markdown(
        f"- **Alerta inmediata** cuando aparece un lead {', '.join(cfg.notificar_segmentos)} o alguien te responde.\n"
        f"- **Resumen diario** a partir de las {cfg.resumen_hora:02d}:00 ({cfg.zona_horaria}). "
        f"Último enviado: {db.get_meta('ultimo_resumen') or 'nunca'}.\n"
        f"- Destinatario: **{cfg.notificar_a or 'sin configurar (NOTIFY_EMAIL)'}**"
        + (f" · copia: {', '.join(cfg.notificar_cc)}" if cfg.notificar_cc else "")
    )
    c1, c2 = st.columns(2)
    if c1.button("📨 Enviar resumen ahora", disabled=not (cfg.gmail_configurado and cfg.notificar_a)):
        estado = notificador.enviar_resumen(db, cfg, forzar=True)
        st.toast({"enviado": "Resumen enviado", "error": "Error al enviar: revisa la bitácora"}.get(estado, estado))
    if c2.button("🔥 Enviar alerta de leads pendientes", disabled=not (cfg.gmail_configurado and cfg.notificar_a)):
        estado = notificador.enviar_alerta(db, cfg)
        st.toast({"enviado": "Alerta enviada", None: "No hay leads nuevos por avisar"}.get(estado, estado))
    if not cfg.gmail_configurado:
        st.info("Configura tu Gmail para recibir estos avisos. Mientras tanto, aquí tienes la vista previa.")

    vista_previa = st.radio("Vista previa", ["Resumen diario", "Alerta de leads calientes"], horizontal=True)
    if vista_previa == "Resumen diario":
        _, html, _ = notificador.construir_resumen(db, cfg)
    else:
        calientes = [l for l in leads if l["segmento"] == "Caliente"][:5]
        _, html, _ = notificador.construir_alerta(db, cfg, calientes, set())
    # El correo es claro por diseño: se muestra sobre su propio fondo también en modo oscuro.
    st.html(f'<div style="background:#f9f9f7;border-radius:12px;overflow:hidden;">{html}</div>')

# ---------------------------------------------------------------------------
# Bitácora
# ---------------------------------------------------------------------------
with tab_log:
    agentes = ["Todos", "Captura", "Analista", "Seguimiento", "Reuniones", "Notificador", "Orquestador", "Tablero"]
    agente = st.selectbox("Agente", agentes)
    log = pd.DataFrame(db.bitacora(1000, None if agente == "Todos" else agente))
    if not log.empty:
        log["fecha"] = log["fecha"].map(local)
        st.dataframe(
            log[["fecha", "agente", "accion", "detalle"]],
            hide_index=True,
            width="stretch",
            column_config={"fecha": "Fecha", "agente": "Agente", "accion": "Acción", "detalle": "Detalle"},
        )
