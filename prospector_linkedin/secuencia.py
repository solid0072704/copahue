"""Agente de Secuencia: decide qué toca hacer hoy con cada prospecto y respeta tus límites diarios.

El agente prepara todo (a quién, qué texto, en qué orden). La acción en LinkedIn —enviar la
invitación o el mensaje— la haces tú con un clic desde el tablero, y la marcas como enviada.
"""

from datetime import timedelta, timezone

from agentes_ebook.db import ahora_utc, desde_iso, iso

from .db import ETAPAS, ETAPAS_CERRADAS
from .mensajes import nombre_tipo, redactar

AGENTE = "Secuencia"
SIN_IA = -1  # texto editado a mano o IA sin resultado: no volver a intentar


def _tipo(paso):
    return "invitacion" if paso == 0 else f"mensaje_{paso}"


def _nota(p, texto):
    return (p.get("notas") + "\n" if p.get("notas") else "") + texto


def planificar(db, cfgp, ahora=None):
    """Crea la próxima acción de cada prospecto activo y cierra los que no respondieron."""
    ahora = ahora or ahora_utc()
    creadas = []
    por_prospecto = {}
    for a in db.acciones():
        por_prospecto.setdefault(a["prospecto_id"], {})[a["paso"]] = a

    for p in db.prospectos():
        if p["etapa"] in ETAPAS_CERRADAS:
            continue
        acciones = por_prospecto.get(p["id"], {})
        iniciado = any(a["estado"] == "hecho" for a in acciones.values())
        if not iniciado and p["prioridad"] not in cfgp.prioridades_activas:
            continue  # bajo tu prioridad mínima: queda en el embudo, pero no entra a la cola

        if p["etapa"] == "Por invitar":
            if 0 not in acciones:
                creadas.append(db.crear_accion(p["id"], 0, "invitacion", iso(ahora), redactar("invitacion", p, cfgp)))
        elif p["etapa"] == "Invitación enviada":
            invitado = desde_iso(p["invitado_en"]) or ahora
            if ahora - invitado > timedelta(days=cfgp.dias_retirar):
                db.actualizar_prospecto(
                    p["id"],
                    etapa="Sin respuesta",
                    notas=_nota(p, f"No aceptó en {cfgp.dias_retirar} días: retira la invitación en LinkedIn."),
                )
                db.log(AGENTE, "Invitación sin aceptar", p["nombre"], p["id"])
        else:  # Conectado / En conversación
            mensajes = [a for a in acciones.values() if a["paso"] >= 1]
            if any(a["estado"] == "pendiente" for a in mensajes):
                continue
            ultimo = max(mensajes, key=lambda a: a["paso"], default=None)
            if ultimo is None:
                creadas.append(db.crear_accion(p["id"], 1, "mensaje_1", iso(ahora), redactar("mensaje_1", p, cfgp)))
                continue
            desde = desde_iso(ultimo["hecho_en"]) or ahora
            if ultimo["paso"] < cfgp.total_mensajes:
                paso = ultimo["paso"] + 1
                cuando = desde + timedelta(days=cfgp.dias_seguimiento[paso - 2])
                creadas.append(db.crear_accion(p["id"], paso, _tipo(paso), iso(cuando), redactar(_tipo(paso), p, cfgp)))
            elif ahora - desde > timedelta(days=cfgp.dias_cierre):
                db.actualizar_prospecto(p["id"], etapa="Sin respuesta")
                db.log(AGENTE, "Secuencia terminada sin respuesta", p["nombre"], p["id"])
    creadas = [c for c in creadas if c]
    if creadas:
        db.log(AGENTE, "Acciones preparadas", f"{len(creadas)} invitaciones o mensajes nuevos en el plan")
    return creadas


# --------------------------------------------------------------------------
# Cola diaria
# --------------------------------------------------------------------------
def cupos(db, cfgp, ahora=None):
    """Cuántas invitaciones y mensajes te quedan hoy según tus límites."""
    ahora = ahora or ahora_utc()
    inicio = ahora.astimezone(cfgp.tz).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    inv_hoy = db.hechas_desde(iso(inicio), invitaciones=True)
    inv_semana = db.hechas_desde(iso(ahora - timedelta(days=7)), invitaciones=True)
    msg_hoy = db.hechas_desde(iso(inicio), invitaciones=False)
    return {
        "invitaciones": max(0, min(cfgp.invitaciones_dia - inv_hoy, cfgp.invitaciones_semana - inv_semana)),
        "mensajes": max(0, cfgp.mensajes_dia - msg_hoy),
        "invitaciones_hoy": inv_hoy,
        "invitaciones_semana": inv_semana,
        "mensajes_hoy": msg_hoy,
    }


def pendientes(db, cfgp, ahora=None):
    """Todas las acciones que ya tocan (sin aplicar límites): (mensajes, invitaciones), por score."""
    ahora = ahora or ahora_utc()
    listas = [
        a for a in db.acciones(estados=["pendiente"])
        if a["etapa"] not in ETAPAS_CERRADAS and (desde_iso(a["programado_para"]) or ahora) <= ahora
    ]
    mensajes = [a for a in listas if a["tipo"] != "invitacion"]
    invitaciones = [
        a for a in listas
        if a["tipo"] == "invitacion" and a["etapa"] == "Por invitar" and a["prioridad"] in cfgp.prioridades_activas
    ]
    return mensajes, invitaciones


def cola_de_hoy(db, cfgp, ahora=None):
    """Lo que te toca hoy: primero los mensajes a quienes ya conectaron, luego las invitaciones."""
    ahora = ahora or ahora_utc()
    mensajes, invitaciones = pendientes(db, cfgp, ahora)
    c = cupos(db, cfgp, ahora)
    return mensajes[: c["mensajes"]] + invitaciones[: c["invitaciones"]]


def personalizar_cola(db, cfgp, asistente, ahora=None):
    """Con IA: reescribe los textos de la cola de hoy que aún son plantilla (hasta el máximo por ciclo)."""
    if not asistente:
        return 0
    hechos = 0
    candidatas = [a for a in cola_de_hoy(db, cfgp, ahora) if a["personalizado_ia"] == 0 and a["fuente"] != "demo"]
    for a in candidatas[: cfgp.base.llm_max_leads_por_ciclo]:
        p = db.prospecto(a["prospecto_id"])
        limite = cfgp.nota_max if a["tipo"] == "invitacion" else None
        texto = asistente.personalizar(p, a["tipo"], a["texto"], limite)
        if texto:
            db.actualizar_accion(a["id"], texto=texto, personalizado_ia=1)
            hechos += 1
        else:
            db.actualizar_accion(a["id"], personalizado_ia=SIN_IA)
    if hechos:
        db.log(AGENTE, "Textos personalizados con IA", f"{hechos} de la cola de hoy")
    return hechos


# --------------------------------------------------------------------------
# Lo que tú (o la sincronización con Gmail) marcan
# --------------------------------------------------------------------------
def _omitir_pendientes(db, prospecto_id, ahora):
    for a in db.acciones(prospecto_id=prospecto_id, estados=["pendiente"]):
        db.actualizar_accion(a["id"], estado="omitido", hecho_en=iso(ahora))


def marcar_hecho(db, accion_id, ahora=None):
    """Ya enviaste la invitación o el mensaje en LinkedIn."""
    ahora = ahora or ahora_utc()
    a = db.accion(accion_id)
    if not a or a["estado"] != "pendiente":
        return False
    db.actualizar_accion(accion_id, estado="hecho", hecho_en=iso(ahora))
    if a["tipo"] == "invitacion":
        db.actualizar_prospecto(a["prospecto_id"], etapa="Invitación enviada", invitado_en=iso(ahora))
    else:
        db.actualizar_prospecto(a["prospecto_id"], etapa="En conversación", ultimo_mensaje_en=iso(ahora))
    db.log(AGENTE, f"Enviado: {nombre_tipo(a['tipo'])}", a["nombre"], a["prospecto_id"])
    return True


def omitir(db, accion_id, ahora=None):
    """Saltar este paso. Omitir la invitación descarta al prospecto; omitir un mensaje sigue con el próximo."""
    ahora = ahora or ahora_utc()
    a = db.accion(accion_id)
    if not a or a["estado"] != "pendiente":
        return False
    db.actualizar_accion(accion_id, estado="omitido", hecho_en=iso(ahora))
    if a["tipo"] == "invitacion":
        db.actualizar_prospecto(a["prospecto_id"], etapa="Descartado")
    db.log(AGENTE, f"Omitido: {nombre_tipo(a['tipo'])}", a["nombre"], a["prospecto_id"])
    return True


def editar_texto(db, accion_id, texto):
    db.actualizar_accion(accion_id, texto=texto.strip(), personalizado_ia=SIN_IA)


def marcar_conectado(db, prospecto_id, ahora=None, origen="manual"):
    """Aceptó tu invitación (o ya eran contactos): pasa a la secuencia de mensajes."""
    ahora = ahora or ahora_utc()
    p = db.prospecto(prospecto_id)
    if not p or p["etapa"] not in ("Por invitar", "Invitación enviada"):
        return False
    _omitir_pendientes(db, prospecto_id, ahora)
    db.actualizar_prospecto(
        prospecto_id, etapa="Conectado", grado=1, conectado_en=iso(ahora), invitado_en=p["invitado_en"] or iso(ahora)
    )
    db.log(AGENTE, f"Aceptó la invitación ({origen})", p["nombre"], prospecto_id)
    return True


def marcar_respondio(db, prospecto_id, ahora=None, origen="manual"):
    """Te escribió: se detiene la secuencia y queda en tus manos."""
    ahora = ahora or ahora_utc()
    p = db.prospecto(prospecto_id)
    if not p or p["etapa"] in ("Respondió", "Reunión agendada", "Cliente", "Descartado"):
        return False
    _omitir_pendientes(db, prospecto_id, ahora)
    db.actualizar_prospecto(
        prospecto_id, etapa="Respondió", grado=1, respondio_en=iso(ahora), conectado_en=p["conectado_en"] or iso(ahora)
    )
    db.log(AGENTE, f"Respondió ({origen})", p["nombre"], prospecto_id)
    return True


def cambiar_etapa(db, prospecto_id, etapa, ahora=None):
    """Cambio manual de etapa desde el tablero."""
    if etapa not in ETAPAS:
        return False
    if etapa == "Respondió":
        return marcar_respondio(db, prospecto_id, ahora)
    actual = db.prospecto(prospecto_id)
    if etapa == "Conectado" and actual and actual["etapa"] in ("Por invitar", "Invitación enviada"):
        return marcar_conectado(db, prospecto_id, ahora)
    if etapa in ETAPAS_CERRADAS:
        _omitir_pendientes(db, prospecto_id, ahora or ahora_utc())
    db.actualizar_prospecto(prospecto_id, etapa=etapa)
    p = db.prospecto(prospecto_id)
    db.log(AGENTE, f"Etapa cambiada a «{etapa}»", p["nombre"] if p else "", prospecto_id)
    return True
