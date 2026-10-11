"""Agente de Seguimiento: arma la secuencia de correos de cada lead y la ejecuta."""

from datetime import datetime, time, timedelta

from . import reuniones
from .db import ETAPAS_SIN_SEGUIMIENTO, ahora_utc, desde_iso, iso
from .gmail import enviar_correo
from .plantillas import CADENCIAS, redactar

AGENTE = "Seguimiento"
ESTADOS_ABIERTOS = ("pendiente", "borrador")
# Días máximos desde la descarga para dar la bienvenida normal; después se usa "reactivación".
DIAS_BIENVENIDA = 3
# Leads que nunca contactaste y descargaron hace más de esto quedan para gestión manual.
DIAS_ANTIGUEDAD_MAX = 60
# Separación mínima entre dos correos al mismo lead.
DIAS_ENTRE_CORREOS = 2


def detener_seguimiento(lead):
    return lead["etapa"] in ETAPAS_SIN_SEGUIMIENTO or bool(lead.get("respondio_en"))


def siguiente_paso(lead, existentes, ahora=None):
    """Decide el próximo correo: (paso, tipo, nombre_paso, programado_para) o None."""
    ahora = ahora or ahora_utc()
    if detener_seguimiento(lead):
        return None
    if any(s["estado"] in ESTADOS_ABIERTOS for s in existentes):
        return None  # Hay un correo esperando tu aprobación: no se acumulan más.
    cadencia = CADENCIAS.get(lead.get("segmento") or "Frío", CADENCIAS["Frío"])
    por_paso = {s["paso"]: s for s in existentes}

    if 1 not in por_paso:
        primera = desde_iso(lead["primera_descarga"]) or ahora
        antiguedad = ahora - primera
        if antiguedad > timedelta(days=DIAS_ANTIGUEDAD_MAX):
            return None
        if antiguedad <= timedelta(days=DIAS_BIENVENIDA):
            _, tipo, nombre = cadencia[0]
        else:
            tipo, nombre = "reactivacion", "Reactivación"
        return 1, tipo, nombre, ahora

    inicio = desde_iso(por_paso[1]["creado_en"])
    ultimo_envio = max(desde_iso(s["actualizado_en"]) for s in existentes if s["estado"] == "enviado") if any(
        s["estado"] == "enviado" for s in existentes
    ) else inicio
    for paso, (dias, tipo, nombre) in enumerate(cadencia, start=1):
        if paso in por_paso:
            continue
        programado = max(inicio + timedelta(days=dias), ultimo_envio + timedelta(days=DIAS_ENTRE_CORREOS))
        if programado <= ahora:
            # Si se saltó algún paso (p. ej. cambió de segmento), se toma el siguiente que falte.
            return paso, tipo, nombre, programado
        return None
    return None


def _inicio_dia_local(cfg, ahora):
    local = ahora.astimezone(cfg.tz)
    return datetime.combine(local.date(), time(0, 0), tzinfo=cfg.tz)


def creados_hoy(db, cfg, ahora=None):
    desde = iso(_inicio_dia_local(cfg, ahora or ahora_utc()))
    return sum(
        1 for s in db.seguimientos() if s["estado"] != "omitido" and (s["creado_en"] or "") >= desde
    )


def ejecutar(db, cfg, cliente_gmail=None, asistente=None, ahora=None):
    """Crea (y según el modo, envía o deja en borrador) los correos que tocan hoy."""
    ahora = ahora or ahora_utc()
    cupo = cfg.seguimiento_max_dia - creados_hoy(db, cfg, ahora)
    creados = []
    for lead in db.leads():  # ordenados por score: primero los más calientes
        existentes = db.seguimientos(lead_id=lead["id"])
        if detener_seguimiento(lead):
            for s in existentes:
                if s["estado"] in ESTADOS_ABIERTOS:
                    db.actualizar_seguimiento(s["id"], estado="omitido")
                    db.log(AGENTE, "Secuencia detenida", f"{lead['email']} · etapa {lead['etapa']}", lead["id"])
            continue
        if cupo <= 0:
            continue
        plan = siguiente_paso(lead, existentes, ahora)
        if not plan:
            continue
        paso, tipo, nombre_paso, programado = plan
        horarios = [] if cfg.link_agenda else reuniones.horarios_de(db, lead["id"], cfg)
        if not horarios and not cfg.link_agenda:
            horarios = reuniones.proximos_horarios(cfg, db.horarios_ocupados(), rotacion=lead["id"])
        asunto, cuerpo = redactar(tipo, lead, cfg, horarios)
        ia = False
        if asistente is not None and lead["segmento"] in ("Caliente", "Tibio"):
            personalizado = asistente.personalizar_correo(lead, tipo, asunto, cuerpo)
            if personalizado:
                cuerpo, ia = personalizado, True

        estado, detalle = "pendiente", "Esperando tu aprobación en el tablero"
        try:
            if cfg.seguimiento_modo == "automatico":
                resultado = enviar_correo(cfg, lead["email"], asunto, cuerpo)
                estado = "enviado"
                detalle = "Enviado automáticamente" + (" (simulado: dato demo)" if resultado == "simulado" else "")
            elif cfg.seguimiento_modo == "borradores" and cliente_gmail is not None:
                resultado = cliente_gmail.crear_borrador(lead["email"], asunto, cuerpo)
                if resultado in ("ok", "simulado"):
                    estado = "borrador"
                    detalle = "Borrador creado en Gmail" + (" (simulado: dato demo)" if resultado == "simulado" else "")
        except Exception as e:  # el correo queda pendiente para reintentar desde el tablero
            detalle = f"No se pudo procesar ({e}); queda pendiente"

        seg_id = db.crear_seguimiento(
            lead["id"], paso, tipo, nombre_paso, iso(programado), estado, asunto, cuerpo, ia
        )
        if seg_id is None:
            continue
        if lead["etapa"] == "Nuevo":
            db.actualizar_lead(lead["id"], etapa="En seguimiento")
        db.log(AGENTE, f"Paso {paso}: {nombre_paso}", f"{lead['email']} · {detalle}", lead["id"])
        creados.append(seg_id)
        cupo -= 1
    return creados


def sincronizar(db, cfg, cliente_gmail):
    """Lee Gmail para saber quién respondió y qué borradores ya enviaste. Devuelve ids de leads que respondieron."""
    leads = {l["email"]: l for l in db.leads()}
    contactados, con_borrador = {}, {}
    for s in db.seguimientos():
        if s["estado"] in ("enviado", "borrador"):
            creado = desde_iso(s["creado_en"])
            previo = contactados.get(s["email"])
            contactados[s["email"]] = min(previo, creado) if previo else creado
        if s["estado"] == "borrador":
            con_borrador.setdefault(s["email"], []).append(s)

    respondieron = []
    pendientes_respuesta = {
        correo: desde for correo, desde in contactados.items() if not leads[correo].get("respondio_en")
    }
    for correo, fecha in cliente_gmail.ultima_respuesta(pendientes_respuesta).items():
        lead = leads[correo]
        db.actualizar_lead(lead["id"], respondio_en=iso(fecha), etapa=(
            lead["etapa"] if lead["etapa"] in ("Reunión agendada", "Cliente") else "Respondió"
        ))
        db.log(AGENTE, "Respuesta detectada", f"{lead['nombre']} <{correo}> te respondió", lead["id"])
        respondieron.append(lead["id"])

    desde_borrador = {
        correo: min(desde_iso(s["creado_en"]) for s in segs) for correo, segs in con_borrador.items()
    }
    for correo, fechas in cliente_gmail.enviados_a(desde_borrador).items():
        segs = sorted(con_borrador[correo], key=lambda s: s["paso"])
        for s, fecha in zip(segs, fechas):
            db.actualizar_seguimiento(s["id"], estado="enviado")
            db.log(AGENTE, "Borrador enviado", f"{correo} · paso {s['paso']}", s["lead_id"])
    return respondieron


def marcar(db, seg_id, estado):
    db.actualizar_seguimiento(seg_id, estado=estado)
    s = db.seguimiento(seg_id)
    if s and estado == "enviado":
        lead = db.lead(s["lead_id"])
        if lead and lead["etapa"] == "Nuevo":
            db.actualizar_lead(lead["id"], etapa="En seguimiento")
    db.log(AGENTE, f"Marcado como {estado}", f"{s['email']} · paso {s['paso']}" if s else "", s["lead_id"] if s else None)
