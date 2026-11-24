"""SLA de la central: reasigna los leads que el vendedor no atendió a tiempo.

Un lead se considera atendido cuando el vendedor completa la tarea que creó la central,
mueve el negocio de la etapa inicial o lo cierra. Si pasan `sla.minutos` hábiles sin
eso, el negocio (y su tarea) pasan al siguiente vendedor del proyecto. Después de
`sla.max_reasignaciones`, el lead se escala al supervisor.

Todo el estado vive en HubSpot (propiedades central_*), así que esta revisión puede
correr desde el servicio web, un cron o GitHub Actions sin base de datos propia.
"""

import logging
from dataclasses import asdict, dataclass, field
from datetime import timedelta

from .distribuidor import ESTADOS_EN_SLA, PROPS_NEGOCIO, iso, ms, parse_fecha
from .hubspot import ErrorHubSpot
from .ruteo import elegir_vendedor

log = logging.getLogger(__name__)


@dataclass
class ResultadoSLA:
    revisados: int = 0
    en_plazo: int = 0
    atendidos: int = 0
    reasignados: int = 0
    escalados: int = 0
    errores: list = field(default_factory=list)

    def texto(self):
        partes = [
            f"{self.revisados} leads en seguimiento",
            f"{self.en_plazo} en plazo",
            f"{self.atendidos} atendidos",
            f"{self.reasignados} reasignados",
            f"{self.escalados} escalados",
        ]
        if self.errores:
            partes.append(f"{len(self.errores)} errores")
        return " · ".join(partes)

    def como_dict(self):
        return asdict(self)


def revisar_sla(dist, ahora=None, dias=7):
    hs, cfg = dist.hs, dist.cfg
    ahora = ahora or dist.reloj()
    r = ResultadoSLA()
    filtros = [
        {"propertyName": "fuente", "operator": "EQ", "value": cfg.fuente},
        {"propertyName": "central_estado", "operator": "IN", "values": sorted(ESTADOS_EN_SLA)},
        {"propertyName": "createdate", "operator": "GTE", "value": ms(ahora - timedelta(days=dias))},
    ]
    negocios = list(hs.buscar("deals", [filtros], PROPS_NEGOCIO, max_paginas=10))
    activos = hs.propietarios()
    for n in negocios:
        r.revisados += 1
        try:
            _revisar(dist, n, ahora, activos, r)
        except ErrorHubSpot as e:
            log.warning("SLA: error con el negocio %s: %s", n.get("id"), e)
            r.errores.append(f"negocio {n.get('id')}: {e.mensaje}")
    if r.reasignados or r.escalados:
        log.info("SLA: %s", r.texto())
    return r


def _revisar(dist, negocio, ahora, activos, r):
    tabla, hs = dist.tabla, dist.hs
    id_ = negocio["id"]
    p = negocio.get("properties", {})
    asignado = parse_fecha(p.get("central_asignado_en")) or parse_fecha(p.get("createdate")) or ahora
    tarea_id = str(p.get("central_tarea_id") or "")

    completada_en = None
    if tarea_id and not tarea_id.startswith("simulado"):
        try:
            t = hs.leer("tasks", tarea_id, ("hs_task_status", "hs_task_completion_date")).get("properties", {})
        except ErrorHubSpot as e:
            if e.status != 404:
                raise
            t = {}  # la tarea fue borrada: se decide solo por la etapa del negocio
        if t.get("hs_task_status") == "COMPLETED":
            completada_en = parse_fecha(t.get("hs_task_completion_date")) or ahora

    if completada_en or p.get("hs_is_closed") == "true" or p.get("dealstage") != tabla.etapa_inicial:
        minutos = round(tabla.minutos_habiles(asignado, min(completada_en or ahora, ahora)))
        dist._actualizar(
            "deals", id_, {"central_estado": "atendido", "central_minutos_respuesta": str(minutos)},
            opcionales=("central_minutos_respuesta",),
        )
        r.atendidos += 1
        return

    if tabla.minutos_habiles(asignado, ahora) < tabla.sla_minutos:
        r.en_plazo += 1
        return

    # --- Vencido: reasignar o escalar
    actual = str(p.get("hubspot_owner_id") or "")
    previos = [x for x in str(p.get("central_vendedores_previos") or "").split(";") if x]
    try:
        reasignaciones = int(float(p.get("central_reasignaciones") or 0))
    except ValueError:
        reasignaciones = 0
    contactos = hs.asociados("deals", id_, "contacts")
    contacto_id = contactos[0] if contactos else ""

    if reasignaciones >= tabla.max_reasignaciones:
        _escalar(dist, negocio, contacto_id, ahora, activos, r, "nadie lo atendió después de las reasignaciones")
        return

    valor = p.get("proyecto_de_interes") or ""
    candidatos, _, _ = tabla.candidatos(valor)
    cargas_p, cargas_t = dist.cargas([v.id for v in candidatos], valor, ahora)
    nuevo = elegir_vendedor(tabla, candidatos, ahora, cargas_p, cargas_t, activos, excluir=set(previos) | {actual})
    if not nuevo:
        _escalar(dist, negocio, contacto_id, ahora, activos, r, "no hay otro vendedor disponible en el proyecto")
        return

    previos.append(actual)
    hs.actualizar(
        "deals", id_,
        {
            "hubspot_owner_id": nuevo.id,
            "central_estado": "reasignado",
            "central_reasignaciones": str(reasignaciones + 1),
            "central_vendedores_previos": ";".join(x for x in previos if x),
            "central_asignado_en": iso(ahora),
        },
    )
    if tarea_id and not tarea_id.startswith("simulado"):
        vence = tabla.sumar_minutos_habiles(ahora, tabla.sla_minutos)
        hs.actualizar("tasks", tarea_id, {"hubspot_owner_id": nuevo.id, "hs_timestamp": iso(vence)})
    anterior = dist._nombre(actual, activos) or actual or "sin vendedor"
    nombre_nuevo = dist._nombre(nuevo.id, activos) or nuevo.id
    _nota(
        dist, id_, contacto_id, ahora,
        f"Central de WhatsApp: {anterior} no atendió el lead en {tabla.sla_minutos} minutos hábiles. "
        f"Se reasignó a {nombre_nuevo}.",
    )
    if contacto_id and dist.cfg.propietario_contacto != "nunca":
        dueño_contacto = hs.leer("contacts", contacto_id, ("hubspot_owner_id",)).get("properties", {}).get("hubspot_owner_id")
        if str(dueño_contacto or "") in ("", actual):
            hs.actualizar("contacts", contacto_id, {"hubspot_owner_id": nuevo.id})
    if dist.notificador and dist.notificador.activo:
        correo = activos.get(nuevo.id, {}).get("email")
        dist.notificador.enviar(
            correo,
            f"Lead reasignado a ti · {valor or 'sin proyecto'}",
            f"{p.get('dealname') or 'Un lead'} no fue atendido a tiempo y ahora es tuyo.\n"
            f"Negocio: {hs.url_registro('deals', id_)}\n"
            f"Tienes {tabla.sla_minutos} minutos hábiles para contactarlo.",
        )
    log.info("SLA: negocio %s reasignado de %s a %s", id_, actual, nuevo.id)
    r.reasignados += 1


def _escalar(dist, negocio, contacto_id, ahora, activos, r, motivo):
    tabla, hs = dist.tabla, dist.hs
    id_ = negocio["id"]
    p = negocio.get("properties", {})
    supervisor = tabla.supervisor_de(tabla.proyecto(p.get("proyecto_de_interes")))
    hs.actualizar("deals", id_, {"central_estado": "escalado"})
    texto = f"Central de WhatsApp: lead escalado al supervisor porque {motivo}."
    if supervisor and supervisor in activos:
        hs.crear(
            "tasks",
            {
                "hs_task_subject": f"Lead sin atender · {p.get('dealname') or id_} · {p.get('proyecto_de_interes') or 'sin proyecto'}",
                "hs_task_body": f"<p>{texto}</p><p>Asígnalo a un vendedor o contáctalo tú.</p>",
                "hs_timestamp": iso(ahora),
                "hubspot_owner_id": supervisor,
                "hs_task_status": "NOT_STARTED",
                "hs_task_priority": "HIGH",
                "hs_task_type": "TODO",
            },
            [("deals", id_), ("contacts", contacto_id)],
        )
        if dist.notificador and dist.notificador.activo:
            dist.notificador.enviar(
                activos.get(supervisor, {}).get("email"),
                f"Lead sin atender · {p.get('proyecto_de_interes') or 'sin proyecto'}",
                f"{texto}\nNegocio: {hs.url_registro('deals', id_)}",
            )
    else:
        texto += " (no hay supervisor activo configurado en sla.supervisor)"
    _nota(dist, id_, contacto_id, ahora, texto)
    log.info("SLA: negocio %s escalado (%s)", id_, motivo)
    r.escalados += 1


def _nota(dist, negocio_id, contacto_id, ahora, texto):
    asociaciones = [("deals", negocio_id)] + ([("contacts", contacto_id)] if contacto_id else [])
    dist.hs.crear("notes", {"hs_note_body": texto, "hs_timestamp": iso(ahora)}, asociaciones)
