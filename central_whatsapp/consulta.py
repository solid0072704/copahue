"""Consulta periódica: reparte los contactos que un workflow de HubSpot marcó como pendientes.

Reemplaza al webhook (la acción "Enviar un webhook" de HubSpot requiere Data Hub Professional).
El workflow usa solo la acción estándar "Editar registro": cuando Breeze califica al cliente,
marca «Pendiente de reparto (central)» = Sí. Este proceso busca esos contactos cada minuto,
reparte cada lead y desmarca la propiedad. Si algo falla, la marca queda y se reintenta en la
siguiente vuelta; el control de avisos repetidos evita negocios duplicados.
"""

import logging
from dataclasses import asdict, dataclass, field

from .hubspot import ErrorHubSpot

log = logging.getLogger(__name__)

PROPIEDAD_PENDIENTE = "central_pendiente_reparto"


@dataclass
class ResultadoConsulta:
    pendientes: int = 0
    asignados: int = 0
    existentes: int = 0
    duplicados: int = 0
    sin_vendedor: int = 0
    errores: list = field(default_factory=list)

    def texto(self):
        partes = [
            f"{self.pendientes} pendientes",
            f"{self.asignados} asignados",
            f"{self.existentes} a su vendedor anterior",
            f"{self.duplicados} repetidos",
        ]
        if self.sin_vendedor:
            partes.append(f"{self.sin_vendedor} sin vendedor disponible")
        if self.errores:
            partes.append(f"{len(self.errores)} errores")
        return " · ".join(partes)

    def como_dict(self):
        return asdict(self)


def consultar_pendientes(dist, limite=100):
    """Una vuelta: busca los contactos marcados, los reparte y los desmarca."""
    r = ResultadoConsulta()
    # En modo simulación nada se desmarca en HubSpot: se recuerda en memoria lo ya revisado
    simulando = hasattr(dist.hs, "escrituras")
    vistos = dist.__dict__.setdefault("_vistos_simulacion", set())
    filtros = [{"propertyName": PROPIEDAD_PENDIENTE, "operator": "EQ", "value": "true"}]
    orden = {"propertyName": "lastmodifieddate", "direction": "ASCENDING"}  # primero los que llevan más esperando
    contactos = []
    for c in dist.hs.buscar("contacts", [filtros], ("hs_object_id",), orden=orden, max_paginas=max(1, limite // 100)):
        if simulando and str(c["id"]) in vistos:
            continue
        contactos.append(c)
        if len(contactos) >= limite:
            break
    r.pendientes = len(contactos)

    for c in contactos:
        id_ = str(c["id"])
        if simulando:
            vistos.add(id_)
        try:
            resultado = dist.procesar(id_)
        except ErrorHubSpot as e:
            log.warning("Reparto: contacto %s, HubSpot respondió %s (%s)", id_, e.status, e.mensaje)
            r.errores.append(f"contacto {id_}: {e.mensaje}")
            if e.reintentable:
                continue  # queda marcado y se reintenta en la próxima vuelta
            # Error permanente (p. ej. datos inválidos): se desmarca para no repetirlo cada minuto
            _desmarcar(dist, id_, r)
            continue
        contador = {
            "asignado": "asignados",
            "existente": "existentes",
            "duplicado": "duplicados",
            "sin_vendedor": "sin_vendedor",
        }.get(resultado.accion)
        if contador:
            setattr(r, contador, getattr(r, contador) + 1)
        if resultado.accion == "sin_vendedor":
            continue  # queda marcado: se asigna apenas haya alguien disponible
        _desmarcar(dist, id_, r)
    return r


def _desmarcar(dist, contacto_id, r):
    try:
        dist.hs.actualizar("contacts", contacto_id, {PROPIEDAD_PENDIENTE: "false"})
    except ErrorHubSpot as e:
        log.warning("Reparto: no se pudo desmarcar el contacto %s (%s)", contacto_id, e.mensaje)
        r.errores.append(f"desmarcar {contacto_id}: {e.mensaje}")
