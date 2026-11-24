"""Arma un archivo de ruteo a partir de cómo se repartieron los negocios en HubSpot.

Para cada valor de "Proyecto de Interés" cuenta los negocios creados en los últimos N días
por propietario. Quienes tienen al menos `participacion_minima` del proyecto y siguen
activos quedan como vendedores, con un peso proporcional a su participación (1 a 10).
Es un punto de partida: revisa el resultado con la gerencia comercial antes de usarlo.
"""

from collections import Counter
from datetime import datetime, timedelta, timezone

import yaml

from .distribuidor import ms

BASE = {
    "zona_horaria": "America/Santiago",
    "pipeline": "default",
    "etapa_inicial": "appointmentscheduled",
    "contar_carga": "todos",
    "horario": {"lunes-viernes": "09:00-19:00", "sabado": "10:00-14:00", "domingo": "cerrado"},
    "feriados": [
        "2026-12-08", "2026-12-25", "2027-01-01", "2027-03-26", "2027-03-27", "2027-05-01", "2027-05-21",
        "2027-07-16", "2027-08-15", "2027-09-18", "2027-09-19", "2027-11-01", "2027-12-08", "2027-12-25",
    ],
    "sla": {"minutos": 15, "max_reasignaciones": 2, "supervisor": None},
    "pool_general": "todos",
}

ENCABEZADO = """\
# Ruteo de la central de WhatsApp: qué vendedor atiende cada proyecto.
# Generado con `python run_central.py sugerir-ruteo` a partir de los negocios de los últimos {dias} días
# ({fecha}). Revísalo con la gerencia comercial: es un punto de partida.
#
# - Las claves de `proyectos` son el valor interno de la propiedad "Proyecto de Interés".
# - `peso`: cuántos leads recibe cada vendedor en relación con los demás del proyecto (1 a 10). Peso 0 = pausado.
# - `vendedores` (catálogo opcional): ausencias (`ausente_hasta: 2026-12-31`), `max_diario` y horario propio.
# - `sla.supervisor`: propietario que recibe los leads que nadie atendió.
# - Este archivo tiene nombres e IDs de tu equipo: no lo subas a un repositorio público (ya está en .gitignore).
"""


def sugerir_ruteo(hs, dias=90, participacion_minima=0.08, ahora=None, supervisor=None):
    ahora = ahora or datetime.now(timezone.utc)
    desde = ahora - timedelta(days=dias)
    activos = hs.propietarios()
    nombres = Counter(info["nombre"] for info in activos.values())

    def nombre(id_):
        n = activos[id_]["nombre"]
        return f"{n} ({id_})" if nombres[n] > 1 else n

    proyectos = {}
    for valor, etiqueta in hs.opciones("contacts", "proyecto_de_interes"):
        conteo, zonas = Counter(), Counter()
        filtros = [
            {"propertyName": "createdate", "operator": "GTE", "value": ms(desde)},
            {"propertyName": "proyecto_de_interes", "operator": "EQ", "value": valor},
        ]
        for n in hs.buscar("deals", [filtros], ("hubspot_owner_id", "zona_ubicacion_del_proyecto"), max_paginas=50):
            p = n.get("properties", {})
            if p.get("hubspot_owner_id"):
                conteo[str(p["hubspot_owner_id"])] += 1
            if p.get("zona_ubicacion_del_proyecto"):
                zonas[p["zona_ubicacion_del_proyecto"]] += 1
        total = sum(conteo.values())
        if not total:
            continue
        vendedores = [
            {"id": int(id_) if id_.isdigit() else id_, "nombre": nombre(id_), "peso": max(1, round(10 * n / total))}
            for id_, n in conteo.most_common()
            if n / total >= participacion_minima and id_ in activos
        ]
        entrada = {"zona": zonas.most_common(1)[0][0] if zonas else "", "vendedores": vendedores}
        if etiqueta != valor:
            entrada["alias"] = [etiqueta]
        entrada["negocios_ultimos_dias"] = total
        proyectos[valor] = entrada

    datos = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in BASE.items()}
    if supervisor:
        datos["sla"]["supervisor"] = int(supervisor) if str(supervisor).isdigit() else supervisor
    datos["vendedores"] = {}
    datos["proyectos"] = dict(sorted(proyectos.items(), key=lambda kv: -kv[1]["negocios_ultimos_dias"]))
    return datos


def a_yaml(datos, dias=90, ahora=None):
    fecha = (ahora or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    return ENCABEZADO.format(dias=dias, fecha=fecha) + "\n" + yaml.safe_dump(
        datos, allow_unicode=True, sort_keys=False, width=120, default_flow_style=False
    )
