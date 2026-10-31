"""Cliente mínimo de la API de HubSpot (CRM v3): pipelines, negocios, propietarios y tareas.

Usa un token de *app privada* (HubSpot → Configuración → Integraciones → Aplicaciones privadas).
Permisos: crm.objects.deals.read, crm.objects.owners.read y crm.schemas.deals.read para leer;
para el modo `tareas`, además el permiso de escritura de tareas (HubSpot lo indica si falta).
"""

import logging
import time
from datetime import datetime, timezone

import requests

log = logging.getLogger(__name__)

PROPIEDADES_BASE = [
    "dealname",
    "dealstage",
    "pipeline",
    "amount",
    "hubspot_owner_id",
    "createdate",
    "closedate",
    "hs_lastmodifieddate",
    "hs_v2_date_entered_current_stage",
    "hs_v2_time_in_current_stage",
    "notes_last_contacted",
    "notes_last_updated",
    "notes_next_activity_date",
    "hs_next_step",
    "num_contacted_notes",
    "hs_is_closed",
    "hs_is_closed_lost",
    "hs_is_closed_won",
]
ASOCIACION_TAREA_NEGOCIO = 216  # id de asociación "tarea → negocio" definido por HubSpot
TAMANO_PAGINA = 200
PAUSA_ENTRE_PAGINAS = 0.25  # la API de búsqueda admite 4 llamadas por segundo


class ErrorHubSpot(RuntimeError):
    """Error al hablar con HubSpot: token, permisos, pipeline inexistente o red."""


def _fecha(valor):
    if valor in (None, ""):
        return None
    texto = str(valor)
    try:
        if texto.lstrip("-").isdigit():
            return datetime.fromtimestamp(int(texto) / 1000, tz=timezone.utc)
        return datetime.fromisoformat(texto.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, OSError, OverflowError):
        return None


def _numero(valor):
    try:
        return float(valor) if valor not in (None, "") else None
    except (TypeError, ValueError):
        return None


def normalizar_negocio(obj, propiedades_razon_perdida=("razon_perdido", "closed_lost_reason")):
    """Convierte un negocio tal como lo devuelve la API en un dict plano con fechas reales."""
    p = obj.get("properties") or {}
    razon = next((p[x] for x in propiedades_razon_perdida if p.get(x)), "")
    entradas = {}
    prefijo = "hs_v2_date_entered_"
    for clave, valor in p.items():
        if clave.startswith(prefijo) and clave != "hs_v2_date_entered_current_stage" and valor:
            fecha = _fecha(valor)
            if fecha:
                entradas[clave[len(prefijo):]] = fecha
    return {
        "id": str(obj.get("id")),
        "nombre": p.get("dealname") or f"Negocio {obj.get('id')}",
        "etapa_id": p.get("dealstage") or "",
        "pipeline": p.get("pipeline") or "",
        "monto": _numero(p.get("amount")),
        "propietario_id": str(p.get("hubspot_owner_id") or ""),
        "creado": _fecha(p.get("createdate")),
        "cerrado": _fecha(p.get("closedate")),
        "entro_etapa": _fecha(p.get("hs_v2_date_entered_current_stage")),
        "ultimo_contacto": _fecha(p.get("notes_last_contacted")),
        "ultima_actividad": _fecha(p.get("notes_last_updated")),
        "proxima_actividad": _fecha(p.get("notes_next_activity_date")),
        "proximo_paso": p.get("hs_next_step") or p.get("proximo_paso_acordado_con_el_cliente") or "",
        "contactos": int(_numero(p.get("num_contacted_notes")) or 0),
        "cerrado_perdido": str(p.get("hs_is_closed_lost")).lower() == "true",
        "cerrado_ganado": str(p.get("hs_is_closed_won")).lower() == "true",
        "razon_perdida": razon,
        "entradas": entradas,  # {id_etapa: fecha en que entró} para medir la conversión
        "propiedades": p,
    }


class ClienteHubSpot:
    def __init__(self, cfg, sesion=None):
        self.cfg = cfg
        self.sesion = sesion or requests.Session()
        self.sesion.headers.update({"Authorization": f"Bearer {cfg.token}", "Content-Type": "application/json"})

    # ------------------------------------------------------------------ transporte
    def _pedir(self, metodo, ruta, intentos=4, **kwargs):
        url = self.cfg.api_base + ruta
        for intento in range(intentos):
            try:
                r = self.sesion.request(metodo, url, timeout=60, **kwargs)
            except requests.RequestException as e:
                if intento == intentos - 1:
                    raise ErrorHubSpot(f"Sin conexión con HubSpot: {e}") from e
                time.sleep(2**intento)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                try:
                    espera = float(r.headers.get("Retry-After", 2**intento))
                except (TypeError, ValueError):
                    espera = 2**intento
                log.warning("HubSpot respondió %s; reintento en %.0f s", r.status_code, espera)
                time.sleep(min(espera, 30))
                continue
            if r.status_code == 401:
                raise ErrorHubSpot("HubSpot rechazó el token (401). Revisa HUBSPOT_TOKEN y que la app privada esté activa.")
            if r.status_code == 403:
                raise ErrorHubSpot(f"HubSpot negó el permiso (403) para {metodo} {ruta}. Revisa los permisos de la app privada.")
            if r.status_code >= 400:
                raise ErrorHubSpot(f"HubSpot respondió {r.status_code} en {metodo} {ruta}: {r.text[:300]}")
            return r.json() if r.text else {}
        raise ErrorHubSpot(f"HubSpot no respondió tras {intentos} intentos ({metodo} {ruta})")

    # ------------------------------------------------------------------ pipeline
    def etapas(self):
        """Devuelve (etapas, id_pipeline, nombre_pipeline) del pipeline configurado (por id o por nombre)."""
        datos = self._pedir("GET", "/crm/v3/pipelines/deals")
        pipelines = datos.get("results", [])
        buscado = self.cfg.pipeline.strip().lower()
        for p in pipelines:
            if p["id"].lower() == buscado or (p.get("label") or "").strip().lower() == buscado:
                etapas = sorted(p.get("stages", []), key=lambda s: s.get("displayOrder", 0))
                return (
                    [
                        {
                            "id": e["id"],
                            "nombre": e.get("label") or e["id"],
                            "orden": i,
                            "probabilidad": float((e.get("metadata") or {}).get("probability") or 0),
                        }
                        for i, e in enumerate(etapas)
                    ],
                    p["id"],
                    p.get("label") or p["id"],
                )
        disponibles = ", ".join(f"{p.get('label')} (id {p['id']})" for p in pipelines) or "ninguno"
        raise ErrorHubSpot(f"No existe el pipeline '{self.cfg.pipeline}'. Disponibles: {disponibles}")

    # ------------------------------------------------------------------ negocios
    def _propiedades(self, etapas):
        props = list(PROPIEDADES_BASE) + [p for p in self.cfg.propiedades_extra if p]
        props += [f"hs_v2_date_entered_{e['id']}" for e in etapas]
        return list(dict.fromkeys(props))

    def _buscar_negocios(self, filtros, propiedades):
        """Recorre todos los resultados paginando por hs_object_id (evita el tope de 10.000 de la API)."""
        resultados, ultimo_id = [], None
        while True:
            filtros_pagina = list(filtros)
            if ultimo_id is not None:
                filtros_pagina.append({"propertyName": "hs_object_id", "operator": "GT", "value": str(ultimo_id)})
            cuerpo = {
                "filterGroups": [{"filters": filtros_pagina}],
                "properties": propiedades,
                "sorts": [{"propertyName": "hs_object_id", "direction": "ASCENDING"}],
                "limit": TAMANO_PAGINA,
            }
            datos = self._pedir("POST", "/crm/v3/objects/deals/search", json=cuerpo)
            pagina = datos.get("results", [])
            if not pagina:
                break
            resultados.extend(normalizar_negocio(n, self.cfg.propiedades_razon_perdida) for n in pagina)
            ultimo_id = int(pagina[-1]["id"])
            if len(pagina) < TAMANO_PAGINA:
                break
            time.sleep(PAUSA_ENTRE_PAGINAS)
        return resultados

    def negocios_abiertos(self, pipeline_id, etapas):
        filtros = [
            {"propertyName": "pipeline", "operator": "EQ", "value": pipeline_id},
            {"propertyName": "hs_is_closed", "operator": "EQ", "value": "false"},
        ]
        return self._buscar_negocios(filtros, self._propiedades(etapas))

    def negocios_creados_desde(self, pipeline_id, etapas, desde):
        filtros = [
            {"propertyName": "pipeline", "operator": "EQ", "value": pipeline_id},
            {"propertyName": "createdate", "operator": "GTE", "value": str(int(desde.timestamp() * 1000))},
        ]
        return self._buscar_negocios(filtros, self._propiedades(etapas))

    # ------------------------------------------------------------------ propietarios y cuenta
    def propietarios(self):
        """{id: {'nombre', 'email'}} de los usuarios del CRM (incluye archivados para nombrar negocios viejos)."""
        resultado = {}
        for archivados in ("false", "true"):
            after = None
            while True:
                params = {"limit": 100, "archived": archivados}
                if after:
                    params["after"] = after
                datos = self._pedir("GET", "/crm/v3/owners", params=params)
                for o in datos.get("results", []):
                    nombre = " ".join(x for x in (o.get("firstName"), o.get("lastName")) if x)
                    resultado.setdefault(str(o["id"]), {"nombre": nombre or o.get("email") or str(o["id"]), "email": o.get("email", "")})
                after = ((datos.get("paging") or {}).get("next") or {}).get("after")
                if not after:
                    break
        return resultado

    def portal_id(self):
        if self.cfg.portal_id:
            return self.cfg.portal_id
        try:
            datos = self._pedir("GET", "/account-info/v3/details")
            return str(datos.get("portalId") or "")
        except ErrorHubSpot as e:
            log.info("No se pudo leer el id del portal (%s); el informe irá sin enlaces a los negocios.", e)
            return ""

    # ------------------------------------------------------------------ tareas
    def crear_tarea(self, negocio_id, propietario_id, asunto, cuerpo, vence, prioridad="HIGH", tipo="CALL"):
        """Crea una tarea asociada al negocio y devuelve su id."""
        propiedades = {
            "hs_timestamp": str(int(vence.timestamp() * 1000)),
            "hs_task_subject": asunto[:250],
            "hs_task_body": cuerpo[:5000],
            "hs_task_status": "NOT_STARTED",
            "hs_task_priority": prioridad,
            "hs_task_type": tipo,
        }
        if propietario_id:
            propiedades["hubspot_owner_id"] = str(propietario_id)
        cuerpo_api = {
            "properties": propiedades,
            "associations": [
                {
                    "to": {"id": str(negocio_id)},
                    "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": ASOCIACION_TAREA_NEGOCIO}],
                }
            ],
        }
        datos = self._pedir("POST", "/crm/v3/objects/tasks", json=cuerpo_api)
        return str(datos.get("id", ""))
