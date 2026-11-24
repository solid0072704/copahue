"""Cliente mínimo de la API de HubSpot (CRM v3/v4) para la central de WhatsApp.

Usa el token de una app privada. Reintenta solo ante límites de tasa (429) y errores
del servidor (5xx). Nunca escribe datos personales en los logs: solo IDs.
"""

import logging
import time

import requests

log = logging.getLogger(__name__)

BASE = "https://api.hubapi.com"

# Tipos de asociación definidos por HubSpot (objeto origen, objeto destino) -> associationTypeId
TIPOS_ASOCIACION = {
    ("deals", "contacts"): 3,
    ("contacts", "deals"): 4,
    ("tasks", "contacts"): 204,
    ("tasks", "deals"): 216,
    ("notes", "contacts"): 202,
    ("notes", "deals"): 214,
}

# Identificadores de tipo de objeto para armar links a los registros
TIPO_OBJETO = {"contacts": "0-1", "deals": "0-3", "tasks": "0-27", "notes": "0-46"}


class ErrorHubSpot(Exception):
    def __init__(self, status, mensaje, detalle=""):
        super().__init__(f"HubSpot {status}: {mensaje}")
        self.status = status
        self.mensaje = mensaje
        self.detalle = detalle

    @property
    def reintentable(self):
        """Errores pasajeros (o de credenciales) que conviene reintentar más tarde."""
        return self.status in (401, 403, 429) or self.status >= 500


class ClienteHubSpot:
    def __init__(self, token, portal_id="", sesion=None, reintentos=4, timeout=20):
        if not token:
            raise ValueError("Falta HUBSPOT_TOKEN (token de la app privada de HubSpot).")
        self.token = token
        self._portal_id = str(portal_id or "")
        self.sesion = sesion or requests.Session()
        self.reintentos = reintentos
        self.timeout = timeout
        self._propietarios = None
        self._propietarios_ts = 0.0

    # ------------------------------------------------------------------ HTTP
    def _req(self, metodo, ruta, **kwargs):
        cabeceras = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        for intento in range(self.reintentos + 1):
            try:
                r = self.sesion.request(metodo, BASE + ruta, headers=cabeceras, timeout=self.timeout, **kwargs)
            except requests.RequestException as e:
                if intento < self.reintentos:
                    time.sleep(min(2**intento, 10))
                    continue
                raise ErrorHubSpot(599, f"sin conexión con HubSpot ({type(e).__name__})") from e
            if (r.status_code == 429 or r.status_code >= 500) and intento < self.reintentos:
                try:
                    espera = float(r.headers.get("Retry-After", "0"))
                except ValueError:
                    espera = 0
                time.sleep(espera or min(2**intento, 10))
                continue
            if r.status_code >= 400:
                try:
                    cuerpo = r.json()
                    mensaje = cuerpo.get("message") or cuerpo.get("category") or r.reason
                except ValueError:
                    mensaje = r.reason
                raise ErrorHubSpot(r.status_code, mensaje, r.text[:800])
            return r.json() if r.content else {}
        raise ErrorHubSpot(599, "sin respuesta de HubSpot")

    # ------------------------------------------------------------- registros
    def leer(self, objeto, id_, propiedades=()):
        params = {"properties": ",".join(propiedades)} if propiedades else None
        return self._req("GET", f"/crm/v3/objects/{objeto}/{id_}", params=params)

    def leer_lote(self, objeto, ids, propiedades=()):
        ids = [str(i) for i in ids if i]
        resultados = []
        for i in range(0, len(ids), 100):
            cuerpo = {"inputs": [{"id": x} for x in ids[i : i + 100]], "properties": list(propiedades)}
            resultados.extend(self._req("POST", f"/crm/v3/objects/{objeto}/batch/read", json=cuerpo).get("results", []))
        return resultados

    def buscar(self, objeto, grupos, propiedades=(), orden=None, limite=200, max_paginas=50):
        """Búsqueda CRM. `grupos` es una lista de listas de filtros (OR entre grupos, AND dentro)."""
        cuerpo = {
            "filterGroups": [{"filters": filtros} for filtros in grupos],
            "properties": list(propiedades),
            "limit": min(limite, 200),
        }
        if orden:
            cuerpo["sorts"] = [orden]
        for _ in range(max_paginas):
            datos = self._req("POST", f"/crm/v3/objects/{objeto}/search", json=cuerpo)
            yield from datos.get("results", [])
            siguiente = datos.get("paging", {}).get("next", {}).get("after")
            if not siguiente:
                return
            cuerpo["after"] = siguiente

    def crear(self, objeto, propiedades, asociaciones=()):
        """Crea un registro. `asociaciones`: lista de (objeto_destino, id_destino)."""
        cuerpo = {"properties": {k: v for k, v in propiedades.items() if v is not None}}
        if asociaciones:
            cuerpo["associations"] = [
                {
                    "to": {"id": str(id_destino)},
                    "types": [
                        {"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": TIPOS_ASOCIACION[(objeto, destino)]}
                    ],
                }
                for destino, id_destino in asociaciones
                if id_destino
            ]
        return self._req("POST", f"/crm/v3/objects/{objeto}", json=cuerpo)

    def actualizar(self, objeto, id_, propiedades):
        return self._req("PATCH", f"/crm/v3/objects/{objeto}/{id_}", json={"properties": propiedades})

    def asociados(self, objeto, id_, destino):
        datos = self._req("GET", f"/crm/v4/objects/{objeto}/{id_}/associations/{destino}", params={"limit": 500})
        return [str(x["toObjectId"]) for x in datos.get("results", [])]

    # ---------------------------------------------------------- propietarios
    def propietarios(self, cache_seg=600):
        """Propietarios activos: {owner_id: {"nombre", "email"}} (en caché 10 minutos)."""
        if self._propietarios is not None and time.time() - self._propietarios_ts < cache_seg:
            return self._propietarios
        resultado, params = {}, {"limit": 500}
        while True:
            datos = self._req("GET", "/crm/v3/owners", params=params)
            for o in datos.get("results", []):
                if o.get("archived"):
                    continue
                nombre = f"{o.get('firstName') or ''} {o.get('lastName') or ''}".strip() or o.get("email", "")
                resultado[str(o["id"])] = {"nombre": nombre, "email": o.get("email", "")}
            siguiente = datos.get("paging", {}).get("next", {}).get("after")
            if not siguiente:
                break
            params["after"] = siguiente
        self._propietarios, self._propietarios_ts = resultado, time.time()
        return resultado

    # ------------------------------------------------------------ propiedades
    def propiedades(self, objeto):
        return {p["name"]: p for p in self._req("GET", f"/crm/v3/properties/{objeto}").get("results", [])}

    def opciones(self, objeto, propiedad):
        datos = self._req("GET", f"/crm/v3/properties/{objeto}/{propiedad}")
        return [(o["value"], o.get("label", o["value"])) for o in datos.get("options", []) if not o.get("hidden")]

    def crear_grupo(self, objeto, nombre, etiqueta):
        try:
            return self._req("POST", f"/crm/v3/properties/{objeto}/groups", json={"name": nombre, "label": etiqueta})
        except ErrorHubSpot as e:
            if e.status == 409:  # ya existe
                return {}
            raise

    def crear_propiedad(self, objeto, definicion):
        return self._req("POST", f"/crm/v3/properties/{objeto}", json=definicion)

    # ------------------------------------------------------------------ links
    def portal(self):
        if not self._portal_id:
            self._portal_id = str(self._req("GET", "/account-info/v3/details").get("portalId", ""))
        return self._portal_id

    def url_registro(self, objeto, id_):
        try:
            portal = self.portal()
        except ErrorHubSpot:
            return ""
        return f"https://app.hubspot.com/contacts/{portal}/record/{TIPO_OBJETO[objeto]}/{id_}"


class ClienteSoloLectura:
    """Envuelve un cliente real: lee de HubSpot, pero las escrituras solo se registran (modo simulación)."""

    def __init__(self, cliente):
        self._c = cliente
        self.escrituras = []

    def __getattr__(self, nombre):
        return getattr(self._c, nombre)

    def crear(self, objeto, propiedades, asociaciones=()):
        self.escrituras.append(("crear", objeto, dict(propiedades), list(asociaciones)))
        return {"id": f"simulado-{objeto}-{len(self.escrituras)}", "properties": dict(propiedades)}

    def actualizar(self, objeto, id_, propiedades):
        self.escrituras.append(("actualizar", objeto, id_, dict(propiedades)))
        return {"id": id_, "properties": dict(propiedades)}

    def crear_grupo(self, objeto, nombre, etiqueta):
        self.escrituras.append(("crear_grupo", objeto, nombre))
        return {}

    def crear_propiedad(self, objeto, definicion):
        self.escrituras.append(("crear_propiedad", objeto, definicion["name"]))
        return {}
