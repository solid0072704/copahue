"""Pruebas de la central de WhatsApp con un HubSpot simulado en memoria (sin internet)."""

import itertools
from collections import Counter
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from central_whatsapp.config import ConfigCentral
from central_whatsapp.distribuidor import Distribuidor, iso, parse_fecha, telefono_whatsapp
from central_whatsapp.hubspot import ClienteSoloLectura, ErrorHubSpot
from central_whatsapp.instalar import PROPIEDADES, instalar
from central_whatsapp.ruteo import TablaRuteo, elegir_vendedor, normalizar
from central_whatsapp.servicio import crear_app, extraer_contacto_id, extraer_proyecto
from central_whatsapp.sla import revisar_sla
from central_whatsapp.sugerir import a_yaml, sugerir_ruteo

TZ = ZoneInfo("America/Santiago")
# Martes 24-nov-2026 10:00 en Santiago (horario de verano, UTC-3)
MARTES_10 = datetime(2026, 11, 24, 10, 0, tzinfo=TZ)

YAML = """
zona_horaria: America/Santiago
pipeline: default
etapa_inicial: appointmentscheduled
horario:
  lunes-viernes: "09:00-19:00"
  sabado: "10:00-14:00"
  domingo: cerrado
feriados: [2026-12-25]
sla: {minutos: 15, max_reasignaciones: 2, supervisor: 900}
vendedores:
  "103": {ausente_hasta: 2026-11-30}
proyectos:
  Raíces de Curicó:
    zona: Zona Centro
    vendedores:
      - {id: 101, nombre: Ana, peso: 2}
      - {id: 102, nombre: Beto, peso: 1}
      - {id: 103, nombre: Carla, peso: 1}
  Raíces de Los Ulmos:
    zona: Zona Sur
    alias: [Raíces Los Ulmos]
    vendedores:
      - {id: 201, nombre: Diego}
      - {id: 202, nombre: Elena}
  Raíces de Frutillar: {zona: Zona Sur}
"""


class FakeHubSpot:
    """Implementa la misma interfaz que ClienteHubSpot, guardando todo en memoria."""

    def __init__(self, propietarios=None):
        self.objetos = {"contacts": {}, "deals": {}, "tasks": {}, "notes": {}}
        self.asoc = set()  # (objeto, id, destino, id_destino)
        self._ids = itertools.count(5000)
        self.owners = propietarios or {
            str(i): {"nombre": f"Vendedor {i}", "email": f"v{i}@empresa.example"}
            for i in (101, 102, 103, 201, 202, 900)
        }
        self.props_existentes = {"contacts": {"firstname": {}}, "deals": {"dealname": {}}}
        self.grupos = []
        self.creadas = []
        self.opciones_proyecto = []
        self.fallar_con = None
        self.reloj = None

    # --- utilidades de prueba
    def agregar(self, objeto, props, id_=None, asociar=()):
        id_ = str(id_ or next(self._ids))
        self.objetos[objeto][id_] = {k: (None if v is None else str(v)) for k, v in props.items()}
        for destino, id_destino in asociar:
            self._asociar(objeto, id_, destino, str(id_destino))
        return id_

    def _asociar(self, a, ida, b, idb):
        self.asoc.add((a, ida, b, idb))
        self.asoc.add((b, idb, a, ida))

    # --- interfaz
    def leer(self, objeto, id_, propiedades=()):
        if str(id_) not in self.objetos[objeto]:
            raise ErrorHubSpot(404, "not found")
        return {"id": str(id_), "properties": dict(self.objetos[objeto][str(id_)])}

    def leer_lote(self, objeto, ids, propiedades=()):
        return [self.leer(objeto, i) for i in ids if str(i) in self.objetos[objeto]]

    def _cumple(self, props, f):
        valor, op = props.get(f["propertyName"]), f["operator"]
        if op == "EQ":
            return valor == f["value"]
        if op == "IN":
            return valor in f["values"]
        if op in ("GTE", "LT"):
            fecha = parse_fecha(valor)
            limite = parse_fecha(f["value"])
            if fecha is None:
                return False
            return fecha >= limite if op == "GTE" else fecha < limite
        if op == "HAS_PROPERTY":
            return bool(valor)
        raise NotImplementedError(op)

    def buscar(self, objeto, grupos, propiedades=(), orden=None, limite=200, max_paginas=50):
        for id_, props in list(self.objetos[objeto].items()):
            if any(all(self._cumple(props, f) for f in filtros) for filtros in grupos):
                yield {"id": id_, "properties": dict(props)}

    def crear(self, objeto, propiedades, asociaciones=()):
        if self.fallar_con:
            raise self.fallar_con
        props = {k: v for k, v in propiedades.items() if v is not None}
        props.setdefault("createdate", iso(self.reloj() if self.reloj else datetime.now(timezone.utc)))
        if objeto == "deals":
            props.setdefault("hs_is_closed", "false")
        id_ = self.agregar(objeto, props)
        for destino, id_destino in asociaciones:
            if id_destino:
                self._asociar(objeto, id_, destino, str(id_destino))
        return {"id": id_, "properties": props}

    def actualizar(self, objeto, id_, propiedades):
        self.objetos[objeto][str(id_)].update({k: str(v) for k, v in propiedades.items()})
        return {"id": str(id_)}

    def asociados(self, objeto, id_, destino):
        return sorted(idb for (a, ida, b, idb) in self.asoc if a == objeto and ida == str(id_) and b == destino)

    def propietarios(self):
        return self.owners

    def propiedades(self, objeto):
        return self.props_existentes.get(objeto, {})

    def opciones(self, objeto, propiedad):
        return self.opciones_proyecto

    def crear_grupo(self, objeto, nombre, etiqueta):
        self.grupos.append((objeto, nombre))

    def crear_propiedad(self, objeto, definicion):
        self.creadas.append((objeto, definicion["name"]))

    def url_registro(self, objeto, id_):
        return f"https://app.hubspot.com/contacts/1/record/0-1/{id_}"

    # --- consultas de prueba
    def negocios_de(self, contacto_id):
        return [self.objetos["deals"][i] | {"id": i} for i in self.asociados("contacts", contacto_id, "deals")]


class Reloj:
    def __init__(self, inicio):
        self.ahora = inicio

    def __call__(self):
        return self.ahora

    def avanzar(self, **kw):
        self.ahora += timedelta(**kw)


@pytest.fixture
def tabla():
    return TablaRuteo.desde_texto(YAML)


@pytest.fixture
def reloj():
    return Reloj(MARTES_10)


@pytest.fixture
def hs(reloj):
    falso = FakeHubSpot()
    falso.reloj = reloj
    return falso


@pytest.fixture
def dist(tabla, hs, reloj):
    cfg = ConfigCentral(hubspot_token="x", webhook_secreto="secreto")
    return Distribuidor(cfg, tabla, hs, reloj=reloj)


def contacto(hs, proyecto="Raíces de Curicó", **extra):
    props = {
        "firstname": "Cliente",
        "lastname": "Prueba",
        "hs_whatsapp_phone_number": "+56 9 5550 1234",
        "proyecto_de_interes": proyecto,
        "central_plazo_compra": "1_3_meses",
    }
    props.update(extra)
    return hs.agregar("contacts", props)


# ------------------------------------------------------------------ ruteo
def test_normalizar_y_alias(tabla):
    assert normalizar("Raíces del Río Paloma") == normalizar("Raices de Rio Paloma") == "rio paloma"
    assert tabla.proyecto("Raíces Los Ulmos").nombre == "Raíces de Los Ulmos"
    assert tabla.proyecto("raices de curico").nombre == "Raíces de Curicó"
    assert tabla.proyecto("Proyecto inexistente") is None
    assert not tabla.mismo_proyecto("Raíces de Curicó", "Raíces de Los Ulmos")


def test_candidatos_por_proyecto_zona_y_general(tabla):
    vendedores, origen, _ = tabla.candidatos("Raíces de Curicó")
    assert origen == "proyecto" and [v.id for v in vendedores] == ["101", "102", "103"]
    vendedores, origen, _ = tabla.candidatos("Raíces de Frutillar")  # sin vendedores: su zona
    assert origen == "zona" and {v.id for v in vendedores} == {"201", "202"}
    vendedores, origen, _ = tabla.candidatos("")
    assert origen == "general" and {v.id for v in vendedores} == {"101", "102", "103", "201", "202"}


def test_reparto_respeta_pesos_y_ausencias(tabla):
    candidatos, _, _ = tabla.candidatos("Raíces de Curicó")
    cargas = Counter()
    for _ in range(30):
        v = elegir_vendedor(tabla, candidatos, MARTES_10, cargas, cargas)
        cargas[v.id] += 1
    assert cargas == {"101": 20, "102": 10}  # Carla (103) está de vacaciones hasta el 30-nov
    diciembre = datetime(2026, 12, 1, 10, 0, tzinfo=TZ)
    assert elegir_vendedor(tabla, candidatos, diciembre, {"101": 2, "102": 1}, {}).id == "103"


def test_elegir_descarta_inactivos_excluidos_y_tope(tabla):
    candidatos, _, _ = tabla.candidatos("Raíces de Curicó")
    assert elegir_vendedor(tabla, candidatos, MARTES_10, {}, {}, activos={"102"}).id == "102"
    assert elegir_vendedor(tabla, candidatos, MARTES_10, {}, {}, excluir={"101", "102"}) is None
    tabla.catalogo["101"] = {"max_diario": 3}
    assert elegir_vendedor(tabla, candidatos, MARTES_10, {}, {"101": 3}).id == "102"


def test_minutos_habiles(tabla):
    viernes_1850 = datetime(2026, 11, 27, 18, 50, tzinfo=TZ)
    sabado_1010 = datetime(2026, 11, 28, 10, 10, tzinfo=TZ)
    assert tabla.minutos_habiles(viernes_1850, sabado_1010) == pytest.approx(20)
    # 15 minutos hábiles desde el sábado 13:55 vencen el lunes 09:10 (domingo cerrado)
    vence = tabla.sumar_minutos_habiles(datetime(2026, 11, 28, 13, 55, tzinfo=TZ), 15)
    assert vence.astimezone(TZ) == datetime(2026, 11, 30, 9, 10, tzinfo=TZ)
    # feriado: el 25-dic no cuenta
    navidad = datetime(2026, 12, 25, 12, 0, tzinfo=TZ)
    assert tabla.minutos_habiles(navidad, navidad + timedelta(hours=2)) == 0


def test_validar_detecta_inactivos(tabla):
    avisos = tabla.validar({"101": {}, "102": {}, "201": {}, "202": {}, "900": {}})
    assert any("103" in a for a in avisos)


# ------------------------------------------------------------- distribuidor
def test_lead_nuevo_crea_negocio_tarea_y_propietario(dist, hs):
    c = contacto(hs)
    r = dist.procesar(c)
    assert r.accion == "asignado" and r.vendedor_id == "101" and r.origen == "proyecto"
    negocio = hs.objetos["deals"][r.negocio_id]
    assert negocio["pipeline"] == "default" and negocio["dealstage"] == "appointmentscheduled"
    assert negocio["proyecto_de_interes"] == "Raíces de Curicó" and negocio["zona_ubicacion_del_proyecto"] == "Zona Centro"
    assert negocio["fuente"] == "WhatsApp Central" and negocio["canal_de_contacto_principal"] == "Whatsapp"
    assert negocio["central_estado"] == "asignado" and negocio["central_tarea_id"] == r.tarea_id
    tarea = hs.objetos["tasks"][r.tarea_id]
    assert tarea["hubspot_owner_id"] == "101" and "wa.me/56955501234" in tarea["hs_task_body"]
    assert "En 1 a 3 meses" in tarea["hs_task_body"]
    vence = parse_fecha(tarea["hs_timestamp"])
    assert vence == MARTES_10 + timedelta(minutes=15)
    assert hs.objetos["contacts"][c]["hubspot_owner_id"] == "101"
    assert c in hs.asociados("deals", r.negocio_id, "contacts")


def test_reparto_entre_vendedores_por_carga_del_dia(dist, hs):
    asignados = Counter(dist.procesar(contacto(hs)).vendedor_id for _ in range(6))
    assert asignados == {"101": 4, "102": 2}


def test_aviso_repetido_no_duplica(dist, hs, reloj):
    c = contacto(hs)
    primero = dist.procesar(c)
    reloj.avanzar(minutes=2)
    segundo = dist.procesar(c)
    assert segundo.accion == "duplicado" and segundo.negocio_id == primero.negocio_id
    assert len(hs.objetos["deals"]) == 1 and len(hs.objetos["tasks"]) == 1


def test_cliente_que_vuelve_queda_con_su_vendedor(dist, hs):
    c = contacto(hs, hubspot_owner_id="102")
    negocio = hs.agregar(
        "deals",
        {"pipeline": "default", "dealstage": "189519371", "hubspot_owner_id": "102", "hs_is_closed": "false",
         "proyecto_de_interes": "Raíces de Curicó", "createdate": "2026-10-01T12:00:00Z", "fuente": "google"},
        asociar=[("contacts", c)],
    )
    r = dist.procesar(c)
    assert r.accion == "existente" and r.negocio_id == negocio and r.vendedor_id == "102"
    assert len(hs.objetos["deals"]) == 1
    assert hs.objetos["tasks"][r.tarea_id]["hs_task_subject"].startswith("Cliente volvió a escribir")
    assert hs.objetos["deals"][negocio]["central_estado"] == "existente"


def test_negocio_abierto_con_vendedor_inactivo_se_reasigna(dist, hs):
    c = contacto(hs, hubspot_owner_id="777")
    negocio = hs.agregar(
        "deals",
        {"pipeline": "default", "dealstage": "appointmentscheduled", "hubspot_owner_id": "777",
         "hs_is_closed": "false", "proyecto_de_interes": "Raíces de Curicó", "createdate": "2026-10-01T12:00:00Z"},
        asociar=[("contacts", c)],
    )
    r = dist.procesar(c)
    assert r.accion == "asignado" and r.negocio_id == negocio and r.vendedor_id == "101"
    assert len(hs.objetos["deals"]) == 1 and hs.objetos["deals"][negocio]["hubspot_owner_id"] == "101"
    assert hs.objetos["contacts"][c]["hubspot_owner_id"] == "101"  # propietario inactivo: se reemplaza


def test_otro_proyecto_no_le_quita_el_contacto_a_su_vendedor(dist, hs):
    c = contacto(hs, proyecto="Raíces de Los Ulmos", hubspot_owner_id="102")
    hs.agregar(
        "deals",
        {"pipeline": "default", "dealstage": "189519371", "hubspot_owner_id": "102", "hs_is_closed": "false",
         "proyecto_de_interes": "Raíces de Curicó", "createdate": "2026-10-01T12:00:00Z"},
        asociar=[("contacts", c)],
    )
    r = dist.procesar(c)
    assert r.accion == "asignado" and r.vendedor_id in ("201", "202")
    assert len(hs.negocios_de(c)) == 2
    assert hs.objetos["contacts"][c]["hubspot_owner_id"] == "102"  # política si_libre


def test_sin_proyecto_va_al_pool_general(dist, hs):
    r = dist.procesar(contacto(hs, proyecto=""))
    assert r.accion == "asignado" and r.origen == "general"
    assert "proyecto_de_interes" not in hs.objetos["deals"][r.negocio_id]


def test_sin_vendedores_va_al_supervisor(dist, hs):
    hs.owners = {"900": {"nombre": "Supervisor", "email": ""}}
    r = dist.procesar(contacto(hs))
    assert r.accion == "asignado" and r.vendedor_id == "900" and r.origen == "supervisor"


def test_propiedades_faltantes_no_bloquean(dist, hs):
    """Si no se corrió `instalar`, HubSpot rechaza las propiedades central_*: el lead igual se asigna."""
    original = hs.crear

    def crear(objeto, propiedades, asociaciones=()):
        if objeto == "deals" and any(k.startswith("central_") for k in propiedades):
            raise ErrorHubSpot(400, "Property values were not valid")
        return original(objeto, propiedades, asociaciones)

    hs.crear = crear
    r = dist.procesar(contacto(hs))
    assert r.accion == "asignado" and hs.objetos["deals"][r.negocio_id]["hubspot_owner_id"] == "101"


def test_telefono_whatsapp():
    assert telefono_whatsapp({"phone": "9 5550 1234"}) == "56955501234"
    assert telefono_whatsapp({"mobilephone": "+56 9 5550 1234"}) == "56955501234"
    assert telefono_whatsapp({}) == ""


def test_modo_simulacion_no_escribe(tabla, hs, reloj):
    c = contacto(hs)
    lectura = ClienteSoloLectura(hs)
    d = Distribuidor(ConfigCentral(hubspot_token="x"), tabla, lectura, reloj=reloj)
    r = d.procesar(c)
    assert r.accion == "asignado" and not hs.objetos["deals"] and not hs.objetos["tasks"]
    assert [e[1] for e in lectura.escrituras if e[0] == "crear"] == ["deals", "tasks"]


# --------------------------------------------------------------------- SLA
def test_sla_reasigna_escala_y_registra_atencion(dist, hs, reloj):
    c = contacto(hs)
    r = dist.procesar(c)
    negocio = hs.objetos["deals"][r.negocio_id]

    reloj.avanzar(minutes=10)
    assert revisar_sla(dist).en_plazo == 1

    reloj.avanzar(minutes=6)  # 16 minutos sin atender
    res = revisar_sla(dist)
    assert res.reasignados == 1
    assert negocio["hubspot_owner_id"] == "102" and negocio["central_estado"] == "reasignado"
    assert negocio["central_vendedores_previos"] == "101" and negocio["central_reasignaciones"] == "1"
    assert hs.objetos["tasks"][r.tarea_id]["hubspot_owner_id"] == "102"
    assert hs.objetos["contacts"][c]["hubspot_owner_id"] == "102"
    assert any("no atendió" in n["hs_note_body"] for n in hs.objetos["notes"].values())

    reloj.avanzar(minutes=16)  # tampoco atendió: no queda nadie más (Carla está ausente) -> supervisor
    res = revisar_sla(dist)
    assert res.escalados == 1 and negocio["central_estado"] == "escalado"
    tareas_supervisor = [t for t in hs.objetos["tasks"].values() if t["hubspot_owner_id"] == "900"]
    assert len(tareas_supervisor) == 1


def test_sla_detecta_tarea_completada(dist, hs, reloj):
    r = dist.procesar(contacto(hs))
    reloj.avanzar(minutes=8)
    hs.objetos["tasks"][r.tarea_id].update({"hs_task_status": "COMPLETED", "hs_task_completion_date": iso(reloj.ahora)})
    reloj.avanzar(minutes=30)
    res = revisar_sla(dist)
    assert res.atendidos == 1
    negocio = hs.objetos["deals"][r.negocio_id]
    assert negocio["central_estado"] == "atendido" and negocio["central_minutos_respuesta"] == "8"
    assert revisar_sla(dist).revisados == 0  # ya no está en seguimiento


def test_sla_no_cuenta_fuera_de_horario(dist, hs, reloj):
    reloj.ahora = datetime(2026, 11, 28, 13, 55, tzinfo=TZ)  # sábado 13:55, cierra a las 14:00
    dist.procesar(contacto(hs))
    reloj.ahora = datetime(2026, 11, 30, 9, 5, tzinfo=TZ)  # lunes 09:05: solo 10 minutos hábiles
    assert revisar_sla(dist).en_plazo == 1


def test_sla_etapa_movida_cuenta_como_atendido(dist, hs, reloj):
    r = dist.procesar(contacto(hs))
    hs.objetos["deals"][r.negocio_id]["dealstage"] = "232657274"  # Interés Confirmado
    reloj.avanzar(minutes=40)
    assert revisar_sla(dist).atendidos == 1


# ----------------------------------------------------------------- servicio
@pytest.fixture
def cliente_web(dist):
    app = crear_app(dist.cfg, distribuidor=dist, iniciar_sla=False)
    with TestClient(app) as cliente:
        yield cliente


def test_webhook_exige_token(cliente_web):
    assert cliente_web.post("/webhooks/hubspot/lead", json={"objectId": 1}).status_code == 401
    assert cliente_web.get("/salud").json()["webhook_protegido"] is True


def test_webhook_asigna_lead(cliente_web, hs):
    c = contacto(hs)
    r = cliente_web.post("/webhooks/hubspot/lead", json={"objectId": int(c)}, headers={"X-Central-Token": "secreto"})
    assert r.status_code == 200 and r.json()["accion"] == "asignado"
    r = cliente_web.post("/webhooks/hubspot/lead", json={"objectId": int(c)}, headers={"X-Central-Token": "secreto"})
    assert r.json()["accion"] == "duplicado"


def test_webhook_errores_de_hubspot(cliente_web, hs):
    headers = {"X-Central-Token": "secreto"}
    assert cliente_web.post("/webhooks/hubspot/lead", json={}, headers=headers).status_code == 400
    assert cliente_web.post("/webhooks/hubspot/lead", json={"objectId": 999999}, headers=headers).status_code == 404
    c = contacto(hs)
    hs.fallar_con = ErrorHubSpot(429, "rate limit")
    assert cliente_web.post("/webhooks/hubspot/lead", json={"objectId": c}, headers=headers).status_code == 503


def test_webhook_sin_secreto_configurado_no_opera(tabla, hs, reloj):
    d = Distribuidor(ConfigCentral(hubspot_token="x"), tabla, hs, reloj=reloj)
    with TestClient(crear_app(d.cfg, distribuidor=d, iniciar_sla=False)) as cliente:
        assert cliente.post("/webhooks/hubspot/lead", json={"objectId": 1}).status_code == 503


def test_extraer_contacto_y_proyecto():
    assert extraer_contacto_id({"objectId": 123}) == "123"
    assert extraer_contacto_id({"vid": 55, "properties": {}}) == "55"
    assert extraer_contacto_id({"properties": {"hs_object_id": {"value": "77"}}}) == "77"
    assert extraer_contacto_id([{"objectId": 9}]) == "9"
    assert extraer_contacto_id({}, {"contacto": "42"}) == "42"
    assert extraer_contacto_id({"objectId": "abc"}) == ""
    assert extraer_proyecto({"properties": {"proyecto_de_interes": {"value": "Raíces de Putu"}}}) == "Raíces de Putu"
    assert extraer_proyecto({}, {"proyecto": "X"}) == "X"


# ---------------------------------------------------------- instalar/sugerir
def test_instalar_es_idempotente(hs):
    hs.props_existentes["contacts"]["central_resumen"] = {}
    resumen = instalar(hs)
    creadas = {n for _, n in hs.creadas}
    assert "central_resumen" not in creadas and "central_estado" in creadas
    assert len(resumen) == sum(len(v) for v in PROPIEDADES.values())


def test_sugerir_ruteo(hs):
    hs.opciones_proyecto = [("Raíces de Curicó", "Raíces de Curicó"), ("Raíces de Los Ulmos", "Raíces Los Ulmos")]
    for dueño, n in (("101", 12), ("102", 6), ("103", 1)):
        for _ in range(n):
            hs.agregar("deals", {"proyecto_de_interes": "Raíces de Curicó", "hubspot_owner_id": dueño,
                                 "zona_ubicacion_del_proyecto": "Zona Centro", "createdate": iso(MARTES_10)})
    hs.agregar("deals", {"proyecto_de_interes": "Raíces de Curicó", "hubspot_owner_id": "999",
                         "createdate": iso(MARTES_10)})  # propietario inactivo
    datos = sugerir_ruteo(hs, ahora=MARTES_10 + timedelta(days=1), supervisor="900")
    curico = datos["proyectos"]["Raíces de Curicó"]
    assert curico["zona"] == "Zona Centro"
    assert [(v["id"], v["peso"]) for v in curico["vendedores"]] == [(101, 6), (102, 3)]  # 103 < 8%; 999 inactivo
    assert "Raíces de Los Ulmos" not in datos["proyectos"]
    tabla = TablaRuteo.desde_texto(a_yaml(datos))  # el YAML generado se puede cargar
    assert tabla.supervisor == "900" and tabla.proyecto("Raíces de Curicó").vendedores[0].id == "101"
