"""Pruebas de los agentes del embudo de HubSpot (sin conexión a HubSpot, Gmail ni internet)."""

import dataclasses
import json
from datetime import datetime, timedelta, timezone

import pytest

from agentes_hubspot import desatascador, embudo, informe
from agentes_hubspot.api import ClienteHubSpot, normalizar_negocio
from agentes_hubspot.config import ConfigHubSpot
from agentes_hubspot.db import MemoriaEmbudo
from agentes_hubspot.demo import ClienteDemo, etapas_demo
from agentes_hubspot.desatascador import PRIORIDAD_CATEGORIA
from agentes_hubspot.orquestador import ejecutar_ciclo, preparar_demo

AHORA = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)
ETAPAS = etapas_demo()
POR_ID = {e["id"]: e for e in ETAPAS}


@pytest.fixture
def cfg(tmp_path):
    return ConfigHubSpot(
        token="",
        modo="lectura",
        ruta_db=str(tmp_path / "embudo.db"),
        carpeta_informes=str(tmp_path / "informes"),
        gmail_usuario="",
        gmail_app_password="",
        notificar_a="",
        anthropic_api_key="",
    )


@pytest.fixture
def db(cfg):
    memoria = MemoriaEmbudo(cfg.ruta_db)
    yield memoria
    memoria.cerrar()


def negocio(id_, etapa, dias_en_etapa, proxima=None, contactos=0, ultimo_contacto_dias=None, **extra):
    """Negocio ficticio ya normalizado. `proxima` son días hasta la próxima actividad (negativo = vencida)."""
    entro = AHORA - timedelta(days=dias_en_etapa)
    creado = entro - timedelta(days=3)
    orden = POR_ID[etapa]["orden"] if etapa in POR_ID else 0
    n = {
        "id": str(id_),
        "nombre": f"Negocio {id_}",
        "etapa_id": etapa,
        "pipeline": "demo",
        "monto": 1000.0,
        "propietario_id": "1",
        "creado": creado,
        "cerrado": None,
        "entro_etapa": entro,
        "ultimo_contacto": AHORA - timedelta(days=ultimo_contacto_dias) if ultimo_contacto_dias is not None else None,
        "ultima_actividad": None,
        "proxima_actividad": AHORA + timedelta(days=proxima) if proxima is not None else None,
        "proximo_paso": "",
        "contactos": contactos,
        "cerrado_perdido": False,
        "cerrado_ganado": False,
        "razon_perdida": "",
        "entradas": {e["id"]: creado for e in ETAPAS if e["orden"] <= orden and e["probabilidad"] > 0},
        "propiedades": {},
    }
    n.update(extra)
    return n


# ----------------------------------------------------------------------------- api
def test_normaliza_negocio_de_la_api():
    obj = {
        "id": "42",
        "properties": {
            "dealname": "Parcela 7",
            "dealstage": "interes",
            "amount": "25000000",
            "hubspot_owner_id": "9",
            "createdate": "2026-10-01T10:00:00Z",
            "hs_v2_date_entered_current_stage": "1761900000000",
            "hs_v2_date_entered_interes": "1761900000000",
            "hs_v2_date_entered_nuevo": "2026-10-01T10:00:00.000Z",
            "notes_next_activity_date": None,
            "num_contacted_notes": "3",
            "hs_is_closed_lost": "false",
            "razon_perdido": None,
            "closed_lost_reason": "Otro",
        },
    }
    n = normalizar_negocio(obj)
    assert n["id"] == "42" and n["monto"] == 25_000_000 and n["contactos"] == 3 and n["propietario_id"] == "9"
    assert n["creado"].year == 2026 and n["entro_etapa"].tzinfo is not None
    assert set(n["entradas"]) == {"interes", "nuevo"}
    assert n["razon_perdida"] == "Otro"  # cae a closed_lost_reason si razon_perdido viene vacía
    assert n["proxima_actividad"] is None and n["cerrado_perdido"] is False


class _Respuesta:
    def __init__(self, status, cuerpo=None, headers=None):
        self.status_code, self._cuerpo, self.headers = status, cuerpo or {}, headers or {}
        self.text = json.dumps(self._cuerpo)

    def json(self):
        return self._cuerpo


class _SesionFalsa:
    """Imita requests.Session: 450 negocios en páginas de 200 y un 429 en la primera búsqueda."""

    def __init__(self):
        self.headers, self.llamadas, self._dio_429 = {}, [], False

    def request(self, metodo, url, timeout=None, json=None, params=None):
        self.llamadas.append((metodo, url, json, params))
        if url.endswith("/pipelines/deals"):
            etapas = [
                {"id": "a", "label": "Nuevo", "displayOrder": 0, "metadata": {"probability": "0.1"}},
                {"id": "closedlost", "label": "Perdido", "displayOrder": 1, "metadata": {"probability": "0.0"}},
            ]
            return _Respuesta(200, {"results": [{"id": "default", "label": "Ventas", "stages": etapas}]})
        if url.endswith("/deals/search"):
            if not self._dio_429:
                self._dio_429 = True
                return _Respuesta(429, {}, {"Retry-After": "0"})
            filtros = json["filterGroups"][0]["filters"]
            desde = next((int(f["value"]) for f in filtros if f["propertyName"] == "hs_object_id"), 0)
            ids = list(range(desde + 1, 451))[:200]
            return _Respuesta(200, {"results": [{"id": str(i), "properties": {"dealname": f"N{i}", "dealstage": "a"}} for i in ids]})
        if url.endswith("/objects/tasks"):
            return _Respuesta(201, {"id": "t1"})
        return _Respuesta(404, {})


def test_cliente_pagina_por_id_y_reintenta_429(monkeypatch):
    monkeypatch.setattr("agentes_hubspot.api.time.sleep", lambda s: None)
    cliente = ClienteHubSpot(ConfigHubSpot(token="x", pipeline="Ventas"), sesion=_SesionFalsa())
    etapas, pid, nombre = cliente.etapas()
    assert pid == "default" and nombre == "Ventas" and [e["id"] for e in etapas] == ["a", "closedlost"]
    negocios = cliente.negocios_abiertos(pid, etapas)
    assert len(negocios) == 450 and negocios[-1]["id"] == "450"
    busquedas = [c for c in cliente.sesion.llamadas if c[1].endswith("/deals/search")]
    assert len(busquedas) == 4  # 1 reintento por el 429 + 3 páginas (200, 200, 50)
    ultimos_filtros = busquedas[-1][2]["filterGroups"][0]["filters"]
    assert any(f["propertyName"] == "hs_object_id" and f["operator"] == "GT" for f in ultimos_filtros)
    assert cliente.crear_tarea("450", "9", "Llamar", "cuerpo", AHORA) == "t1"
    cuerpo = cliente.sesion.llamadas[-1][2]
    assert cuerpo["associations"][0]["types"][0]["associationTypeId"] == 216
    assert cuerpo["properties"]["hubspot_owner_id"] == "9" and cuerpo["properties"]["hs_task_status"] == "NOT_STARTED"


def test_pipeline_inexistente_explica_los_disponibles(monkeypatch):
    cliente = ClienteHubSpot(ConfigHubSpot(token="x", pipeline="otro"), sesion=_SesionFalsa())
    with pytest.raises(Exception, match="Ventas"):
        cliente.etapas()


# ----------------------------------------------------------------------------- desatascador
def test_categorias_y_umbrales(cfg):
    assert desatascador.categoria_etapa(POR_ID["sin_contacto"]) == "primer_contacto"
    assert desatascador.categoria_etapa(POR_ID["seguimiento"]) == "prospeccion"
    assert desatascador.categoria_etapa(POR_ID["visita_agendada"]) == "visita"
    assert desatascador.categoria_etapa(POR_ID["reserva"]) == "negociacion"
    assert desatascador.categoria_etapa(POR_ID["pagado"]) == "administrativo"
    assert desatascador.umbral_dias(POR_ID["sin_contacto"], "primer_contacto", cfg) == 2
    cfg2 = dataclasses.replace(cfg, dias_por_etapa={"sin contacto": 1, "pagado": 10})
    assert desatascador.umbral_dias(POR_ID["sin_contacto"], "primer_contacto", cfg2) == 1
    assert desatascador.umbral_dias(POR_ID["pagado"], "administrativo", cfg2) == 10


def test_detecta_estancados_y_propone_accion(cfg):
    negocios = [
        negocio(1, "sin_contacto", 5),  # estancado: primer contacto
        negocio(2, "sin_contacto", 5, proxima=2),  # tiene actividad futura: no está estancado
        negocio(3, "sin_contacto", 1),  # recién llegó: no
        negocio(4, "sin_contacto", 9, contactos=4),  # muchos intentos: último intento o cerrar
        negocio(5, "seguimiento", 20, ultimo_contacto_dias=15),
        negocio(6, "visita_agendada", 4, proxima=-1),  # la visita venció y nadie actualizó
        negocio(7, "reserva", 20, proximo_paso="Firmar promesa"),
        negocio(8, "pagado", 120),
        negocio(9, "promesa", 60),
        negocio(10, "closedlost", 300),
    ]
    est = desatascador.detectar(negocios, ETAPAS, cfg, AHORA)
    por_id = {e["negocio"]["id"]: e for e in est}
    assert set(por_id) == {"1", "4", "5", "6", "7", "8", "9"}
    assert por_id["1"]["accion"].startswith("Llamar hoy")
    assert "cerrar" in por_id["4"]["accion"].lower()
    assert por_id["5"]["categoria"] == "prospeccion" and "hace 15 días" in por_id["5"]["detalle"]
    assert por_id["6"]["categoria"] == "visita"
    assert "Firmar promesa" in por_id["7"]["detalle"]
    assert por_id["8"]["accion"].startswith("Marcar como ganado")
    assert por_id["9"]["accion"] == "Destrabar el cierre legal"
    # Primero lo que tiene más valor en juego (negociación y visitas); lo administrativo al final.
    assert [e["negocio"]["id"] for e in est][:2] == ["7", "6"]
    assert [e["negocio"]["id"] for e in est][-2:] == ["8", "9"]
    prioridades = [PRIORIDAD_CATEGORIA[e["categoria"]] for e in est]
    assert prioridades == sorted(prioridades)


def test_modo_lectura_informa_sin_crear_tareas(cfg, db):
    cliente = ClienteDemo(AHORA)
    resultado = ejecutar_ciclo(cfg, db, cliente, AHORA, enviar=True, guardar=True, usar_ia=False)
    assert not resultado.errores
    assert cliente.tareas == [] and resultado.desatascador.creadas == 0
    assert resultado.desatascador.estancados
    assert resultado.diagnostico["total_abiertos"] == len(cliente.negocios_abiertos("demo", ETAPAS))
    assert "## Hallazgos" in resultado.markdown and "Sin contacto" in resultado.markdown
    assert resultado.informe_ruta.endswith(".md") and resultado.correo == "sin_configurar"
    with open(resultado.informe_ruta, encoding="utf-8") as f:
        assert f.read() == resultado.markdown
    foto = db.foto_anterior("demo", 7, AHORA + timedelta(days=1))
    assert foto["total_abiertos"] == resultado.diagnostico["total_abiertos"]


def test_modo_tareas_crea_sin_repetir_y_escala(cfg, db):
    cfg = dataclasses.replace(cfg, modo="tareas", max_tareas_por_ciclo=5, supervisor_id="99")
    cliente = ClienteDemo(AHORA)
    r1 = ejecutar_ciclo(cfg, db, cliente, AHORA, enviar=False, guardar=False, usar_ia=False)
    assert r1.desatascador.creadas == 5 and len(cliente.tareas) == 5
    ids_primera = {t["negocio_id"] for t in cliente.tareas}
    assert all(t["asunto"] for t in cliente.tareas) and all("Desatascador" in t["cuerpo"] for t in cliente.tareas)
    assert all(t["propietario_id"] in ("1", "2", "3") for t in cliente.tareas)
    # Una hora después: esos 5 ya tienen tarea, así que se crean 5 para otros negocios.
    r2 = ejecutar_ciclo(cfg, db, cliente, AHORA + timedelta(hours=1), enviar=False, guardar=False, usar_ia=False)
    assert r2.desatascador.con_tarea_reciente == 5 and r2.desatascador.creadas == 5
    assert len({t["negocio_id"] for t in cliente.tareas}) == 10
    # Ocho días después siguen estancados: los 10 se escalan al supervisor (una sola vez cada uno).
    r3 = ejecutar_ciclo(cfg, db, cliente, AHORA + timedelta(days=8), enviar=False, guardar=False, usar_ia=False)
    escalados = {e["negocio"]["id"] for e in r3.desatascador.escalados}
    assert ids_primera <= escalados and len(escalados) == 10
    supervisor = [t for t in cliente.tareas if t["propietario_id"] == "99"]
    assert len(supervisor) == 10 and all(t["asunto"].startswith("Escalado") for t in supervisor)
    assert all(e["estado"] == "escalado" for e in r3.desatascador.escalados)
    assert "Para escalar" in r3.markdown


class _ClienteRoto:
    def etapas(self):
        raise RuntimeError("token inválido")


def test_ciclo_reporta_error_de_lectura(cfg, db):
    r = ejecutar_ciclo(cfg, db, _ClienteRoto(), AHORA, enviar=False, guardar=False, usar_ia=False)
    assert r.diagnostico is None and r.errores and "token inválido" in r.errores[0]
    assert "sin diagnóstico" in r.texto()


# ----------------------------------------------------------------------------- analista de embudo
def test_diagnostico_conversion_y_hallazgos(cfg):
    abiertos = [negocio(i, "sin_contacto", 6) for i in range(1, 21)] + [negocio(i, "promesa", 100) for i in range(21, 33)]
    periodo = list(abiertos)
    for i in range(100, 160):  # 60 perdidos: 20 llegaron a 'visita agendada', 40 solo a 'sin contacto'
        n = negocio(i, "visita_agendada" if i < 120 else "sin_contacto", 1)
        n.update({"etapa_id": "closedlost", "cerrado_perdido": True, "razon_perdida": "No responde" if i % 3 else "Solo cotizando"})
        periodo.append(n)
    diag = embudo.diagnosticar(abiertos, periodo, ETAPAS, cfg, AHORA, "Pipeline demo", {"1": {"nombre": "Ana"}})
    etapas = {e["id"]: e for e in diag["etapas"]}
    assert diag["total_abiertos"] == 32 and diag["perdidos_periodo"] == 60 and diag["creados_periodo"] == 92
    assert etapas["nuevo"]["alcanzaron"] == 92 and etapas["sin_contacto"]["alcanzaron"] == 92
    assert etapas["visita_agendada"]["alcanzaron"] == 32  # 20 perdidos + 12 promesas
    assert etapas["visita_realizada"]["alcanzaron"] == 12 and etapas["closedwon"]["alcanzaron"] == 0
    assert etapas["interes"]["conversion"] == pytest.approx(100 * 32 / 92, abs=0.1)
    assert etapas["sin_contacto"]["sin_proxima_actividad"] == 20 and etapas["sin_contacto"]["mediana_dias"] == 6
    assert diag["razones_perdida"][0] == ["No responde", 40]
    assert any("Sin contacto" in h for h in diag["hallazgos"])
    assert any("cierre administrativo" in h for h in diag["hallazgos"])
    assert any("No responde" in h for h in diag["hallazgos"])
    assert any("Visita agendada" in h and "Visita realizada" in h for h in diag["hallazgos"])
    assert 1 <= len(diag["recomendaciones"]) <= 3
    assert diag["propietarios"][0]["nombre"] == "Ana" and diag["propietarios"][0]["abiertos"] == 32


def test_comparacion_con_foto_anterior(cfg, db):
    abiertos = [negocio(i, "seguimiento", 20) for i in range(1, 11)]
    viejo = embudo.diagnosticar(abiertos[:6], [], ETAPAS, cfg, AHORA - timedelta(days=10), "demo")
    db.guardar_foto(viejo, "demo")
    anterior = db.foto_anterior("demo", 7, AHORA)
    diag = embudo.diagnosticar(abiertos, [], ETAPAS, cfg, AHORA, "demo", anterior=anterior)
    assert diag["comparacion"]["abiertos"] == 4 and diag["comparacion"]["por_etapa"] == {"En seguimiento": 4}
    assert any("Frente a la foto" in h for h in diag["hallazgos"])
    assert db.foto_anterior("otro", 7, AHORA) is None


# ----------------------------------------------------------------------------- informe
def test_informe_markdown_html_y_texto(cfg):
    cliente = ClienteDemo(AHORA)
    etapas, pid, nombre = cliente.etapas()
    abiertos = cliente.negocios_abiertos(pid, etapas)
    est = desatascador.detectar(abiertos, etapas, cfg, AHORA)
    resultado = desatascador.ejecutar(MemoriaEmbudo(":memory:"), cfg, cliente, est, AHORA)
    periodo = cliente.negocios_creados_desde(pid, etapas, AHORA - timedelta(days=30))
    diag = embudo.diagnosticar(abiertos, periodo, etapas, cfg, AHORA, nombre, cliente.propietarios(), None, est)
    titulo, md, html, texto = informe.construir(diag, resultado, cfg, "123", cliente.propietarios(), "Lectura de prueba.", AHORA)
    assert titulo.startswith("Embudo HubSpot · Pipeline demo")
    assert "## Lectura" in md and "Lectura de prueba." in md
    assert "Primer contacto pendiente" in md and "Cierre administrativo" in md and "Razones de pérdida" in md
    assert "https://app.hubspot.com/contacts/123/record/0-3/demo-" in md and "record/0-3/demo-" in html
    assert "Ana Demo" in md and "<table" in html and "Por etapa" in html and "Lectura de prueba." in html
    assert texto.startswith(titulo) and "Hallazgos:" in texto
    ruta = informe.guardar(md, cfg, AHORA, nombre)
    assert ruta.endswith("embudo_pipeline-demo_2026-10-31.md")
    assert informe.enviar(cfg, titulo, texto, html) == "sin_configurar"


def test_preparar_demo_no_toca_hubspot(cfg, db):
    resultado, cliente = preparar_demo(cfg, db, AHORA)
    assert resultado.diagnostico["total_abiertos"] > 100 and cliente.tareas == []
    assert "estancados" in resultado.texto() and not resultado.errores


# ----------------------------------------------------------------------------- configuración
def test_config_desde_entorno(monkeypatch):
    monkeypatch.setenv("DESATASCAR_DIAS_POR_ETAPA", "Sin Contacto=1, Promesa Firmada=30,mal,otra=x")
    monkeypatch.setenv("HUBSPOT_MODO", "cualquiera")
    monkeypatch.setenv("HUBSPOT_TOKEN", "pat-xyz")
    cfg = ConfigHubSpot.desde_entorno()
    assert cfg.dias_por_etapa == {"sin contacto": 1, "promesa firmada": 30}
    assert cfg.modo == "lectura" and not cfg.escribe_en_hubspot and cfg.hubspot_configurado
