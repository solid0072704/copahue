"""Pruebas de Copahue: calendario, automatizaciones, aprobaciones y aislamiento entre estudios."""

from datetime import date

import pytest
from starlette.testclient import TestClient

from copahue import agentes, asistente
from copahue import calendario as cal
from copahue.agentes.acciones import ejecutar_propuesta
from copahue.api import crear_app
from copahue.config import Config
from copahue.db import conectar, fila, filas, inicializar, json_cargar
from copahue.demo import crear_demo
from copahue.llm import _datos_protegidos

HOY = date(2026, 11, 20)
H = {"X-Copahue": "1"}


@pytest.fixture
def cfg(tmp_path):
    return Config(db=str(tmp_path / "copahue.db"), hoy_fijo=HOY.isoformat(), intervalo_min=0)


@pytest.fixture
def estudio(cfg):
    inicializar(cfg.db)
    with conectar(cfg.db) as conn:
        return crear_demo(conn, cfg, HOY)


@pytest.fixture
def cliente_web(cfg):
    return TestClient(crear_app(cfg))


def entrar_demo(c):
    assert c.post("/api/auth/demo", headers=H).status_code == 200
    return c


# --- Calendario tributario ---------------------------------------------------------------------
def test_vencimientos_se_corren_al_dia_habil():
    assert cal.vencimiento("F29", "2026-10") == date(2026, 11, 20)
    assert cal.vencimiento("PREVIRED", "2026-10") == date(2026, 11, 13)
    assert cal.vencimiento("LRE", "2026-10") == date(2026, 11, 16)  # el 15 es domingo
    assert cal.vencimiento("F29", "2026-11") == date(2026, 12, 21)  # el 20 es domingo
    assert cal.vencimiento("F22", "AT2027") == date(2027, 4, 30)
    assert cal.vencimiento("PREVIRED", "2026-11") == date(2026, 12, 14)  # el 13 es domingo


def test_dias_habiles_y_feriados():
    assert not cal.es_habil(date(2026, 12, 8))  # Inmaculada Concepción
    assert cal.sumar_habiles(date(2026, 12, 7), 1) == date(2026, 12, 9)
    assert cal.habiles_hasta(date(2026, 11, 20), date(2026, 11, 23)) == 1
    assert cal.habiles_hasta(date(2026, 11, 23), date(2026, 11, 20)) == -1


def test_obligaciones_segun_servicios():
    assert cal.tipos_mensuales(["contabilidad", "iva"], 5) == ["F29"]
    assert cal.tipos_mensuales(["iva", "remuneraciones"], 3) == ["F29", "PREVIRED", "LRE"]
    assert cal.tipos_mensuales(["iva", "remuneraciones"], 0) == ["F29"]


# --- Automatizaciones ------------------------------------------------------------------------
def test_demo_genera_propuestas_de_cada_automatizacion(cfg, estudio):
    with conectar(cfg.db) as conn:
        por_agente = {r["agente"]: r["n"] for r in filas(conn, """
            SELECT agente, COUNT(*) AS n FROM propuestas WHERE estudio_id = ? AND estado = 'pendiente' GROUP BY agente""", estudio)}
    assert set(por_agente) == {a.clave for a in agentes.AGENTES}


def test_automatizaciones_no_duplican_propuestas(cfg, estudio):
    with conectar(cfg.db) as conn:
        antes = fila(conn, "SELECT COUNT(*) AS n FROM propuestas WHERE estudio_id = ?", estudio)["n"]
        resultados = agentes.ejecutar(conn, cfg, estudio, HOY)
        despues = fila(conn, "SELECT COUNT(*) AS n FROM propuestas WHERE estudio_id = ?", estudio)["n"]
    assert antes == despues
    assert all(r["creadas"] == 0 for r in resultados)


def test_aprobar_solicitud_de_documentos(cfg, estudio):
    with conectar(cfg.db) as conn:
        p = fila(conn, """SELECT * FROM propuestas WHERE estudio_id = ? AND agente = 'documentos'
                          AND titulo LIKE 'Pedir%'""", estudio)
        estudio_fila = fila(conn, "SELECT * FROM estudios WHERE id = ?", estudio)
        usuario = fila(conn, "SELECT * FROM usuarios WHERE estudio_id = ? LIMIT 1", estudio)
        resultado = ejecutar_propuesta(conn, cfg, estudio_fila, p, usuario=usuario)
        ids = json_cargar(p["datos"])["efectos"][0]["ids"]
        docs = filas(conn, f"SELECT estado, recordatorios FROM documentos WHERE id IN ({','.join('?' * len(ids))})", *ids)
        correo = fila(conn, "SELECT * FROM correos WHERE estudio_id = ? ORDER BY id DESC LIMIT 1", estudio)
    assert "simulado" in resultado
    assert all(d["estado"] == "solicitado" and d["recordatorios"] == 1 for d in docs)
    assert correo["estado"] == "simulado"  # los dominios .example nunca reciben correos


def test_propuesta_se_retira_si_el_cliente_ya_envio_todo(cfg, estudio):
    with conectar(cfg.db) as conn:
        p = fila(conn, """SELECT * FROM propuestas WHERE estudio_id = ? AND agente = 'documentos'
                          AND titulo LIKE 'Pedir%'""", estudio)
        conn.execute("UPDATE documentos SET estado = 'recibido' WHERE cliente_id = ?", (p["cliente_id"],))
        agentes.ejecutar(conn, cfg, estudio, HOY, claves=["documentos"])
        assert fila(conn, "SELECT estado FROM propuestas WHERE id = ?", p["id"])["estado"] == "retirada"


def test_modo_automatico_ejecuta_sin_esperar(cfg, estudio):
    with conectar(cfg.db) as conn:
        conn.execute("DELETE FROM propuestas WHERE estudio_id = ? AND agente = 'cobranza'", (estudio,))
        conn.execute("UPDATE automatizaciones SET modo = 'automatico' WHERE estudio_id = ? AND clave = 'cobranza'", (estudio,))
        r = agentes.ejecutar(conn, cfg, estudio, HOY, claves=["cobranza"])[0]
        pendientes = fila(conn, """SELECT COUNT(*) AS n FROM propuestas WHERE estudio_id = ? AND agente = 'cobranza'
                                   AND estado = 'pendiente'""", estudio)["n"]
    assert r["ejecutadas"] == 4 and pendientes == 0


def test_cobranza_escala_al_socio_desde_45_dias(cfg, estudio):
    with conectar(cfg.db) as conn:
        p = fila(conn, "SELECT * FROM propuestas WHERE estudio_id = ? AND clave LIKE 'cobro:%:3'", estudio)
        socio = fila(conn, "SELECT id FROM usuarios WHERE estudio_id = ? AND rol = 'socio'", estudio)
    assert p["tipo"] == "tarea"
    assert json_cargar(p["datos"])["tarea"]["responsable_id"] == socio["id"]


def test_instruccion_sin_ia_reconoce_pedidos(cfg, estudio):
    with conectar(cfg.db) as conn:
        e = fila(conn, "SELECT * FROM estudios WHERE id = ?", estudio)
        u = fila(conn, "SELECT * FROM usuarios WHERE estudio_id = ? LIMIT 1", estudio)
        r = asistente.responder(conn, cfg, e, u, "pide los documentos que faltan", HOY)
        assert r["modo"] == "reglas" and "solicitud de documentos" in r["respuesta"]
        r = asistente.responder(conn, cfg, e, u, "ferreteria", HOY)
        assert r["navegar"].startswith("#/clientes/")


def test_redaccion_asistida_debe_conservar_cifras():
    original = "Factura N.º 1270 por $230.000, venció el 16 de septiembre."
    assert _datos_protegidos(original) <= _datos_protegidos("Hola: la factura N.º 1270 ($230.000) venció el 16 de septiembre.")
    assert not _datos_protegidos(original) <= _datos_protegidos("Hola: tienes una factura pendiente.")


# --- API web ---------------------------------------------------------------------------------
def test_mutaciones_exigen_encabezado(cliente_web):
    assert cliente_web.post("/api/auth/demo").status_code == 403


def test_registro_ingreso_y_clave_incorrecta(cliente_web):
    datos = {"estudio": "Estudio Prueba", "nombre": "Ana Prueba", "email": "ana@prueba.example", "clave": "secreta123"}
    assert cliente_web.post("/api/auth/registro", json=datos, headers=H).status_code == 200
    assert cliente_web.get("/api/sesion").json()["estudio"]["nombre"] == "Estudio Prueba"
    cliente_web.post("/api/auth/salir", headers=H)
    assert cliente_web.get("/api/sesion").status_code == 401
    malo = cliente_web.post("/api/auth/ingresar", json={"email": "ana@prueba.example", "clave": "otra"}, headers=H)
    assert malo.status_code == 401
    bueno = cliente_web.post("/api/auth/ingresar", json={"email": "ANA@prueba.example", "clave": "secreta123"}, headers=H)
    assert bueno.status_code == 200


def test_un_estudio_no_ve_datos_de_otro(cfg):
    a = entrar_demo(TestClient(crear_app(cfg)))
    b = entrar_demo(TestClient(crear_app(cfg)))
    cliente_a = a.get("/api/clientes").json()["clientes"][0]["id"]
    propuesta_a = a.get("/api/propuestas").json()["propuestas"][0]["id"]
    assert b.get(f"/api/clientes/{cliente_a}").status_code == 404
    assert b.post(f"/api/propuestas/{propuesta_a}/aprobar", json={}, headers=H).status_code == 404
    assert cliente_a not in {c["id"] for c in b.get("/api/clientes").json()["clientes"]}


def test_flujo_completo_en_la_api(cliente_web):
    c = entrar_demo(cliente_web)
    for ruta in ("/api/hoy", "/api/clientes", "/api/vencimientos", "/api/documentos", "/api/cobranza",
                 "/api/propuestas", "/api/automatizaciones", "/api/tareas", "/api/actividad", "/api/ajustes"):
        assert c.get(ruta).status_code == 200, ruta
    pendientes = c.get("/api/sesion").json()["para_revisar"]
    correo = next(p for p in c.get("/api/propuestas").json()["propuestas"] if p["tipo"] == "correo")
    r = c.post(f"/api/propuestas/{correo['id']}/aprobar", json={"asunto": "Asunto editado"}, headers=H)
    assert r.status_code == 200
    assert c.get("/api/sesion").json()["para_revisar"] == pendientes - 1
    assert c.post(f"/api/propuestas/{correo['id']}/aprobar", json={}, headers=H).status_code == 409
    nuevo = c.post("/api/clientes", json={"razon_social": "Nueva Empresa SpA", "servicios": ["contabilidad", "iva", "remuneraciones"],
                                          "trabajadores": 3, "contacto_email": "hola@nueva.example"}, headers=H)
    ficha = c.get(f"/api/clientes/{nuevo.json()['id']}").json()
    assert {o["tipo"] for o in ficha["obligaciones"]} == {"F29", "PREVIRED", "LRE"}
    assert len(ficha["documentos"]) == 4
    assert c.patch("/api/obligaciones/999999", json={"estado": "presentada"}, headers=H).status_code == 404
