"""Pruebas del agente de caja (sin conexión a Forpay, HubSpot, Gmail ni internet)."""

import csv
import dataclasses
from datetime import date, datetime, timedelta, timezone

import pytest

from agentes_caja import analista, cobranza, fuentes, informe, proyeccion
from agentes_caja.config import ConfigCaja
from agentes_caja.demo import FuentesDemo
from agentes_caja.orquestador import ejecutar_ciclo, preparar_demo
from agentes_hubspot.db import MemoriaEmbudo
from agentes_hubspot.demo import ClienteDemo as ClienteHubSpotDemo

AHORA = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)
HOY = date(2026, 10, 31)


@pytest.fixture
def cfg(tmp_path):
    return ConfigCaja(
        ruta_db=str(tmp_path / "caja.db"),
        carpeta_informes=str(tmp_path / "informes"),
        egresos_archivo=str(tmp_path / "egresos.csv"),
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


def cuota(empresa, compromiso, numero, monto, vence_en_dias, estado="por_vencer", pagado_en_dias=None, cliente=""):
    vence = HOY + timedelta(days=vence_en_dias)
    return {
        "empresa": empresa,
        "compromiso": compromiso,
        "cliente": cliente or f"Cliente {compromiso}",
        "proyecto": "Parcela 1",
        "numero": numero,
        "monto": float(monto),
        "monto_pagado": float(monto) if estado == "pagada" else 0.0,
        "vence": vence,
        "pagado_en": HOY + timedelta(days=pagado_en_dias) if pagado_en_dias is not None else None,
        "estado": estado,
        "estado_original": estado,
        "dias_mora": max(0, -vence_en_dias) if estado in ("moroso", "en_cobro") else 0,
    }


# ----------------------------------------------------------------------------- parseo
def test_parsea_montos_y_fechas():
    assert fuentes.parsear_monto("$ 1.250.000") == 1_250_000
    assert fuentes.parsear_monto("1.250,50") == 1250.5
    assert fuentes.parsear_monto("1250.50") == 1250.5
    assert fuentes.parsear_monto("1,250,000") == 1_250_000
    assert fuentes.parsear_monto(980000) == 980000 and fuentes.parsear_monto("-") == 0 and fuentes.parsear_monto(None) == 0
    assert fuentes.parsear_fecha("2026-10-05") == date(2026, 10, 5)
    assert fuentes.parsear_fecha("05-10-2026") == date(2026, 10, 5)
    assert fuentes.parsear_fecha("05/10/2026 14:30:00") == date(2026, 10, 5)
    assert fuentes.parsear_fecha("2026-10-05T14:30:00Z") == date(2026, 10, 5)
    assert fuentes.parsear_fecha(1_759_622_400_000) == date(2025, 10, 5)
    assert fuentes.parsear_fecha("-") is None and fuentes.parsear_fecha("") is None


def test_clasifica_estados_de_forpay():
    f = fuentes.clasificar_estado
    assert f("Pendiente-Moroso", None, None, HOY) == "moroso"
    assert f("Pendiente-No-Moroso", None, None, HOY) == "por_vencer"
    assert f("Pendiente-Hoy", None, None, HOY) == "por_vencer"
    assert f("Pagada", None, None, HOY) == "pagada"
    assert f("En Proceso de Cobro", None, None, HOY) == "en_cobro"
    assert f("Repactada", None, None, HOY) == "repactada"
    assert f("Anulada", None, None, HOY) == "anulada" and f("Reversada", None, None, HOY) == "anulada"
    assert f("Pendiente", HOY - timedelta(days=3), None, HOY) == "moroso"
    assert f("Pendiente", HOY + timedelta(days=3), None, HOY) == "por_vencer"
    assert f("", None, HOY, HOY) == "pagada"


RESPUESTA_FORPAY = {
    "data": [
        {
            "id": 501,
            "estado_compromiso": "Activa",
            "data": {"nombre_cliente": "Juan Demo", "proyecto": "Parcela 12", "monto_total": "36.000.000"},
            "cuotas": [
                {"numero_cuota": 1, "monto_cuota": "$ 1.000.000", "fecha_vencimiento": "2026-09-30", "estado_cuota": "Pendiente-Moroso", "fecha_pago": None},
                {"numero_cuota": 2, "monto_cuota": "$ 1.000.000", "fecha_vencimiento": "2026-11-30", "estado_cuota": "Pendiente-No-Moroso", "fecha_pago": None},
                {"numero_cuota": 0, "monto_cuota": "$ 1.000.000", "fecha_vencimiento": "2026-08-30", "estado_cuota": "Pagada", "fecha_pago": "2026-09-02"},
            ],
        },
        {"id": 502, "cliente": "Ana Demo", "valor": "750000", "vencimiento": "15/10/2026", "estado": "Pendiente"},  # registro plano
    ]
}


def test_aplana_respuesta_de_forpay_con_deteccion_de_campos():
    cuotas = fuentes.aplanar_forpay(RESPUESTA_FORPAY, HOY, "Sociedad Demo", "Pendiente")
    assert len(cuotas) == 4
    primera = cuotas[0]
    assert primera["empresa"] == "Sociedad Demo" and primera["compromiso"] == "501" and primera["cliente"] == "Juan Demo"
    assert primera["proyecto"] == "Parcela 12" and primera["monto"] == 1_000_000 and primera["numero"] == 1
    assert primera["estado"] == "moroso" and primera["dias_mora"] == 31
    assert cuotas[1]["estado"] == "por_vencer" and cuotas[2]["estado"] == "pagada" and cuotas[2]["pagado_en"] == date(2026, 9, 2)
    plana = cuotas[3]
    assert plana["cliente"] == "Ana Demo" and plana["monto"] == 750_000 and plana["vence"] == date(2026, 10, 15)
    assert plana["estado"] == "moroso" and plana["dias_mora"] == 16  # "Pendiente" ya vencida


def test_campos_fijados_por_configuracion():
    datos = {"data": [{"folio": "A1", "cuotas": [{"valor_pactado": "500", "fecha_limite": "2026-12-01", "situacion": "Pendiente"}]}]}
    sin = fuentes.aplanar_forpay(datos, HOY, "X")
    assert sin[0]["vence"] is None  # "fecha_limite" no se parece a ningún candidato de vencimiento
    con = fuentes.aplanar_forpay(datos, HOY, "X", fijados={"monto": "valor_pactado", "vencimiento": "fecha_limite", "estado": "situacion"})
    assert con[0]["monto"] == 500 and con[0]["vence"] == date(2026, 12, 1) and con[0]["estado"] == "por_vencer"


class _Respuesta:
    def __init__(self, cuerpo, status=200):
        self._cuerpo, self.status_code = cuerpo, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise fuentes.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._cuerpo


class _SesionForpay:
    def __init__(self):
        self.llamadas = []

    def get(self, url, params=None, timeout=None):
        self.llamadas.append(params)
        if params["token"] == "rota":
            return _Respuesta({}, 500)
        if params["estado_cuota"] == "Pagada":
            return _Respuesta({"data": []})
        return _Respuesta(RESPUESTA_FORPAY)


def test_cliente_forpay_recorre_sociedades_y_aisla_errores(monkeypatch):
    monkeypatch.setattr("agentes_caja.fuentes.time.sleep", lambda s: None)
    cfg = ConfigCaja(forpay_empresas={"Buena SpA": "ok", "Rota SpA": "rota"}, forpay_estados=["Pendiente", "Pagada"])
    cliente = fuentes.ClienteForpay(cfg, sesion=_SesionForpay())
    cuotas, errores = cliente.cuotas(HOY)
    assert len(cuotas) == 4 and {c["empresa"] for c in cuotas} == {"Buena SpA"}
    assert len(errores) == 2 and all("Rota SpA" in e for e in errores)
    assert len(cliente.sesion.llamadas) == 4


def test_lee_archivo_de_cuentas_por_cobrar(tmp_path):
    ruta = tmp_path / "Morosos forpay La Oración SpA.csv"
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "empresa", "nombre", "estado", "numero_cuota", "monto", "fecha_vencimiento", "estado_cuota", "fecha_pago"])
        w.writerow(["77", "La Oración SpA", "Pedro Demo", "Activa", "5", "890000", "2026-10-01", "Pendiente-Moroso", "-"])
        w.writerow(["77", "La Oración SpA", "Pedro Demo", "Activa", "6", "890000", "2026-11-01", "Pendiente-No-Moroso", "-"])
    cuotas = fuentes.leer_archivo_cuotas(str(ruta), HOY)
    assert len(cuotas) == 2 and cuotas[0]["empresa"] == "La Oración SpA" and cuotas[0]["cliente"] == "Pedro Demo"
    assert cuotas[0]["estado"] == "moroso" and cuotas[0]["dias_mora"] == 30 and cuotas[1]["estado"] == "por_vencer"
    assert cuotas[0]["compromiso"] == "77" and cuotas[0]["monto"] == 890_000


# ----------------------------------------------------------------------------- egresos
def test_plantilla_de_cobros(tmp_path):
    ruta = fuentes.escribir_plantilla_cobros(str(tmp_path / "cobros.csv"))
    cuotas = fuentes.leer_archivo_cuotas(ruta, HOY)
    assert len(cuotas) == 5 and {c["empresa"] for c in cuotas} == {"Sociedad Ejemplo SpA", "Otra Sociedad SpA"}
    assert [c["estado"] for c in cuotas] == ["moroso", "por_vencer", "pagada", "moroso", "por_vencer"]
    assert cuotas[0]["cliente"] == "Cliente Uno" and cuotas[0]["compromiso"] == "CV-0001" and cuotas[0]["numero"] == 7
    assert cuotas[2]["pagado_en"] == date(2026, 9, 22) and cuotas[4]["monto"] == 4_500_000


def test_plantilla_y_expansion_de_egresos(tmp_path):
    ruta = fuentes.escribir_plantilla_egresos(str(tmp_path / "egresos.csv"))
    egresos, errores = fuentes.leer_egresos(ruta)
    assert not errores and len(egresos) == 5 and egresos[0]["recurrencia"] == "mensual"
    eventos = fuentes.expandir_egresos(egresos, date(2026, 11, 1), date(2027, 1, 31))
    remuneraciones = [e for e in eventos if e["concepto"] == "Remuneraciones"]
    assert [e["fecha"] for e in remuneraciones] == [date(2026, 11, 5), date(2026, 12, 5), date(2027, 1, 5)]
    assert sum(1 for e in eventos if e["concepto"].startswith("Movimiento")) == 1
    # fin de mes y recurrencias cortas
    assert fuentes._sumar_mes(date(2026, 1, 31), 1) == date(2026, 2, 28)
    semanal = [{"fecha": date(2026, 11, 2), "monto": 10.0, "concepto": "x", "categoria": "c", "empresa": "", "recurrencia": "semanal", "estado": "programado", "hasta": date(2026, 11, 20)}]
    assert [e["fecha"].day for e in fuentes.expandir_egresos(semanal, date(2026, 11, 1), date(2026, 12, 31))] == [2, 9, 16]
    pagado = [{**semanal[0], "estado": "pagado"}]
    assert fuentes.expandir_egresos(pagado, date(2026, 11, 1), date(2026, 12, 31)) == []
    assert fuentes.leer_egresos(str(tmp_path / "no_existe.csv"))[0] == []


# ----------------------------------------------------------------------------- cobranza
def test_cobranza_tramos_prioridad_y_efectividad():
    cuotas = [
        cuota("A", "C1", 1, 1_000_000, -10, "moroso"),
        cuota("A", "C1", 2, 1_000_000, 20),
        cuota("A", "C2", 1, 3_000_000, -100, "moroso"),
        cuota("A", "C2", 2, 3_000_000, -70, "moroso"),
        cuota("B", "C3", 1, 500_000, -40, "moroso"),
        cuota("B", "C3", 2, 500_000, 3),
        cuota("B", "C4", 1, 800_000, 5),
        cuota("A", "C5", 1, 2_000_000, -20, "en_cobro"),
        cuota("B", "C6", 1, 400_000, -5, "repactada"),
        cuota("A", "C1", 0, 1_000_000, -40, "pagada", pagado_en_dias=-38),
        cuota("A", "C7", 1, 1_000_000, -45, "pagada", pagado_en_dias=-44),
        cuota("A", "C8", 1, 1_000_000, -75, "moroso"),
    ]
    cb = cobranza.analizar(cuotas, HOY, top=10)
    assert cb["total_moroso"] == 8_500_000 and cb["total_por_vencer"] == 2_300_000
    assert cb["pct_mora"] == pytest.approx(100 * 8.5 / 10.8, abs=0.1)
    assert cb["compromisos_morosos"] == 4 and cb["en_cobro"]["monto"] == 2_000_000 and cb["repactadas"]["cuotas"] == 1
    tramos = {t["tramo"]: t for t in cb["tramos"]}
    assert tramos["1-15 días"]["monto"] == 1_000_000 and tramos["16-45 días"]["monto"] == 500_000
    assert tramos["46-90 días"]["monto"] == 4_000_000 and tramos["más de 90 días"]["monto"] == 3_000_000
    primero = cb["prioritarios"][0]
    assert primero["compromiso"] == "C2" and primero["dias_mora"] == 100 and primero["tramo"] == "más de 90 días" and primero["nivel"] == "crítica"
    assert cb["prioritarios"][-1]["compromiso"] == "C3" or cb["prioritarios"][-1]["monto"] <= 1_000_000
    assert cb["vence_7d"]["cuotas"] == 2 and cb["vence_7d"]["con_mora_cuotas"] == 1  # C3 ya tiene mora, C4 no
    assert [e["empresa"] for e in cb["por_empresa"]] == ["A", "B"]
    assert cb["efectividad"][-1]["mes"] == "2026-09" and cb["tasa_cobro_historica"] is not None
    sin_pagadas = cobranza.analizar([c for c in cuotas if c["estado"] != "pagada"], HOY)
    assert sin_pagadas["tasa_cobro_historica"] is None


# ----------------------------------------------------------------------------- proyección
def test_proyeccion_saldo_alerta_y_escenarios(cfg):
    cfg = dataclasses.replace(cfg, saldo_inicial=1_000_000, caja_minima=500_000, horizonte_semanas=4, tasa_recuperacion_mora=0.5, semanas_recuperacion_mora=2)
    cuotas = [cuota("A", "C1", 1, 1_000_000, 2), cuota("A", "C2", 1, 2_000_000, 9), cuota("A", "C3", 1, 400_000, -3, "moroso")]
    egresos = [{"fecha": HOY + timedelta(days=1), "monto": 1_500_000, "concepto": "Sueldos", "categoria": "Personal", "empresa": ""}]
    py = proyeccion.proyectar(cuotas, egresos, [], cfg, HOY, tasa_puntual=0.8)
    s1, s2 = py["semanas"][0], py["semanas"][1]
    assert s1["programado"] == 1_000_000 and s1["esperado"] == 800_000 and s1["mora"] == 100_000 and s1["egresos"] == 1_500_000
    assert s1["saldo"] == 1_000_000 + 800_000 + 100_000 - 1_500_000 == 400_000
    assert s2["saldo"] == 400_000 + 1_600_000 + 100_000 == 2_100_000
    assert py["alerta"] == {"semana": 1, "inicio": HOY.isoformat(), "saldo": 400_000, "brecha": 100_000}
    esc = py["escenarios"]
    assert esc["pesimista"]["saldo_final"] < esc["base"]["saldo_final"] <= esc["optimista"]["saldo_final"]
    assert esc["base"]["saldo_minimo"] == 400_000 and esc["base"]["semana_minima"] == HOY.isoformat()
    sin_saldos = proyeccion.proyectar(cuotas, egresos, [], dataclasses.replace(cfg, saldo_inicial=0, caja_minima=0), HOY)
    assert sin_saldos["alerta"] is None


def test_ventas_comprometidas_pondera_por_probabilidad(cfg):
    hubspot = ClienteHubSpotDemo(AHORA)
    eventos, errores = fuentes.ventas_comprometidas(dataclasses.replace(cfg, pie_porcentaje=10, pipeline_prob_minima=0.5), HOY, cliente_hubspot=hubspot)
    assert not errores and eventos
    etapas, pid, _ = hubspot.etapas()
    prob = {e["id"]: e["probabilidad"] for e in etapas}
    elegibles = [n for n in hubspot.negocios_abiertos(pid, etapas) if 0.5 <= prob[n["etapa_id"]] < 1 and n["monto"]]
    assert len(eventos) == len(elegibles)
    assert all(e["fecha"] >= HOY and e["monto"] > 0 for e in eventos)


# ----------------------------------------------------------------------------- analista e informe
def test_hallazgos_y_acciones(cfg):
    cfg = dataclasses.replace(cfg, saldo_inicial=5_000_000, caja_minima=4_000_000, horizonte_semanas=4)
    cuotas = [cuota("A", f"C{i}", 1, 1_000_000, -100 - i, "moroso") for i in range(12)] + [cuota("B", "V1", 1, 300_000, 3)]
    egresos = [{"fecha": HOY + timedelta(days=2), "monto": 9_000_000, "concepto": "Obra", "categoria": "Obras", "empresa": ""}]
    cb = cobranza.analizar(cuotas, HOY)
    py = proyeccion.proyectar(cuotas, egresos, [], cfg, HOY)
    diag = analista.diagnosticar(cuotas, cb, py, egresos, cfg, AHORA, HOY, None, ["Forpay X: timeout"])
    texto = " ".join(diag["hallazgos"])
    assert "cae a" in texto and "bajo el mínimo" in texto
    assert "superan los ingresos" in texto and "más de 90 días" in texto and "fuente(s) no respondieron" in texto
    assert 1 <= len(diag["acciones"]) <= 3 and "brecha" in diag["acciones"][0]
    assert diag["egresos_categorias_4s"] == [{"categoria": "Obras", "monto": 9_000_000}]


def test_informe_markdown_html_texto_y_guardado(cfg, db):
    resultado, fuentes_demo = preparar_demo(cfg, db, AHORA)
    assert not resultado.errores and resultado.diagnostico["cuotas_total"] > 1000
    md, diag = resultado.markdown, resultado.diagnostico
    for seccion in ("## Hallazgos", "## Proyección de caja", "### Escenarios", "## Por sociedad", "## Morosidad por antigüedad", "## Cobranza prioritaria", "## Efectividad de cobranza", "## Egresos de las próximas 4 semanas"):
        assert seccion in md
    assert "Boulevard Demo SpA" in md and "S1 ·" in md
    titulo, md2, html, texto = informe.construir(diag, cfg, "Lectura de prueba.", AHORA)
    assert md2.startswith("# Flujo de caja") and "Lectura de prueba." in md2 and "Lectura de prueba." in html
    assert "<table" in html and texto.startswith(titulo) and "Hallazgos:" in texto
    assert resultado.informe_ruta.endswith(".md") and resultado.correo is None
    with open(resultado.informe_ruta, encoding="utf-8") as f:
        assert f.read() == md
    assert db.foto_anterior("caja", 7, AHORA + timedelta(days=1))["cobranza"]["total_moroso"] == diag["cobranza"]["total_moroso"]
    assert fuentes_demo.saldo_inicial == diag["proyeccion"]["saldo_inicial"]


def test_ciclo_sin_cuotas_informa_error(cfg, db):
    class _Vacias:
        def cuotas(self, hoy):
            return [], ["Forpay A: HTTP 500"]

        def egresos(self):
            return [], []

        def ventas(self, hoy):
            return [], []

    r = ejecutar_ciclo(cfg, db, _Vacias(), AHORA, enviar=False, guardar=False, usar_ia=False)
    assert r.diagnostico is None and r.errores == ["Forpay A: HTTP 500"] and "sin diagnóstico" in r.texto()


def test_comparacion_con_foto_anterior(cfg, db):
    fd = FuentesDemo(HOY)
    r1 = ejecutar_ciclo(cfg, db, fd, AHORA - timedelta(days=10), enviar=False, guardar=False, usar_ia=False)
    r2 = ejecutar_ciclo(cfg, db, fd, AHORA, enviar=False, guardar=False, usar_ia=False)
    assert r1.diagnostico["comparacion"] is None and r2.diagnostico["comparacion"] is not None
    assert any("Frente a la foto" in h for h in r2.diagnostico["hallazgos"])


# ----------------------------------------------------------------------------- configuración
def test_config_desde_entorno(monkeypatch):
    monkeypatch.setenv("FORPAY_EMPRESAS", "Boulevard Macul SpA=abc123, La Oración SpA=def456,sin_token")
    monkeypatch.setenv("CAJA_SALDOS", "Boulevard Macul SpA=12000000,La Oración SpA=8000000")
    monkeypatch.setenv("CAJA_MINIMA", "15.000.000".replace(".", ""))
    monkeypatch.setenv("COBROS_CAMPOS", "monto=valor_pactado")
    monkeypatch.setenv("COBROS_ARCHIVOS", "data/cobros.csv, data/cobros_2.xlsx")
    monkeypatch.setenv("CAJA_INCLUIR_PIPELINE", "true")
    cfg = ConfigCaja.desde_entorno()
    assert cfg.forpay_empresas == {"Boulevard Macul SpA": "abc123", "La Oración SpA": "def456"}
    assert cfg.saldo_inicial_total == 20_000_000 and cfg.caja_minima == 15_000_000
    assert cfg.cobros_campos == {"monto": "valor_pactado"} and cfg.incluir_pipeline and cfg.cobros_configurado
    assert cfg.cobros_archivos == ["data/cobros.csv", "data/cobros_2.xlsx"]
