"""Pruebas del agente de embudo semanal (sin conexión a HubSpot, Sheets ni Gmail)."""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agentes_hubspot" / "embudo_semanal"))
import embudo  # noqa: E402

HOY = date(2026, 11, 23)  # lunes

# Encabezado y tres filas tal como las devuelve Google Sheets get_values de la hoja real.
PLAN = {
    "range": "'05 Performance Embudo'!A1:I30",
    "values": [
        ["Escenario Pesimista", "", "IO a IC", "IC a VA", "VA a VR", "VR a PR", "PR a V",
         "Considerar que es una estimación y puede variar según performance"],
        ["Proyecto", "IO TOTAL", "%MQL", "%SQL", "%Opportunity", "%Closed", "%WON", "Venta Cohorte Potencial", "Venta Total"],
        ["Fundo Las Ortigas", "3288", "17.23%", "18.74%", "46.67%", "53.12%", "82.84%", "22", "14"],
        ["Peor es Nada", "1932", "22.03%", "21.26%", "68.39%", "26.86%", "87.10%", "14", "18"],
        ["Raíces de Osorno", "2280", "21.56%", "19.38%", "55.40%", "66.13%", "85.39%", "30", "35"],
        ["Raíces de Putú", "2496", "23.31%", "15.09%", "57.64%", "37.81%", "88.66%", "17", "10"],
        ["Raíces de San Javier", "0", "21.87%", "14.13%", "67.55%", "48.84%", "83.66%", "0", "0"],
        ["Total", "", "", "", "", "", "", "337", "348"],
    ],
}


def tsv(filas):
    lineas = ["Proyecto de Interés [proyecto_de_interes]\t(Count) Deals [count]"]
    lineas += [f"{n}\t{c}" for n, c in filas]
    return "\n".join(lineas) + "\n"


# Conteos por etapa (IO, IC, VA, VR, PR, V). Ortigas cierra mal (VR→PR muy bajo); Osorno sigue el plan.
CONTEOS = {
    "Fundo Las Ortigas": [3000, 540, 100, 50, 10, 8],
    "Raíces de Osorno": [2200, 480, 95, 55, 36, 31],
    "Raíces de Putu": [2500, 560, 90, 50, 20, 18],
    "Raíces de San Javier": [1300, 230, 30, 20, 8, 6],
    "Viñedos de Santo Domingo (Vinedos de Santo Domingo)": [780, 100, 15, 10, 2, 1],
}
RITMO = {"Fundo Las Ortigas": 100, "Raíces de Osorno": 180, "Raíces de Putu": 190, "Raíces de San Javier": 90}


@pytest.fixture
def cfg():
    return embudo.cargar_config()


@pytest.fixture
def entrada(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps(PLAN), encoding="utf-8")
    dir_entrada = tmp_path / "entrada"
    dir_entrada.mkdir()
    for i, etapa in enumerate(embudo.ORDEN):
        filas = [(n, c[i]) for n, c in CONTEOS.items() if c[i]]
        (dir_entrada / f"cohorte_{etapa}.tsv").write_text(tsv(filas), encoding="utf-8")
    (dir_entrada / "ritmo_io.tsv").write_text(tsv(RITMO.items()), encoding="utf-8")
    return tmp_path


def correr(tmp_path, cfg, hoy=HOY, historial=None):
    return embudo.construir(hoy, tmp_path / "plan.json", tmp_path / "entrada", historial, cfg, tmp_path / "salida")


# ----------------------------------------------------------------------------- planilla
def test_lee_la_planilla_y_valida_la_venta_potencial(tmp_path):
    ruta = tmp_path / "plan.json"
    ruta.write_text(json.dumps(PLAN), encoding="utf-8")
    plan, advertencias = embudo.leer_plan(ruta)
    assert [p["nombre"] for p in plan] == [
        "Fundo Las Ortigas", "Peor es Nada", "Raíces de Osorno", "Raíces de Putú", "Raíces de San Javier"]
    ortigas = plan[0]
    assert ortigas["io_plan"] == 3288
    assert ortigas["tasas_plan"]["mql"] == pytest.approx(0.1723)
    assert ortigas["pot_plan"] == pytest.approx(21.8, abs=0.1)  # 3288 × tasas ≈ 22 de la planilla
    assert advertencias == []


def test_avisa_si_la_venta_potencial_no_calza(tmp_path):
    datos = json.loads(json.dumps(PLAN))
    datos["values"][2][7] = "40"
    ruta = tmp_path / "plan.json"
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    _, advertencias = embudo.leer_plan(ruta)
    assert any("Fundo Las Ortigas" in a for a in advertencias)


def test_falla_si_la_hoja_cambio_de_formato(tmp_path):
    ruta = tmp_path / "plan.json"
    ruta.write_text(json.dumps({"values": [["Proyecto", "Leads"], ["A", "1"]]}), encoding="utf-8")
    with pytest.raises(embudo.ErrorEntrada) as e:
        embudo.leer_plan(ruta)
    assert e.value.codigo == 2


def test_porcentajes_con_coma_decimal():
    assert embudo._pct("17,23%") == pytest.approx(0.1723)
    assert embudo._pct("0.5") == pytest.approx(0.5)
    assert embudo._num("3,288") == 3288


# ----------------------------------------------------------------------------- HubSpot
def test_lee_tsv_y_resultado_json_de_hubspot(tmp_path):
    texto = tsv([("Raíces del Río Paloma (Raíces de Río Paloma)", 130), ("Fundo Las Ortigas", 551)])
    plano = tmp_path / "a.tsv"
    plano.write_text(texto, encoding="utf-8")
    envuelto = tmp_path / "b.tsv"
    envuelto.write_text(json.dumps({"results": [{"content": "Dataset fetched.\n\nDataset TSV:\n" + texto}]}), encoding="utf-8")
    for ruta in (plano, envuelto):
        assert embudo.leer_conteos(ruta) == {"Raíces del Río Paloma (Raíces de Río Paloma)": 130, "Fundo Las Ortigas": 551}


def test_claves_de_etiqueta_y_valor():
    claves = embudo.claves_hubspot("Raíces Los Ulmos (Raíces de Los Ulmos)")
    assert "raices de los ulmos" in claves and "raices los ulmos" in claves


def test_consultas_usan_la_cohorte_madura_y_etapas_anidadas(cfg):
    c = embudo.consultas(HOY, cfg)
    assert c["cohorte"] == {"desde": "2025-10-13", "hasta_exclusivo": "2026-10-12"}
    assert c["ritmo"] == {"desde": "2026-10-26", "hasta_exclusivo": "2026-11-23"}
    assert [q["archivo"] for q in c["consultas"]] == [
        "cohorte_io.tsv", "cohorte_ic.tsv", "cohorte_va.tsv", "cohorte_vr.tsv", "cohorte_pr.tsv", "cohorte_v.tsv", "ritmo_io.tsv"]
    sql = {q["archivo"]: q["sql"] for q in c["consultas"]}
    assert "hs_v2_date_entered" not in sql["cohorte_io.tsv"]
    # Quien llegó a venta también cuenta como reserva, visita e interés.
    for archivo in ("cohorte_ic.tsv", "cohorte_va.tsv", "cohorte_vr.tsv", "cohorte_pr.tsv", "cohorte_v.tsv"):
        assert "hs_v2_date_entered_1026814100 IS NOT NULL" in sql[archivo]
        assert "dealstage = '1433853017'" in sql[archivo]
    assert "hs_v2_date_entered_decisionmakerboughtin" not in sql["cohorte_v.tsv"]
    assert "pipeline = 'default'" in sql["cohorte_io.tsv"] and "createdate < '2026-10-12'" in sql["cohorte_io.tsv"]


# ----------------------------------------------------------------------------- análisis
def test_detecta_el_punto_critico_y_su_responsable(entrada, cfg):
    res = correr(entrada, cfg)
    ortigas = next(p for p in res["proyectos"] if p["nombre"] == "Fundo Las Ortigas")
    cierre = ortigas["tramos"]["closed"]
    assert cierre["real"] == pytest.approx(0.2)
    assert cierre["estado"] == "rojo"
    assert cierre["impacto"] == pytest.approx(ortigas["pot_plan"] * (0.2 / 0.5312 - 1))
    primero = res["puntos"][0]
    assert (primero["proyecto"], primero["clave"]) == ("Fundo Las Ortigas", "closed")
    assert primero["responsable"]["nombre"] == "Gonzalo Guerra"
    assert all(p["impacto"] <= -cfg["umbrales"]["impacto_minimo"] for p in res["puntos"])
    assert [p["impacto"] for p in res["puntos"]] == sorted(p["impacto"] for p in res["puntos"])
    # Los puntos se agrupan por tramo, y el foco de mayor efecto va primero.
    foco = res["focos"][0]
    assert foco["clave"] == "closed" and foco["responsable"]["nombre"] == "Gonzalo Guerra"
    assert [f["impacto"] for f in res["focos"]] == sorted(f["impacto"] for f in res["focos"])
    assert embudo.asunto(res) == f"Embudo semanal 23-11-2026: {embudo.ICONO[res['semaforo']]} ventas al " \
        f"{embudo.f_pct(res['total']['razon'], 0)} del plan · foco en VR → PR"


def test_diferencia_dentro_del_margen_no_es_punto_critico(cfg):
    umbrales = cfg["umbrales"]
    # 30% vs 40% del plan con 25 casos: −25%, pero dentro de la variación normal (z ≈ −1,0).
    z = embudo._z_tasa(0.30, 0.40, 25)
    assert -1.1 < z < -0.9
    assert embudo._estado(0.30 / 0.40 - 1, z, umbrales) == "muestra"
    # La misma brecha con 400 casos sí es roja.
    assert embudo._estado(0.30 / 0.40 - 1, embudo._z_tasa(0.30, 0.40, 400), umbrales) == "rojo"
    assert embudo._estado(-0.05, -5, umbrales) == "verde"


def test_muestra_chica_no_genera_punto_y_usa_tasa_del_plan(entrada, cfg):
    res = correr(entrada, cfg)
    ortigas = next(p for p in res["proyectos"] if p["nombre"] == "Fundo Las Ortigas")
    reservas = ortigas["tramos"]["won"]  # solo 10 reservas: bajo el mínimo de 20
    assert reservas["estado"] == "muestra"
    assert not any(x["proyecto"] == "Fundo Las Ortigas" and x["clave"] == "won" for x in res["puntos"])


def test_proyectos_sin_datos_y_fuera_del_plan(entrada, cfg):
    res = correr(entrada, cfg)
    peor = next(p for p in res["proyectos"] if p["nombre"] == "Peor es Nada")
    assert not peor["tiene_datos"] and peor["pot_real"] is None
    assert "Viñedos de Santo Domingo" in res["fuera"]
    putu = next(p for p in res["proyectos"] if p["nombre"] == "Raíces de Putú")
    assert putu["c"]["io"] == 2500  # «Raíces de Putu» en HubSpot
    # San Javier no tiene IO en el plan: no suma al portafolio.
    assert res["total"]["io_real"] == 3000 + 2200 + 2500


def test_conteos_incoherentes_detienen_el_informe(entrada, cfg):
    (entrada / "entrada" / "cohorte_va.tsv").write_text(tsv([("Fundo Las Ortigas", 900)]), encoding="utf-8")
    with pytest.raises(embudo.ErrorEntrada) as e:
        correr(entrada, cfg)
    assert e.value.codigo == 3


def test_salidas_del_correo(entrada, cfg):
    res = correr(entrada, cfg)
    salida = entrada / "salida"
    html = (salida / "correo.html").read_text(encoding="utf-8")
    assert "Puntos críticos y plan de acción" in html and "Gonzalo Guerra" in html
    assert "viernes 27-11" in html and "lunes 30-11" in html
    assert embudo.asunto(res).startswith("Embudo semanal 23-11-2026: ")
    pagina = (salida / "pagina.html").read_text(encoding="utf-8")
    assert pagina.startswith("<title>Embudo semanal TPDV</title>")
    historial = json.loads((salida / "historial.json").read_text(encoding="utf-8"))
    assert historial["semanas"][-1]["fecha"] == "2026-11-23"
    assert historial["semanas"][-1]["proyectos"]["Fundo Las Ortigas"]["estados"]["closed"] == "rojo"


def test_escala_tras_tres_semanas_en_rojo(entrada, cfg):
    hist = entrada / "historial.json"
    for semanas_atras in (2, 1):
        fecha = HOY - timedelta(weeks=semanas_atras)
        correr(entrada, cfg, hoy=fecha, historial=hist if hist.exists() else None)
        hist.write_text((entrada / "salida" / "historial.json").read_text(encoding="utf-8"), encoding="utf-8")
    res = correr(entrada, cfg, historial=hist)
    punto = next(x for x in res["puntos"] if x["proyecto"] == "Fundo Las Ortigas" and x["clave"] == "closed")
    assert punto["semanas_rojo"] == 3 and punto["escalar"]
    html = (entrada / "salida" / "correo.html").read_text(encoding="utf-8")
    assert "Escalamiento:" in html and "3 semanas seguidas en rojo" in html
    # Volver a correr el mismo día reemplaza la foto en vez de duplicarla.
    hist.write_text((entrada / "salida" / "historial.json").read_text(encoding="utf-8"), encoding="utf-8")
    correr(entrada, cfg, historial=hist)
    fechas = [s["fecha"] for s in json.loads((entrada / "salida" / "historial.json").read_text())["semanas"]]
    assert fechas == sorted(set(fechas)) and len(fechas) == 3


def test_un_hueco_en_el_historial_corta_la_racha(entrada, cfg):
    hist = entrada / "historial.json"
    correr(entrada, cfg, hoy=HOY - timedelta(weeks=3))
    hist.write_text((entrada / "salida" / "historial.json").read_text(encoding="utf-8"), encoding="utf-8")
    res = correr(entrada, cfg, historial=hist)
    punto = next(x for x in res["puntos"] if x["proyecto"] == "Fundo Las Ortigas" and x["clave"] == "closed")
    assert punto["semanas_rojo"] == 1 and not punto["escalar"]


def test_cli_imprime_asunto_y_resumen(entrada, cfg, capsys):
    codigo = embudo.main([
        "build", "--hoy", HOY.isoformat(), "--plan", str(entrada / "plan.json"),
        "--entrada", str(entrada / "entrada"), "--salida", str(entrada / "salida")])
    salida = capsys.readouterr().out
    assert codigo == 0
    assert salida.startswith("ASUNTO=Embudo semanal 23-11-2026")
    assert "RESUMEN=" in salida
