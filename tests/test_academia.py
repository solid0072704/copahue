"""El contenido del primer curso de la academia: su temario y sus evaluaciones."""

import re

from academia.contenido.claude_finanzas import CURSO, evaluaciones, programa


def test_el_programa_tiene_modulos_numerados_y_lecciones_con_duracion_razonable():
    modulos = programa.modulos()
    assert [m["clave"] for m in modulos] == [f"m{n:02d}" for n in range(1, len(modulos) + 1)]
    ids = [x["id"] for x in programa.lecciones()]
    assert len(ids) == len(set(ids)) and all(re.fullmatch(r"m\d\d-l\d\d", i) for i in ids)
    for m in modulos:
        assert 5 <= len(m["lecciones"]) <= 8 and len(m["logras"]) >= 3 and m["practica"], m["clave"]
        assert all(5 <= x["min"] <= 10 and x["tipo"] in ("clase", "demo", "laboratorio") and x["d"] for x in m["lecciones"]), m["clave"]
        assert m["min"] == sum(x["min"] for x in m["lecciones"])


def test_las_etapas_suman_el_programa_y_el_presupuesto_de_voz_sale_de_los_minutos():
    total, por_etapa = programa.totales(), [programa.totales(e) for e in programa.ETAPAS]
    for clave in ("modulos", "lecciones", "minutos"):
        assert sum(t[clave] for t in por_etapa) == total[clave]
    assert total["creditos"] == round(total["minutos"] * programa.CARACTERES_POR_MINUTO * programa.CREDITOS_POR_CARACTER)
    assert 10 <= total["horas"] <= 14 and por_etapa[0]["modulos"] == 9          # la etapa 1 es lo pedido primero


def test_el_programa_cubre_lo_pedido():
    """Cómo funciona Claude, artefactos, skills, conectores y MCP, CRM y Power BI: cada tema tiene su módulo en la etapa 1."""
    por_tema = {m["clave"]: " ".join([m["t"], m["d"]] + [x["t"] + " " + x["d"] for x in m["lecciones"]]) for m in programa.modulos() if m["etapa"] == 1}
    for clave, palabras in (("m01", ["modelo de lenguaje", "contexto", "privacidad"]), ("m05", ["Artefactos", "simulador"]), ("m06", ["skill", "plugin de finanzas"]),
                            ("m07", ["MCP", "conectores", "Permisos"]), ("m08", ["CRM", "HubSpot", "flujo de caja", "Cobranza"]),
                            ("m09", ["Power BI", "DAX", "Power Query", "servidores MCP"])):
        assert all(p.lower() in por_tema[clave].lower() for p in palabras), clave


def test_las_evaluaciones_son_validas_y_cada_pregunta_apunta_a_su_modulo():
    evaluaciones.validar()
    assert len(evaluaciones.DIAGNOSTICO) == 10 and {q["tema"] for q in evaluaciones.DIAGNOSTICO} >= {"m01", "m05", "m07", "m08", "m09"}
    for clave, banco in evaluaciones.BANCOS.items():
        assert all(q["l"].startswith(clave + "-") for q in banco), clave
        assert len({q["p"] for q in banco}) == len(banco)                         # sin preguntas repetidas
        assert {q["tipo"] for q in banco} >= {"unica", "multiple", "numero", "orden"} or clave != "m01"
    umbrales = [u for u, _ in evaluaciones.RECOMENDACIONES]
    assert umbrales == sorted(umbrales) and umbrales[0] == 0


def test_la_ficha_del_curso_trae_lo_que_la_academia_necesita_para_importarlo():
    assert re.fullmatch(r"[a-z0-9-]+", CURSO["slug"]) and CURSO["titulo"] == programa.NOMBRE and CURSO["logros"] == programa.LOGROS
    assert CURSO["instructor"]["nombre"] == programa.AUTOR["nombre"] and all(len(x) == 2 for x in CURSO["preguntas"])
    assert "precio" not in CURSO and "publicado" not in CURSO                    # eso se decide en el panel, no en el contenido
