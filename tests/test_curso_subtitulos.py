"""Subtítulos del curso: cómo se parte el texto y cómo se ajusta cada cambio a las pausas de la voz."""

import re

import pytest

pytest.importorskip("numpy")
pytest.importorskip("PIL")

from curso_youtube import deck  # noqa: E402
from curso_youtube import ensamblar as E  # noqa: E402


def _pasos():
    return [p["texto"] for g in deck.segmentos() for p in g["pasos"]]


def test_no_parte_las_cifras():
    assert E._trozos("Los activos suman 6.105 millones. Los pasivos, 4.065. Cuadra.") == [
        "Los activos suman 6.105 millones. Los pasivos, 4.065. Cuadra."]
    assert E._trozos("La liquidez bajó de 2,00 a 1,65. ¿Qué pasaría si…? Sirve para abrir escenarios y medir la sensibilidad.") == [
        "La liquidez bajó de 2,00 a 1,65. ¿Qué pasaría si…?", "Sirve para abrir escenarios y medir la sensibilidad."]


def test_trozos_del_curso_completos_y_cortos():
    for texto in _pasos():
        trozos = E._trozos(texto)
        assert " ".join(trozos) == " ".join(texto.split()), texto[:60]
        for a, b in zip(trozos, trozos[1:]):
            assert not (re.search(r"\d[.,]$", a) and re.match(r"\d", b)), f"cifra partida: «{a}» | «{b}»"
        for t in trozos:
            assert len(t) <= E.LARGO_SUBTITULO, t
            lineas = E._dos_lineas(t).split("\n")
            assert len(lineas) <= 2 and max(map(len, lineas)) <= 48, lineas


def test_frase_larga_se_corta_en_la_coma():
    trozos = E._trozos("Si solo escribes «analiza este balance», recibes un análisis genérico, que podría ser de cualquier empresa.")
    assert trozos[0].endswith("»,") and len(trozos) == 2


def test_dos_lineas_prefiere_la_puntuacion():
    assert E._dos_lineas("Repasemos. Ocho pasos. Preparar la información y el contexto.") == (
        "Repasemos. Ocho pasos.\nPreparar la información y el contexto.")
    assert E._dos_lineas("La pregunta es qué haces con ellos.") == "La pregunta es qué haces con ellos."


def test_cada_cambio_de_frase_cae_en_la_pausa():
    trozos = ["Primera frase del paso, con algo de texto.", "Segunda frase, un poco más larga que la anterior.",
              "Y la tercera."]
    # la voz hace una pausa corta en una coma (2,0 s) y pausas largas entre frases (3,1 s y 7,0 s)
    pausas = [(1.90, 2.05), (2.75, 3.16), (4.2, 4.33), (6.55, 7.06), (7.9, 8.6)]
    inicios = E.tiempos_de_subtitulos(0.0, 8.54, trozos, pausas)
    assert inicios == pytest.approx([0.0, 3.10, 7.00])


def test_sin_pausas_reparte_por_texto_hablado():
    trozos = ["Vendió 11.800 millones de pesos en el año", "y ganó 485."]
    inicios = E.tiempos_de_subtitulos(10.0, 16.0, trozos, [])
    # «485» son tres caracteres escritos y treinta hablados: el cambio llega antes que repartiendo por lo escrito (14,7 s)
    assert inicios[0] == 10.0 and 13.2 < inicios[1] < 14.0


def test_subtitulos_en_orden_y_sin_traslapes():
    tl = dict(subtitulos=[(0.35, 9.0, "Primera frase del paso, con algo de texto. Segunda frase, un poco más larga que la anterior. Y la tercera."),
                          (9.0, 12.0, "Un paso corto.")],
              pausas=[(3.1, 3.5), (7.3, 7.9), (8.5, 9.06)])
    subs = E.subtitulos_con_tiempo(tl)
    assert [s[2] for s in subs][-1] == "Un paso corto."
    for (a0, a1, _), (b0, _, _) in zip(subs, subs[1:]):
        assert a0 < a1 <= b0
    assert subs[0][0] == 0.35 and subs[-1][1] <= 12.0
