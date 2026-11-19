"""Pruebas del conversor de guion a texto para la voz y de las cuadraturas del caso."""

import pytest

from curso_youtube import caso
from curso_youtube.voz import a_voz, entero_a_palabras, numeros_a_palabras


@pytest.mark.parametrize("n, esperado", [
    (0, "cero"), (1, "uno"), (16, "dieciséis"), (21, "veintiuno"), (30, "treinta"), (31, "treinta y uno"),
    (65, "sesenta y cinco"), (100, "cien"), (101, "ciento uno"), (118, "ciento dieciocho"),
    (485, "cuatrocientos ochenta y cinco"), (555, "quinientos cincuenta y cinco"),
    (1000, "mil"), (1005, "mil cinco"), (1150, "mil ciento cincuenta"),
    (1344, "mil trescientos cuarenta y cuatro"), (2025, "dos mil veinticinco"),
    (6105, "seis mil ciento cinco"), (11800, "once mil ochocientos"), (21000, "veintiún mil"),
    (1_000_000, "un millón"), (2_500_000, "dos millones quinientos mil"),
])
def test_entero_a_palabras(n, esperado):
    assert entero_a_palabras(n) == esperado


@pytest.mark.parametrize("texto, esperado", [
    ("vendió 11.800 millones de pesos", "vendió once mil ochocientos millones de pesos"),
    ("creció 18 %", "creció dieciocho por ciento"),
    ("el 1 % de los activos", "el uno por ciento de los activos"),
    ("bajó de 2 a 1,65", "bajó de dos a uno coma sesenta y cinco"),
    ("cayó de 0,39 a 0,03", "cayó de cero coma treinta y nueve a cero coma cero tres"),
    ("subió de 1 a 1,19", "subió de uno a uno coma diecinueve"),
    ("pagan a 101 días", "pagan a ciento un días"),
    ("recuperar 1 punto de margen", "recuperar un punto de margen"),
    ("son 21 millones", "son veintiún millones"),
    ("una proyección a 13 semanas", "una proyección a trece semanas"),
    ("en 21 semanas", "en veintiuna semanas"),
    ("en 200 cuotas", "en doscientas cuotas"),
    ("Paso 1: preparar", "Paso uno: preparar"),
    ("En 2025 son 92.", "En dos mil veinticinco son noventa y dos."),
    ("entre 19,6 % y 3,9 %", "entre diecinueve coma seis por ciento y tres coma nueve por ciento"),
    ("75 pesos por cada 100.", "Setenta y cinco pesos por cada cien."),
    ("Crecer cuesta caja. Son 75 pesos.", "Crecer cuesta caja. Son setenta y cinco pesos."),
    ("Bajó mucho. 156 menos 118: los 38 que vimos.", "Bajó mucho. Ciento cincuenta y seis menos ciento dieciocho: los treinta y ocho que vimos."),
    ("de cada 100 pesos", "de cada cien pesos"),
    ("casi 1.500 millones", "casi mil quinientos millones"),
])
def test_numeros_en_contexto(texto, esperado):
    assert a_voz(texto) == esperado


def test_pronunciacion():
    assert a_voz("analizar con Claude en claude.ai") == "analizar con Clod en clod punto a i"
    assert a_voz("sobre EBITDA") == "sobre ébitda"
    assert a_voz("un PDF o un CSV") == "un pe de efe o un ce ese ve"
    assert a_voz("«analiza este balance»") == "analiza este balance"
    assert a_voz("descarga el ebook") == "descarga el íbuk"
    assert a_voz("gerente de finanzas o controller") == "gerente de finanzas o contróler"
    assert a_voz("conecta conmigo en LinkedIn") == "conecta conmigo en Línkedin"


def test_no_toca_palabras_con_digitos():
    assert numeros_a_palabras("ratio v2") == "ratio v2"


def test_caso_cuadra():
    caso.verificar()
    b = caso.balance("2025")
    assert b["activos"] == b["pasivo_patrimonio"] == 6105
    assert caso.flujo_2025()["efectivo_final"] == 65
    assert caso.eerr("2025")["utilidad"] == 485
    assert round(caso.ratios("2025")["ciclo_caja"]) == 92
