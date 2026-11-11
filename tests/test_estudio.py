"""Pruebas de la plataforma del estudio contable (sin red ni datos reales)."""

import dataclasses
import io
from datetime import date

import pandas as pd
import pytest

from estudio import calendario, cobranza, demo, indicadores, intercambio, tema
from estudio import rut as rutlib
from estudio.config import Config, a_numero
from estudio.db import BaseDatos, ErrorValidacion, estado_cobro

HOY = date(2026, 11, 11)  # miércoles


@pytest.fixture
def cfg(tmp_path):
    return dataclasses.replace(
        Config(), ruta_db=str(tmp_path / "estudio.db"), ruta_db_ebook=str(tmp_path / "ebook.db"),
        valor_uf=40000.0, honorarios_dia_vence=5, firma="Carolina", estudio_nombre="Estudio Prueba",
    )


@pytest.fixture
def db(cfg):
    base = BaseDatos(cfg.ruta_db)
    yield base
    base.cerrar()


def rut_demo(n):
    return f"{55555000 + n}-{rutlib.digito_verificador(55555000 + n)}"


def nuevo_cliente(db, n=1, **campos):
    datos = dict(
        rut=rut_demo(n), razon_social=f"Cliente {n} SpA", responsable="Carolina", honorario=100000, moneda="CLP",
        servicios=["IVA mensual (F29)", "Remuneraciones", "Renta anual (F22)"], email=f"contacto{n}@cliente.example",
        contacto="Ana Soto", telefono="+56 9 5550 0001",
    )
    datos.update(campos)
    return db.crear_cliente(**datos)


# ---------------------------------------------------------------- RUT y números
def test_rut_digito_verificador():
    assert rutlib.digito_verificador(11111111) == "1"
    assert rutlib.digito_verificador(55555009) == "K"
    assert rutlib.digito_verificador(55555003) == "0"


@pytest.mark.parametrize("texto", ["11.111.111-1", "111111111", " 11111111-1 ", "55.555.009-k", "055555009K"])
def test_rut_valido_en_cualquier_formato(texto):
    assert rutlib.validar(texto)


@pytest.mark.parametrize("texto", ["", None, "11.111.111-2", "abc", "1-9", "123", "11111111", "K"])
def test_rut_invalido(texto):
    assert not rutlib.validar(texto)
    with pytest.raises(rutlib.RutInvalido):
        rutlib.normalizar(texto)


def test_rut_normaliza_y_formatea():
    assert rutlib.normalizar("55.555.009-k") == "55555009-K"
    assert rutlib.formatear("55555009K") == "55.555.009-K"
    assert rutlib.formatear("no es rut") == "no es rut"


@pytest.mark.parametrize(
    "texto, esperado",
    [("39.500,25", 39500.25), ("1.234.567", 1234567), ("2,5", 2.5), ("2.5", 2.5), ("$ 150.000", 150000), ("", 0), (7, 7), (float("nan"), 0)],
)
def test_a_numero_formato_chileno(texto, esperado):
    assert a_numero(texto) == esperado


# ---------------------------------------------------------------- clientes
def test_crear_cliente_normaliza_y_lee(db):
    cid = nuevo_cliente(db, 1, rut="55.555.001-4", email="  Contacto@Cliente.Example ")
    c = db.cliente(cid)
    assert c["rut"] == "55555001-4"
    assert c["email"] == "contacto@cliente.example"
    assert c["servicios"] == ["IVA mensual (F29)", "Remuneraciones", "Renta anual (F22)"]
    assert db.cliente_por_rut("55555001-4")["id"] == cid


def test_cliente_rut_duplicado_o_invalido(db):
    nuevo_cliente(db, 1)
    with pytest.raises(ErrorValidacion, match="Ya existe"):
        nuevo_cliente(db, 1, razon_social="Otro")
    with pytest.raises(ErrorValidacion, match="no es válido"):
        nuevo_cliente(db, 2, rut="55.555.002-9")
    assert len(db.clientes()) == 1


def test_solo_los_prospectos_pueden_no_tener_rut(db):
    with pytest.raises(ErrorValidacion, match="RUT es obligatorio"):
        db.crear_cliente(razon_social="Sin RUT", estado="Activo")
    pid = db.crear_cliente(razon_social="Prospecto 1", estado="Prospecto")
    db.crear_cliente(razon_social="Prospecto 2", estado="Prospecto")  # varios sin RUT no chocan
    with pytest.raises(ErrorValidacion, match="RUT es obligatorio"):
        db.actualizar_cliente(pid, estado="Activo")
    db.actualizar_cliente(pid, estado="Activo", rut=rut_demo(7))
    assert db.cliente(pid)["estado"] == "Activo"


def test_actualizar_cliente_valida(db):
    a, b = nuevo_cliente(db, 1), nuevo_cliente(db, 2)
    db.actualizar_cliente(a, rut=rut_demo(1), notas="mismo RUT, sin conflicto")
    with pytest.raises(ErrorValidacion, match="Ya existe"):
        db.actualizar_cliente(b, rut=rut_demo(1))
    with pytest.raises(ErrorValidacion, match="razón social"):
        db.actualizar_cliente(a, razon_social="  ")
    with pytest.raises(ErrorValidacion, match="no permitidos"):
        db.actualizar_cliente(a, id=99)
    with pytest.raises(ErrorValidacion, match="correo"):
        db.actualizar_cliente(a, email="sin arroba")
    with pytest.raises(ErrorValidacion, match="negativo"):
        db.actualizar_cliente(a, honorario=-1)


def test_eliminar_cliente_borra_su_historial(db, cfg):
    a, b = nuevo_cliente(db, 1), nuevo_cliente(db, 2)
    calendario.generar(db, cfg, "2026-10")
    cobranza.emitir_honorarios(db, cfg, "2026-10", HOY)
    for c in db.cobros():
        db.registrar_pago(c["id"], 1000, HOY)
    db.registrar_interaccion(a, "nota")
    db.eliminar_cliente(a)
    assert [c["id"] for c in db.clientes()] == [b]
    assert {o["cliente_id"] for o in db.obligaciones()} == {b}
    assert {c["cliente_id"] for c in db.cobros()} == {b}
    assert {p["cliente_id"] for p in db.pagos()} == {b}
    assert db.interacciones() == []


# ---------------------------------------------------------------- calendario
def test_feriados_moviles():
    assert date(2026, 4, 3) in calendario.feriados(2026)  # Viernes Santo
    assert date(2026, 4, 4) in calendario.feriados(2026)  # Sábado Santo
    assert date(2027, 6, 28) in calendario.feriados(2027)  # San Pedro y San Pablo: martes 29 → lunes 28
    assert date(2027, 10, 11) in calendario.feriados(2027)  # 12 de octubre: martes → lunes 11
    assert date(2028, 10, 27) in calendario.feriados(2028)  # 31 de octubre: martes → viernes anterior
    assert date(2026, 9, 18) in calendario.feriados(2026)


def test_siguiente_habil():
    assert calendario.siguiente_habil(date(2026, 11, 13)) == date(2026, 11, 13)  # viernes
    assert calendario.siguiente_habil(date(2026, 11, 14)) == date(2026, 11, 16)  # sábado → lunes
    assert calendario.siguiente_habil(date(2026, 9, 18)) == date(2026, 9, 21)  # fiestas patrias + fin de semana
    assert calendario.siguiente_habil(date(2026, 11, 13), [date(2026, 11, 13)]) == date(2026, 11, 16)


@pytest.mark.parametrize(
    "codigo, periodo, esperado",
    [
        ("F29", "2026-10", date(2026, 11, 20)),
        ("F29", "2026-12", date(2027, 1, 20)),  # diciembre se declara en enero del año siguiente
        ("PREVIRED", "2026-10", date(2026, 11, 13)),
        ("PREVIRED", "2026-11", date(2026, 12, 14)),  # el 13 cae domingo
        ("LRE", "2026-10", date(2026, 11, 16)),  # el 15 cae domingo
        ("F22", "2025", date(2026, 4, 30)),
        ("DJ1887", "2025", date(2026, 3, 20)),
        ("PATENTE1", "2026", date(2026, 7, 31)),
        ("PATENTE2", "2026", date(2027, 2, 1)),  # el 31 de enero de 2027 cae domingo
    ],
)
def test_vencimiento(codigo, periodo, esperado):
    assert calendario.vencimiento(calendario.CATALOGO[codigo], periodo) == esperado


def test_vencimiento_recorta_al_ultimo_dia_del_mes():
    tipo = dataclasses.replace(calendario.CATALOGO["F29"], dia=31)
    assert calendario.vencimiento(tipo, "2027-01") == date(2027, 3, 1)  # 28 de febrero de 2027 es domingo
    assert calendario.vencimiento(tipo, "2026-03") == date(2026, 4, 30)


@pytest.mark.parametrize("codigo, periodo", [("F29", "2026"), ("F22", "2026-10"), ("F29", "2026-13"), ("F29", "octubre")])
def test_vencimiento_periodo_invalido(codigo, periodo):
    with pytest.raises(ValueError):
        calendario.vencimiento(calendario.CATALOGO[codigo], periodo)


def test_periodos():
    assert calendario.periodo_anterior(date(2026, 1, 15)) == "2025-12"
    assert calendario.periodo_de(date(2026, 1, 15)) == "2026-01"
    assert calendario.mes_siguiente(2026, 12) == (2027, 1)
    assert calendario.mes_siguiente(2026, 1, -1) == (2025, 12)
    assert calendario.nombre_periodo("2026-10") == "octubre 2026"
    assert calendario.nombre_periodo("2025") == "año 2025"


def test_catalogo_con_ajustes():
    tipos = calendario.catalogo({"F29": {"dia": 12}, "F22": {"dia": 29, "mes": 4}, "LRE": {"dia": "x"}})
    assert tipos["F29"].dia == 12 and tipos["F22"].dia == 29 and tipos["LRE"].dia == 15
    assert calendario.vencimiento(tipos["F29"], "2026-10") == date(2026, 11, 12)


def test_generar_segun_servicios_y_sin_duplicar(db, cfg):
    a = nuevo_cliente(db, 1)
    b = nuevo_cliente(db, 2, servicios=["Renta anual (F22)"], responsable="Matías")
    nuevo_cliente(db, 3, estado="Suspendido")
    assert calendario.generar(db, cfg, "2026-10") == 3  # F29, Previred y LRE solo para el cliente 1
    assert calendario.generar(db, cfg, "2026-10") == 0
    assert calendario.generar(db, cfg, "2025") == 2  # F22 para los dos activos
    obligaciones = db.obligaciones()
    assert {(o["cliente_id"], o["tipo"]) for o in obligaciones} == {
        (a, "F29"), (a, "PREVIRED"), (a, "LRE"), (a, "F22"), (b, "F22"),
    }
    assert {o["responsable"] for o in obligaciones if o["cliente_id"] == b} == {"Matías"}
    assert all(o["estado"] == "Pendiente" for o in obligaciones)


def test_generar_usa_ajustes_y_feriados_extra(db, cfg):
    nuevo_cliente(db, 1, servicios=["IVA mensual (F29)"])
    db.set_json(calendario.META_AJUSTES, {"F29": {"dia": 12}})
    cfg = dataclasses.replace(cfg, feriados_extra=[date(2026, 11, 12)])
    calendario.generar(db, cfg, "2026-10")
    assert db.obligaciones()[0]["vence"] == "2026-11-13"


def test_semaforo_y_cierre(db, cfg):
    cid = nuevo_cliente(db, 1)
    ids = {
        nombre: db.crear_obligacion(cid, "OTRO", nombre, vence)
        for nombre, vence in [("atrasada", "2026-11-10"), ("hoy", HOY), ("proxima", "2026-11-18"), ("lejana", "2026-11-19")]
    }
    estados = {o["nombre"]: calendario.semaforo(o, HOY, 7) for o in db.obligaciones()}
    assert estados == {"atrasada": "Atrasada", "hoy": "Vence hoy", "proxima": "Próxima", "lejana": "Al día"}

    db.actualizar_obligacion(ids["atrasada"], estado="Presentada")
    cerrada = db.obligacion(ids["atrasada"])
    assert cerrada["cerrada_en"] and calendario.semaforo(cerrada, HOY) == "Cerrada"
    assert len(db.obligaciones(abiertas=True)) == 3 and len(db.obligaciones(abiertas=False)) == 1
    db.actualizar_obligacion(ids["atrasada"], estado="En proceso")
    assert db.obligacion(ids["atrasada"])["cerrada_en"] is None

    with pytest.raises(ErrorValidacion):
        db.actualizar_obligacion(ids["hoy"], estado="Inventado")
    with pytest.raises(ErrorValidacion):
        db.actualizar_obligacion(ids["hoy"], cliente_id=5)
    with pytest.raises(ErrorValidacion):
        db.actualizar_obligacion(ids["hoy"], vence="mañana")
    db.actualizar_obligacion(ids["hoy"], vence="2026-12-01", responsable=None, notas=None)
    assert db.obligacion(ids["hoy"])["vence"] == "2026-12-01"


def test_tareas_libres_no_tienen_limite_por_periodo(db):
    cid = nuevo_cliente(db, 1)
    assert db.crear_obligacion(cid, "OTRO", "Trámite A", HOY)
    assert db.crear_obligacion(cid, "OTRO", "Trámite B", HOY)
    assert db.crear_obligacion(cid, "F29", "F29", HOY, periodo="2026-10")
    assert db.crear_obligacion(cid, "F29", "F29", HOY, periodo="2026-10") is None
    with pytest.raises(ErrorValidacion):
        db.crear_obligacion(cid, "OTRO", "  ", HOY)


def test_texto_plazo():
    assert calendario.texto_plazo("2026-11-11", HOY) == "hoy"
    assert calendario.texto_plazo("2026-11-12", HOY) == "mañana"
    assert calendario.texto_plazo("2026-11-10", HOY) == "ayer"
    assert calendario.texto_plazo("2026-11-20", HOY) == "en 9 días"
    assert calendario.texto_plazo("2026-11-01", HOY) == "hace 10 días"


# ---------------------------------------------------------------- honorarios
def test_emitir_honorarios_clp_y_uf(db, cfg):
    a = nuevo_cliente(db, 1, honorario=150000)
    b = nuevo_cliente(db, 2, honorario=2.5, moneda="UF")
    nuevo_cliente(db, 3, honorario=0)
    nuevo_cliente(db, 4, estado="De baja")
    creados, avisos = cobranza.emitir_honorarios(db, cfg, "2026-10", HOY)
    assert (creados, avisos) == (2, [])
    montos = {c["cliente_id"]: c for c in db.cobros()}
    assert montos[a]["monto"] == 150000 and montos[b]["monto"] == 100000  # 2,5 UF × $40.000
    assert montos[a]["vence"] == "2026-11-05" and montos[a]["emitido"] == "2026-11-11"
    assert cobranza.emitir_honorarios(db, cfg, "2026-10", HOY) == (0, [])  # no duplica
    assert cobranza.emitir_honorarios(db, cfg, "2026-12", HOY)[0] == 2
    assert {c["vence"] for c in db.cobros() if c["periodo"] == "2026-12"} == {"2027-01-05"}
    with pytest.raises(ValueError):
        cobranza.emitir_honorarios(db, cfg, "2026", HOY)


def test_honorario_en_uf_sin_valor_de_uf(db, cfg):
    nuevo_cliente(db, 1, honorario=3, moneda="UF", razon_social="Cliente UF")
    cfg = dataclasses.replace(cfg, valor_uf=0.0)
    creados, avisos = cobranza.emitir_honorarios(db, cfg, "2026-10", HOY)
    assert creados == 0 and len(avisos) == 1 and "Cliente UF" in avisos[0]
    db.set_meta(cobranza.META_UF, "39.500,50")  # el valor guardado en la app tiene prioridad
    assert cobranza.valor_uf(db, cfg) == 39500.5
    assert cobranza.emitir_honorarios(db, cfg, "2026-10", HOY) == (1, [])
    assert db.cobros()[0]["monto"] == 118502  # 3 × 39.500,5 = 118.501,5 → redondeado


def test_pagos_parciales_y_validaciones(db, cfg):
    nuevo_cliente(db, 1, honorario=100000)
    cobranza.emitir_honorarios(db, cfg, "2026-10", HOY)
    cobro = db.cobros()[0]
    assert estado_cobro(cobro) == "Pendiente" and cobro["saldo"] == 100000

    db.registrar_pago(cobro["id"], 40000, "2026-11-06", "Transferencia", "abono")
    cobro = db.cobro(cobro["id"])
    assert (cobro["pagado"], cobro["saldo"], cobro["ultimo_pago"]) == (40000, 60000, "2026-11-06")
    with pytest.raises(ErrorValidacion, match="supera"):
        db.registrar_pago(cobro["id"], 60001, HOY)
    with pytest.raises(ErrorValidacion, match="mayor que cero"):
        db.registrar_pago(cobro["id"], 0, HOY)
    with pytest.raises(ErrorValidacion, match="ya tiene pagos"):
        db.actualizar_cobro(cobro["id"], anulado=True)
    with pytest.raises(ErrorValidacion, match="menor que lo que ya se pagó"):
        db.actualizar_cobro(cobro["id"], monto=30000)

    pago = db.registrar_pago(cobro["id"], 60000, HOY)
    assert estado_cobro(db.cobro(cobro["id"])) == "Pagado"
    assert db.cobros(pendientes=True) == [] and len(db.cobros(pendientes=False)) == 1
    db.eliminar_pago(pago)
    assert db.cobro(cobro["id"])["saldo"] == 60000


def test_anular_cobro(db, cfg):
    cid = nuevo_cliente(db, 1)
    cobro_id = db.crear_cobro(cid, "Constitución de sociedad", 350000, HOY, "2026-11-30")
    assert db.crear_cobro(cid, "Constitución de sociedad", 350000, HOY, "2026-11-30")  # sin período no hay tope
    db.actualizar_cobro(cobro_id, anulado=True)
    assert estado_cobro(db.cobro(cobro_id)) == "Anulado"
    assert len(db.cobros()) == 1 and len(db.cobros(incluir_anulados=True)) == 2
    with pytest.raises(ErrorValidacion, match="anulado"):
        db.registrar_pago(cobro_id, 1000, HOY)
    with pytest.raises(ErrorValidacion):
        db.crear_cobro(cid, "Gratis", 0, HOY, HOY)


def test_cartera_resumen_y_tramos(db, cfg):
    a, b = nuevo_cliente(db, 1), nuevo_cliente(db, 2)
    vencimientos = {"2026-11-30": 10000, "2026-11-05": 20000, "2026-10-01": 30000, "2026-08-20": 40000, "2026-07-01": 50000}
    for vence, monto in vencimientos.items():
        db.crear_cobro(a, f"Cobro {vence}", monto, "2026-06-01", vence)
    pagado = db.crear_cobro(b, "Pagado", 70000, "2026-10-01", "2026-10-05")
    db.registrar_pago(pagado, 70000, "2026-11-02")
    db.registrar_pago(pagado - 1, 5000, "2026-10-30")  # abono al cobro más antiguo del cliente a

    cartera = cobranza.cartera(db, HOY)
    assert {c["vence"]: (c["dias_mora"], c["tramo"]) for c in cartera} == {
        "2026-11-30": (0, "Por vencer"), "2026-11-05": (6, "1 a 30 días"), "2026-10-01": (41, "31 a 60 días"),
        "2026-08-20": (83, "61 a 90 días"), "2026-07-01": (133, "Más de 90 días"),
    }
    assert cobranza.por_tramo(cartera) == {
        "Por vencer": 10000, "1 a 30 días": 20000, "31 a 60 días": 30000, "61 a 90 días": 40000, "Más de 90 días": 45000,
    }
    assert cobranza.resumen(db, HOY) == {"por_cobrar": 145000, "vencido": 135000, "clientes_morosos": 1, "cobrado_mes": 70000}
    serie = {m["mes"]: (m["emitido"], m["cobrado"]) for m in cobranza.serie_mensual(db, HOY, 6)}
    assert list(serie) == ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-11"]
    assert serie["2026-06"] == (150000, 0) and serie["2026-10"] == (70000, 5000) and serie["2026-11"] == (0, 70000)


def test_formatos_de_dinero():
    assert cobranza.formato_clp(1234567) == "$1.234.567"
    assert cobranza.formato_clp(None) == "$0"
    assert cobranza.formato_corto(1704000) == "$1,7 M"
    assert cobranza.formato_corto(850400) == "$850 mil"
    assert cobranza.formato_corto(9000) == "$9.000"
    assert cobranza.formato_honorario({"honorario": 6.5, "moneda": "UF"}) == "6,5 UF"
    assert cobranza.formato_honorario({"honorario": 180000, "moneda": "CLP"}) == "$180.000"
    assert cobranza.formato_honorario({"honorario": 0}) == "—"


def test_mensajes_para_el_cliente(db, cfg):
    cid = nuevo_cliente(db, 1, honorario=100000)
    cobranza.emitir_honorarios(db, cfg, "2026-09", date(2026, 10, 1))
    cobranza.emitir_honorarios(db, cfg, "2026-10", HOY)
    cliente = db.cliente(cid)
    cfg = dataclasses.replace(cfg, datos_pago="Banco Ejemplo · Cta. Cte. 000")
    asunto, cuerpo = cobranza.mensaje_cobranza(cliente, cobranza.cartera(db, HOY, cid), cfg)
    assert asunto.startswith("Recordatorio de pago") and "Cliente 1 SpA" in asunto
    assert "Hola Ana:" in cuerpo and "septiembre 2026: $100.000 (vence el 05/10/2026)" in cuerpo
    assert "Total: $200.000" in cuerpo and "Banco Ejemplo" in cuerpo and cuerpo.rstrip().endswith("Estudio Prueba")

    db.crear_cobro(cid, "Asesoría", 50000, HOY, "2026-12-01")
    futuro = [c for c in cobranza.cartera(db, HOY, cid) if c["dias_mora"] == 0]
    assert cobranza.mensaje_cobranza(cliente, futuro, cfg)[0].startswith("Aviso de vencimiento")

    calendario.generar(db, cfg, "2026-10")
    asunto, cuerpo = cobranza.mensaje_documentos(cliente, db.obligaciones(cliente_id=cid), cfg)
    assert asunto.startswith("Información pendiente") and "F29 · IVA, PPM y retenciones (octubre 2026): vence el 20/11/2026" in cuerpo

    mailto = cobranza.link_mailto(cliente["email"], asunto, cuerpo)
    assert mailto.startswith("mailto:contacto1%40cliente.example?subject=Informaci%C3%B3n%20pendiente")
    assert "\n" not in mailto and " " not in mailto


@pytest.mark.parametrize(
    "telefono, esperado",
    [("+56 9 5550 1234", "56955501234"), ("955501234", "56955501234"), ("56955501234", "56955501234"), ("2 2555 1234", None), ("", None), (None, None)],
)
def test_telefono_whatsapp(telefono, esperado):
    assert cobranza.telefono_whatsapp(telefono) == esperado
    link = cobranza.link_whatsapp(telefono, "Hola, ¿cómo está?")
    assert (link is None) if esperado is None else link.startswith(f"https://wa.me/{esperado}?text=Hola%2C%20")


# ---------------------------------------------------------------- importar / exportar
def test_importar_clientes_desde_planilla(db):
    nuevo_cliente(db, 1)
    df = pd.DataFrame(
        {
            "RUT": [rut_demo(1), "55.555.002-2", "55.555.003-9", "", "", None],
            "Razón Social": ["Ya existe", "Nueva Ltda.", "RUT malo SpA", "Prospecto sin RUT", "", None],
            "Correo": ["", "nueva@cliente.example", "", "", "", None],
            "Servicios": ["", "iva mensual; REMUNERACIONES, renta anual", "", "", "", None],
            "Honorario mensual": ["", "$ 180.000", "", "", "", None],
            "Moneda": ["", "clp", "", "", "", None],
            "Fecha inicio": ["", "01/03/2025", "", "", "", None],
            "Columna desconocida": ["x", "x", "x", "x", "", None],
        }
    )
    creados, existentes, errores = intercambio.importar_clientes(db, df)
    assert (creados, existentes) == (2, 1)
    assert len(errores) == 1 and errores[0].startswith("Fila 4:") and "no es válido" in errores[0]
    nueva = db.cliente_por_rut("55555002-2")
    assert nueva["razon_social"] == "Nueva Ltda." and nueva["honorario"] == 180000 and nueva["estado"] == "Activo"
    assert nueva["servicios"] == ["IVA mensual (F29)", "Remuneraciones", "Renta anual (F22)"]
    assert nueva["fecha_inicio"] == "2025-03-01" and nueva["fuente"] == "importado"
    assert [c["estado"] for c in db.clientes() if c["razon_social"] == "Prospecto sin RUT"] == ["Prospecto"]
    assert intercambio.importar_clientes(db, df)[:2] == (0, 3)  # repetir no duplica, tampoco al prospecto sin RUT


def test_importar_exige_razon_social(db):
    with pytest.raises(ErrorValidacion):
        intercambio.importar_clientes(db, pd.DataFrame({"rut": [rut_demo(1)]}))


def test_plantilla_de_importacion_se_puede_importar(db):
    df = pd.read_csv(io.BytesIO(intercambio.plantilla_importacion()), dtype=str)
    assert intercambio.importar_clientes(db, df) == (1, 0, [])
    assert db.clientes()[0]["servicios"] == ["Contabilidad mensual", "IVA mensual (F29)", "Remuneraciones", "Renta anual (F22)"]


def test_exportar_excel(db, cfg):
    demo.cargar(db, cfg, HOY)
    hojas = pd.read_excel(io.BytesIO(intercambio.exportar_excel(db, HOY)), sheet_name=None)
    assert list(hojas) == ["Clientes", "Vencimientos", "Honorarios", "Pagos"]
    assert len(hojas["Clientes"]) == 11 and "55.555.001-4" in set(hojas["Clientes"]["RUT"])
    assert len(hojas["Vencimientos"]) == len(db.obligaciones())
    assert hojas["Honorarios"]["Saldo"].sum() == cobranza.resumen(db, HOY)["por_cobrar"]
    assert hojas["Pagos"]["Monto"].sum() == sum(p["monto"] for p in db.pagos())


def test_exportar_excel_sin_datos(db):
    hojas = pd.read_excel(io.BytesIO(intercambio.exportar_excel(db, HOY)), sheet_name=None)
    assert all(h.empty for h in hojas.values())


def test_importar_leads_del_ebook(db, cfg):
    from agentes_ebook.db import BaseDatos as BaseEbook

    assert intercambio.importar_leads_ebook(db, cfg.ruta_db_ebook) == 0  # aún no existe la base del ebook
    ebook = BaseEbook(cfg.ruta_db_ebook)
    for i, (email, empresa, fuente) in enumerate(
        [("ganado@empresa.example", "Empresa Ganada", "gmail"), ("otro@empresa.example", "Sigue en conversación", "gmail"),
         ("demo@empresa.example", "Lead demo", "demo")]
    ):
        lead_id, _ = ebook.registrar_descarga(
            {"mensaje_id": f"m{i}", "email": email, "fecha": "2026-11-01T10:00:00+00:00", "nombre": "Ana Pérez",
             "empresa": empresa, "cargo": "Gerente", "motivo": "Ordenar la contabilidad", "fuente": fuente}
        )
        ebook.actualizar_lead(lead_id, etapa="Sigue" if i == 1 else "Cliente")
    ebook.cerrar()

    assert intercambio.importar_leads_ebook(db, cfg.ruta_db_ebook) == 1
    assert intercambio.importar_leads_ebook(db, cfg.ruta_db_ebook) == 0  # no duplica
    prospecto = db.clientes()[0]
    assert (prospecto["razon_social"], prospecto["estado"], prospecto["fuente"]) == ("Empresa Ganada", "Prospecto", "ebook")
    assert prospecto["contacto"] == "Ana Pérez" and "Gerente" in prospecto["notas"]


# ---------------------------------------------------------------- UF en línea
class _Respuesta:
    def __init__(self, datos=None, estado=200):
        self._datos, self.estado = datos, estado

    def raise_for_status(self):
        if self.estado >= 400:
            raise RuntimeError(f"HTTP {self.estado}")

    def json(self):
        if self._datos is None:
            raise ValueError("no es JSON")
        return self._datos


def _get(datos=None, estado=200, error=None, llamadas=None):
    def get(url, timeout=None, headers=None):
        if llamadas is not None:
            llamadas.append(url)
        if error:
            raise error
        return _Respuesta(datos, estado)

    return get


# Misma forma que la respuesta de mindicador.cl/api/uf (valores inventados).
RESPUESTA_UF = {
    "version": "1.7.0", "autor": "mindicador.cl", "codigo": "uf", "nombre": "Unidad de fomento (UF)", "unidad_medida": "Pesos",
    "serie": [
        {"fecha": "2026-11-12T03:00:00.000Z", "valor": 39512.5},
        {"fecha": "2026-11-11T03:00:00.000Z", "valor": 39510.25},
        {"fecha": "2026-11-10T03:00:00.000Z", "valor": 39508.0},
    ],
}


def test_uf_toma_el_valor_de_hoy_y_no_uno_futuro():
    assert indicadores.valor_del_dia(RESPUESTA_UF, HOY) == (39510.25, HOY)
    assert indicadores.valor_del_dia(RESPUESTA_UF, date(2026, 11, 14)) == (39512.5, date(2026, 11, 12))  # fin de semana sin dato nuevo
    desordenada = {"serie": list(reversed(RESPUESTA_UF["serie"]))}
    assert indicadores.valor_del_dia(desordenada, HOY) == (39510.25, HOY)


@pytest.mark.parametrize(
    "datos",
    [
        {}, {"serie": []}, {"serie": "x"}, None, [],
        {"serie": [{"fecha": "2026-11-11", "valor": 395}]},  # fuera de rango: no es una UF
        {"serie": [{"fecha": "2026-11-11", "valor": "abc"}, {"fecha": "ayer", "valor": 39500}, {"valor": 39500}]},
        {"serie": [{"fecha": "2026-10-20T03:00:00.000Z", "valor": 39400}]},  # demasiado antiguo
        {"serie": [{"fecha": "2026-11-12T03:00:00.000Z", "valor": 39512.5}]},  # solo valores futuros
    ],
)
def test_uf_rechaza_respuestas_inutiles(datos):
    with pytest.raises(indicadores.ErrorIndicador):
        indicadores.valor_del_dia(datos, HOY)


def test_consultar_uf_traduce_los_errores_de_red():
    assert indicadores.consultar_uf(HOY, get=_get(RESPUESTA_UF)) == (39510.25, HOY)
    for get in (_get(estado=503), _get(error=TimeoutError("lento")), _get(datos=None)):
        with pytest.raises(indicadores.ErrorIndicador, match="No se pudo consultar"):
            indicadores.consultar_uf(HOY, get=get)


def test_actualizar_uf_una_vez_al_dia(db, cfg):
    llamadas = []
    assert indicadores.actualizar_uf(db, cfg, HOY, get=_get(RESPUESTA_UF, llamadas=llamadas)) == "actualizada"
    assert cobranza.valor_uf(db, cfg) == 39510.25
    estado = indicadores.estado_uf(db, cfg)
    assert estado["fecha"] == "2026-11-11" and "mindicador.cl" in estado["fuente"] and estado["error"] == ""
    assert indicadores.actualizar_uf(db, cfg, HOY, get=_get(RESPUESTA_UF, llamadas=llamadas)) == "vigente"
    assert len(llamadas) == 1  # con el valor de hoy guardado no vuelve a consultar
    assert indicadores.actualizar_uf(db, cfg, HOY, forzar=True, get=_get(RESPUESTA_UF, llamadas=llamadas)) == "actualizada"
    assert len(llamadas) == 2


def test_actualizar_uf_sin_conexion_conserva_el_valor_y_no_insiste(db, cfg):
    from datetime import datetime, timedelta, timezone

    ahora = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
    llamadas = []
    caido = _get(error=ConnectionError("sin internet"), llamadas=llamadas)
    assert indicadores.actualizar_uf(db, cfg, HOY, get=caido, ahora=ahora) == "error"
    assert cobranza.valor_uf(db, cfg) == 40000.0  # sigue el valor configurado
    assert "No se pudo consultar" in indicadores.estado_uf(db, cfg)["error"]
    assert indicadores.actualizar_uf(db, cfg, HOY, get=caido, ahora=ahora + timedelta(minutes=30)) == "en_espera"
    assert len(llamadas) == 1  # no reintenta en cada clic
    assert indicadores.actualizar_uf(db, cfg, HOY, get=_get(RESPUESTA_UF, llamadas=llamadas), ahora=ahora + timedelta(hours=4)) == "actualizada"
    assert indicadores.estado_uf(db, cfg)["error"] == ""


def test_uf_manual_vale_por_el_dia(db, cfg):
    indicadores.guardar_uf_manual(db, 39999, HOY)
    llamadas = []
    assert indicadores.actualizar_uf(db, cfg, HOY, get=_get(RESPUESTA_UF, llamadas=llamadas)) == "vigente"
    assert cobranza.valor_uf(db, cfg) == 39999 and not llamadas
    assert indicadores.estado_uf(db, cfg)["fuente"] == "ingresada a mano"
    manana = date(2026, 11, 12)
    assert indicadores.actualizar_uf(db, cfg, manana, get=_get(RESPUESTA_UF, llamadas=llamadas)) == "actualizada"
    assert cobranza.valor_uf(db, cfg) == 39512.5


# ---------------------------------------------------------------- apariencia
def test_tema_escapa_el_texto_del_usuario():
    malo = '<img src=x onerror="alert(1)">'
    for fragmento in (
        tema.fila(malo, lado=malo, lado_sub=malo), tema.kpi(malo, malo, malo), tema.chip(malo), tema.chip(malo, tema.ROJO),
        tema.vacio(malo, malo), tema.datos([(malo, malo)]),
    ):
        assert "<img" not in fragmento and "&lt;img" in fragmento
    assert "<img" not in tema.avatar(malo) and ">SO<" in tema.avatar(malo)


def test_tema_iniciales_y_datos():
    assert tema.iniciales("Café Barrio Italia SpA") == "CB"
    assert tema.iniciales("ñandú") == "Ñ"
    assert tema.iniciales("") == "?" and tema.iniciales(None) == "?" and tema.iniciales("  ...  ") == "?"
    assert tema.color_avatar("Cliente") == tema.color_avatar("Cliente")
    assert tema.datos([("RUT", "1-9"), ("Giro", ""), ("Trabajadores", 0), ("Notas", None)]).count("<span>") == 2


# ---------------------------------------------------------------- demo y aplicación
def test_demo_carga_una_sola_vez_y_se_borra_completa(db, cfg):
    assert demo.cargar(db, cfg, HOY) == 11
    assert demo.cargar(db, cfg, HOY) == 0
    clientes = db.clientes()
    assert all(c["fuente"] == "demo" and c["email"].endswith(".example") for c in clientes)
    assert all(c["email"].isascii() for c in clientes)
    assert all(rutlib.validar(c["rut"]) for c in clientes if c["rut"])
    situacion = {calendario.semaforo(o, HOY) for o in db.obligaciones()}
    assert {"Atrasada", "Próxima", "Cerrada"} <= situacion
    res = cobranza.resumen(db, HOY)
    assert 0 < res["vencido"] <= res["por_cobrar"] and res["cobrado_mes"] > 0

    propio = nuevo_cliente(db, 99, razon_social="Cliente real")
    assert demo.borrar(db) == 11
    assert [c["id"] for c in db.clientes()] == [propio]
    assert db.obligaciones() == [] and db.cobros() == [] and db.pagos() == [] and db.interacciones() == []


def test_config_desde_entorno(monkeypatch):
    monkeypatch.setenv("ESTUDIO_NOMBRE", "Estudio Uno")
    monkeypatch.setenv("ESTUDIO_EQUIPO", "Ana, Luis ,")
    monkeypatch.setenv("ESTUDIO_VALOR_UF", "39.500,25")
    monkeypatch.setenv("ESTUDIO_FERIADOS_EXTRA", "2026-06-21, no-es-fecha")
    monkeypatch.setenv("ESTUDIO_HONORARIOS_DIA_VENCE", "45")
    cfg = Config.desde_entorno()
    assert cfg.estudio_nombre == "Estudio Uno" and cfg.equipo == ["Ana", "Luis"]
    assert cfg.valor_uf == 39500.25 and cfg.feriados_extra == [date(2026, 6, 21)]
    assert cfg.honorarios_dia_vence == 28
    assert isinstance(cfg.hoy(), date)


def _app(monkeypatch, tmp_path, **entorno):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ESTUDIO_DB", str(tmp_path / "app.db"))
    monkeypatch.setenv("TABLERO_DB", str(tmp_path / "ebook.db"))
    monkeypatch.setenv("ESTUDIO_UF_AUTOMATICA", "no")  # las pruebas no salen a internet
    monkeypatch.delenv("ESTUDIO_CLAVE", raising=False)
    for clave, valor in entorno.items():
        monkeypatch.setenv(clave, valor)
    import streamlit as st

    st.cache_resource.clear()
    return AppTest.from_file(str(__import__("pathlib").Path(__file__).parent.parent / "gestion_clientes.py"), default_timeout=60)


def _boton(at, etiqueta=None, clave=None):
    """Primer botón con esa etiqueta, o cuya clave empieza con `clave`."""
    return next(b for b in at.button if (etiqueta and b.label == etiqueta) or (clave and str(b.key or "").startswith(clave)))


def _html(at):
    return " ".join(h.proto.body for h in at.get("html") if "<style>" not in h.proto.body)


def test_app_vacia_demo_y_todas_las_pantallas(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path).run()
    assert not at.exception
    assert "Empecemos por tus clientes" in _html(at)

    _boton(at, "Ver con datos de ejemplo").click().run()
    assert not at.exception and not at.error
    assert "Lo que viene" in [s.value for s in at.subheader]
    assert "Atrasados" in _html(at) and "Por cobrar" in _html(at)

    for pantalla, texto in (("clientes", "Agrícola Valle Verde"), ("vencimientos", "Previred"), ("cobros", "Transportes Cordillera"), ("mas", "")):
        _boton(at, clave=f"nav-{pantalla}").click().run()
        assert not at.exception, pantalla
        assert at.session_state["pagina"] == pantalla and texto in _html(at)

    _boton(at, "Borrar demo").click().run()
    _boton(at, clave="nav-inicio").click().run()
    assert not at.exception and "Empecemos por tus clientes" in _html(at)


def test_app_listo_cierra_el_vencimiento_y_los_indicadores_navegan(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path).run()
    _boton(at, "Ver con datos de ejemplo").click().run()
    base = BaseDatos(str(tmp_path / "app.db"))
    try:
        abiertas = len(base.obligaciones(abiertas=True))
        _boton(at, clave="accion-listo-ini-").click().run()
        assert not at.exception
        assert len(base.obligaciones(abiertas=True)) == abiertas - 1
    finally:
        base.cerrar()

    _boton(at, clave="abrir-kpi-atrasados").click().run()
    assert not at.exception
    assert at.session_state["pagina"] == "vencimientos" and at.session_state["filtro_venc"] == "Atrasados"


def test_app_abre_la_ficha_de_un_cliente(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path).run()
    _boton(at, "Ver con datos de ejemplo").click().run()
    _boton(at, clave="nav-clientes").click().run()
    _boton(at, "Abrir Café Barrio Italia SpA").click().run()
    assert not at.exception
    assert at.session_state["pagina"] == "cliente"
    assert 'class="heroe"' in _html(at) and "Café Barrio Italia SpA" in _html(at)
    for seccion in ("Cobros", "Notas", "Datos"):
        next(c for c in at.button_group if c.key and str(c.key).startswith("sec-")).set_value(seccion).run()
        assert not at.exception, seccion
    assert "tomas.herrera@barrio.example" in _html(at)

    _boton(at, clave="volver").click().run()
    assert at.session_state["pagina"] == "clientes"


def test_app_escapa_los_datos_del_cliente(monkeypatch, tmp_path):
    base = BaseDatos(str(tmp_path / "app.db"))
    base.crear_cliente(rut="55.555.001-4", razon_social="<script>alert(1)</script> & Cía.")
    base.cerrar()
    at = _app(monkeypatch, tmp_path).run()
    _boton(at, clave="nav-clientes").click().run()
    assert not at.exception
    assert "<script>alert(1)" not in _html(at) and "&lt;script&gt;alert(1)" in _html(at)


def test_app_pide_clave(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path, ESTUDIO_CLAVE="secreta").run()
    assert not at.exception and not any(b.key == "nav-inicio" for b in at.button)
    at.text_input[0].input("incorrecta")
    at.button[0].click().run()
    assert any("incorrecta" in e.value for e in at.error) and not any(b.key == "nav-inicio" for b in at.button)
    at.text_input[0].input("secreta")
    at.button[0].click().run()
    assert not at.exception and any(b.key == "nav-inicio" for b in at.button)
