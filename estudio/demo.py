"""Datos de demostración: un estudio ficticio para recorrer la plataforma sin cargar clientes reales."""

import unicodedata
from datetime import date, timedelta

from . import calendario, cobranza
from .db import REGIMENES
from .rut import digito_verificador

FUENTE = "demo"
EQUIPO_DEMO = ["Carolina", "Matías", "Fernanda"]
UF_DEMO = 39500.0

# (razón social, tipo, giro, régimen, contacto, servicios, honorario, moneda, trabajadores, comuna)
_TODO = ["Contabilidad mensual", "IVA mensual (F29)", "Remuneraciones", "Renta anual (F22)", "Declaraciones juradas"]
_CLIENTES = [
    ("Panadería Los Aromos SpA", "Empresa", "Elaboración de pan y pasteles", 0, "Rosa Fuentes", _TODO + ["Patente municipal"], 180000, "CLP", 8, "Ñuñoa"),
    ("Constructora Río Claro Ltda.", "Empresa", "Obras menores de construcción", 2, "Javier Soto", _TODO, 12, "UF", 24, "Maipú"),
    ("Transportes Cordillera SpA", "Empresa", "Transporte de carga por carretera", 0, "Marcela Pino", _TODO, 250000, "CLP", 11, "Quilicura"),
    ("Clínica Dental Sonrisa SpA", "Empresa", "Servicios odontológicos", 0, "Dra. Paula Reyes", _TODO, 6.5, "UF", 5, "Providencia"),
    ("Ferretería El Maestro EIRL", "Empresa", "Venta de artículos de ferretería", 1, "Hugo Castillo", _TODO[:2] + _TODO[3:], 120000, "CLP", 0, "La Florida"),
    ("Agrícola Valle Verde Ltda.", "Empresa", "Cultivo de frutales", 3, "Ignacio Vera", _TODO[:1] + _TODO[3:], 150000, "CLP", 0, "Buin"),
    ("Estudio de Arquitectura Trazo SpA", "Sociedad de profesionales", "Servicios de arquitectura", 0, "Antonia Lagos", _TODO, 5, "UF", 3, "Las Condes"),
    ("Café Barrio Italia SpA", "Empresa", "Cafetería y restaurante", 1, "Tomás Herrera", _TODO + ["Patente municipal"], 160000, "CLP", 6, "Providencia"),
    ("Camila Rojas Diseño", "Persona natural", "Servicios de diseño gráfico", 4, "Camila Rojas", ["Renta anual (F22)", "Asesoría tributaria"], 45000, "CLP", 0, "Santiago"),
    ("Fundación Manos Abiertas", "Organización sin fines de lucro", "Actividades de asistencia social", 5, "Pedro Aguirre", _TODO[:1] + _TODO[2:], 90000, "CLP", 4, "Recoleta"),
]
_PROSPECTO = ("Importadora Pacífico Sur SpA", "Empresa", "Importación de artículos electrónicos", 0, "Valentina Muñoz")


def _rut(n):
    cuerpo = 55555000 + n
    return f"{cuerpo}-{digito_verificador(cuerpo)}"


def _ascii(texto):
    """Minúsculas sin tildes ni eñes, para armar correos de ejemplo."""
    return unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()


def _dominio(nombre):
    return "".join(ch for ch in _ascii(nombre).split(" ")[1] if ch.isalpha()) + ".example"


def cargar(db, cfg, hoy=None):
    """Carga el estudio de demostración. Devuelve cuántos clientes creó (0 si ya estaba cargado)."""
    hoy = hoy or cfg.hoy()
    if any(c["fuente"] == FUENTE for c in db.clientes()):
        return 0
    if not db.get_json("equipo"):
        db.set_json("equipo", EQUIPO_DEMO)
    if not cobranza.valor_uf(db, cfg):
        db.set_meta(cobranza.META_UF, str(UF_DEMO))
    equipo = db.get_json("equipo")

    ids = []
    for n, (razon, tipo, giro, regimen, contacto, servicios, honorario, moneda, trabajadores, comuna) in enumerate(_CLIENTES, start=1):
        nombre = _ascii(contacto.replace("Dra. ", "")).split(" ")
        ids.append(
            db.crear_cliente(
                rut=_rut(n), razon_social=razon, tipo=tipo, giro=giro, regimen=REGIMENES[regimen], contacto=contacto,
                representante=contacto.replace("Dra. ", ""), email=f"{nombre[0]}.{nombre[-1]}@{_dominio(razon)}",
                telefono=f"+56 9 5550 {1000 + n}", responsable=equipo[n % len(equipo)], estado="Activo",
                fecha_inicio=date(hoy.year - 1 - n % 3, 1 + n % 12, 1), servicios=servicios, honorario=honorario,
                moneda=moneda, trabajadores=trabajadores, comuna=comuna, fuente=FUENTE,
            )
        )
    razon, tipo, giro, regimen, contacto = _PROSPECTO
    prospecto = db.crear_cliente(
        razon_social=razon, tipo=tipo, giro=giro, regimen=REGIMENES[regimen], contacto=contacto,
        email="valentina.munoz@pacificosur.example", telefono="+56 9 5550 2000", responsable=equipo[0],
        estado="Prospecto", fuente=FUENTE, notas="Pidió cotización por contabilidad y remuneraciones (6 trabajadores).",
    )
    db.registrar_interaccion(prospecto, "Reunión inicial: enviar propuesta de honorarios esta semana.", "Reunión", equipo[0])

    # --- Vencimientos: los dos últimos períodos mensuales y las obligaciones anuales del año comercial anterior ---
    actual = calendario.periodo_anterior(hoy)  # se declara este mes
    previo = calendario.periodo_anterior(hoy.replace(day=1) - timedelta(days=1))
    for periodo in (previo, actual, str(hoy.year - 1)):
        calendario.generar(db, cfg, periodo)
    for i, o in enumerate(db.obligaciones()):
        vence = date.fromisoformat(o["vence"])
        if o["periodo"] != actual:
            # Lo antiguo ya está presentado, salvo un par de casos que quedaron atrasados.
            atrasada = o["periodo"] == previo and i % 11 == 4
            if vence < hoy and not atrasada:
                db.actualizar_obligacion(o["id"], estado="Presentada")
            elif atrasada:
                db.actualizar_obligacion(o["id"], estado="Esperando cliente", notas="Faltan facturas de compra del mes.")
        else:
            estado = ("Presentada", "En proceso", "Pendiente", "Esperando cliente", "Pendiente")[i % 5]
            if vence < hoy and estado != "Esperando cliente":
                estado = "Presentada"
            notas = "Falta el libro de asistencia y las licencias médicas." if estado == "Esperando cliente" else ""
            db.actualizar_obligacion(o["id"], estado=estado, notas=notas)
    db.crear_obligacion(ids[1], "OTRO", "Renovar poder del representante ante el SII", hoy + timedelta(days=5), responsable=equipo[1])
    db.crear_obligacion(ids[3], "OTRO", "Inicio de actividades de nueva sucursal", hoy + timedelta(days=12), responsable=equipo[0])

    # --- Honorarios: seis meses emitidos, con pagos al día, parciales y morosos ---
    for atras in range(6, 0, -1):
        anio, mes = calendario.mes_siguiente(hoy.year, hoy.month, -atras)
        anio_e, mes_e = calendario.mes_siguiente(anio, mes)
        emision = min(hoy, date(anio_e, mes_e, 1))
        cobranza.emitir_honorarios(db, cfg, f"{anio:04d}-{mes:02d}", emision)
    for c in db.cobros():
        n = ids.index(c["cliente_id"])
        vence = date.fromisoformat(c["vence"])
        antiguedad = (hoy.year - vence.year) * 12 + hoy.month - vence.month  # 0 = vence este mes
        if n == 2 and antiguedad <= 2:  # moroso: debe los tres últimos meses
            continue
        if n == 7 and antiguedad <= 1:  # atrasado: debe los dos últimos, el más antiguo con abono
            if antiguedad == 1:
                db.registrar_pago(c["id"], c["monto"] // 2, min(hoy, vence + timedelta(days=9)), "Transferencia", "Abono")
            continue
        if antiguedad == 0 and n % 4 == 1:  # el cobro más reciente aún no lo pagan todos
            continue
        db.registrar_pago(c["id"], c["monto"], min(hoy, vence - timedelta(days=n % 4)), ("Transferencia", "Depósito")[n % 2])

    db.registrar_interaccion(ids[2], "Se llamó por honorarios pendientes. Compromete pago para fin de mes.", "Cobranza", equipo[2 % len(equipo)])
    db.registrar_interaccion(ids[0], "Envió facturas de compra del mes por correo.", "Correo", equipo[1 % len(equipo)])
    db.registrar_interaccion(ids[6], "Consulta por cambio a régimen transparente para el próximo año.", "Llamada", equipo[0])
    return len(ids) + 1


def borrar(db):
    return db.borrar_fuente(FUENTE)
