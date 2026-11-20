"""Estudio de demostración con clientes ficticios (dominios .example, que nunca reciben correos).

Cada visitante recibe su propio estudio de demostración, armado alrededor de la fecha de hoy:
declaraciones por vencer, documentos que faltan, facturas atrasadas y cifras fuera de lo normal.
"""

import random
import secrets
import unicodedata
from datetime import date, datetime, time, timedelta

from . import agentes, planes
from . import calendario as cal
from .db import filas, insertar, json_texto, registrar_actividad
from .seguridad import hash_clave

EQUIPO = [
    ("Camila Rojas", "socio", "Socia", "#0a84ff"),
    ("Diego Muñoz", "contador", "Contador senior", "#ff9f0a"),
    ("Valentina Soto", "analista", "Analista de remuneraciones", "#30d158"),
    ("Matías Pérez", "asistente", "Asistente contable", "#ff375f"),
]

# razón social, nombre corto, giro, régimen, servicios, trabajadores, contacto, comuna,
# responsable (índice en EQUIPO), honorario, años como cliente, ventas netas base, escenario
CLIENTES = [
    ("Panadería Los Aromos SpA", "Panadería Los Aromos", "Elaboración de pan y pastelería", "14D3",
     ["contabilidad", "iva", "remuneraciones", "renta"], 8, "Carolina Fuentes", "Ñuñoa", 1, 280_000, 4, 38_000_000,
     {"docs": "falta", "f29": "preparacion"}),
    ("Transportes Cordillera Limitada", "Transportes Cordillera", "Transporte de carga por carretera", "14A",
     ["contabilidad", "iva", "remuneraciones", "renta"], 22, "Rodrigo Valenzuela", "Pudahuel", 0, 690_000, 7, 142_000_000,
     {"docs": "completo", "f29": "revision", "anomalia": "rem_alza"}),
    ("Clínica Dental Sonrisa Sur SpA", "Clínica Sonrisa Sur", "Servicios odontológicos", "14D8",
     ["contabilidad", "iva", "remuneraciones", "renta"], 5, "Francisca Herrera", "Providencia", 2, 240_000, 3, 21_500_000,
     {"docs": "completo", "f29": "presentada"}),
    ("Ferretería El Roble Limitada", "Ferretería El Roble", "Venta al por menor de artículos de ferretería", "14D3",
     ["contabilidad", "iva", "remuneraciones", "renta"], 11, "Jorge Castillo", "Maipú", 1, 380_000, 9, 64_000_000,
     {"docs": "completo", "f29": "preparacion", "anomalia": "compras_alza"}),
    ("Pehuén Arquitectos SpA", "Pehuén Arquitectos", "Servicios de arquitectura", "14D8",
     ["contabilidad", "iva", "renta"], 0, "Ignacio Lagos", "Vitacura", 3, 150_000, 2, 12_800_000,
     {"docs": "completo", "f29": "presentada"}),
    ("Viña Altos de Colchagua SpA", "Viña Altos de Colchagua", "Elaboración de vinos", "14A",
     ["contabilidad", "iva", "remuneraciones", "renta"], 30, "María José Aravena", "Santa Cruz", 0, 850_000, 6, 118_000_000,
     {"docs": "solicitado", "f29": "revision", "anomalia": "remanente"}),
    ("Café Volcán SpA", "Café Volcán", "Cafetería y venta de café de especialidad", "14D3",
     ["contabilidad", "iva", "renta"], 6, "Tomás Riquelme", "Puerto Varas", 3, 190_000, 2, 17_200_000,
     {"docs": "completo", "f29": "preparacion", "anomalia": "ventas_baja", "pago": "atraso_corto"}),
    ("Inversiones Llaima SpA", "Inversiones Llaima", "Sociedad de inversiones", "14A",
     ["contabilidad", "iva", "renta"], 0, "Andrés Montt", "Las Condes", 0, 320_000, 5, 6_400_000,
     {"docs": "completo", "f29": "presentada"}),
    ("Agrícola Santa Elena Limitada", "Agrícola Santa Elena", "Cultivo de frutales", "14D3",
     ["contabilidad", "iva", "remuneraciones", "renta"], 15, "Patricio Ortúzar", "Rancagua", 1, 420_000, 8, 52_000_000,
     {"docs": "sin_respuesta", "f29": "pendiente", "pago": "atraso_medio"}),
    ("Consultora Andes Digital SpA", "Andes Digital", "Desarrollo de software y consultoría", "14D8",
     ["contabilidad", "iva", "remuneraciones", "renta"], 4, "Daniela Paredes", "Santiago", 3, 210_000, 3, 26_000_000,
     {"docs": "completo", "f29": "presentada", "anomalia": "crecimiento"}),
    ("Lavandería Magallanes EIRL", "Lavandería Magallanes", "Servicios de lavandería", "14D3",
     ["contabilidad", "iva", "remuneraciones"], 2, "Rosa Gallardo", "Punta Arenas", 2, 130_000, 4, 5_900_000,
     {"docs": "completo", "f29": "presentada", "previred": "atrasada", "pago": "atraso_medio"}),
    ("Taller Mecánico Ruta 5 SpA", "Taller Ruta 5", "Mantención y reparación de vehículos", "14D3",
     ["contabilidad", "iva", "remuneraciones", "renta"], 7, "Luis Cárcamo", "San Bernardo", 1, 230_000, 5, 28_500_000,
     {"docs": "sin_respuesta", "f29": "pendiente", "pago": "atraso_largo"}),
    ("Óptica Visión Clara SpA", "Óptica Visión Clara", "Venta de artículos ópticos", "14D8",
     ["contabilidad", "iva", "remuneraciones", "renta"], 3, "Paula Cifuentes", "Concepción", 2, 170_000, 2, 14_300_000,
     {"docs": "completo", "f29": "revision"}),
    ("Constructora Villarrica SpA", "Constructora Villarrica", "Construcción de edificios", "14A",
     ["contabilidad", "iva", "remuneraciones", "renta"], 40, "Felipe Sandoval", "Temuco", 0, 980_000, 10, 210_000_000,
     {"docs": "completo", "f29": "revision"}),
    ("Jardín Infantil Pequeños Pasos Limitada", "Pequeños Pasos", "Educación preescolar", "14D3",
     ["contabilidad", "iva", "remuneraciones", "renta"], 9, "Verónica Leiva", "La Florida", 2, 260_000, 6, 19_800_000,
     {"docs": "completo", "f29": "revision"}),
    ("Marisol Fuentes Ibáñez", "Marisol Fuentes", "Diseñadora independiente", "PN",
     ["renta"], 0, "Marisol Fuentes", "Valparaíso", 3, 45_000, 1, 0,
     {"docs": "completo"}),
]

TAREAS = [
    (8, "Preparar carpeta tributaria para crédito bancario", "El banco pidió carpeta tributaria y balance 2025.", 4, 1),
    (7, "Actualizar representante legal en el SII", "Cambio de administración informado por el cliente.", 7, 0),
    (None, "Revisar honorarios de la cartera para 2027", "Propuesta de reajuste anual por IPC.", 15, 0),
    (13, "Enviar contrato de servicios firmado", "Falta la copia firmada del nuevo alcance.", 2, 3),
]


def _dv(numero):
    suma, factor = 0, 2
    for digito in reversed(str(numero)):
        suma += int(digito) * factor
        factor = 2 if factor == 7 else factor + 1
    resto = 11 - suma % 11
    return "0" if resto == 11 else "K" if resto == 10 else str(resto)


def _rut(numero):
    return f"{numero:,}".replace(",", ".") + "-" + _dv(numero)


def _ascii(texto):
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()


def _correo(nombre, dominio):
    partes = _ascii(nombre).split()
    return f"{partes[0]}.{partes[-1]}@{dominio}"


def _momento(cfg, d, hora=10, minuto=0):
    return datetime.combine(d, time(hora, minuto), tzinfo=cfg.tz).isoformat(timespec="seconds")


def crear_demo(conn, cfg, hoy):
    rnd = random.Random(20261120)
    sufijo = secrets.token_hex(4)
    ahora = cal.ahora(cfg.tz)
    eid = insertar(conn, "estudios", {
        "nombre": "Rojas & Muñoz Contadores", "rut": _rut(77_412_905), "email": "contacto@rojasmunoz.example",
        "telefono": "+56 2 2555 0140", "firma": "Av. Providencia 1208, of. 702 · +56 2 2555 0140",
        "demo": 1, "creado": ahora, **planes.iniciar_prueba(hoy),
    })
    equipo = []
    for nombre, rol, cargo, color in EQUIPO:
        uid = insertar(conn, "usuarios", {
            "estudio_id": eid, "nombre": nombre, "email": f"{_correo(nombre, 'rojasmunoz.example').split('@')[0]}+{sufijo}@rojasmunoz.example",
            "clave_hash": hash_clave(secrets.token_urlsafe(18)), "rol": rol, "cargo": cargo, "color": color,
            "creado": ahora,
        })
        equipo.append(uid)

    periodo = cal.periodo_cerrado(hoy)
    folio = 1201
    ids = []
    for i, (razon, corto, giro, regimen, servicios, trabajadores, contacto, comuna, resp, honorario,
            anios, ventas_base, esc) in enumerate(CLIENTES):
        dominio = _ascii(corto).replace(" ", "") + ".example"
        cliente_desde = date(hoy.year - anios, rnd.randint(1, 12), 1)
        honorario_desde = date(hoy.year - (2 if esc.get("anomalia") == "crecimiento" else 1), 3, 1)
        cid = insertar(conn, "clientes", {
            "estudio_id": eid, "razon_social": razon, "nombre_corto": corto,
            "rut": _rut(rnd.randint(9_000_000, 15_000_000)) if regimen == "PN" else _rut(rnd.randint(76_100_000, 77_900_000)),
            "giro": giro, "regimen": regimen, "servicios": json_texto(servicios), "trabajadores": trabajadores,
            "contacto_nombre": contacto, "contacto_email": _correo(contacto, dominio),
            "contacto_telefono": f"+56 9 5550 {rnd.randint(1000, 9999)}", "comuna": comuna,
            "responsable_id": equipo[resp], "honorario": honorario, "honorario_desde": honorario_desde.isoformat(),
            "estado": "activo", "cliente_desde": cliente_desde.isoformat(), "notas": "", "creado": ahora,
        })
        ids.append(cid)
        cliente = {"id": cid, "servicios": servicios, "trabajadores": trabajadores}

        if ventas_base:
            _indicadores(conn, rnd, cid, periodo, ventas_base, trabajadores, esc.get("anomalia"))
        _obligaciones(conn, rnd, cfg, eid, cliente, equipo[resp], corto, hoy, esc)
        _documentos(conn, cfg, eid, cliente, _correo(contacto, dominio), corto, periodo, hoy, esc)
        folio = _facturas(conn, rnd, eid, cid, honorario, periodo, hoy, folio, esc.get("pago", "al_dia"))

    for indice, titulo, detalle, dias, resp in TAREAS:
        insertar(conn, "tareas", {
            "estudio_id": eid, "cliente_id": ids[indice] if indice is not None else None, "titulo": titulo,
            "detalle": detalle, "vence": cal.sumar_habiles(hoy, dias).isoformat(), "responsable_id": equipo[resp],
            "estado": "abierta", "origen": "manual", "creada": _momento(cfg, hoy - timedelta(days=3), 9, 30),
        })

    _uso_ilustrativo(conn, rnd, cfg, eid, equipo, hoy)
    agentes.asegurar_configuracion(conn, eid)
    agentes.ejecutar(conn, cfg, eid, hoy)
    return eid


# Instrucciones y correos pulidos en lo que va del mes, por persona del equipo.
USO_DEMO = [(9, 18), (6, 52), (2, 14), (11, 31)]


def _uso_ilustrativo(conn, rnd, cfg, eid, equipo, hoy):
    """Consumo de IA de ejemplo para que Plan y uso muestre cómo se ve un mes real."""
    inicio = date(hoy.year, hoy.month, 1)
    dias = max(1, (hoy - inicio).days + 1)
    for uid, (instrucciones, correos) in zip(equipo, USO_DEMO):
        for tipo, n in (("instruccion", instrucciones), ("redaccion", correos)):
            for _ in range(n):
                dia = inicio + timedelta(days=rnd.randrange(dias))
                if tipo == "instruccion":
                    uso = {"entrada": rnd.randint(6_000, 11_000), "cache_lectura": rnd.randint(3_000, 7_000),
                           "cache_escritura": rnd.randint(0, 2_000), "salida": rnd.randint(900, 2_200)}
                else:
                    uso = {"entrada": rnd.randint(700, 1_100), "salida": rnd.randint(350, 800)}
                planes.registrar_uso(conn, eid, uid, tipo, uso, _momento(cfg, dia, rnd.randint(9, 19), rnd.randint(0, 59)))


def _indicadores(conn, rnd, cid, periodo, base, trabajadores, anomalia):
    sueldo = rnd.randint(720_000, 1_150_000)
    meses = 18
    crecimiento = 1.036 if anomalia == "crecimiento" else rnd.uniform(0.998, 1.006)
    for k in range(meses):
        p = cal.mover_periodo(periodo, -(meses - 1 - k))
        mes = int(p[5:7])
        estacional = 1 + 0.08 * (1 if mes in (12, 1) else -0.5 if mes in (6, 7) else 0)
        ventas = base * (crecimiento ** (k - meses + 1)) * estacional * rnd.uniform(0.93, 1.07)
        compras = ventas * rnd.uniform(0.56, 0.68)
        dotacion = trabajadores
        remuneraciones = trabajadores * sueldo * rnd.uniform(0.98, 1.03)
        atras = meses - 1 - k
        if anomalia == "compras_alza" and atras == 0:
            compras *= 2.15
        if anomalia == "ventas_baja" and atras == 0:
            ventas *= 0.52
            compras = ventas * 0.66
        if anomalia == "remanente" and atras <= 2:
            compras = ventas * rnd.uniform(1.18, 1.3)
        if anomalia == "rem_alza" and atras == 0:
            remuneraciones *= 1.38
        conn.execute("""
            INSERT INTO indicadores (cliente_id, periodo, ventas, compras, iva_debito, iva_credito, remuneraciones, trabajadores)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (cid, p, round(ventas, -3), round(compras, -3), round(ventas * 0.19), round(compras * 0.19),
             round(remuneraciones, -3), dotacion))


def _obligaciones(conn, rnd, cfg, eid, cliente, responsable_id, corto, hoy, esc):
    periodo = cal.periodo_cerrado(hoy)
    tipos = cal.tipos_mensuales(cliente["servicios"], cliente["trabajadores"])
    for atras in range(3, -2, -1):
        p = cal.mover_periodo(periodo, -atras)
        for tipo in tipos:
            vence = cal.vencimiento(tipo, p)
            estado, presentada = "pendiente", None
            if atras >= 1 or (atras == 0 and vence < hoy):
                estado = "presentada"
                inicio = max(vence - timedelta(days=6), date(vence.year, vence.month, 1))
                presentada = min(hoy, inicio + timedelta(days=rnd.randint(0, (vence - inicio).days)))
            if atras == 0:
                if tipo == "PREVIRED" and esc.get("previred") == "atrasada" and vence < hoy:
                    estado, presentada = "pendiente", None
                elif tipo == "F29" and vence >= hoy:
                    deseado = esc.get("f29", "preparacion")
                    if deseado == "presentada":
                        presentada = max(hoy - timedelta(days=rnd.randint(0, 4)), date(hoy.year, hoy.month, 1))
                        estado = "presentada"
                    else:
                        estado = deseado
            monto = None
            if tipo == "F29":
                ind = conn.execute("SELECT iva_debito, iva_credito FROM indicadores WHERE cliente_id = ? AND periodo = ?",
                                   (cliente["id"], p)).fetchone()
                if ind:
                    monto = max(0, ind["iva_debito"] - ind["iva_credito"])
            insertar(conn, "obligaciones", {
                "estudio_id": eid, "cliente_id": cliente["id"], "tipo": tipo, "periodo": p, "vence": vence.isoformat(),
                "estado": estado, "monto": monto, "presentada_el": presentada.isoformat() if presentada else None,
            })
            if presentada and (hoy - presentada).days <= 6:
                registrar_actividad(conn, eid, _momento(cfg, presentada, rnd.randint(9, 18), rnd.randint(0, 59)),
                                    f"usuario:{responsable_id}",
                                    f"{cal.TIPOS[tipo]['nombre']} de {cal.nombre_periodo(p)} presentado · {corto}", cliente["id"])
    if "renta" in cliente["servicios"]:
        anterior = hoy.year if hoy > date(hoy.year, 4, 30) else hoy.year - 1
        vence = cal.vencimiento("F22", f"AT{anterior}")
        insertar(conn, "obligaciones", {
            "estudio_id": eid, "cliente_id": cliente["id"], "tipo": "F22", "periodo": f"AT{anterior}",
            "vence": vence.isoformat(), "estado": "presentada", "monto": None,
            "presentada_el": (vence - timedelta(days=rnd.randint(2, 12))).isoformat(),
        })


def _documentos(conn, cfg, eid, cliente, email, corto, periodo, hoy, esc):
    escenario = esc.get("docs", "completo")
    requeridos = cal.documentos_requeridos(cliente["servicios"], cliente["trabajadores"])
    inicio_mes = date(hoy.year, hoy.month, 1)
    for n, tipo in enumerate(requeridos):
        estado, solicitado, recibido, recordatorios = "recibido", None, None, 0
        if escenario == "completo":
            solicitado = cal.habil_desde(inicio_mes)
            recibido = min(hoy, solicitado + timedelta(days=2 + n))
            recordatorios = 1
        elif escenario == "falta":
            estado = "recibido" if n == 1 else "pendiente"
            recibido = min(hoy, inicio_mes + timedelta(days=4)) if estado == "recibido" else None
        elif escenario == "solicitado":
            estado = "recibido" if n == 0 else "solicitado"
            solicitado = hoy - timedelta(days=5)
            recibido = hoy - timedelta(days=3) if estado == "recibido" else None
            recordatorios = 1
        elif escenario == "sin_respuesta":
            estado = "solicitado"
            solicitado = hoy - timedelta(days=4)
            recordatorios = 2
        conn.execute("""
            INSERT INTO documentos (estudio_id, cliente_id, periodo, tipo, estado, solicitado_el, recibido_el, recordatorios)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (eid, cliente["id"], periodo, tipo, estado, solicitado.isoformat() if solicitado else None,
             recibido.isoformat() if recibido else None, recordatorios))
    if escenario in ("solicitado", "sin_respuesta"):
        mes = cal.nombre_periodo(periodo)
        envios = [hoy - timedelta(days=9), hoy - timedelta(days=4)] if escenario == "sin_respuesta" else [hoy - timedelta(days=5)]
        for k, d in enumerate(envios):
            insertar(conn, "correos", {
                "estudio_id": eid, "cliente_id": cliente["id"], "para": email,
                "asunto": (f"Recordatorio: documentos de {mes}" if k else f"Documentos de {mes}") + f" · {corto}",
                "cuerpo": f"Solicitud de documentos de {mes}.", "estado": "simulado",
                "detalle": "Destinatario de demostración: el correo no se envió.", "fecha": _momento(cfg, d, 9, 12),
            })
            registrar_actividad(conn, eid, _momento(cfg, d, 9, 12), "agente:documentos",
                                f"{'Recordatorio' if k else 'Solicitud'} de documentos de {mes} enviada · {corto}", cliente["id"])


def _facturas(conn, rnd, eid, cid, honorario, periodo, hoy, folio, pago):
    impagas = {"al_dia": 0, "atraso_corto": 1, "atraso_medio": 2, "atraso_largo": 3}[pago]
    for atras in range(5, -1, -1):
        p = cal.mover_periodo(periodo, -atras)
        anio, mes = int(cal.mover_periodo(p, 1)[:4]), int(cal.mover_periodo(p, 1)[5:7])
        emitida = cal.habil_desde(date(anio, mes, 1))
        vence = emitida + timedelta(days=15)
        if emitida > hoy:
            continue
        pagada = None
        if atras >= impagas:
            pagada = emitida + timedelta(days=rnd.randint(2, 14))
            if pagada > hoy:
                pagada = None if vence >= hoy else hoy - timedelta(days=1)
        insertar(conn, "facturas", {
            "estudio_id": eid, "cliente_id": cid, "folio": folio, "periodo": p, "monto": honorario,
            "emitida": emitida.isoformat(), "vence": vence.isoformat(),
            "pagada_el": pagada.isoformat() if pagada else None, "recordatorios": 1 if (not pagada and atras >= 1) else 0,
        })
        folio += 1
    return folio


def limpiar_demos_antiguas(conn, hoy, dias=2):
    limite = (hoy - timedelta(days=dias)).isoformat()
    for e in filas(conn, "SELECT id FROM estudios WHERE demo = 1 AND substr(creado, 1, 10) < ?", limite):
        eliminar_estudio(conn, e["id"])


def eliminar_estudio(conn, eid):
    for tabla in ("propuestas", "correos", "actividad", "automatizaciones", "tareas", "facturas", "documentos", "obligaciones"):
        conn.execute(f"DELETE FROM {tabla} WHERE estudio_id = ?", (eid,))
    conn.execute("DELETE FROM indicadores WHERE cliente_id IN (SELECT id FROM clientes WHERE estudio_id = ?)", (eid,))
    conn.execute("DELETE FROM clientes WHERE estudio_id = ?", (eid,))
    conn.execute("DELETE FROM sesiones WHERE usuario_id IN (SELECT id FROM usuarios WHERE estudio_id = ?)", (eid,))
    conn.execute("DELETE FROM usuarios WHERE estudio_id = ?", (eid,))
    conn.execute("DELETE FROM estudios WHERE id = ?", (eid,))
