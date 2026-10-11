"""Agente de Captura: trae las descargas desde Gmail, un CSV o datos de demostración."""

import hashlib
import random
from datetime import timedelta

import pandas as pd

from .db import ahora_utc, desde_iso, iso
from .plantillas import CADENCIAS

AGENTE = "Captura"


def registrar(db, registros, origen):
    nuevas, nuevos_leads = 0, []
    for reg in registros:
        lead_id, es_nuevo = db.registrar_descarga(reg)
        if lead_id is None:
            continue
        nuevas += 1
        if es_nuevo:
            nuevos_leads.append(lead_id)
            db.log(AGENTE, "Nueva descarga", f"{reg.get('nombre')} <{reg['email']}> · {origen}", lead_id)
        else:
            db.log(AGENTE, "Descarga repetida", f"{reg['email']} volvió a descargar", lead_id)
    return nuevas, nuevos_leads


def desde_gmail(db, cfg, cliente):
    """Lee Gmail desde la última lectura (con 2 días de margen); la primera vez, `descargas_dias`."""
    ultima = desde_iso(db.get_meta("ultima_lectura_gmail"))
    dias = cfg.descargas_dias
    if ultima:
        dias = min(dias, (ahora_utc() - ultima).days + 2)
    inicio = ahora_utc()
    resultado = registrar(db, cliente.leer_descargas(dias), "Gmail")
    db.set_meta("ultima_lectura_gmail", iso(inicio))
    return resultado


COLUMNAS_CSV = {
    "email": ("email", "correo", "e-mail", "mail"),
    "nombre": ("nombre", "name", "nombre completo"),
    "telefono": ("telefono", "teléfono", "phone", "celular", "whatsapp"),
    "cargo": ("cargo", "puesto", "job title"),
    "empresa": ("empresa", "company", "compañía"),
    "motivo": ("motivo", "motivo_de_descarga", "motivo de descarga"),
    "fecha": ("fecha", "date", "submitted at", "fecha de descarga"),
}


def desde_csv(db, df, cfg):
    """Importa un CSV/Excel exportado (FormSubmit, Google Sheets, etc.)."""
    columnas = {c.strip().lower(): c for c in df.columns}
    mapa = {}
    for destino, alias in COLUMNAS_CSV.items():
        for a in alias:
            if a in columnas:
                mapa[destino] = columnas[a]
                break
    if "email" not in mapa:
        raise ValueError("El archivo necesita una columna 'email' (o 'correo').")
    registros = []
    for _, fila in df.iterrows():
        correo = str(fila[mapa["email"]]).strip().lower()
        if "@" not in correo or correo in cfg.excluir_correos:
            continue
        fecha = pd.to_datetime(fila[mapa["fecha"]], errors="coerce", utc=True) if "fecha" in mapa else None
        fecha = ahora_utc() if fecha is None or pd.isna(fecha) else fecha.to_pydatetime()
        datos = {k: ("" if pd.isna(fila[v]) else str(fila[v]).strip()) for k, v in mapa.items() if k not in ("email", "fecha")}
        registros.append(
            {
                "mensaje_id": "csv:" + hashlib.sha1(f"{correo}|{iso(fecha)}".encode()).hexdigest(),
                "fecha": iso(fecha),
                "email": correo,
                "fuente": "csv",
                **datos,
            }
        )
    return registrar(db, registros, "CSV")


# --------------------------------------------------------------------------
# Datos de demostración (ficticios; los dominios .example nunca reciben correos)
# --------------------------------------------------------------------------
NOMBRES = [
    "Camila Rojas", "Matías González", "Valentina Muñoz", "Benjamín Soto", "Fernanda Contreras",
    "Joaquín Silva", "Antonia Morales", "Tomás Fuentes", "Isidora Araya", "Diego Castillo",
    "Javiera Espinoza", "Nicolás Herrera", "Catalina Vargas", "Felipe Reyes", "Martina Núñez",
    "Sebastián Torres", "Florencia Díaz", "Vicente Paredes", "Josefa Carrasco", "Agustín Pizarro",
    "Renata Salazar", "Cristóbal Medina", "Paula Gutiérrez", "Ignacio Vega", "Daniela Campos",
    "Lucas Bravo", "Carolina Ortiz", "Andrés Sepúlveda", "María José Lagos", "Pablo Cárdenas",
    "Constanza Riquelme", "Rodrigo Alarcón", "Sofía Jiménez", "Gabriel Navarro", "Emilia Figueroa",
    "Maximiliano Ríos", "Trinidad Cortés", "Francisco Leiva", "Amanda Tapia", "Alonso Valenzuela",
]
CARGOS = [
    "Gerente de Finanzas", "CFO", "Socio", "Fundador", "Gerente General", "Jefe de Control de Gestión",
    "Subgerente de Administración", "Analista Financiero", "Controller", "Contador", "Consultor Financiero",
    "Estudiante de Ingeniería Comercial", "Asistente Contable", "Dueña", "Coordinador de Finanzas",
]
EMPRESAS = [
    "Constructora Los Andes", "Transportes del Sur", "Agrícola Valle Central", "Clínica Integral",
    "Retail Express", "Inmobiliaria Pacífico", "Minera Atacama", "Distribuidora Norte", "Viña Santa Clara",
    "Logística Austral", "Independiente", "Colegio San Marcos", "Tecnologías Cóndor", "Alimentos del Maule", "",
]
MOTIVOS_DEMO = [
    ("Mejorar el análisis financiero de mi empresa", 6),
    ("Soy consultor/a o asesor/a financiero", 2),
    ("Estoy estudiando y quiero aprender", 1),
    ("Otro", 2),
]
PREFIJOS_DEMO = [("+56 9", 7), ("+51 9", 1), ("+52 33", 1), ("", 1)]


def _elegir(rng, opciones):
    valores, pesos = zip(*opciones)
    return rng.choices(valores, weights=pesos, k=1)[0]


def registros_demo(cantidad=40, semilla=7):
    rng = random.Random(semilla)
    ahora = ahora_utc()
    registros = []
    for i, nombre in enumerate(NOMBRES[:cantidad]):
        empresa = rng.choice(EMPRESAS)
        usuario = nombre.lower().replace(" ", ".").translate(str.maketrans("áéíóúñ", "aeioun"))
        if empresa and empresa != "Independiente" and rng.random() < 0.6:
            dominio = empresa.lower().replace(" ", "").translate(str.maketrans("áéíóúñ", "aeioun")) + ".example"
        else:
            dominio = rng.choice(["gmail.example", "hotmail.example", "outlook.example"])
        prefijo = _elegir(rng, PREFIJOS_DEMO)
        telefono = f"{prefijo} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}" if prefijo else ""
        # Más descargas recientes que antiguas
        dias = min(29.9, rng.expovariate(1 / 9))
        fecha = ahora - timedelta(days=dias, hours=rng.randint(0, 10))
        reg = {
            "mensaje_id": f"demo-{i}",
            "fecha": iso(fecha),
            "email": f"{usuario}@{dominio}",
            "nombre": nombre,
            "telefono": telefono,
            "cargo": rng.choice(CARGOS),
            "empresa": empresa,
            "motivo": _elegir(rng, MOTIVOS_DEMO),
            "fuente": "demo",
        }
        registros.append(reg)
        if rng.random() < 0.12:  # algunos vuelven a descargar
            repetido = dict(reg, mensaje_id=f"demo-{i}-b", fecha=iso(fecha + timedelta(days=rng.randint(1, 4))))
            if repetido["fecha"] < iso(ahora):
                registros.append(repetido)
    return registros


def cargar_demo(db):
    return registrar(db, registros_demo(), "demo")


def simular_historial_demo(db, cfg):
    """Crea un historial creíble para la demo: correos ya enviados, respuestas y reuniones."""
    from . import reuniones
    from .plantillas import redactar

    rng = random.Random(11)
    ahora = ahora_utc()
    demo = [l for l in db.leads() if l["fuente"] == "demo"]
    for lead in demo:
        inicio = max(ahora - timedelta(days=60), desde_iso(lead["primera_descarga"]))
        cadencia = CADENCIAS.get(lead["segmento"] or "Frío")
        for paso, (dias, tipo, nombre) in enumerate(cadencia, start=1):
            fecha = inicio + timedelta(days=dias, hours=1)
            if fecha > ahora - timedelta(hours=2):
                break
            asunto, cuerpo = redactar(tipo, lead, cfg, [])
            db.con.execute(
                "INSERT OR IGNORE INTO seguimientos(lead_id, paso, tipo, nombre_paso, programado_para, estado, "
                "asunto, cuerpo, creado_en, actualizado_en) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (lead["id"], paso, tipo, nombre, iso(fecha), "enviado", asunto, cuerpo, iso(fecha), iso(fecha)),
            )
        if db.seguimientos(lead_id=lead["id"]):
            db.actualizar_lead(lead["id"], etapa="En seguimiento", notificado=1)
    db.con.commit()
    contactados = [l for l in demo if db.seguimientos(lead_id=l["id"])]
    calientes = sorted(contactados, key=lambda l: -l["score"])
    for lead in calientes[:3]:
        db.actualizar_lead(lead["id"], etapa="Respondió", respondio_en=iso(ahora - timedelta(days=rng.randint(1, 3))))
    for lead in calientes[3:5]:
        inicio = reuniones.proximos_horarios(cfg, db.horarios_ocupados(), cantidad=1)[0]
        db.crear_reunion(lead["id"], [iso(inicio)])
        reuniones.agendar(db, lead["id"], inicio)
    if len(calientes) > 5:
        db.actualizar_lead(calientes[5]["id"], etapa="Cliente")
    if len(contactados) > 8:
        db.actualizar_lead(contactados[-1]["id"], etapa="Descartado")
    db.log(AGENTE, "Demo cargada", f"{len(demo)} leads ficticios con historial simulado")
