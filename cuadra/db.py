"""Persistencia en SQLite. Cada tabla de negocio lleva estudio_id: un estudio nunca ve datos de otro."""

import json
import os
import sqlite3
from contextlib import contextmanager

ESQUEMA = """
CREATE TABLE IF NOT EXISTS estudios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL,
    rut TEXT DEFAULT '',
    email TEXT DEFAULT '',
    telefono TEXT DEFAULT '',
    firma TEXT DEFAULT '',
    demo INTEGER DEFAULT 0,
    plan TEXT DEFAULT 'profesional',
    estado_suscripcion TEXT DEFAULT 'prueba',
    prueba_hasta TEXT,
    pagado_hasta TEXT,
    creado TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usuarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estudio_id INTEGER NOT NULL REFERENCES estudios(id),
    nombre TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    clave_hash TEXT NOT NULL,
    rol TEXT DEFAULT 'contador',
    cargo TEXT DEFAULT '',
    color TEXT DEFAULT '',
    creado TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sesiones (
    token_hash TEXT PRIMARY KEY,
    usuario_id INTEGER NOT NULL REFERENCES usuarios(id),
    expira TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clientes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estudio_id INTEGER NOT NULL REFERENCES estudios(id),
    razon_social TEXT NOT NULL,
    nombre_corto TEXT DEFAULT '',
    rut TEXT DEFAULT '',
    giro TEXT DEFAULT '',
    regimen TEXT DEFAULT '14D3',
    servicios TEXT DEFAULT '["contabilidad","iva"]',
    trabajadores INTEGER DEFAULT 0,
    contacto_nombre TEXT DEFAULT '',
    contacto_email TEXT DEFAULT '',
    contacto_telefono TEXT DEFAULT '',
    comuna TEXT DEFAULT '',
    responsable_id INTEGER REFERENCES usuarios(id),
    honorario INTEGER DEFAULT 0,
    honorario_desde TEXT,
    estado TEXT DEFAULT 'activo',
    cliente_desde TEXT,
    notas TEXT DEFAULT '',
    creado TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS obligaciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estudio_id INTEGER NOT NULL,
    cliente_id INTEGER NOT NULL REFERENCES clientes(id),
    tipo TEXT NOT NULL,
    periodo TEXT NOT NULL,
    vence TEXT NOT NULL,
    estado TEXT DEFAULT 'pendiente',
    monto INTEGER,
    presentada_el TEXT,
    UNIQUE (cliente_id, tipo, periodo)
);
CREATE TABLE IF NOT EXISTS documentos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estudio_id INTEGER NOT NULL,
    cliente_id INTEGER NOT NULL REFERENCES clientes(id),
    periodo TEXT NOT NULL,
    tipo TEXT NOT NULL,
    estado TEXT DEFAULT 'pendiente',
    solicitado_el TEXT,
    recibido_el TEXT,
    recordatorios INTEGER DEFAULT 0,
    UNIQUE (cliente_id, periodo, tipo)
);
CREATE TABLE IF NOT EXISTS indicadores (
    cliente_id INTEGER NOT NULL REFERENCES clientes(id),
    periodo TEXT NOT NULL,
    ventas INTEGER DEFAULT 0,
    compras INTEGER DEFAULT 0,
    iva_debito INTEGER DEFAULT 0,
    iva_credito INTEGER DEFAULT 0,
    remuneraciones INTEGER DEFAULT 0,
    trabajadores INTEGER DEFAULT 0,
    PRIMARY KEY (cliente_id, periodo)
);
CREATE TABLE IF NOT EXISTS facturas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estudio_id INTEGER NOT NULL,
    cliente_id INTEGER NOT NULL REFERENCES clientes(id),
    folio INTEGER NOT NULL,
    periodo TEXT NOT NULL,
    monto INTEGER NOT NULL,
    emitida TEXT NOT NULL,
    vence TEXT NOT NULL,
    pagada_el TEXT,
    recordatorios INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tareas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estudio_id INTEGER NOT NULL,
    cliente_id INTEGER,
    titulo TEXT NOT NULL,
    detalle TEXT DEFAULT '',
    vence TEXT,
    responsable_id INTEGER,
    estado TEXT DEFAULT 'abierta',
    origen TEXT DEFAULT 'manual',
    creada TEXT NOT NULL,
    hecha_el TEXT
);
CREATE TABLE IF NOT EXISTS propuestas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estudio_id INTEGER NOT NULL,
    agente TEXT NOT NULL,
    tipo TEXT NOT NULL,
    clave TEXT NOT NULL,
    cliente_id INTEGER,
    titulo TEXT NOT NULL,
    resumen TEXT DEFAULT '',
    motivos TEXT DEFAULT '[]',
    prioridad INTEGER DEFAULT 2,
    datos TEXT DEFAULT '{}',
    estado TEXT DEFAULT 'pendiente',
    creada TEXT NOT NULL,
    resuelta TEXT,
    resuelta_por INTEGER,
    resultado TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS correos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estudio_id INTEGER NOT NULL,
    cliente_id INTEGER,
    para TEXT NOT NULL,
    asunto TEXT NOT NULL,
    cuerpo TEXT NOT NULL,
    estado TEXT NOT NULL,
    detalle TEXT DEFAULT '',
    fecha TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS actividad (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estudio_id INTEGER NOT NULL,
    fecha TEXT NOT NULL,
    autor TEXT NOT NULL,
    cliente_id INTEGER,
    texto TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS automatizaciones (
    estudio_id INTEGER NOT NULL,
    clave TEXT NOT NULL,
    activo INTEGER DEFAULT 1,
    modo TEXT DEFAULT 'revisar',
    ultima_ejecucion TEXT,
    ultimo_resultado TEXT DEFAULT '',
    PRIMARY KEY (estudio_id, clave)
);
CREATE TABLE IF NOT EXISTS uso_ia (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estudio_id INTEGER NOT NULL,
    usuario_id INTEGER,
    periodo TEXT NOT NULL,
    fecha TEXT NOT NULL,
    tipo TEXT NOT NULL,
    entrada INTEGER DEFAULT 0,
    salida INTEGER DEFAULT 0,
    cache_lectura INTEGER DEFAULT 0,
    cache_escritura INTEGER DEFAULT 0,
    tokens INTEGER DEFAULT 0,
    costo_usd REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS paquetes_ia (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estudio_id INTEGER NOT NULL,
    periodo TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    fecha TEXT NOT NULL,
    nota TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_uso_ia ON uso_ia(estudio_id, periodo);
CREATE INDEX IF NOT EXISTS ix_clientes_estudio ON clientes(estudio_id);
CREATE INDEX IF NOT EXISTS ix_obligaciones_estudio ON obligaciones(estudio_id, vence);
CREATE INDEX IF NOT EXISTS ix_documentos_estudio ON documentos(estudio_id, periodo);
CREATE INDEX IF NOT EXISTS ix_facturas_estudio ON facturas(estudio_id, cliente_id);
CREATE INDEX IF NOT EXISTS ix_propuestas_estudio ON propuestas(estudio_id, estado, clave);
CREATE INDEX IF NOT EXISTS ix_actividad_estudio ON actividad(estudio_id, fecha);
"""


def _dict(cursor, fila):
    return {col[0]: fila[i] for i, col in enumerate(cursor.description)}


# Columnas agregadas después de la primera versión: (tabla, columna, definición).
MIGRACIONES = [
    ("estudios", "plan", "TEXT DEFAULT 'profesional'"),
    ("estudios", "estado_suscripcion", "TEXT DEFAULT 'prueba'"),
    ("estudios", "prueba_hasta", "TEXT"),
    ("estudios", "pagado_hasta", "TEXT"),
]


def inicializar(ruta):
    carpeta = os.path.dirname(ruta)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    with conectar(ruta) as conn:
        for tabla, columna, definicion in MIGRACIONES:
            existe_tabla = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (tabla,)).fetchone()
            columnas = {c["name"] for c in conn.execute(f"PRAGMA table_info({tabla})").fetchall()}
            if existe_tabla and columna not in columnas:
                conn.execute(f"ALTER TABLE {tabla} ADD COLUMN {columna} {definicion}")
        conn.executescript(ESQUEMA)


@contextmanager
def conectar(ruta):
    conn = sqlite3.connect(ruta, timeout=30)
    conn.row_factory = _dict
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def filas(conn, sql, *params):
    return conn.execute(sql, params).fetchall()


def fila(conn, sql, *params):
    return conn.execute(sql, params).fetchone()


def valor(conn, sql, *params):
    resultado = conn.execute(sql, params).fetchone()
    return next(iter(resultado.values())) if resultado else None


def insertar(conn, tabla, datos):
    columnas = ", ".join(datos)
    marcas = ", ".join("?" for _ in datos)
    cursor = conn.execute(f"INSERT INTO {tabla} ({columnas}) VALUES ({marcas})", tuple(datos.values()))
    return cursor.lastrowid


def json_cargar(texto, defecto=None):
    try:
        return json.loads(texto) if texto else defecto
    except (TypeError, json.JSONDecodeError):
        return defecto


def json_texto(datos):
    return json.dumps(datos, ensure_ascii=False)


def registrar_actividad(conn, estudio_id, fecha, autor, texto, cliente_id=None):
    insertar(conn, "actividad", {
        "estudio_id": estudio_id, "fecha": fecha, "autor": autor,
        "cliente_id": cliente_id, "texto": texto,
    })
