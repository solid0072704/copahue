"""Persistencia en SQLite: supermercados, listas, historial de precios, compras y bitácora."""

import json
import os
import sqlite3
from datetime import date

from agentes_ebook.db import ahora_utc, iso

FRECUENCIAS = {"unica": "Una vez", "semanal": "Cada semana", "quincenal": "Cada 2 semanas", "mensual": "Cada mes"}

ESQUEMA = """
CREATE TABLE IF NOT EXISTS tiendas (
    nombre TEXT PRIMARY KEY,
    tipo TEXT NOT NULL,
    url TEXT DEFAULT '',
    activa INTEGER DEFAULT 1,
    despacho REAL DEFAULT 0,
    gratis_desde REAL DEFAULT 0,
    minimo REAL DEFAULT 0,
    parametros TEXT DEFAULT '{}',
    ultimo_estado TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS listas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT NOT NULL,
    frecuencia TEXT DEFAULT 'unica',
    proxima_fecha TEXT,
    activa INTEGER DEFAULT 1,
    max_tiendas INTEGER DEFAULT 2,
    notas TEXT DEFAULT '',
    creado_en TEXT
);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lista_id INTEGER NOT NULL REFERENCES listas(id) ON DELETE CASCADE,
    consulta TEXT NOT NULL,
    cantidad REAL DEFAULT 1,
    formato TEXT DEFAULT '',
    marcas TEXT DEFAULT '',
    excluir TEXT DEFAULT '',
    orden INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS precios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT NOT NULL,
    tienda TEXT NOT NULL,
    consulta TEXT NOT NULL,
    sku TEXT,
    nombre TEXT,
    marca TEXT,
    precio REAL,
    precio_normal REAL,
    cantidad REAL,
    unidad TEXT,
    url TEXT
);
CREATE INDEX IF NOT EXISTS ix_precios_consulta ON precios(consulta, fecha);
CREATE TABLE IF NOT EXISTS compras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lista_id INTEGER NOT NULL,
    fecha_objetivo TEXT NOT NULL,
    estado TEXT DEFAULT 'propuesta',
    total REAL DEFAULT 0,
    ahorro REAL DEFAULT 0,
    tiendas TEXT DEFAULT '',
    detalle TEXT DEFAULT '{}',
    notificada INTEGER DEFAULT 0,
    creado_en TEXT,
    actualizado_en TEXT,
    UNIQUE(lista_id, fecha_objetivo)
);
CREATE TABLE IF NOT EXISTS alertas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT NOT NULL,
    consulta TEXT NOT NULL,
    tienda TEXT,
    nombre TEXT,
    precio_unidad REAL,
    mediana REAL,
    url TEXT,
    notificada INTEGER DEFAULT 0,
    UNIQUE(fecha, consulta)
);
CREATE TABLE IF NOT EXISTS ia_coincidencias (
    consulta TEXT NOT NULL,
    nombre TEXT NOT NULL,
    ok INTEGER NOT NULL,
    PRIMARY KEY (consulta, nombre)
);
CREATE TABLE IF NOT EXISTS bitacora (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT NOT NULL,
    agente TEXT,
    accion TEXT,
    detalle TEXT
);
CREATE TABLE IF NOT EXISTS meta (clave TEXT PRIMARY KEY, valor TEXT);
"""


class BaseSuper:
    def __init__(self, ruta):
        if ruta != ":memory:":
            os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
        self.con = sqlite3.connect(ruta, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA foreign_keys = ON")
        self.con.executescript(ESQUEMA)
        self.con.commit()

    # --- utilidades ---
    def _filas(self, sql, params=()):
        return [dict(f) for f in self.con.execute(sql, params).fetchall()]

    def _fila(self, sql, params=()):
        f = self.con.execute(sql, params).fetchone()
        return dict(f) if f else None

    def _ejecutar(self, sql, params=()):
        cur = self.con.execute(sql, params)
        self.con.commit()
        return cur

    def cerrar(self):
        self.con.commit()
        self.con.close()

    def get_meta(self, clave, defecto=None):
        f = self._fila("SELECT valor FROM meta WHERE clave = ?", (clave,))
        return f["valor"] if f else defecto

    def set_meta(self, clave, valor):
        self._ejecutar("INSERT OR REPLACE INTO meta (clave, valor) VALUES (?, ?)", (clave, str(valor)))

    def log(self, agente, accion, detalle=""):
        self._ejecutar(
            "INSERT INTO bitacora (fecha, agente, accion, detalle) VALUES (?, ?, ?, ?)",
            (iso(ahora_utc()), agente, accion, detalle),
        )

    def bitacora(self, limite=300):
        return self._filas("SELECT * FROM bitacora ORDER BY id DESC LIMIT ?", (limite,))

    # --- tiendas ---
    def tiendas(self, solo_activas=False):
        filas = self._filas("SELECT * FROM tiendas" + (" WHERE activa = 1" if solo_activas else "") + " ORDER BY nombre")
        for f in filas:
            f["parametros"] = json.loads(f["parametros"] or "{}")
        return filas

    def guardar_tienda(self, nombre, tipo, url="", activa=True, despacho=0, gratis_desde=0, minimo=0, parametros=None):
        self._ejecutar(
            """INSERT INTO tiendas (nombre, tipo, url, activa, despacho, gratis_desde, minimo, parametros)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(nombre) DO UPDATE SET tipo=excluded.tipo, url=excluded.url, activa=excluded.activa,
               despacho=excluded.despacho, gratis_desde=excluded.gratis_desde, minimo=excluded.minimo,
               parametros=excluded.parametros""",
            (nombre, tipo, url, int(bool(activa)), despacho or 0, gratis_desde or 0, minimo or 0,
             json.dumps(parametros or {})),
        )

    def actualizar_tienda(self, nombre, **campos):
        if "parametros" in campos:
            campos["parametros"] = json.dumps(campos["parametros"] or {})
        if "activa" in campos:
            campos["activa"] = int(bool(campos["activa"]))
        sets = ", ".join(f"{k} = ?" for k in campos)
        self._ejecutar(f"UPDATE tiendas SET {sets} WHERE nombre = ?", (*campos.values(), nombre))

    def borrar_tienda(self, nombre):
        self._ejecutar("DELETE FROM tiendas WHERE nombre = ?", (nombre,))

    # --- listas e ítems ---
    def listas(self, solo_activas=False):
        return self._filas("SELECT * FROM listas" + (" WHERE activa = 1" if solo_activas else "") + " ORDER BY id")

    def lista(self, lista_id):
        return self._fila("SELECT * FROM listas WHERE id = ?", (lista_id,))

    def crear_lista(self, nombre, frecuencia="unica", proxima_fecha=None, max_tiendas=2, notas=""):
        cur = self._ejecutar(
            "INSERT INTO listas (nombre, frecuencia, proxima_fecha, max_tiendas, notas, creado_en) VALUES (?, ?, ?, ?, ?, ?)",
            (nombre, frecuencia, (proxima_fecha or date.today()).isoformat(), max_tiendas, notas, iso(ahora_utc())),
        )
        return cur.lastrowid

    def actualizar_lista(self, lista_id, **campos):
        if isinstance(campos.get("proxima_fecha"), date):
            campos["proxima_fecha"] = campos["proxima_fecha"].isoformat()
        sets = ", ".join(f"{k} = ?" for k in campos)
        self._ejecutar(f"UPDATE listas SET {sets} WHERE id = ?", (*campos.values(), lista_id))

    def borrar_lista(self, lista_id):
        self._ejecutar("DELETE FROM items WHERE lista_id = ?", (lista_id,))
        self._ejecutar("DELETE FROM compras WHERE lista_id = ?", (lista_id,))
        self._ejecutar("DELETE FROM listas WHERE id = ?", (lista_id,))

    def items(self, lista_id):
        return self._filas("SELECT * FROM items WHERE lista_id = ? ORDER BY orden, id", (lista_id,))

    def agregar_item(self, lista_id, consulta, cantidad=1, formato="", marcas="", excluir=""):
        cur = self._ejecutar(
            "INSERT INTO items (lista_id, consulta, cantidad, formato, marcas, excluir) VALUES (?, ?, ?, ?, ?, ?)",
            (lista_id, consulta.strip(), cantidad, formato or "", marcas or "", excluir or ""),
        )
        return cur.lastrowid

    def reemplazar_items(self, lista_id, items):
        self.con.execute("DELETE FROM items WHERE lista_id = ?", (lista_id,))
        for orden, it in enumerate(items):
            if not str(it.get("consulta") or "").strip():
                continue
            self.con.execute(
                "INSERT INTO items (lista_id, consulta, cantidad, formato, marcas, excluir, orden) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (lista_id, str(it["consulta"]).strip(), float(it.get("cantidad") or 1), it.get("formato") or "",
                 it.get("marcas") or "", it.get("excluir") or "", orden),
            )
        self.con.commit()

    # --- precios ---
    def guardar_precios(self, fecha, consulta, productos, tienda=None):
        """Guarda la foto de precios del día (reemplaza la del mismo día, tienda y consulta)."""
        for t in {tienda} if tienda else {p.tienda for p in productos}:
            self.con.execute("DELETE FROM precios WHERE fecha = ? AND tienda = ? AND consulta = ?", (fecha, t, consulta))
        self.con.executemany(
            """INSERT INTO precios (fecha, tienda, consulta, sku, nombre, marca, precio, precio_normal, cantidad, unidad, url)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (fecha, p.tienda, consulta, p.sku, p.nombre, p.marca, p.precio, p.precio_normal, p.cantidad, p.unidad, p.url)
                for p in productos
            ],
        )
        self.con.commit()

    def historial(self, consulta=None, desde=None):
        sql, params = "SELECT * FROM precios WHERE 1=1", []
        if consulta:
            sql += " AND consulta = ?"
            params.append(consulta)
        if desde:
            sql += " AND fecha >= ?"
            params.append(desde)
        return self._filas(sql + " ORDER BY fecha", params)

    def consultas_con_historial(self):
        return [f["consulta"] for f in self._filas("SELECT DISTINCT consulta FROM precios ORDER BY consulta")]

    # --- compras ---
    def compra(self, compra_id):
        f = self._fila("SELECT * FROM compras WHERE id = ?", (compra_id,))
        if f:
            f["detalle"] = json.loads(f["detalle"] or "{}")
        return f

    def compras(self, lista_id=None, estados=None):
        sql, params = "SELECT * FROM compras WHERE 1=1", []
        if lista_id:
            sql += " AND lista_id = ?"
            params.append(lista_id)
        if estados:
            sql += f" AND estado IN ({','.join('?' * len(estados))})"
            params += list(estados)
        filas = self._filas(sql + " ORDER BY fecha_objetivo DESC, id DESC", params)
        for f in filas:
            f["detalle"] = json.loads(f["detalle"] or "{}")
        return filas

    def guardar_compra(self, lista_id, fecha_objetivo, total, ahorro, tiendas, detalle):
        """Crea o actualiza la propuesta de compra de esa lista y fecha. Devuelve el id."""
        ahora = iso(ahora_utc())
        existente = self._fila(
            "SELECT id, estado FROM compras WHERE lista_id = ? AND fecha_objetivo = ?", (lista_id, fecha_objetivo)
        )
        datos = (total, ahorro, ", ".join(tiendas), json.dumps(detalle, ensure_ascii=False), ahora)
        if existente:
            if existente["estado"] != "propuesta":
                return existente["id"]
            self._ejecutar(
                "UPDATE compras SET total=?, ahorro=?, tiendas=?, detalle=?, actualizado_en=? WHERE id=?",
                (*datos, existente["id"]),
            )
            return existente["id"]
        cur = self._ejecutar(
            """INSERT INTO compras (total, ahorro, tiendas, detalle, actualizado_en, lista_id, fecha_objetivo, creado_en)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (*datos, lista_id, fecha_objetivo, ahora),
        )
        return cur.lastrowid

    def actualizar_compra(self, compra_id, **campos):
        campos["actualizado_en"] = iso(ahora_utc())
        sets = ", ".join(f"{k} = ?" for k in campos)
        self._ejecutar(f"UPDATE compras SET {sets} WHERE id = ?", (*campos.values(), compra_id))

    # --- alertas de oferta ---
    def registrar_alerta(self, fecha, consulta, tienda, nombre, precio_unidad, mediana, url=""):
        cur = self._ejecutar(
            """INSERT OR IGNORE INTO alertas (fecha, consulta, tienda, nombre, precio_unidad, mediana, url)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (fecha, consulta, tienda, nombre, precio_unidad, mediana, url),
        )
        return cur.rowcount > 0

    def alertas(self, solo_pendientes=False, limite=100):
        sql = "SELECT * FROM alertas" + (" WHERE notificada = 0" if solo_pendientes else "")
        return self._filas(sql + " ORDER BY fecha DESC, id DESC LIMIT ?", (limite,))

    def marcar_alertas_notificadas(self, ids):
        self.con.executemany("UPDATE alertas SET notificada = 1 WHERE id = ?", [(i,) for i in ids])
        self.con.commit()

    # --- caché de decisiones de la IA ---
    def ia_decision(self, consulta, nombre):
        f = self._fila("SELECT ok FROM ia_coincidencias WHERE consulta = ? AND nombre = ?", (consulta, nombre))
        return None if f is None else bool(f["ok"])

    def guardar_ia_decisiones(self, consulta, decisiones):
        self.con.executemany(
            "INSERT OR REPLACE INTO ia_coincidencias (consulta, nombre, ok) VALUES (?, ?, ?)",
            [(consulta, n, int(ok)) for n, ok in decisiones.items()],
        )
        self.con.commit()
