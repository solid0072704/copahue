"""Memoria de los agentes del embudo (SQLite): fotos del diagnóstico, tareas creadas y bitácora."""

import json
import os
import sqlite3
from datetime import timedelta

from agentes_ebook.db import ahora_utc, iso

ESQUEMA = """
CREATE TABLE IF NOT EXISTS fotos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT NOT NULL,
    pipeline TEXT DEFAULT '',
    datos TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tareas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    negocio_id TEXT NOT NULL,
    etapa_id TEXT DEFAULT '',
    accion TEXT DEFAULT '',
    tarea_id TEXT DEFAULT '',
    propietario_id TEXT DEFAULT '',
    creada_en TEXT NOT NULL,
    escalada_en TEXT
);
CREATE TABLE IF NOT EXISTS bitacora (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT NOT NULL,
    agente TEXT NOT NULL,
    accion TEXT NOT NULL,
    detalle TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS meta (
    clave TEXT PRIMARY KEY,
    valor TEXT
);
CREATE INDEX IF NOT EXISTS idx_tareas_negocio ON tareas(negocio_id);
CREATE INDEX IF NOT EXISTS idx_fotos_fecha ON fotos(pipeline, fecha);
"""


class MemoriaEmbudo:
    def __init__(self, ruta):
        self.ruta = ruta
        if ruta != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
        self.con = sqlite3.connect(ruta, timeout=30, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        if ruta != ":memory:":
            self.con.execute("PRAGMA journal_mode=WAL")
        self.con.executescript(ESQUEMA)
        self.con.commit()

    def _filas(self, sql, params=()):
        return [dict(f) for f in self.con.execute(sql, params).fetchall()]

    def _fila(self, sql, params=()):
        f = self.con.execute(sql, params).fetchone()
        return dict(f) if f else None

    def cerrar(self):
        self.con.close()

    # ---------------- fotos del diagnóstico ----------------
    def guardar_foto(self, diagnostico, pipeline=""):
        self.con.execute(
            "INSERT INTO fotos(fecha, pipeline, datos) VALUES(?,?,?)",
            (diagnostico["fecha"], pipeline, json.dumps(diagnostico, ensure_ascii=False, default=str)),
        )
        self.con.commit()

    def foto_anterior(self, pipeline="", hace_dias=7, ahora=None):
        """La foto más reciente de hace al menos `hace_dias`; si no la hay, la más antigua guardada."""
        ahora = ahora or ahora_utc()
        limite = iso(ahora - timedelta(days=hace_dias))
        fila = self._fila(
            "SELECT datos FROM fotos WHERE pipeline=? AND fecha<=? ORDER BY fecha DESC LIMIT 1", (pipeline, limite)
        )
        if not fila:
            fila = self._fila("SELECT datos FROM fotos WHERE pipeline=? ORDER BY fecha ASC LIMIT 1", (pipeline,))
        return json.loads(fila["datos"]) if fila else None

    def fotos(self, pipeline="", limite=60):
        return self._filas(
            "SELECT id, fecha, pipeline FROM fotos WHERE pipeline=? ORDER BY fecha DESC LIMIT ?", (pipeline, limite)
        )

    # ---------------- tareas creadas en HubSpot ----------------
    def ultima_tarea(self, negocio_id):
        return self._fila("SELECT * FROM tareas WHERE negocio_id=? ORDER BY id DESC LIMIT 1", (str(negocio_id),))

    def registrar_tarea(self, negocio_id, etapa_id, accion, tarea_id, propietario_id, ahora=None):
        cur = self.con.execute(
            "INSERT INTO tareas(negocio_id, etapa_id, accion, tarea_id, propietario_id, creada_en) VALUES(?,?,?,?,?,?)",
            (str(negocio_id), etapa_id or "", accion, str(tarea_id or ""), str(propietario_id or ""), iso(ahora or ahora_utc())),
        )
        self.con.commit()
        return cur.lastrowid

    def marcar_escalada(self, id_registro, ahora=None):
        self.con.execute("UPDATE tareas SET escalada_en=? WHERE id=?", (iso(ahora or ahora_utc()), id_registro))
        self.con.commit()

    def tareas_desde(self, desde):
        return self._filas("SELECT * FROM tareas WHERE creada_en>=? ORDER BY id DESC", (iso(desde),))

    # ---------------- bitácora y meta ----------------
    def log(self, agente, accion, detalle=""):
        self.con.execute(
            "INSERT INTO bitacora(fecha, agente, accion, detalle) VALUES(?,?,?,?)", (iso(ahora_utc()), agente, accion, detalle)
        )
        self.con.commit()

    def bitacora(self, limite=200):
        return self._filas("SELECT * FROM bitacora ORDER BY id DESC LIMIT ?", (limite,))

    def get_meta(self, clave, defecto=None):
        f = self._fila("SELECT valor FROM meta WHERE clave=?", (clave,))
        return f["valor"] if f else defecto

    def set_meta(self, clave, valor):
        self.con.execute(
            "INSERT INTO meta(clave, valor) VALUES(?, ?) ON CONFLICT(clave) DO UPDATE SET valor=excluded.valor",
            (clave, valor),
        )
        self.con.commit()
