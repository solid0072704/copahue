"""Persistencia en SQLite: leads, descargas, seguimientos, reuniones y bitácora."""

import json
import os
import sqlite3
from datetime import datetime, timezone

ETAPAS = [
    "Nuevo",
    "En seguimiento",
    "Respondió",
    "Reunión propuesta",
    "Reunión agendada",
    "Cliente",
    "Descartado",
]
# Etapas en las que los agentes ya no envían más correos de seguimiento.
ETAPAS_SIN_SEGUIMIENTO = {"Respondió", "Reunión agendada", "Cliente", "Descartado"}

ESQUEMA = """
CREATE TABLE IF NOT EXISTS leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    nombre TEXT DEFAULT '',
    telefono TEXT DEFAULT '',
    cargo TEXT DEFAULT '',
    empresa TEXT DEFAULT '',
    motivo TEXT DEFAULT '',
    pais TEXT DEFAULT '',
    fuente TEXT DEFAULT '',
    primera_descarga TEXT,
    ultima_descarga TEXT,
    num_descargas INTEGER DEFAULT 0,
    score INTEGER DEFAULT 0,
    segmento TEXT DEFAULT '',
    nivel_cargo TEXT DEFAULT '',
    score_detalle TEXT DEFAULT '[]',
    lectura TEXT DEFAULT '',
    angulo TEXT DEFAULT '',
    lectura_ia INTEGER DEFAULT 0,
    etapa TEXT DEFAULT 'Nuevo',
    notificado INTEGER DEFAULT 0,
    respondio_en TEXT,
    notas TEXT DEFAULT '',
    creado_en TEXT,
    actualizado_en TEXT
);
CREATE TABLE IF NOT EXISTS descargas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id INTEGER NOT NULL REFERENCES leads(id),
    mensaje_id TEXT UNIQUE NOT NULL,
    fecha TEXT NOT NULL,
    datos TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS seguimientos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id INTEGER NOT NULL REFERENCES leads(id),
    paso INTEGER NOT NULL,
    tipo TEXT NOT NULL,
    nombre_paso TEXT DEFAULT '',
    programado_para TEXT,
    estado TEXT DEFAULT 'pendiente',
    asunto TEXT DEFAULT '',
    cuerpo TEXT DEFAULT '',
    personalizado_ia INTEGER DEFAULT 0,
    creado_en TEXT,
    actualizado_en TEXT,
    UNIQUE(lead_id, paso)
);
CREATE TABLE IF NOT EXISTS reuniones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id INTEGER UNIQUE NOT NULL REFERENCES leads(id),
    estado TEXT DEFAULT 'propuesta',
    propuestas TEXT DEFAULT '[]',
    fecha_agendada TEXT,
    creado_en TEXT,
    actualizado_en TEXT
);
CREATE TABLE IF NOT EXISTS bitacora (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT NOT NULL,
    agente TEXT NOT NULL,
    accion TEXT NOT NULL,
    detalle TEXT DEFAULT '',
    lead_id INTEGER
);
CREATE TABLE IF NOT EXISTS meta (
    clave TEXT PRIMARY KEY,
    valor TEXT
);
CREATE INDEX IF NOT EXISTS idx_seg_estado ON seguimientos(estado);
CREATE INDEX IF NOT EXISTS idx_bitacora_fecha ON bitacora(fecha);
"""


def ahora_utc():
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def desde_iso(texto):
    if not texto:
        return None
    dt = datetime.fromisoformat(texto)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class BaseDatos:
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

    # ---------------- utilidades ----------------
    def _filas(self, sql, params=()):
        return [dict(f) for f in self.con.execute(sql, params).fetchall()]

    def _fila(self, sql, params=()):
        f = self.con.execute(sql, params).fetchone()
        return dict(f) if f else None

    def cerrar(self):
        self.con.close()

    # ---------------- meta ----------------
    def get_meta(self, clave, defecto=None):
        f = self._fila("SELECT valor FROM meta WHERE clave=?", (clave,))
        return f["valor"] if f else defecto

    def set_meta(self, clave, valor):
        self.con.execute(
            "INSERT INTO meta(clave, valor) VALUES(?, ?) "
            "ON CONFLICT(clave) DO UPDATE SET valor=excluded.valor",
            (clave, valor),
        )
        self.con.commit()

    # ---------------- bitácora ----------------
    def log(self, agente, accion, detalle="", lead_id=None):
        self.con.execute(
            "INSERT INTO bitacora(fecha, agente, accion, detalle, lead_id) VALUES(?,?,?,?,?)",
            (iso(ahora_utc()), agente, accion, detalle, lead_id),
        )
        self.con.commit()

    def bitacora(self, limite=500, agente=None):
        if agente:
            return self._filas(
                "SELECT * FROM bitacora WHERE agente=? ORDER BY id DESC LIMIT ?", (agente, limite)
            )
        return self._filas("SELECT * FROM bitacora ORDER BY id DESC LIMIT ?", (limite,))

    # ---------------- leads y descargas ----------------
    def existe_descarga(self, mensaje_id):
        return self._fila("SELECT id FROM descargas WHERE mensaje_id=?", (mensaje_id,)) is not None

    def registrar_descarga(self, registro):
        """Guarda una descarga y crea/actualiza el lead.

        Devuelve (lead_id, es_lead_nuevo) o (None, False) si ya estaba registrada.
        """
        if self.existe_descarga(registro["mensaje_id"]):
            return None, False
        email = registro["email"].strip().lower()
        fecha = registro["fecha"]
        ahora = iso(ahora_utc())
        lead = self._fila("SELECT * FROM leads WHERE email=?", (email,))
        nuevo = lead is None
        if nuevo:
            cur = self.con.execute(
                """INSERT INTO leads(email, nombre, telefono, cargo, empresa, motivo, fuente,
                       primera_descarga, ultima_descarga, num_descargas, creado_en, actualizado_en)
                   VALUES(?,?,?,?,?,?,?,?,?,1,?,?)""",
                (
                    email,
                    registro.get("nombre", ""),
                    registro.get("telefono", ""),
                    registro.get("cargo", ""),
                    registro.get("empresa", ""),
                    registro.get("motivo", ""),
                    registro.get("fuente", ""),
                    fecha,
                    fecha,
                    ahora,
                    ahora,
                ),
            )
            lead_id = cur.lastrowid
        else:
            lead_id = lead["id"]
            # Completa datos vacíos y conserva los más recientes cuando vienen informados.
            campos = {}
            for c in ("nombre", "telefono", "cargo", "empresa", "motivo"):
                valor = (registro.get(c) or "").strip()
                if valor:
                    campos[c] = valor
            primera = min(lead["primera_descarga"] or fecha, fecha)
            ultima = max(lead["ultima_descarga"] or fecha, fecha)
            sets = ", ".join(f"{c}=?" for c in campos)
            sql = (
                "UPDATE leads SET "
                + (sets + ", " if sets else "")
                + "primera_descarga=?, ultima_descarga=?, num_descargas=num_descargas+1, actualizado_en=? WHERE id=?"
            )
            self.con.execute(sql, (*campos.values(), primera, ultima, ahora, lead_id))
        self.con.execute(
            "INSERT INTO descargas(lead_id, mensaje_id, fecha, datos) VALUES(?,?,?,?)",
            (lead_id, registro["mensaje_id"], fecha, json.dumps(registro.get("extra", {}), ensure_ascii=False)),
        )
        self.con.commit()
        return lead_id, nuevo

    def leads(self):
        return self._filas("SELECT * FROM leads ORDER BY score DESC, ultima_descarga DESC")

    def lead(self, lead_id):
        return self._fila("SELECT * FROM leads WHERE id=?", (lead_id,))

    def actualizar_lead(self, lead_id, **campos):
        if not campos:
            return
        campos["actualizado_en"] = iso(ahora_utc())
        sets = ", ".join(f"{c}=?" for c in campos)
        self.con.execute(f"UPDATE leads SET {sets} WHERE id=?", (*campos.values(), lead_id))
        self.con.commit()

    def descargas(self):
        return self._filas(
            "SELECT d.*, l.email, l.nombre, l.fuente FROM descargas d JOIN leads l ON l.id=d.lead_id "
            "ORDER BY d.fecha"
        )

    def borrar_fuente(self, fuente):
        ids = [f["id"] for f in self._filas("SELECT id FROM leads WHERE fuente=?", (fuente,))]
        for tabla in ("descargas", "seguimientos", "reuniones", "bitacora"):
            self.con.executemany(f"DELETE FROM {tabla} WHERE lead_id=?", [(i,) for i in ids])
        self.con.executemany("DELETE FROM leads WHERE id=?", [(i,) for i in ids])
        self.con.commit()
        return len(ids)

    # ---------------- seguimientos ----------------
    def seguimientos(self, lead_id=None, estados=None):
        sql = (
            "SELECT s.*, l.nombre, l.email, l.empresa, l.cargo, l.telefono, l.score, l.segmento, l.fuente "
            "FROM seguimientos s JOIN leads l ON l.id=s.lead_id WHERE 1=1"
        )
        params = []
        if lead_id is not None:
            sql += " AND s.lead_id=?"
            params.append(lead_id)
        if estados:
            sql += f" AND s.estado IN ({','.join('?' * len(estados))})"
            params.extend(estados)
        sql += " ORDER BY l.score DESC, s.programado_para"
        return self._filas(sql, params)

    def seguimiento(self, seg_id):
        return self._fila(
            "SELECT s.*, l.nombre, l.email, l.empresa, l.cargo, l.telefono, l.fuente "
            "FROM seguimientos s JOIN leads l ON l.id=s.lead_id WHERE s.id=?",
            (seg_id,),
        )

    def crear_seguimiento(self, lead_id, paso, tipo, nombre_paso, programado_para, estado, asunto="", cuerpo="", ia=False):
        ahora = iso(ahora_utc())
        cur = self.con.execute(
            """INSERT OR IGNORE INTO seguimientos(lead_id, paso, tipo, nombre_paso, programado_para, estado,
                   asunto, cuerpo, personalizado_ia, creado_en, actualizado_en)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (lead_id, paso, tipo, nombre_paso, programado_para, estado, asunto, cuerpo, int(ia), ahora, ahora),
        )
        self.con.commit()
        return cur.lastrowid if cur.rowcount else None

    def actualizar_seguimiento(self, seg_id, **campos):
        campos["actualizado_en"] = iso(ahora_utc())
        sets = ", ".join(f"{c}=?" for c in campos)
        self.con.execute(f"UPDATE seguimientos SET {sets} WHERE id=?", (*campos.values(), seg_id))
        self.con.commit()

    def enviados_hoy(self, desde_iso_utc):
        f = self._fila(
            "SELECT COUNT(*) AS n FROM seguimientos WHERE estado='enviado' AND actualizado_en>=?",
            (desde_iso_utc,),
        )
        return f["n"] if f else 0

    # ---------------- reuniones ----------------
    def reuniones(self):
        filas = self._filas(
            "SELECT r.*, l.nombre, l.email, l.empresa, l.cargo, l.telefono, l.score, l.segmento, l.etapa, l.fuente "
            "FROM reuniones r JOIN leads l ON l.id=r.lead_id ORDER BY r.creado_en DESC"
        )
        for f in filas:
            f["propuestas"] = json.loads(f["propuestas"] or "[]")
        return filas

    def reunion_de(self, lead_id):
        f = self._fila("SELECT * FROM reuniones WHERE lead_id=?", (lead_id,))
        if f:
            f["propuestas"] = json.loads(f["propuestas"] or "[]")
        return f

    def crear_reunion(self, lead_id, propuestas):
        ahora = iso(ahora_utc())
        self.con.execute(
            "INSERT OR IGNORE INTO reuniones(lead_id, estado, propuestas, creado_en, actualizado_en) VALUES(?,?,?,?,?)",
            (lead_id, "propuesta", json.dumps(propuestas), ahora, ahora),
        )
        self.con.commit()

    def actualizar_reunion(self, lead_id, **campos):
        if "propuestas" in campos:
            campos["propuestas"] = json.dumps(campos["propuestas"])
        campos["actualizado_en"] = iso(ahora_utc())
        sets = ", ".join(f"{c}=?" for c in campos)
        self.con.execute(f"UPDATE reuniones SET {sets} WHERE lead_id=?", (*campos.values(), lead_id))
        self.con.commit()

    def horarios_ocupados(self):
        """Horarios con reunión ya agendada (no se vuelven a ofrecer)."""
        return {
            r["fecha_agendada"]
            for r in self.reuniones()
            if r["estado"] == "agendada" and r["fecha_agendada"]
        }
