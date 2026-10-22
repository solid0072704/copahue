"""Persistencia en SQLite: depósitos, propuestas de inversión, cartera simulada, historial y bitácora."""

import os
import sqlite3
from datetime import datetime, timezone

# Ciclo de vida de una propuesta. Solo 'aprobada' puede pasar a ejecutarse.
ESTADOS = {
    "pendiente": "🕒 Esperando tu decisión",
    "aprobada": "👍 Aprobada, por ejecutar",
    "ejecutando": "⚙️ Ejecutando",
    "enviada": "📤 Enviada al broker",
    "ejecutada": "✅ Ejecutada",
    "rechazada": "❌ Rechazada",
    "vencida": "⌛ Vencida",
    "bloqueada": "🛡️ Bloqueada por riesgo",
    "fallida": "⚠️ Falló",
}
ESTADOS_ABIERTOS = ("pendiente", "aprobada", "ejecutando", "enviada")

ESQUEMA = """
CREATE TABLE IF NOT EXISTS depositos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT NOT NULL,
    monto REAL NOT NULL,
    nota TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS propuestas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT NOT NULL,
    tipo TEXT NOT NULL,
    simbolo TEXT NOT NULL,
    lado TEXT NOT NULL CHECK (lado IN ('compra', 'venta')),
    cantidad REAL NOT NULL,
    precio_ref REAL NOT NULL,
    monto REAL NOT NULL,
    motivo TEXT DEFAULT '',
    riesgo TEXT DEFAULT '',
    opinion_ia TEXT DEFAULT '',
    explicacion_ia TEXT DEFAULT '',
    estado TEXT NOT NULL DEFAULT 'pendiente',
    creada_en TEXT NOT NULL,
    vence_en TEXT NOT NULL,
    decidida_en TEXT,
    decidida_por TEXT DEFAULT '',
    ejecutada_en TEXT,
    precio_ejecucion REAL,
    cantidad_ejecutada REAL,
    id_broker TEXT DEFAULT '',
    error TEXT DEFAULT '',
    telegram_msg_id INTEGER
);
CREATE TABLE IF NOT EXISTS sim_posiciones (
    simbolo TEXT PRIMARY KEY,
    cantidad REAL NOT NULL,
    costo_promedio REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS valorizaciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT NOT NULL,
    valor_total REAL NOT NULL,
    efectivo REAL NOT NULL,
    aportes REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS bitacora (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT NOT NULL,
    agente TEXT NOT NULL,
    accion TEXT NOT NULL,
    detalle TEXT DEFAULT '',
    propuesta_id INTEGER
);
CREATE TABLE IF NOT EXISTS meta (
    clave TEXT PRIMARY KEY,
    valor TEXT
);
CREATE INDEX IF NOT EXISTS idx_prop_estado ON propuestas(estado);
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
            (clave, str(valor)),
        )
        self.con.commit()

    # ---------------- bitácora ----------------
    def log(self, agente, accion, detalle="", propuesta_id=None):
        self.con.execute(
            "INSERT INTO bitacora(fecha, agente, accion, detalle, propuesta_id) VALUES(?,?,?,?,?)",
            (iso(ahora_utc()), agente, accion, detalle, propuesta_id),
        )
        self.con.commit()

    def bitacora(self, limite=500):
        return self._filas("SELECT * FROM bitacora ORDER BY id DESC LIMIT ?", (limite,))

    # ---------------- depósitos ----------------
    def registrar_deposito(self, monto, nota=""):
        self.con.execute(
            "INSERT INTO depositos(fecha, monto, nota) VALUES(?,?,?)", (iso(ahora_utc()), float(monto), nota)
        )
        self.con.commit()

    def depositos(self):
        return self._filas("SELECT * FROM depositos ORDER BY id DESC")

    def total_aportes(self):
        return self._fila("SELECT COALESCE(SUM(monto), 0) AS t FROM depositos")["t"]

    # ---------------- propuestas ----------------
    def crear_propuesta(self, **campos):
        columnas = ", ".join(campos)
        marcas = ", ".join("?" for _ in campos)
        cur = self.con.execute(f"INSERT INTO propuestas({columnas}) VALUES({marcas})", tuple(campos.values()))
        self.con.commit()
        return cur.lastrowid

    def propuesta(self, prop_id):
        return self._fila("SELECT * FROM propuestas WHERE id=?", (prop_id,))

    def propuestas(self, estados=None, limite=500):
        if estados:
            marcas = ",".join("?" for _ in estados)
            return self._filas(
                f"SELECT * FROM propuestas WHERE estado IN ({marcas}) ORDER BY id", tuple(estados)
            )
        return self._filas("SELECT * FROM propuestas ORDER BY id DESC LIMIT ?", (limite,))

    def actualizar_propuesta(self, prop_id, **campos):
        asignaciones = ", ".join(f"{c}=?" for c in campos)
        self.con.execute(f"UPDATE propuestas SET {asignaciones} WHERE id=?", (*campos.values(), prop_id))
        self.con.commit()

    def cambiar_estado(self, prop_id, desde, hacia, **campos):
        """Transición atómica: solo cambia si la propuesta sigue en uno de los estados 'desde'.

        Devuelve True si cambió. Así una orden nunca se aprueba ni se ejecuta dos veces,
        aunque el tablero y Telegram respondan al mismo tiempo.
        """
        desde = (desde,) if isinstance(desde, str) else tuple(desde)
        asignaciones = ", ".join(["estado=?"] + [f"{c}=?" for c in campos])
        marcas = ",".join("?" for _ in desde)
        cur = self.con.execute(
            f"UPDATE propuestas SET {asignaciones} WHERE id=? AND estado IN ({marcas})",
            (hacia, *campos.values(), prop_id, *desde),
        )
        self.con.commit()
        return cur.rowcount == 1

    def simbolos_con_stop_desde(self, desde_iso_utc, estados=None):
        sql = "SELECT DISTINCT simbolo FROM propuestas WHERE tipo='stop' AND creada_en >= ?"
        params = [desde_iso_utc]
        if estados:
            sql += f" AND estado IN ({','.join('?' for _ in estados)})"
            params += list(estados)
        return {f["simbolo"] for f in self._filas(sql, params)}

    def ejecutadas_desde(self, desde_iso_utc):
        return self._fila(
            "SELECT COUNT(*) AS n FROM propuestas WHERE estado IN ('ejecutada','enviada','ejecutando') AND ejecutada_en >= ?",
            (desde_iso_utc,),
        )["n"]

    # ---------------- cartera simulada ----------------
    def sim_posiciones(self):
        return {f["simbolo"]: f for f in self._filas("SELECT * FROM sim_posiciones WHERE cantidad > 0")}

    def sim_guardar_posicion(self, simbolo, cantidad, costo_promedio):
        if cantidad <= 1e-9:
            self.con.execute("DELETE FROM sim_posiciones WHERE simbolo=?", (simbolo,))
        else:
            self.con.execute(
                "INSERT INTO sim_posiciones(simbolo, cantidad, costo_promedio) VALUES(?,?,?) "
                "ON CONFLICT(simbolo) DO UPDATE SET cantidad=excluded.cantidad, costo_promedio=excluded.costo_promedio",
                (simbolo, cantidad, costo_promedio),
            )
        self.con.commit()

    # ---------------- valorizaciones ----------------
    def registrar_valorizacion(self, valor_total, efectivo, aportes):
        self.con.execute(
            "INSERT INTO valorizaciones(fecha, valor_total, efectivo, aportes) VALUES(?,?,?,?)",
            (iso(ahora_utc()), valor_total, efectivo, aportes),
        )
        self.con.commit()

    def valorizaciones(self):
        return self._filas("SELECT * FROM valorizaciones ORDER BY id")

    def mejor_ganancia(self):
        f = self._fila("SELECT MAX(valor_total - aportes) AS g FROM valorizaciones")
        return f["g"] if f and f["g"] is not None else 0.0
