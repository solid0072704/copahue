"""Persistencia en SQLite: clientes, vencimientos, honorarios, pagos y bitácora."""

import json
import os
import sqlite3
from datetime import date, datetime, timezone

from . import rut as rutlib

ESTADOS_CLIENTE = ["Activo", "Prospecto", "Suspendido", "De baja"]
TIPOS_CLIENTE = ["Empresa", "Persona natural", "Sociedad de profesionales", "Organización sin fines de lucro"]
REGIMENES = [
    "Pro Pyme General (14 D N°3)",
    "Pro Pyme Transparente (14 D N°8)",
    "General semi integrado (14 A)",
    "Renta presunta",
    "Segunda categoría (honorarios)",
    "Sin fines de lucro / exento",
    "Otro",
]
MONEDAS = ["CLP", "UF"]

ESTADOS_OBLIGACION = ["Pendiente", "Esperando cliente", "En proceso", "Presentada", "No aplica"]
ESTADOS_CERRADOS = {"Presentada", "No aplica"}

ESTADOS_COBRO = ["Pendiente", "Pagado", "Anulado"]
MEDIOS_PAGO = ["Transferencia", "Depósito", "Cheque", "Efectivo", "Tarjeta", "Otro"]
TIPOS_INTERACCION = ["Nota", "Llamada", "Correo", "WhatsApp", "Reunión", "Cobranza"]

CAMPOS_CLIENTE = {
    "rut", "razon_social", "nombre_fantasia", "tipo", "giro", "regimen", "direccion", "comuna",
    "representante", "contacto", "email", "telefono", "responsable", "estado", "fecha_inicio",
    "servicios", "honorario", "moneda", "trabajadores", "notas", "fuente",
}
CAMPOS_OBLIGACION = {"nombre", "vence", "estado", "responsable", "notas"}
CAMPOS_COBRO = {"concepto", "monto", "emitido", "vence", "documento", "notas", "anulado"}

ESQUEMA = """
CREATE TABLE IF NOT EXISTS clientes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rut TEXT UNIQUE,
    razon_social TEXT NOT NULL,
    nombre_fantasia TEXT DEFAULT '',
    tipo TEXT DEFAULT 'Empresa',
    giro TEXT DEFAULT '',
    regimen TEXT DEFAULT '',
    direccion TEXT DEFAULT '',
    comuna TEXT DEFAULT '',
    representante TEXT DEFAULT '',
    contacto TEXT DEFAULT '',
    email TEXT DEFAULT '',
    telefono TEXT DEFAULT '',
    responsable TEXT DEFAULT '',
    estado TEXT DEFAULT 'Activo',
    fecha_inicio TEXT,
    servicios TEXT DEFAULT '[]',
    honorario REAL DEFAULT 0,
    moneda TEXT DEFAULT 'CLP',
    trabajadores INTEGER DEFAULT 0,
    notas TEXT DEFAULT '',
    fuente TEXT DEFAULT '',
    creado_en TEXT,
    actualizado_en TEXT
);
CREATE TABLE IF NOT EXISTS obligaciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cliente_id INTEGER NOT NULL REFERENCES clientes(id),
    tipo TEXT NOT NULL,
    nombre TEXT NOT NULL,
    periodo TEXT DEFAULT '',
    vence TEXT NOT NULL,
    estado TEXT DEFAULT 'Pendiente',
    responsable TEXT DEFAULT '',
    notas TEXT DEFAULT '',
    cerrada_en TEXT,
    creado_en TEXT,
    actualizado_en TEXT
);
-- Una sola obligación por cliente, tipo y período (las tareas libres 'OTRO' no tienen límite).
CREATE UNIQUE INDEX IF NOT EXISTS idx_obl_unica ON obligaciones(cliente_id, tipo, periodo) WHERE tipo <> 'OTRO';
CREATE INDEX IF NOT EXISTS idx_obl_vence ON obligaciones(vence);
CREATE TABLE IF NOT EXISTS cobros (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cliente_id INTEGER NOT NULL REFERENCES clientes(id),
    periodo TEXT DEFAULT '',
    concepto TEXT NOT NULL,
    monto INTEGER NOT NULL,
    emitido TEXT NOT NULL,
    vence TEXT NOT NULL,
    documento TEXT DEFAULT '',
    notas TEXT DEFAULT '',
    anulado INTEGER DEFAULT 0,
    creado_en TEXT,
    actualizado_en TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_cobro_unico ON cobros(cliente_id, periodo, concepto) WHERE periodo <> '';
CREATE TABLE IF NOT EXISTS pagos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cobro_id INTEGER NOT NULL REFERENCES cobros(id),
    fecha TEXT NOT NULL,
    monto INTEGER NOT NULL,
    medio TEXT DEFAULT '',
    nota TEXT DEFAULT '',
    creado_en TEXT
);
CREATE TABLE IF NOT EXISTS interacciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cliente_id INTEGER NOT NULL REFERENCES clientes(id),
    fecha TEXT NOT NULL,
    tipo TEXT DEFAULT 'Nota',
    resumen TEXT NOT NULL,
    autor TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS meta (
    clave TEXT PRIMARY KEY,
    valor TEXT
);
"""

# Saldo de cada cobro = monto − pagos. El estado se deduce, nunca se guarda a mano.
_SQL_COBROS = """
SELECT c.*, cl.razon_social, cl.rut, cl.email, cl.telefono, cl.contacto, cl.responsable, cl.fuente,
       COALESCE(p.pagado, 0) AS pagado,
       c.monto - COALESCE(p.pagado, 0) AS saldo,
       p.ultimo_pago
FROM cobros c
JOIN clientes cl ON cl.id = c.cliente_id
LEFT JOIN (SELECT cobro_id, SUM(monto) AS pagado, MAX(fecha) AS ultimo_pago FROM pagos GROUP BY cobro_id) p
       ON p.cobro_id = c.id
"""


class ErrorValidacion(ValueError):
    """Dato ingresado por el usuario que no se puede guardar (el mensaje se muestra tal cual)."""


def ahora_utc():
    return datetime.now(timezone.utc).replace(microsecond=0)


def _ahora():
    return ahora_utc().isoformat()


def fecha_iso(valor):
    """Acepta date, datetime o texto 'AAAA-MM-DD' y devuelve 'AAAA-MM-DD' (o None)."""
    if valor is None or valor == "":
        return None
    if isinstance(valor, datetime):
        return valor.date().isoformat()
    if isinstance(valor, date):
        return valor.isoformat()
    try:
        return date.fromisoformat(str(valor)[:10]).isoformat()
    except ValueError:
        raise ErrorValidacion(f"La fecha «{valor}» no es válida (usa AAAA-MM-DD).") from None


def estado_cobro(cobro):
    if cobro["anulado"]:
        return "Anulado"
    return "Pagado" if cobro["saldo"] <= 0 else "Pendiente"


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

    def get_json(self, clave, defecto=None):
        texto = self.get_meta(clave)
        try:
            return json.loads(texto) if texto else defecto
        except ValueError:
            return defecto

    def set_json(self, clave, valor):
        self.set_meta(clave, json.dumps(valor, ensure_ascii=False))

    # ---------------- clientes ----------------
    @staticmethod
    def _cliente(fila):
        if fila:
            try:
                fila["servicios"] = json.loads(fila["servicios"] or "[]")
            except ValueError:
                fila["servicios"] = []
        return fila

    def _validar_cliente(self, campos, cliente_id=None):
        """Limpia y valida los campos de un cliente antes de guardarlos."""
        desconocidos = set(campos) - CAMPOS_CLIENTE
        if desconocidos:
            raise ErrorValidacion(f"Campos no permitidos: {', '.join(sorted(desconocidos))}")
        actual = self.cliente(cliente_id) if cliente_id else {}
        datos = {c: (v.strip() if isinstance(v, str) else v) for c, v in campos.items()}

        if "razon_social" in datos or not cliente_id:
            if not datos.get("razon_social"):
                raise ErrorValidacion("La razón social (o nombre) es obligatoria.")
        estado = datos.get("estado", actual.get("estado", "Activo"))
        if estado not in ESTADOS_CLIENTE:
            raise ErrorValidacion(f"Estado desconocido: {estado}")

        if "rut" in datos or not cliente_id:
            if datos.get("rut"):
                try:
                    datos["rut"] = rutlib.normalizar(datos["rut"])
                except rutlib.RutInvalido as e:
                    raise ErrorValidacion(str(e)) from None
                otro = self._fila("SELECT id, razon_social FROM clientes WHERE rut=?", (datos["rut"],))
                if otro and otro["id"] != cliente_id:
                    raise ErrorValidacion(
                        f"Ya existe un cliente con el RUT {rutlib.formatear(datos['rut'])}: {otro['razon_social']}."
                    )
            else:
                datos["rut"] = None
        rut_final = datos["rut"] if "rut" in datos else actual.get("rut")
        if not rut_final and estado != "Prospecto":
            raise ErrorValidacion("El RUT es obligatorio (solo los prospectos pueden quedar sin RUT).")

        if "moneda" in datos and datos["moneda"] not in MONEDAS:
            raise ErrorValidacion(f"Moneda desconocida: {datos['moneda']}")
        if "honorario" in datos:
            datos["honorario"] = float(datos["honorario"] or 0)
            if datos["honorario"] < 0:
                raise ErrorValidacion("El honorario no puede ser negativo.")
        if "trabajadores" in datos:
            datos["trabajadores"] = max(0, int(datos["trabajadores"] or 0))
        if "fecha_inicio" in datos:
            datos["fecha_inicio"] = fecha_iso(datos["fecha_inicio"])
        if "servicios" in datos:
            datos["servicios"] = json.dumps(list(datos["servicios"] or []), ensure_ascii=False)
        if "email" in datos and datos["email"]:
            datos["email"] = datos["email"].lower()
            if "@" not in datos["email"] or " " in datos["email"]:
                raise ErrorValidacion(f"El correo «{datos['email']}» no parece válido.")
        return datos

    def crear_cliente(self, **campos):
        datos = self._validar_cliente(campos)
        ahora = _ahora()
        datos.update(creado_en=ahora, actualizado_en=ahora)
        columnas = ", ".join(datos)
        cur = self.con.execute(
            f"INSERT INTO clientes({columnas}) VALUES({','.join('?' * len(datos))})", tuple(datos.values())
        )
        self.con.commit()
        return cur.lastrowid

    def actualizar_cliente(self, cliente_id, **campos):
        if not campos:
            return
        datos = self._validar_cliente(campos, cliente_id)
        datos["actualizado_en"] = _ahora()
        sets = ", ".join(f"{c}=?" for c in datos)
        self.con.execute(f"UPDATE clientes SET {sets} WHERE id=?", (*datos.values(), cliente_id))
        self.con.commit()

    def cliente(self, cliente_id):
        return self._cliente(self._fila("SELECT * FROM clientes WHERE id=?", (cliente_id,)))

    def cliente_por_rut(self, rut):
        try:
            return self._cliente(self._fila("SELECT * FROM clientes WHERE rut=?", (rutlib.normalizar(rut),)))
        except rutlib.RutInvalido:
            return None

    def clientes(self, estados=None):
        sql, params = "SELECT * FROM clientes", []
        if estados:
            sql += f" WHERE estado IN ({','.join('?' * len(estados))})"
            params = list(estados)
        return [self._cliente(f) for f in self._filas(sql + " ORDER BY razon_social COLLATE NOCASE", params)]

    def eliminar_cliente(self, cliente_id):
        """Borra el cliente y todo su historial. Para dejar de atenderlo usa el estado «De baja»."""
        self._eliminar_clientes([cliente_id])

    def borrar_fuente(self, fuente):
        ids = [f["id"] for f in self._filas("SELECT id FROM clientes WHERE fuente=?", (fuente,))]
        self._eliminar_clientes(ids)
        return len(ids)

    def _eliminar_clientes(self, ids):
        pares = [(i,) for i in ids]
        self.con.executemany("DELETE FROM pagos WHERE cobro_id IN (SELECT id FROM cobros WHERE cliente_id=?)", pares)
        for tabla in ("cobros", "obligaciones", "interacciones"):
            self.con.executemany(f"DELETE FROM {tabla} WHERE cliente_id=?", pares)
        self.con.executemany("DELETE FROM clientes WHERE id=?", pares)
        self.con.commit()

    # ---------------- obligaciones (vencimientos) ----------------
    def crear_obligacion(self, cliente_id, tipo, nombre, vence, periodo="", responsable="", estado="Pendiente", notas=""):
        """Crea la obligación. Devuelve su id, o None si ya existía para ese cliente, tipo y período."""
        if not (nombre or "").strip():
            raise ErrorValidacion("La tarea necesita un nombre.")
        if estado not in ESTADOS_OBLIGACION:
            raise ErrorValidacion(f"Estado desconocido: {estado}")
        ahora = _ahora()
        cur = self.con.execute(
            """INSERT OR IGNORE INTO obligaciones(cliente_id, tipo, nombre, periodo, vence, estado, responsable,
                   notas, cerrada_en, creado_en, actualizado_en)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                cliente_id, tipo, nombre.strip(), periodo, fecha_iso(vence), estado, responsable, notas,
                ahora if estado in ESTADOS_CERRADOS else None, ahora, ahora,
            ),
        )
        self.con.commit()
        return cur.lastrowid if cur.rowcount else None

    def actualizar_obligacion(self, obligacion_id, **campos):
        desconocidos = set(campos) - CAMPOS_OBLIGACION
        if desconocidos:
            raise ErrorValidacion(f"Campos no permitidos: {', '.join(sorted(desconocidos))}")
        if not campos:
            return
        if "vence" in campos:
            campos["vence"] = fecha_iso(campos["vence"])
            if not campos["vence"]:
                raise ErrorValidacion("El vencimiento no puede quedar vacío.")
        if "estado" in campos:
            if campos["estado"] not in ESTADOS_OBLIGACION:
                raise ErrorValidacion(f"Estado desconocido: {campos['estado']}")
            campos["cerrada_en"] = _ahora() if campos["estado"] in ESTADOS_CERRADOS else None
        for c in ("responsable", "notas", "nombre"):
            if c in campos:
                campos[c] = campos[c] or ""
        campos["actualizado_en"] = _ahora()
        sets = ", ".join(f"{c}=?" for c in campos)
        self.con.execute(f"UPDATE obligaciones SET {sets} WHERE id=?", (*campos.values(), obligacion_id))
        self.con.commit()

    def eliminar_obligacion(self, obligacion_id):
        self.con.execute("DELETE FROM obligaciones WHERE id=?", (obligacion_id,))
        self.con.commit()

    def obligacion(self, obligacion_id):
        return self._fila("SELECT * FROM obligaciones WHERE id=?", (obligacion_id,))

    def obligaciones(self, cliente_id=None, abiertas=None, desde=None, hasta=None, periodo=None):
        sql = (
            "SELECT o.*, c.razon_social, c.rut, c.email, c.telefono, c.contacto, c.fuente "
            "FROM obligaciones o JOIN clientes c ON c.id=o.cliente_id WHERE 1=1"
        )
        params = []
        if cliente_id is not None:
            sql += " AND o.cliente_id=?"
            params.append(cliente_id)
        if abiertas is not None:
            marcas = ",".join("?" * len(ESTADOS_CERRADOS))
            sql += f" AND o.estado {'NOT IN' if abiertas else 'IN'} ({marcas})"
            params.extend(sorted(ESTADOS_CERRADOS))
        if desde:
            sql += " AND o.vence>=?"
            params.append(fecha_iso(desde))
        if hasta:
            sql += " AND o.vence<=?"
            params.append(fecha_iso(hasta))
        if periodo:
            sql += " AND o.periodo=?"
            params.append(periodo)
        return self._filas(sql + " ORDER BY o.vence, c.razon_social COLLATE NOCASE, o.nombre", params)

    # ---------------- cobros y pagos ----------------
    def crear_cobro(self, cliente_id, concepto, monto, emitido, vence, periodo="", documento="", notas=""):
        """Crea el cobro. Devuelve su id, o None si ya existía ese concepto para el cliente y período."""
        monto = int(round(float(monto or 0)))
        if monto <= 0:
            raise ErrorValidacion("El monto del cobro debe ser mayor que cero.")
        if not (concepto or "").strip():
            raise ErrorValidacion("El cobro necesita un concepto.")
        ahora = _ahora()
        cur = self.con.execute(
            """INSERT OR IGNORE INTO cobros(cliente_id, periodo, concepto, monto, emitido, vence, documento, notas,
                   creado_en, actualizado_en)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (cliente_id, periodo, concepto.strip(), monto, fecha_iso(emitido), fecha_iso(vence), documento, notas, ahora, ahora),
        )
        self.con.commit()
        return cur.lastrowid if cur.rowcount else None

    def actualizar_cobro(self, cobro_id, **campos):
        desconocidos = set(campos) - CAMPOS_COBRO
        if desconocidos:
            raise ErrorValidacion(f"Campos no permitidos: {', '.join(sorted(desconocidos))}")
        if not campos:
            return
        actual = self.cobro(cobro_id)
        if "monto" in campos:
            campos["monto"] = int(round(float(campos["monto"] or 0)))
            if campos["monto"] <= 0:
                raise ErrorValidacion("El monto del cobro debe ser mayor que cero.")
            if campos["monto"] < actual["pagado"]:
                raise ErrorValidacion("El monto no puede ser menor que lo que ya se pagó.")
        if campos.get("anulado") and actual["pagado"] > 0:
            raise ErrorValidacion("No se puede anular un cobro que ya tiene pagos registrados.")
        for c in ("emitido", "vence"):
            if c in campos:
                campos[c] = fecha_iso(campos[c])
        if "anulado" in campos:
            campos["anulado"] = int(bool(campos["anulado"]))
        campos["actualizado_en"] = _ahora()
        sets = ", ".join(f"{c}=?" for c in campos)
        self.con.execute(f"UPDATE cobros SET {sets} WHERE id=?", (*campos.values(), cobro_id))
        self.con.commit()

    def cobro(self, cobro_id):
        return self._fila(_SQL_COBROS + " WHERE c.id=?", (cobro_id,))

    def cobros(self, cliente_id=None, pendientes=None, incluir_anulados=False):
        sql, params = _SQL_COBROS + " WHERE 1=1", []
        if cliente_id is not None:
            sql += " AND c.cliente_id=?"
            params.append(cliente_id)
        if not incluir_anulados or pendientes:
            sql += " AND c.anulado=0"
        if pendientes is True:
            sql += " AND c.monto - COALESCE(p.pagado, 0) > 0"
        elif pendientes is False:
            sql += " AND c.monto - COALESCE(p.pagado, 0) <= 0"
        return self._filas(sql + " ORDER BY c.vence, cl.razon_social COLLATE NOCASE", params)

    def registrar_pago(self, cobro_id, monto, fecha, medio="Transferencia", nota=""):
        cobro = self.cobro(cobro_id)
        if not cobro:
            raise ErrorValidacion("El cobro no existe.")
        if cobro["anulado"]:
            raise ErrorValidacion("El cobro está anulado.")
        monto = int(round(float(monto or 0)))
        if monto <= 0:
            raise ErrorValidacion("El pago debe ser mayor que cero.")
        if monto > cobro["saldo"]:
            raise ErrorValidacion(f"El pago supera el saldo pendiente del cobro ({cobro['saldo']}).")
        cur = self.con.execute(
            "INSERT INTO pagos(cobro_id, fecha, monto, medio, nota, creado_en) VALUES(?,?,?,?,?,?)",
            (cobro_id, fecha_iso(fecha), monto, medio, nota, _ahora()),
        )
        self.con.commit()
        return cur.lastrowid

    def eliminar_pago(self, pago_id):
        self.con.execute("DELETE FROM pagos WHERE id=?", (pago_id,))
        self.con.commit()

    def pagos(self, cliente_id=None, desde=None, hasta=None):
        sql = (
            "SELECT p.*, c.concepto, c.periodo, c.cliente_id, cl.razon_social, cl.rut "
            "FROM pagos p JOIN cobros c ON c.id=p.cobro_id JOIN clientes cl ON cl.id=c.cliente_id WHERE 1=1"
        )
        params = []
        if cliente_id is not None:
            sql += " AND c.cliente_id=?"
            params.append(cliente_id)
        if desde:
            sql += " AND p.fecha>=?"
            params.append(fecha_iso(desde))
        if hasta:
            sql += " AND p.fecha<=?"
            params.append(fecha_iso(hasta))
        return self._filas(sql + " ORDER BY p.fecha DESC, p.id DESC", params)

    # ---------------- bitácora de clientes ----------------
    def registrar_interaccion(self, cliente_id, resumen, tipo="Nota", autor="", fecha=None):
        if not (resumen or "").strip():
            raise ErrorValidacion("Escribe un resumen de lo conversado.")
        cur = self.con.execute(
            "INSERT INTO interacciones(cliente_id, fecha, tipo, resumen, autor) VALUES(?,?,?,?,?)",
            (cliente_id, fecha or _ahora(), tipo, resumen.strip(), autor),
        )
        self.con.commit()
        return cur.lastrowid

    def interacciones(self, cliente_id=None, limite=500):
        sql = (
            "SELECT i.*, c.razon_social FROM interacciones i JOIN clientes c ON c.id=i.cliente_id"
            + (" WHERE i.cliente_id=?" if cliente_id is not None else "")
            + " ORDER BY i.fecha DESC, i.id DESC LIMIT ?"
        )
        return self._filas(sql, (cliente_id, limite) if cliente_id is not None else (limite,))
