"""Persistencia del prospector: prospectos de LinkedIn y sus acciones (invitación y mensajes).

Usa el mismo archivo SQLite que los agentes del ebook (tablas con prefijo `li_`),
así comparten bitácora y un solo archivo de memoria.
"""

from agentes_ebook.db import BaseDatos, ahora_utc, iso

ETAPAS = [
    "Por invitar",
    "Invitación enviada",
    "Conectado",
    "En conversación",
    "Respondió",
    "Reunión agendada",
    "Cliente",
    "Sin respuesta",
    "Descartado",
]
# Etapas en las que el prospector ya no propone más acciones.
ETAPAS_CERRADAS = {"Respondió", "Reunión agendada", "Cliente", "Sin respuesta", "Descartado"}

ESQUEMA_LI = """
CREATE TABLE IF NOT EXISTS li_prospectos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clave TEXT UNIQUE NOT NULL,
    nombre TEXT DEFAULT '',
    cargo TEXT DEFAULT '',
    empresa TEXT DEFAULT '',
    ubicacion TEXT DEFAULT '',
    industria TEXT DEFAULT '',
    url TEXT DEFAULT '',
    email TEXT DEFAULT '',
    fuente TEXT DEFAULT '',
    grado INTEGER DEFAULT 2,
    lead_id INTEGER,
    score INTEGER DEFAULT 0,
    prioridad TEXT DEFAULT '',
    nivel TEXT DEFAULT '',
    score_detalle TEXT DEFAULT '[]',
    angulo TEXT DEFAULT '',
    etapa TEXT DEFAULT 'Por invitar',
    invitado_en TEXT,
    conectado_en TEXT,
    ultimo_mensaje_en TEXT,
    respondio_en TEXT,
    notas TEXT DEFAULT '',
    creado_en TEXT,
    actualizado_en TEXT
);
CREATE TABLE IF NOT EXISTS li_acciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prospecto_id INTEGER NOT NULL REFERENCES li_prospectos(id),
    paso INTEGER NOT NULL,
    tipo TEXT NOT NULL,
    programado_para TEXT,
    estado TEXT DEFAULT 'pendiente',
    texto TEXT DEFAULT '',
    personalizado_ia INTEGER DEFAULT 0,
    hecho_en TEXT,
    creado_en TEXT,
    actualizado_en TEXT,
    UNIQUE(prospecto_id, paso)
);
CREATE INDEX IF NOT EXISTS idx_li_acc_estado ON li_acciones(estado);
CREATE INDEX IF NOT EXISTS idx_li_pros_etapa ON li_prospectos(etapa);
"""

CAMPOS_PERFIL = ("nombre", "cargo", "empresa", "ubicacion", "industria", "url", "email")
_JOIN = (
    "SELECT a.*, p.nombre, p.cargo, p.empresa, p.url, p.score, p.prioridad, p.etapa, p.grado, p.fuente "
    "FROM li_acciones a JOIN li_prospectos p ON p.id=a.prospecto_id"
)


class BaseProspector(BaseDatos):
    def __init__(self, ruta):
        super().__init__(ruta)
        self.con.executescript(ESQUEMA_LI)
        self.con.commit()

    # ---------------- prospectos ----------------
    def guardar_prospecto(self, datos):
        """Inserta o completa un prospecto por su clave. Devuelve (id, es_nuevo)."""
        ahora = iso(ahora_utc())
        existente = self._fila("SELECT * FROM li_prospectos WHERE clave=?", (datos["clave"],))
        if existente:
            # Solo completa lo que faltaba: no pisa lo que ya editaste ni retrocede la etapa.
            cambios = {c: datos[c] for c in CAMPOS_PERFIL if datos.get(c) and not existente.get(c)}
            if datos.get("lead_id") and not existente.get("lead_id"):
                cambios["lead_id"] = datos["lead_id"]
            if datos.get("grado") == 1 and existente["grado"] != 1:
                cambios["grado"] = 1
            if cambios:
                self.actualizar_prospecto(existente["id"], **cambios)
            return existente["id"], False
        grado = 1 if datos.get("grado") == 1 else 2
        cur = self.con.execute(
            """INSERT INTO li_prospectos(clave, nombre, cargo, empresa, ubicacion, industria, url, email,
                   fuente, grado, lead_id, etapa, conectado_en, creado_en, actualizado_en)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datos["clave"], datos.get("nombre", ""), datos.get("cargo", ""), datos.get("empresa", ""),
                datos.get("ubicacion", ""), datos.get("industria", ""), datos.get("url", ""),
                datos.get("email", ""), datos.get("fuente", ""), grado, datos.get("lead_id"),
                "Conectado" if grado == 1 else "Por invitar", datos.get("conectado_en"), ahora, ahora,
            ),
        )
        self.con.commit()
        return cur.lastrowid, True

    def prospectos(self, etapas=None):
        sql, params = "SELECT * FROM li_prospectos", []
        if etapas:
            sql += f" WHERE etapa IN ({','.join('?' * len(etapas))})"
            params = list(etapas)
        return self._filas(sql + " ORDER BY score DESC, id", params)

    def prospecto(self, prospecto_id):
        return self._fila("SELECT * FROM li_prospectos WHERE id=?", (prospecto_id,))

    def actualizar_prospecto(self, prospecto_id, **campos):
        if not campos:
            return
        campos["actualizado_en"] = iso(ahora_utc())
        sets = ", ".join(f"{c}=?" for c in campos)
        self.con.execute(f"UPDATE li_prospectos SET {sets} WHERE id=?", (*campos.values(), prospecto_id))
        self.con.commit()

    def borrar_fuente_li(self, fuente):
        ids = [f["id"] for f in self._filas("SELECT id FROM li_prospectos WHERE fuente=?", (fuente,))]
        self.con.executemany("DELETE FROM li_acciones WHERE prospecto_id=?", [(i,) for i in ids])
        self.con.executemany("DELETE FROM li_prospectos WHERE id=?", [(i,) for i in ids])
        self.con.commit()
        return len(ids)

    # ---------------- acciones ----------------
    def acciones(self, prospecto_id=None, estados=None):
        sql, params = _JOIN + " WHERE 1=1", []
        if prospecto_id is not None:
            sql += " AND a.prospecto_id=?"
            params.append(prospecto_id)
        if estados:
            sql += f" AND a.estado IN ({','.join('?' * len(estados))})"
            params.extend(estados)
        return self._filas(sql + " ORDER BY p.score DESC, a.programado_para, a.id", params)

    def accion(self, accion_id):
        return self._fila(_JOIN + " WHERE a.id=?", (accion_id,))

    def crear_accion(self, prospecto_id, paso, tipo, programado_para, texto="", ia=False):
        ahora = iso(ahora_utc())
        cur = self.con.execute(
            """INSERT OR IGNORE INTO li_acciones(prospecto_id, paso, tipo, programado_para, estado, texto,
                   personalizado_ia, creado_en, actualizado_en)
               VALUES(?,?,?,?,'pendiente',?,?,?,?)""",
            (prospecto_id, paso, tipo, programado_para, texto, int(ia), ahora, ahora),
        )
        self.con.commit()
        return cur.lastrowid if cur.rowcount else None

    def actualizar_accion(self, accion_id, **campos):
        campos["actualizado_en"] = iso(ahora_utc())
        sets = ", ".join(f"{c}=?" for c in campos)
        self.con.execute(f"UPDATE li_acciones SET {sets} WHERE id=?", (*campos.values(), accion_id))
        self.con.commit()

    def hechas_desde(self, desde_iso_utc, invitaciones):
        """Cuántas invitaciones (o mensajes) marcaste como enviadas desde una fecha."""
        operador = "=" if invitaciones else "!="
        f = self._fila(
            f"SELECT COUNT(*) AS n FROM li_acciones WHERE estado='hecho' AND tipo {operador} 'invitacion' "
            "AND hecho_en>=?",
            (desde_iso_utc,),
        )
        return f["n"] if f else 0
