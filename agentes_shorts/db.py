"""Memoria del agente en SQLite: videos descubiertos, cortes, cuota de la API y bitácora."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# La cuota de la YouTube Data API se reinicia a medianoche, hora del Pacífico.
ZONA_CUOTA = ZoneInfo("America/Los_Angeles")

ESTADOS_VIDEO = ("descubierto", "seleccionado", "cortado", "publicado", "descartado", "error")
ESTADOS_CORTE = ("renderizado", "subido", "error")

ESQUEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    titulo TEXT DEFAULT '',
    canal TEXT DEFAULT '',
    canal_id TEXT DEFAULT '',
    publicado_en TEXT,
    duracion_s INTEGER DEFAULT 0,
    vistas INTEGER DEFAULT 0,
    likes INTEGER DEFAULT 0,
    comentarios INTEGER DEFAULT 0,
    licencia TEXT DEFAULT '',
    categoria TEXT DEFAULT '',
    idioma TEXT DEFAULT '',
    fuente TEXT DEFAULT '',
    score INTEGER DEFAULT 0,
    score_detalle TEXT DEFAULT '[]',
    estado TEXT DEFAULT 'descubierto',
    motivo TEXT DEFAULT '',
    creado_en TEXT,
    actualizado_en TEXT
);
CREATE TABLE IF NOT EXISTS cortes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    indice INTEGER NOT NULL,
    inicio_s REAL NOT NULL,
    fin_s REAL NOT NULL,
    metodo TEXT DEFAULT '',
    titulo TEXT DEFAULT '',
    descripcion TEXT DEFAULT '',
    etiquetas TEXT DEFAULT '[]',
    archivo TEXT DEFAULT '',
    estado TEXT DEFAULT 'renderizado',
    youtube_id TEXT DEFAULT '',
    programado_para TEXT,
    subido_en TEXT,
    error TEXT DEFAULT '',
    creado_en TEXT,
    actualizado_en TEXT,
    UNIQUE(video_id, indice)
);
CREATE TABLE IF NOT EXISTS cuota (
    fecha TEXT PRIMARY KEY,
    unidades INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS bitacora (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha TEXT NOT NULL,
    agente TEXT DEFAULT '',
    nivel TEXT DEFAULT 'info',
    mensaje TEXT DEFAULT ''
);
"""


def ahora():
    return datetime.now(timezone.utc)


def iso(fecha=None):
    return (fecha or ahora()).isoformat(timespec="seconds")


def fecha_cuota(momento=None):
    return (momento or ahora()).astimezone(ZONA_CUOTA).date().isoformat()


class BaseDatos:
    def __init__(self, ruta=":memory:"):
        if ruta != ":memory:":
            carpeta = os.path.dirname(ruta)
            if carpeta:
                os.makedirs(carpeta, exist_ok=True)
        self.con = sqlite3.connect(ruta, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(ESQUEMA)
        self.con.commit()

    def cerrar(self):
        self.con.commit()
        self.con.close()

    # ---------- videos ----------
    def guardar_video(self, datos):
        """Inserta o actualiza las métricas de un video sin tocar su estado ni su score."""
        fila = self.video(datos["video_id"])
        ts = iso()
        if fila is None:
            self.con.execute(
                """INSERT INTO videos (video_id, titulo, canal, canal_id, publicado_en, duracion_s, vistas, likes,
                   comentarios, licencia, categoria, idioma, fuente, creado_en, actualizado_en)
                   VALUES (:video_id, :titulo, :canal, :canal_id, :publicado_en, :duracion_s, :vistas, :likes,
                   :comentarios, :licencia, :categoria, :idioma, :fuente, :ts, :ts)""",
                {**_defaults_video(datos), "ts": ts},
            )
            nuevo = True
        else:
            self.con.execute(
                """UPDATE videos SET titulo=:titulo, canal=:canal, canal_id=:canal_id, publicado_en=:publicado_en,
                   duracion_s=:duracion_s, vistas=:vistas, likes=:likes, comentarios=:comentarios,
                   licencia=:licencia, categoria=:categoria, idioma=:idioma, actualizado_en=:ts
                   WHERE video_id=:video_id""",
                {**_defaults_video(datos), "ts": ts},
            )
            nuevo = False
        self.con.commit()
        return nuevo

    def puntuar_video(self, video_id, score, detalle):
        self.con.execute(
            "UPDATE videos SET score=?, score_detalle=?, actualizado_en=? WHERE video_id=?",
            (int(score), json.dumps(detalle, ensure_ascii=False), iso(), video_id),
        )
        self.con.commit()

    def cambiar_estado_video(self, video_id, estado, motivo=""):
        assert estado in ESTADOS_VIDEO, estado
        self.con.execute(
            "UPDATE videos SET estado=?, motivo=?, actualizado_en=? WHERE video_id=?",
            (estado, motivo, iso(), video_id),
        )
        self.con.commit()

    def video(self, video_id):
        fila = self.con.execute("SELECT * FROM videos WHERE video_id=?", (video_id,)).fetchone()
        return _dict(fila)

    def videos(self, estado=None, limite=None, orden="score DESC, vistas DESC"):
        sql = "SELECT * FROM videos"
        params = []
        if estado:
            estados = [estado] if isinstance(estado, str) else list(estado)
            sql += " WHERE estado IN (%s)" % ",".join("?" * len(estados))
            params += estados
        sql += f" ORDER BY {orden}"
        if limite:
            sql += " LIMIT ?"
            params.append(int(limite))
        return [_dict(f) for f in self.con.execute(sql, params).fetchall()]

    def candidatos(self, umbral, limite):
        return [
            v
            for v in self.videos(estado="descubierto", orden="score DESC, vistas DESC", limite=limite * 3)
            if v["score"] >= umbral
        ][:limite]

    # ---------- cortes ----------
    def guardar_corte(self, corte):
        ts = iso()
        self.con.execute(
            """INSERT INTO cortes (video_id, indice, inicio_s, fin_s, metodo, titulo, descripcion, etiquetas, archivo,
               estado, creado_en, actualizado_en)
               VALUES (:video_id, :indice, :inicio_s, :fin_s, :metodo, :titulo, :descripcion, :etiquetas, :archivo,
               'renderizado', :ts, :ts)
               ON CONFLICT(video_id, indice) DO UPDATE SET inicio_s=excluded.inicio_s, fin_s=excluded.fin_s,
               metodo=excluded.metodo, titulo=excluded.titulo, descripcion=excluded.descripcion,
               etiquetas=excluded.etiquetas, archivo=excluded.archivo, estado='renderizado', error='',
               actualizado_en=excluded.actualizado_en""",
            {
                "video_id": corte["video_id"],
                "indice": int(corte["indice"]),
                "inicio_s": float(corte["inicio_s"]),
                "fin_s": float(corte["fin_s"]),
                "metodo": corte.get("metodo", ""),
                "titulo": corte.get("titulo", ""),
                "descripcion": corte.get("descripcion", ""),
                "etiquetas": json.dumps(corte.get("etiquetas", []), ensure_ascii=False),
                "archivo": corte.get("archivo", ""),
                "ts": ts,
            },
        )
        self.con.commit()
        return self.con.execute(
            "SELECT id FROM cortes WHERE video_id=? AND indice=?", (corte["video_id"], int(corte["indice"]))
        ).fetchone()[0]

    def marcar_corte_subido(self, corte_id, youtube_id, programado_para=None):
        self.con.execute(
            "UPDATE cortes SET estado='subido', youtube_id=?, programado_para=?, subido_en=?, error='', actualizado_en=? WHERE id=?",
            (youtube_id, programado_para, iso(), iso(), corte_id),
        )
        self.con.commit()

    def marcar_corte_error(self, corte_id, error):
        self.con.execute(
            "UPDATE cortes SET estado='error', error=?, actualizado_en=? WHERE id=?", (str(error)[:500], iso(), corte_id)
        )
        self.con.commit()

    def cortes(self, estado=None, video_id=None, limite=None):
        sql = "SELECT c.*, v.titulo AS video_titulo, v.canal, v.licencia FROM cortes c JOIN videos v USING (video_id)"
        cond, params = [], []
        if estado:
            cond.append("c.estado=?")
            params.append(estado)
        if video_id:
            cond.append("c.video_id=?")
            params.append(video_id)
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        sql += " ORDER BY v.score DESC, c.video_id, c.indice"
        if limite:
            sql += " LIMIT ?"
            params.append(int(limite))
        filas = [_dict(f) for f in self.con.execute(sql, params).fetchall()]
        for f in filas:
            f["etiquetas"] = json.loads(f.get("etiquetas") or "[]")
        return filas

    def subidos_hoy(self):
        desde = datetime.now(ZONA_CUOTA).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        return self.con.execute(
            "SELECT COUNT(*) FROM cortes WHERE estado='subido' AND subido_en >= ?", (iso(desde),)
        ).fetchone()[0]

    def ultimo_programado(self):
        fila = self.con.execute(
            "SELECT MAX(programado_para) FROM cortes WHERE estado='subido' AND programado_para IS NOT NULL"
        ).fetchone()
        return fila[0] if fila else None

    # ---------- cuota ----------
    def sumar_cuota(self, unidades):
        self.con.execute(
            "INSERT INTO cuota (fecha, unidades) VALUES (?, ?) ON CONFLICT(fecha) DO UPDATE SET unidades = unidades + excluded.unidades",
            (fecha_cuota(), int(unidades)),
        )
        self.con.commit()

    def cuota_usada(self):
        fila = self.con.execute("SELECT unidades FROM cuota WHERE fecha=?", (fecha_cuota(),)).fetchone()
        return fila[0] if fila else 0

    # ---------- bitácora ----------
    def anotar(self, agente, mensaje, nivel="info"):
        self.con.execute(
            "INSERT INTO bitacora (fecha, agente, nivel, mensaje) VALUES (?, ?, ?, ?)", (iso(), agente, nivel, mensaje)
        )
        self.con.commit()

    def bitacora(self, limite=50):
        return [
            _dict(f)
            for f in self.con.execute("SELECT * FROM bitacora ORDER BY id DESC LIMIT ?", (int(limite),)).fetchall()
        ]

    def resumen(self):
        filas = self.con.execute("SELECT estado, COUNT(*) AS n FROM videos GROUP BY estado").fetchall()
        videos = {f["estado"]: f["n"] for f in filas}
        filas = self.con.execute("SELECT estado, COUNT(*) AS n FROM cortes GROUP BY estado").fetchall()
        cortes = {f["estado"]: f["n"] for f in filas}
        return {"videos": videos, "cortes": cortes, "cuota_hoy": self.cuota_usada(), "subidos_hoy": self.subidos_hoy()}


def _defaults_video(d):
    return {
        "video_id": d["video_id"],
        "titulo": d.get("titulo", "") or "",
        "canal": d.get("canal", "") or "",
        "canal_id": d.get("canal_id", "") or "",
        "publicado_en": d.get("publicado_en"),
        "duracion_s": int(d.get("duracion_s") or 0),
        "vistas": int(d.get("vistas") or 0),
        "likes": int(d.get("likes") or 0),
        "comentarios": int(d.get("comentarios") or 0),
        "licencia": d.get("licencia", "") or "",
        "categoria": str(d.get("categoria", "") or ""),
        "idioma": d.get("idioma", "") or "",
        "fuente": d.get("fuente", "") or "",
    }


def _dict(fila):
    if fila is None:
        return None
    d = dict(fila)
    if "score_detalle" in d:
        try:
            d["score_detalle"] = json.loads(d["score_detalle"] or "[]")
        except (TypeError, ValueError):
            d["score_detalle"] = []
    return d


def abrir_db(cfg):
    return BaseDatos(cfg.db_path)
