"""Memoria de la tienda en archivos JSON legibles (carpeta tienda_datos/).

Se guardan en el repositorio a propósito: son datos públicos (lo mismo que muestra la
tienda), así cada cambio que hacen los agentes queda en el historial de git y las URLs
de los productos nunca se pierden, algo clave para no perder posicionamiento.
"""

import json
import os
import tempfile
from datetime import datetime, timezone

ARCHIVOS = ("productos", "nichos", "guias", "keywords", "estado")
MAX_BITACORA = 3000


def ahora():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def desde_iso(texto):
    if not texto:
        return None
    try:
        dt = datetime.fromisoformat(texto)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def dias_desde(texto):
    dt = desde_iso(texto)
    if not dt:
        return 10_000
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400


def _escribir_atomico(ruta, contenido):
    carpeta = os.path.dirname(ruta) or "."
    os.makedirs(carpeta, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=carpeta, prefix=".tmp-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(contenido)
    os.replace(tmp, ruta)


class Almacen:
    """Diccionarios en memoria que se guardan como JSON ordenado (diffs limpios en git)."""

    def __init__(self, carpeta):
        self.carpeta = carpeta
        os.makedirs(carpeta, exist_ok=True)
        for nombre in ARCHIVOS:
            setattr(self, nombre, self._leer(nombre))
        self._bitacora_nueva = []

    def _ruta(self, nombre, ext="json"):
        return os.path.join(self.carpeta, f"{nombre}.{ext}")

    def _leer(self, nombre):
        try:
            with open(self._ruta(nombre), encoding="utf-8") as f:
                datos = json.load(f)
            return datos if isinstance(datos, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def guardar(self):
        for nombre in ARCHIVOS:
            texto = json.dumps(getattr(self, nombre), ensure_ascii=False, indent=1, sort_keys=True) + "\n"
            ruta = self._ruta(nombre)
            try:
                with open(ruta, encoding="utf-8") as f:
                    if f.read() == texto:
                        continue  # sin cambios: no tocamos el archivo
            except FileNotFoundError:
                pass
            _escribir_atomico(ruta, texto)
        if self._bitacora_nueva:
            ruta = self._ruta("bitacora", "jsonl")
            lineas = []
            try:
                with open(ruta, encoding="utf-8") as f:
                    lineas = f.read().splitlines()
            except FileNotFoundError:
                pass
            lineas += [json.dumps(e, ensure_ascii=False, sort_keys=True) for e in self._bitacora_nueva]
            _escribir_atomico(ruta, "\n".join(lineas[-MAX_BITACORA:]) + "\n")
            self._bitacora_nueva = []

    # --- Bitácora: qué hizo cada agente y por qué ---
    def log(self, agente, accion, detalle="", ref=None):
        self._bitacora_nueva.append(
            {"fecha": ahora(), "agente": agente, "accion": accion, "detalle": str(detalle)[:500], "ref": ref}
        )

    def bitacora(self, limite=500):
        eventos = []
        try:
            with open(self._ruta("bitacora", "jsonl"), encoding="utf-8") as f:
                for linea in f:
                    try:
                        eventos.append(json.loads(linea))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        eventos += self._bitacora_nueva
        return list(reversed(eventos))[:limite]

    # --- Consultas frecuentes ---
    def publicados(self, nicho=None):
        lista = [p for p in self.productos.values() if p.get("estado") == "publicado"]
        if nicho:
            lista = [p for p in lista if p.get("nicho_slug") == nicho]
        return sorted(lista, key=lambda p: (-p.get("score", 0), p["id"]))

    def visibles(self):
        """Productos con página propia. Toda URL que alguna vez se creó sigue existiendo (sin indexar si el
        producto ya no está a la venta), para no generar errores 404 ni perder posicionamiento."""
        return [p for p in self.productos.values() if p.get("slug")]

    def hay_demo(self):
        return any(p.get("demo") for p in self.productos.values())

    def borrar_demo(self):
        ids = [k for k, p in self.productos.items() if p.get("demo")]
        for k in ids:
            del self.productos[k]
        for k in [k for k, g in self.guias.items() if g.get("demo")]:
            del self.guias[k]
        return len(ids)
