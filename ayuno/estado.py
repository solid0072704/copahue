"""Memoria del agente: qué avisos ya envió (para no repetirlos)."""

import json
import os
from datetime import datetime, timedelta, timezone

DIAS_MEMORIA = 10


class Estado:
    def __init__(self, ruta):
        self.ruta = ruta
        self.enviados = {}
        self.cambio = False
        try:
            with open(ruta, encoding="utf-8") as f:
                datos = json.load(f)
            self.enviados = dict(datos.get("enviados", {}))
        except FileNotFoundError:
            pass
        except (OSError, ValueError, AttributeError):
            # Archivo dañado: partimos de cero (a lo sumo se repite un aviso).
            self.enviados = {}

    def ya_enviado(self, clave):
        return clave in self.enviados

    def marcar(self, clave, cuando):
        self.enviados[clave] = cuando.astimezone(timezone.utc).isoformat(timespec="seconds")
        self.cambio = True

    def guardar(self, ahora=None):
        if not self.cambio:
            return False
        ahora = ahora or datetime.now(timezone.utc)
        limite = ahora - timedelta(days=DIAS_MEMORIA)
        self.enviados = {
            clave: iso for clave, iso in self.enviados.items() if datetime.fromisoformat(iso) >= limite
        }
        carpeta = os.path.dirname(self.ruta)
        if carpeta:
            os.makedirs(carpeta, exist_ok=True)
        temporal = f"{self.ruta}.tmp"
        with open(temporal, "w", encoding="utf-8") as f:
            json.dump({"enviados": self.enviados}, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(temporal, self.ruta)
        return True
