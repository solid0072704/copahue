"""Datos ficticios para probar los agentes del embudo sin conectarse a HubSpot (ni enviar nada)."""

import random
from datetime import timedelta

ETAPAS_DEMO = [
    ("nuevo", "Nuevo lead", 0.1),
    ("sin_contacto", "Sin contacto", 0.1),
    ("interes", "Interés confirmado", 0.1),
    ("seguimiento", "En seguimiento", 0.2),
    ("visita_agendada", "Visita agendada", 0.4),
    ("visita_realizada", "Visita realizada", 0.5),
    ("reserva", "Reserva", 0.6),
    ("promesa", "Promesa firmada", 0.9),
    ("pagado", "Pagado", 0.9),
    ("closedwon", "Negocio ganado", 1.0),
    ("closedlost", "Negocio perdido", 0.0),
]
PROPIETARIOS_DEMO = {
    "1": {"nombre": "Ana Demo", "email": "ana@demo.example"},
    "2": {"nombre": "Bruno Demo", "email": "bruno@demo.example"},
    "3": {"nombre": "Carla Demo", "email": "carla@demo.example"},
}
RAZONES_DEMO = ["No responde", "No responde", "No responde", "No está interesado", "Solo cotizando", "Busca crédito", "Compró en otro proyecto"]
# (negocios abiertos, mediana de días en la etapa) por etapa: así se ven cuellos de botella en la demo.
PERFIL_ABIERTOS = {
    "nuevo": (12, 1),
    "sin_contacto": (80, 5),
    "interes": (35, 8),
    "seguimiento": (40, 30),
    "visita_agendada": (8, 2),
    "visita_realizada": (6, 6),
    "reserva": (5, 5),
    "promesa": (12, 90),
    "pagado": (25, 300),
}
# Fracción acumulada de los negocios del periodo que llega al menos hasta cada paso del embudo.
ALCANCE_PERIODO = [1.0, 0.95, 0.25, 0.12, 0.05, 0.03, 0.015, 0.008, 0.004]
APELLIDOS = ("Soto", "Rojas", "Díaz", "Muñoz", "Pérez", "González", "Araya")


def etapas_demo():
    return [{"id": i, "nombre": n, "orden": k, "probabilidad": p} for k, (i, n, p) in enumerate(ETAPAS_DEMO)]


class ClienteDemo:
    """Imita al cliente de HubSpot con datos ficticios. No hace ninguna llamada de red."""

    def __init__(self, ahora, semilla=7):
        self.ahora = ahora
        self.rnd = random.Random(semilla)
        self.tareas = []
        self._etapas = etapas_demo()
        self._abiertos = self._generar_abiertos()
        self._perdidos = self._generar_perdidos()

    # ---- generación
    def _base(self, idx, etapa_idx, creado, entro):
        etapas = self._etapas
        return {
            "id": f"demo-{idx}",
            "nombre": f"Parcela {idx:03d} · Familia {self.rnd.choice(APELLIDOS)}",
            "etapa_id": etapas[etapa_idx]["id"],
            "pipeline": "demo",
            "monto": self.rnd.choice([0, 25_000_000, 32_000_000, 45_000_000]),
            "propietario_id": self.rnd.choice(list(PROPIETARIOS_DEMO)),
            "creado": creado,
            "cerrado": None,
            "entro_etapa": entro,
            "ultimo_contacto": None,
            "ultima_actividad": None,
            "proxima_actividad": None,
            "proximo_paso": "",
            "contactos": 0,
            "cerrado_perdido": False,
            "cerrado_ganado": False,
            "razon_perdida": "",
            "entradas": {etapas[k]["id"]: creado + (entro - creado) * (k + 1) / (etapa_idx + 1) for k in range(etapa_idx + 1)},
            "propiedades": {},
        }

    def _generar_abiertos(self):
        negocios, idx = [], 1
        for etapa_idx, (eid, _, _) in enumerate(ETAPAS_DEMO):
            if eid not in PERFIL_ABIERTOS:
                continue
            cantidad, mediana = PERFIL_ABIERTOS[eid]
            for _ in range(cantidad):
                dias = self.rnd.expovariate(1 / mediana) if mediana > 1 else self.rnd.uniform(0, 2)
                entro = self.ahora - timedelta(days=dias)
                creado = entro - timedelta(days=self.rnd.uniform(0, 30) * etapa_idx / 3)
                n = self._base(idx, etapa_idx, creado, entro)
                if self.rnd.random() < 0.3:
                    n["proxima_actividad"] = self.ahora + timedelta(days=self.rnd.uniform(0.5, 6))
                if eid != "sin_contacto" or self.rnd.random() < 0.4:
                    n["contactos"] = self.rnd.randint(1, 5)
                    n["ultimo_contacto"] = entro + timedelta(days=self.rnd.uniform(0, max(0.5, dias)))
                if eid == "reserva" and self.rnd.random() < 0.5:
                    n["proximo_paso"] = "Firmar promesa"
                idx += 1
                negocios.append(n)
        return negocios

    def _generar_perdidos(self):
        negocios = []
        inicio = len(self._abiertos) + 1
        for i in range(400):
            creado = self.ahora - timedelta(days=self.rnd.uniform(0.5, 29))
            u = self.rnd.random()
            etapa_idx = max(k for k, fraccion in enumerate(ALCANCE_PERIODO) if u <= fraccion)
            n = self._base(inicio + i, etapa_idx, creado, creado + timedelta(days=self.rnd.uniform(0, 5)))
            cerrado = self.ahora - timedelta(days=self.rnd.uniform(0, 20))
            n.update({"etapa_id": "closedlost", "cerrado_perdido": True, "razon_perdida": self.rnd.choice(RAZONES_DEMO), "cerrado": cerrado})
            n["entradas"]["closedlost"] = cerrado
            negocios.append(n)
        return negocios

    # ---- misma interfaz que ClienteHubSpot
    def etapas(self):
        return list(self._etapas), "demo", "Pipeline demo"

    def negocios_abiertos(self, pipeline_id, etapas):
        return list(self._abiertos)

    def negocios_creados_desde(self, pipeline_id, etapas, desde):
        return [n for n in self._perdidos + self._abiertos if n["creado"] and n["creado"] >= desde]

    def propietarios(self):
        return {k: dict(v) for k, v in PROPIETARIOS_DEMO.items()}

    def portal_id(self):
        return ""

    def crear_tarea(self, negocio_id, propietario_id, asunto, cuerpo, vence, prioridad="HIGH", tipo="CALL"):
        self.tareas.append(
            {
                "negocio_id": negocio_id,
                "propietario_id": propietario_id,
                "asunto": asunto,
                "cuerpo": cuerpo,
                "vence": vence,
                "prioridad": prioridad,
                "tipo": tipo,
            }
        )
        return f"tarea-demo-{len(self.tareas)}"
