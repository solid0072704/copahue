"""Datos ficticios para probar el agente de caja sin conectarse a Forpay ni a HubSpot."""

import random
from datetime import timedelta

from .fuentes import _sumar_mes

EMPRESAS_DEMO = ["Boulevard Demo SpA", "Costa Demo SpA", "Barbechos Demo SpA"]
PERFILES = (("bueno", 0.95), ("regular", 0.75), ("malo", 0.35))
PESOS_PERFIL = (0.6, 0.25, 0.15)


class FuentesDemo:
    """Misma interfaz que FuentesReales: cuotas(hoy), egresos(), ventas(hoy)."""

    saldo_inicial = 60_000_000
    caja_minima = 40_000_000

    def __init__(self, hoy, semilla=11):
        self.hoy = hoy
        self.rnd = random.Random(semilla)
        self._cuotas = self._generar_cuotas()

    def _generar_cuotas(self):
        cuotas, idx = [], 1
        for empresa in EMPRESAS_DEMO:
            for _ in range(60):
                perfil, prob_pago = self.rnd.choices(PERFILES, PESOS_PERFIL)[0]
                total = self.rnd.choice([24_000_000, 30_000_000, 36_000_000, 45_000_000])
                n = self.rnd.choice([36, 48, 60])
                inicio = _sumar_mes(self.hoy.replace(day=self.rnd.randint(1, 28)), -self.rnd.randint(1, 40))
                monto = round(total / n)
                compromiso = f"C-{idx:04d}"
                cliente = f"Cliente {idx:03d}"
                proyecto = f"Parcela {self.rnd.randint(1, 120)}"
                especial = None
                if perfil == "malo" and self.rnd.random() < 0.3:
                    especial = "en_cobro"
                elif perfil != "bueno" and self.rnd.random() < 0.08:
                    especial = "repactada"
                for k in range(n):
                    vence = _sumar_mes(inicio, k)
                    if vence > self.hoy + timedelta(days=400):
                        break
                    base = {
                        "empresa": empresa, "compromiso": compromiso, "cliente": cliente, "proyecto": proyecto,
                        "numero": k + 1, "monto": float(monto), "monto_pagado": 0.0, "vence": vence, "pagado_en": None,
                        "estado": "por_vencer", "estado_original": "Pendiente-No-Moroso", "dias_mora": 0,
                    }
                    if vence < self.hoy:
                        if self.rnd.random() < prob_pago:
                            base.update({"estado": "pagada", "estado_original": "Pagada", "monto_pagado": float(monto), "pagado_en": vence + timedelta(days=self.rnd.randint(-3, 12))})
                        else:
                            dias = (self.hoy - vence).days
                            base.update({"estado": especial or "moroso", "estado_original": "Pendiente-Moroso", "dias_mora": dias})
                    elif vence == self.hoy:
                        base["estado_original"] = "Pendiente-Hoy"
                    cuotas.append(base)
                idx += 1
        return cuotas

    def cuotas(self, hoy):
        return list(self._cuotas), []

    def egresos(self):
        hoy = self.hoy
        egresos = [
            {"fecha": hoy + timedelta(days=4), "concepto": "Remuneraciones", "categoria": "Personal", "empresa": "", "monto": 42_000_000.0, "recurrencia": "mensual", "estado": "programado", "hasta": None},
            {"fecha": hoy + timedelta(days=11), "concepto": "PPM e IVA", "categoria": "Impuestos", "empresa": "", "monto": 14_000_000.0, "recurrencia": "mensual", "estado": "programado", "hasta": None},
            {"fecha": hoy + timedelta(days=14), "concepto": "Cuota crédito bancario", "categoria": "Financiamiento", "empresa": EMPRESAS_DEMO[0], "monto": 21_000_000.0, "recurrencia": "mensual", "estado": "programado", "hasta": None},
            {"fecha": hoy + timedelta(days=18), "concepto": "Urbanización etapa 3", "categoria": "Obras", "empresa": EMPRESAS_DEMO[2], "monto": 65_000_000.0, "recurrencia": "unica", "estado": "programado", "hasta": None},
            {"fecha": hoy + timedelta(days=7), "concepto": "Publicidad digital", "categoria": "Marketing", "empresa": "", "monto": 6_500_000.0, "recurrencia": "mensual", "estado": "programado", "hasta": None},
            {"fecha": hoy + timedelta(days=25), "concepto": "Honorarios notaría y CBR", "categoria": "Legal", "empresa": "", "monto": 2_800_000.0, "recurrencia": "mensual", "estado": "estimado", "hasta": None},
            {"fecha": hoy - timedelta(days=3), "concepto": "Arriendo oficina", "categoria": "Administración", "empresa": "", "monto": 3_200_000.0, "recurrencia": "mensual", "estado": "pagado", "hasta": None},
        ]
        return egresos, []

    def ventas(self, hoy):
        return [], []
