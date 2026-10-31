"""Proyección semanal de caja: cobros esperados, recuperación de mora, ventas comprometidas y egresos, en tres escenarios."""

from datetime import timedelta

ESCENARIOS = ("base", "pesimista", "optimista")


def semanas(hoy, n):
    return [(hoy + timedelta(days=7 * i), hoy + timedelta(days=7 * i + 6)) for i in range(n)]


def _en(fecha, inicio, fin):
    return fecha is not None and inicio <= fecha <= fin


def proyectar(cuotas, egresos_eventos, ventas_eventos, cfg, hoy, tasa_puntual=None, saldo_inicial=None):
    """Devuelve la proyección base semana a semana más el resumen de los escenarios pesimista y optimista."""
    tasa = tasa_puntual if tasa_puntual is not None else cfg.tasa_cobro_puntual
    saldo_inicial = cfg.saldo_inicial_total if saldo_inicial is None else saldo_inicial
    por_vencer = [c for c in cuotas if c["estado"] == "por_vencer"]
    atrasadas = sum(c["monto"] for c in por_vencer if c["vence"] and c["vence"] < hoy)  # "Pendiente-Hoy" o ya vencidas: semana 1
    total_moroso = sum(c["monto"] for c in cuotas if c["estado"] == "moroso")
    parametros = {
        "base": {"tasa": tasa, "mora": cfg.tasa_recuperacion_mora, "egresos": 1.0, "ventas": 1.0},
        "pesimista": {"tasa": tasa * 0.8, "mora": cfg.tasa_recuperacion_mora * 0.5, "egresos": 1.1, "ventas": 0.5},
        "optimista": {"tasa": min(1.0, tasa * 1.1), "mora": min(1.0, cfg.tasa_recuperacion_mora * 1.5), "egresos": 1.0, "ventas": 1.0},
    }
    resultado = {
        "saldo_inicial": round(saldo_inicial),
        "caja_minima": round(cfg.caja_minima),
        "tasa_cobro": round(tasa, 3),
        "tasa_mora": cfg.tasa_recuperacion_mora,
        "horizonte_semanas": cfg.horizonte_semanas,
        "semanas": [],
        "escenarios": {},
        "alerta": None,
    }
    rangos = semanas(hoy, cfg.horizonte_semanas)
    for nombre in ESCENARIOS:
        p = parametros[nombre]
        saldo, filas, minimo = saldo_inicial, [], (saldo_inicial, None)
        for i, (inicio, fin) in enumerate(rangos):
            programado = sum(c["monto"] for c in por_vencer if _en(c["vence"], inicio, fin)) + (atrasadas if i == 0 else 0)
            esperado = programado * p["tasa"]
            mora = total_moroso * p["mora"] / cfg.semanas_recuperacion_mora if i < cfg.semanas_recuperacion_mora else 0.0
            ventas = sum(e["monto"] for e in ventas_eventos if _en(e["fecha"], inicio, fin)) * p["ventas"]
            egresos = sum(e["monto"] for e in egresos_eventos if _en(e["fecha"], inicio, fin)) * p["egresos"]
            ingresos = esperado + mora + ventas
            neto = ingresos - egresos
            saldo += neto
            if saldo < minimo[0]:
                minimo = (saldo, inicio)
            filas.append(
                {
                    "semana": i + 1,
                    "inicio": inicio.isoformat(),
                    "fin": fin.isoformat(),
                    "programado": round(programado),
                    "esperado": round(esperado),
                    "mora": round(mora),
                    "ventas": round(ventas),
                    "ingresos": round(ingresos),
                    "egresos": round(egresos),
                    "neto": round(neto),
                    "saldo": round(saldo),
                }
            )
        resultado["escenarios"][nombre] = {
            "ingresos": round(sum(f["ingresos"] for f in filas)),
            "egresos": round(sum(f["egresos"] for f in filas)),
            "saldo_minimo": round(minimo[0]),
            "semana_minima": minimo[1].isoformat() if minimo[1] else None,
            "saldo_final": round(saldo),
        }
        if nombre == "base":
            resultado["semanas"] = filas
            if cfg.caja_minima or saldo_inicial:
                bajo = next((f for f in filas if f["saldo"] < cfg.caja_minima), None)
                if bajo:
                    resultado["alerta"] = {
                        "semana": bajo["semana"],
                        "inicio": bajo["inicio"],
                        "saldo": bajo["saldo"],
                        "brecha": round(cfg.caja_minima - min(f["saldo"] for f in filas)),
                    }
    primeras = resultado["semanas"][:4]
    resultado["proximas_4"] = {
        "programado": sum(f["programado"] for f in primeras),
        "ingresos": sum(f["ingresos"] for f in primeras),
        "egresos": sum(f["egresos"] for f in primeras),
        "neto": sum(f["neto"] for f in primeras),
    }
    return resultado
