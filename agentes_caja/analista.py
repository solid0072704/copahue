"""Analista de caja: convierte la cobranza y la proyección en hallazgos y acciones para la gerencia de finanzas."""

from collections import defaultdict
from datetime import date, timedelta

from agentes_ebook.db import iso

AGENTE = "Analista de caja"


def fmt_monto(valor, moneda="CLP"):
    """1234567 -> '$1.234.567' (CLP) · otros: 'USD 1.234.567'."""
    texto = f"{abs(valor):,.0f}".replace(",", ".")
    signo = "-" if valor < 0 else ""
    return f"{signo}${texto}" if moneda.upper() == "CLP" else f"{signo}{moneda} {texto}"


def fecha_corta(texto_iso):
    try:
        d = date.fromisoformat(str(texto_iso)[:10])
        return f"{d.day:02d}/{d.month:02d}"
    except ValueError:
        return str(texto_iso)


def _signo(valor, moneda):
    return ("+" if valor > 0 else "") + fmt_monto(valor, moneda)


def egresos_por_categoria(eventos, hoy, dias=28):
    fin = hoy + timedelta(days=dias - 1)
    totales = defaultdict(float)
    for e in eventos:
        if hoy <= e["fecha"] <= fin:
            totales[e["categoria"] or "Otros"] += e["monto"]
    return sorted(({"categoria": k, "monto": round(v)} for k, v in totales.items()), key=lambda x: -x["monto"])


def diagnosticar(cuotas, cobranza, proyeccion, egresos_eventos, cfg, ahora, hoy, anterior=None, errores_fuentes=()):
    fin_4 = hoy + timedelta(days=27)
    proximos = sorted((e for e in egresos_eventos if hoy <= e["fecha"] <= fin_4), key=lambda e: -e["monto"])
    diag = {
        "fecha": iso(ahora),
        "hoy": hoy.isoformat(),
        "moneda": cfg.moneda,
        "cuotas_total": len(cuotas),
        "empresas": sorted({c["empresa"] for c in cuotas}),
        "cobranza": cobranza,
        "proyeccion": proyeccion,
        "egresos_categorias_4s": egresos_por_categoria(egresos_eventos, hoy),
        "egresos_mayores_4s": [
            {"fecha": e["fecha"].isoformat(), "concepto": e["concepto"], "categoria": e["categoria"], "empresa": e["empresa"], "monto": round(e["monto"])}
            for e in proximos[:10]
        ],
        "errores_fuentes": list(errores_fuentes),
        "comparacion": None,
        "hallazgos": [],
        "acciones": [],
    }
    diag["comparacion"] = comparar(diag, anterior)
    diag["hallazgos"], diag["acciones"] = analizar(diag, cfg)
    return diag


def comparar(diag, anterior):
    if not anterior or anterior.get("fecha") == diag["fecha"]:
        return None
    cb, cb0 = diag["cobranza"], anterior.get("cobranza", {})
    py, py0 = diag["proyeccion"], anterior.get("proyeccion", {})
    return {
        "fecha": anterior["fecha"],
        "moroso": cb["total_moroso"] - cb0.get("total_moroso", 0),
        "pendiente": cb["total_pendiente"] - cb0.get("total_pendiente", 0),
        "pct_mora": round(cb["pct_mora"] - cb0.get("pct_mora", 0), 1),
        "saldo_minimo": py["escenarios"]["base"]["saldo_minimo"] - py0.get("escenarios", {}).get("base", {}).get("saldo_minimo", 0),
    }


def analizar(diag, cfg):
    """Devuelve (hallazgos, acciones) ordenados por importancia."""
    m = diag["moneda"]
    cb, py = diag["cobranza"], diag["proyeccion"]
    reglas = []  # (peso, hallazgo, acción o None)

    if py["alerta"]:
        a = py["alerta"]
        reglas.append(
            (
                100,
                f"La caja proyectada cae a {fmt_monto(a['saldo'], m)} en la semana del {fecha_corta(a['inicio'])}, bajo el mínimo de "
                f"{fmt_monto(py['caja_minima'], m)}: faltan {fmt_monto(a['brecha'], m)}.",
                f"Cubre la brecha de {fmt_monto(a['brecha'], m)} antes del {fecha_corta(a['inicio'])}: adelanta la cobranza de los compromisos "
                f"prioritarios (los 10 primeros suman {fmt_monto(cb['monto_top10'], m)}) o reprograma los egresos no críticos de esas semanas.",
            )
        )
    elif not py["saldo_inicial"] and not py["caja_minima"]:
        reglas.append(
            (
                1,
                "No hay saldo inicial ni caja mínima configurados (CAJA_SALDO_INICIAL, CAJA_MINIMA): la proyección muestra el flujo neto, no el saldo.",
                None,
            )
        )

    p4 = py["proximas_4"]
    if p4["egresos"] and p4["ingresos"] < p4["egresos"]:
        deficit = p4["egresos"] - p4["ingresos"]
        reglas.append(
            (
                80,
                f"En las próximas 4 semanas los egresos ({fmt_monto(p4['egresos'], m)}) superan los ingresos esperados "
                f"({fmt_monto(p4['ingresos'], m)}): déficit de {fmt_monto(deficit, m)}.",
                "Revisa los egresos de las próximas 4 semanas: posterga los no críticos y confirma fecha de pago con los clientes que vencen en ese periodo.",
            )
        )
    elif not p4["egresos"]:
        reglas.append((2, "No hay egresos cargados: la proyección solo muestra cobros. Completa data/egresos.csv (plantilla con --plantilla-egresos).", None))

    if cb["total_moroso"] > 0:
        reglas.append(
            (
                min(90, cb["pct_mora"] * 2 + 10),
                f"La morosidad es {fmt_monto(cb['total_moroso'], m)} ({cb['pct_mora']:.0f}% de la cartera pendiente): "
                f"{cb['cuotas_morosas']} cuotas de {cb['compromisos_morosos']} compromisos.",
                f"Ejecuta la lista de cobranza prioritaria: los 10 primeros compromisos concentran el {cb['concentracion_top10']:.0f}% del moroso "
                f"({fmt_monto(cb['monto_top10'], m)}).",
            )
        )
        if cb["concentracion_top10"] >= 40 and cb["compromisos_morosos"] >= 10:
            reglas.append(
                (
                    cb["concentracion_top10"] / 2,
                    f"La mora está concentrada: 10 compromisos reúnen el {cb['concentracion_top10']:.0f}% del total.",
                    "Gestiona personalmente esos 10 casos esta semana; el resto puede ir por recordatorios automáticos.",
                )
            )
        viejo = cb["tramos"][-1]
        if viejo["pct"] >= 30:
            reglas.append(
                (
                    viejo["pct"],
                    f"El {viejo['pct']:.0f}% del moroso ({fmt_monto(viejo['monto'], m)}) tiene más de 90 días: es caja poco probable.",
                    "Provisiona la mora mayor a 90 días y decide caso a caso: repactación con pie, cobranza externa o resciliación.",
                )
            )
        reciente = cb["tramos"][0]
        if reciente["pct"] >= 40:
            reglas.append(
                (
                    reciente["pct"] / 2,
                    f"El {reciente['pct']:.0f}% de la mora es reciente (1-15 días, {fmt_monto(reciente['monto'], m)}): se recupera con recordatorios rápidos.",
                    "Automatiza el recordatorio al día 1 y al día 7 de atraso con link de pago; es la mora más barata de recuperar.",
                )
            )

    ef = [f for f in cb["efectividad"] if f["puntualidad"] is not None]
    if len(ef) >= 2:
        ultimo, previos = ef[-1], ef[:-1]
        promedio = sum(f["puntualidad"] for f in previos) / len(previos)
        texto = f"En {ultimo['mes']} se cobró el {ultimo['puntualidad']:.0f}% de lo que vencía (promedio de los meses anteriores: {promedio:.0f}%)."
        if ultimo["puntualidad"] - promedio <= -5:
            reglas.append((60, texto, "La cobranza cayó: revisa qué cambió (canal, equipo, fechas de vencimiento) y refuerza el recordatorio previo al vencimiento."))
        else:
            reglas.append((15, texto, None))

    v7 = cb["vence_7d"]
    if v7["monto"]:
        reglas.append(
            (
                30,
                f"Vencen {fmt_monto(v7['monto'], m)} en {v7['cuotas']} cuotas los próximos 7 días; {fmt_monto(v7['con_mora_monto'], m)} "
                f"corresponden a compromisos que ya tienen cuotas en mora.",
                "Envía hoy el recordatorio previo al vencimiento y llama a los que ya tienen mora: son los que más probablemente vuelvan a fallar.",
            )
        )

    con_cartera = [e for e in cb["por_empresa"] if e["pendiente"] > 0]
    if len(con_cartera) >= 2:
        peor = max(con_cartera, key=lambda e: e["pct_mora"])
        reglas.append(
            (
                peor["pct_mora"] / 2,
                f"{peor['empresa']} tiene la mayor morosidad relativa: {peor['pct_mora']:.0f}% de su cartera ({fmt_monto(peor['moroso'], m)}).",
                None,
            )
        )

    if cb["en_cobro"]["monto"] or cb["repactadas"]["monto"]:
        reglas.append(
            (
                10,
                f"{fmt_monto(cb['en_cobro']['monto'], m)} está en proceso de cobro y {fmt_monto(cb['repactadas']['monto'], m)} repactado; "
                "la proyección base no cuenta con ese dinero.",
                None,
            )
        )

    comp = diag.get("comparacion")
    if comp:
        reglas.append(
            (
                5,
                f"Frente a la foto del {comp['fecha'][:10]}: morosidad {_signo(comp['moroso'], m)} y cartera pendiente {_signo(comp['pendiente'], m)}.",
                None,
            )
        )

    if diag["errores_fuentes"]:
        reglas.append((95, f"{len(diag['errores_fuentes'])} fuente(s) no respondieron; el diagnóstico puede estar incompleto (ver al final).", None))

    reglas.sort(key=lambda r: -r[0])
    hallazgos = [h for _, h, _ in reglas][:8]
    acciones = []
    for _, _, accion in reglas:
        if accion and accion not in acciones:
            acciones.append(accion)
        if len(acciones) == 3:
            break
    if not hallazgos:
        hallazgos = ["Sin alertas: la cartera está al día y la caja proyectada se mantiene sobre el mínimo."]
    return hallazgos, acciones
