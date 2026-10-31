"""Agente Analista de embudo: mide dónde se acumulan y dónde se pierden los negocios del pipeline.

Trabaja solo con lectura: negocios por etapa, tiempo en la etapa, negocios sin próxima actividad,
conversión entre etapas, razones de pérdida y carga por propietario. Con eso redacta hallazgos
y recomendaciones por reglas; la IA, si está configurada, agrega una lectura ejecutiva.
"""

import unicodedata
from collections import Counter, defaultdict
from statistics import median, quantiles

from agentes_ebook.db import iso

AGENTE = "Analista de embudo"


# --------------------------------------------------------------------------- medidas por negocio
def dias_en_etapa(negocio, ahora):
    if negocio.get("entro_etapa"):
        return max(0.0, (ahora - negocio["entro_etapa"]).total_seconds() / 86400)
    ms = (negocio.get("propiedades") or {}).get("hs_v2_time_in_current_stage")
    try:
        if ms not in (None, ""):
            return max(0.0, float(ms) / 86_400_000)
    except (TypeError, ValueError):
        pass
    if negocio.get("creado"):
        return max(0.0, (ahora - negocio["creado"]).total_seconds() / 86400)
    return 0.0


def dias_sin_contacto(negocio, ahora):
    referencia = negocio.get("ultimo_contacto") or negocio.get("ultima_actividad") or negocio.get("creado")
    if not referencia:
        return None
    return max(0.0, (ahora - referencia).total_seconds() / 86400)


def sin_proxima_actividad(negocio, ahora):
    """True si no hay nada agendado a futuro (o lo agendado ya venció sin actualizarse)."""
    proxima = negocio.get("proxima_actividad")
    return proxima is None or proxima < ahora


def _p75(valores):
    if not valores:
        return 0.0
    if len(valores) < 4:
        return float(max(valores))
    return float(quantiles(valores, n=4)[2])


def _normalizar(texto):
    sin_tildes = unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode()
    return sin_tildes.lower()


def _es_no_respuesta(razon):
    r = _normalizar(razon)
    return any(p in r for p in ("respond", "contest", "contact", "no answer", "unreachable", "ubic"))


def _signo(n):
    return f"+{n}" if n > 0 else str(n)


# --------------------------------------------------------------------------- embudo
def pasos_embudo(etapas):
    """Etapas que forman el embudo, en orden. Se excluye la de 'perdido' (probabilidad 0)."""
    return [e for e in sorted(etapas, key=lambda e: e["orden"]) if e["probabilidad"] > 0]


def _orden_maximo(negocio, orden_por_id):
    """Hasta qué paso del embudo llegó el negocio (por las fechas de entrada a cada etapa)."""
    ordenes = [orden_por_id[i] for i in (negocio.get("entradas") or {}) if i in orden_por_id]
    if negocio.get("etapa_id") in orden_por_id:
        ordenes.append(orden_por_id[negocio["etapa_id"]])
    return max(ordenes) if ordenes else None


def _fila_etapa(etapa, lista, ahora, cfg):
    dias = [dias_en_etapa(n, ahora) for n in lista]
    return {
        "id": etapa["id"],
        "nombre": etapa["nombre"],
        "orden": etapa["orden"],
        "probabilidad": etapa["probabilidad"],
        "abiertos": len(lista),
        "monto": round(sum(n.get("monto") or 0 for n in lista), 2),
        "mediana_dias": round(median(dias), 1) if dias else 0.0,
        "p75_dias": round(_p75(dias), 1),
        "sin_proxima_actividad": sum(1 for n in lista if sin_proxima_actividad(n, ahora)),
        "sin_contacto": sum(1 for n in lista if (dias_sin_contacto(n, ahora) or 0) >= cfg.sin_contacto_dias),
        "alcanzaron": 0,
        "conversion": None,
    }


def diagnosticar(abiertos, periodo, etapas, cfg, ahora, pipeline="", propietarios=None, anterior=None, estancados=None):
    """Arma el diagnóstico completo (un dict serializable) a partir de los negocios leídos."""
    propietarios = propietarios or {}
    pasos = pasos_embudo(etapas)
    orden_por_id = {e["id"]: e["orden"] for e in pasos}

    por_etapa = defaultdict(list)
    for n in abiertos:
        por_etapa[n["etapa_id"]].append(n)

    maximos = [m for m in (_orden_maximo(n, orden_por_id) for n in periodo) if m is not None]
    filas, alcanzaron_previa = [], None
    for e in sorted(etapas, key=lambda e: e["orden"]):
        fila = _fila_etapa(e, por_etapa.get(e["id"], []), ahora, cfg)
        if e["id"] in orden_por_id:
            fila["alcanzaron"] = sum(1 for m in maximos if m >= e["orden"])
            if alcanzaron_previa:
                fila["conversion"] = round(100 * fila["alcanzaron"] / alcanzaron_previa, 1)
            alcanzaron_previa = fila["alcanzaron"]
        filas.append(fila)

    conteo = defaultdict(lambda: {"abiertos": 0, "sin_proxima": 0, "estancados": 0, "monto": 0.0})
    for n in abiertos:
        c = conteo[n.get("propietario_id") or ""]
        c["abiertos"] += 1
        c["monto"] += n.get("monto") or 0
        if sin_proxima_actividad(n, ahora):
            c["sin_proxima"] += 1
    for est in estancados or []:
        conteo[est["negocio"].get("propietario_id") or ""]["estancados"] += 1
    lista_propietarios = sorted(
        (
            {
                "id": pid,
                "nombre": propietarios.get(pid, {}).get("nombre") or ("Sin propietario" if not pid else f"Propietario {pid}"),
                **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()},
            }
            for pid, c in conteo.items()
        ),
        key=lambda x: (-x["sin_proxima"], -x["abiertos"]),
    )

    perdidos = [n for n in periodo if n.get("cerrado_perdido")]
    razones = Counter((n.get("razon_perdida") or "Sin razón registrada") for n in perdidos).most_common(10)

    diag = {
        "fecha": iso(ahora),
        "pipeline": pipeline,
        "periodo_dias": cfg.embudo_dias,
        "total_abiertos": len(abiertos),
        "monto_abierto": round(sum(n.get("monto") or 0 for n in abiertos), 2),
        "sin_proxima_total": sum(1 for n in abiertos if sin_proxima_actividad(n, ahora)),
        "estancados": len(estancados or []),
        "creados_periodo": len(periodo),
        "perdidos_periodo": len(perdidos),
        "ganados_periodo": sum(1 for n in periodo if n.get("cerrado_ganado")),
        "etapas": filas,
        "razones_perdida": [[r, c] for r, c in razones],
        "propietarios": lista_propietarios[:15],
        "comparacion": None,
        "hallazgos": [],
        "recomendaciones": [],
    }
    diag["comparacion"] = comparar(diag, anterior)
    diag["hallazgos"], diag["recomendaciones"] = analizar(diag, cfg)
    return diag


def comparar(diag, anterior):
    """Diferencias frente a una foto anterior del mismo pipeline (o None si no hay)."""
    if not anterior or anterior.get("fecha") == diag.get("fecha"):
        return None
    previas = {e["id"]: e for e in anterior.get("etapas", [])}
    por_etapa = {}
    for e in diag["etapas"]:
        p = previas.get(e["id"])
        if p and p["abiertos"] != e["abiertos"]:
            por_etapa[e["nombre"]] = e["abiertos"] - p["abiertos"]
    sin_proxima_previa = sum(e.get("sin_proxima_actividad", 0) for e in anterior.get("etapas", []))
    return {
        "fecha": anterior["fecha"],
        "abiertos": diag["total_abiertos"] - anterior.get("total_abiertos", 0),
        "sin_proxima": diag["sin_proxima_total"] - sin_proxima_previa,
        "por_etapa": por_etapa,
    }


# --------------------------------------------------------------------------- hallazgos por reglas
def analizar(diag, cfg):
    """Devuelve (hallazgos, recomendaciones) ordenados por importancia."""
    reglas = []  # (peso, hallazgo, recomendación o None)
    etapas = diag["etapas"]
    periodo = diag["periodo_dias"]
    activas = [e for e in etapas if 0 < e["probabilidad"] < 0.9 and e["abiertos"] >= 5]

    if activas:
        e = max(activas, key=lambda x: x["sin_proxima_actividad"])
        if e["sin_proxima_actividad"] >= 5:
            pct = round(100 * e["sin_proxima_actividad"] / e["abiertos"])
            reglas.append(
                (
                    e["sin_proxima_actividad"],
                    f"{e['sin_proxima_actividad']} de los {e['abiertos']} negocios en «{e['nombre']}» no tienen "
                    f"próxima actividad agendada ({pct}%).",
                    f"Agenda una próxima actividad en cada negocio de «{e['nombre']}»; el Desatascador propone la "
                    "acción negocio por negocio más abajo.",
                )
            )
        lentas = [x for x in activas if x["mediana_dias"] >= 7]
        if lentas:
            e = max(lentas, key=lambda x: x["abiertos"] * x["mediana_dias"])
            reglas.append(
                (
                    e["abiertos"] * e["mediana_dias"] / 10,
                    f"«{e['nombre']}» acumula {e['abiertos']} negocios con una mediana de {e['mediana_dias']:.0f} días en la etapa.",
                    f"Revisa «{e['nombre']}» negocio por negocio: reactivar con una acción concreta o cerrar como "
                    "perdido con su razón.",
                )
            )

    pasos = [e for e in etapas if e["probabilidad"] > 0]
    caidas = [(p, a) for p, a in zip(pasos, pasos[1:]) if a["conversion"] is not None and p["alcanzaron"] >= 20]
    if caidas:
        previa, actual = min(caidas, key=lambda par: par[1]["conversion"])
        perdidos = previa["alcanzaron"] - actual["alcanzaron"]
        reglas.append(
            (
                perdidos / 10,
                f"Solo el {actual['conversion']:.0f}% de los negocios que llegan a «{previa['nombre']}» pasan a "
                f"«{actual['nombre']}»: {perdidos} se quedaron en el camino en {periodo} días.",
                f"Revisa qué pasa entre «{previa['nombre']}» y «{actual['nombre']}»: es el paso del embudo donde más se pierde.",
            )
        )
        avanzadas = [par for par in caidas if par[0]["probabilidad"] >= 0.3 and par[0] is not previa]
        if avanzadas:
            p2, a2 = min(avanzadas, key=lambda par: par[1]["conversion"])
            reglas.append(
                (
                    (p2["alcanzaron"] - a2["alcanzaron"]) / 5,
                    f"En etapas avanzadas, el salto más débil es de «{p2['nombre']}» a «{a2['nombre']}»: "
                    f"{a2['conversion']:.0f}% de conversión.",
                    f"Protege el paso de «{p2['nombre']}» a «{a2['nombre']}» con confirmaciones, recordatorios y "
                    "seguimiento el mismo día: ahí se pierde valor que ya costó conseguir.",
                )
            )

    if diag["razones_perdida"] and diag["perdidos_periodo"]:
        razon, n = diag["razones_perdida"][0]
        pct = round(100 * n / diag["perdidos_periodo"])
        if pct >= 25:
            if _es_no_respuesta(razon):
                rec = (
                    "Acorta el primer contacto: la mayoría de las pérdidas son leads que nunca respondieron. "
                    "Llamada y WhatsApp en la primera hora, y reasignación del negocio a las 48 horas."
                )
            else:
                rec = f"Analiza los negocios perdidos por «{razon}» para ajustar la calificación de entrada o el discurso comercial."
            reglas.append(
                (pct / 2, f"«{razon}» explica el {pct}% de las pérdidas de los últimos {periodo} días ({n} negocios).", rec)
            )

    administrativas = [e for e in etapas if 0.9 <= e["probabilidad"] < 1 and e["abiertos"] > 0]
    total_adm = sum(e["abiertos"] for e in administrativas)
    if total_adm >= 10:
        peor = max(administrativas, key=lambda x: x["mediana_dias"])
        nombres = ", ".join(f"«{e['nombre']}»" for e in sorted(administrativas, key=lambda x: -x["abiertos"])[:3])
        reglas.append(
            (
                total_adm / 4,
                f"{total_adm} negocios siguen abiertos en etapas de cierre administrativo ({nombres}); en "
                f"«{peor['nombre']}» la mediana es de {peor['mediana_dias']:.0f} días.",
                "Marca como ganados los negocios ya pagados o muévelos a un pipeline de postventa: inflan el "
                "pipeline abierto y distorsionan el pronóstico.",
            )
        )

    if diag["propietarios"]:
        p = diag["propietarios"][0]
        if p["sin_proxima"] >= 10 and p["id"]:
            reglas.append(
                (
                    p["sin_proxima"] / 3,
                    f"{p['nombre']} concentra {p['sin_proxima']} negocios sin próxima actividad (de {p['abiertos']} abiertos).",
                    f"Revisa la cartera de {p['nombre']} en la próxima reunión de pipeline.",
                )
            )

    comp = diag.get("comparacion")
    if comp:
        reglas.append(
            (
                0,
                f"Frente a la foto del {comp['fecha'][:10]}: {_signo(comp['abiertos'])} negocios abiertos y "
                f"{_signo(comp['sin_proxima'])} sin próxima actividad.",
                None,
            )
        )

    reglas.sort(key=lambda r: -r[0])
    hallazgos = [h for _, h, _ in reglas][:7]
    recomendaciones = []
    for _, _, rec in reglas:
        if rec and rec not in recomendaciones:
            recomendaciones.append(rec)
        if len(recomendaciones) == 3:
            break
    if not hallazgos:
        hallazgos = ["No se detectaron acumulaciones relevantes: el pipeline se está moviendo."]
    return hallazgos, recomendaciones
