"""Agente Analista: puntúa y segmenta cada descarga, y resume tendencias."""

import json
import re
from collections import Counter
from datetime import timedelta

from .db import ahora_utc, desde_iso
from .gmail import normalizar

AGENTE = "Analista"

PROVEEDORES_GRATUITOS = {
    "gmail", "googlemail", "hotmail", "outlook", "live", "msn", "yahoo", "ymail",
    "icloud", "me", "mac", "aol", "proton", "protonmail", "gmx", "zoho", "mail", "yandex",
}

# El orden importa: la primera coincidencia gana (estudiante antes que gerente).
NIVELES_CARGO = [
    ("Estudiante", 0, r"\b(estudiante|alumn[oa]|practicante|tesista|egresad[oa]|student)\b"),
    ("Decisor", 30, r"\b(ceo|cfo|coo|cto|cio|gerente|director[a]?|due[nñ]o|fundador[a]?|founder|co-?founder|socio|socia|partner|presidente|owner|propietari[oa]|vp|vicepresidente|head|country manager|administrador[a]? general)\b"),
    ("Jefatura", 20, r"\b(jefe|jefa|subgerente|manager|lider|lead|coordinador[a]?|supervisor[a]?|encargad[oa]|responsable)\b"),
    ("Especialista", 10, r"\b(analista|controller|control|contador[a]?|contable|ingenier[oa]|consultor[a]?|asesor[a]?|especialista|ejecutiv[oa]|asistente|tesorer[oa]|auditor[a]?|planner|financ)"),
]

MOTIVOS = [
    ("Necesidad en su empresa", 25, r"(mi empresa|empresa|negocio|implementar|mejorar|equipo|pyme|compan)"),
    ("Consultor / asesor", 10, r"(consultor|asesor|aliad|partner)"),
    ("Aprendizaje", 3, r"(aprend|estudi|curso|conocimiento|curiosidad|personal|universidad)"),
]

EMPRESA_VACIA = {"", "-", "na", "n/a", "no", "ninguna", "ninguno", "none", "x", "xx", "test", "prueba"}
EMPRESA_INDEPENDIENTE = {"independiente", "freelance", "particular", "personal", "propia", "emprendimiento"}

PREFIJOS_PAIS = [
    ("593", "Ecuador"), ("591", "Bolivia"), ("595", "Paraguay"), ("598", "Uruguay"),
    ("506", "Costa Rica"), ("507", "Panamá"), ("502", "Guatemala"), ("503", "El Salvador"),
    ("504", "Honduras"), ("505", "Nicaragua"), ("1809", "Rep. Dominicana"),
    ("56", "Chile"), ("51", "Perú"), ("52", "México"), ("54", "Argentina"), ("57", "Colombia"),
    ("58", "Venezuela"), ("34", "España"), ("1", "EE.UU./Canadá"),
]
TLD_PAIS = {
    "cl": "Chile", "pe": "Perú", "mx": "México", "ar": "Argentina", "co": "Colombia",
    "ec": "Ecuador", "bo": "Bolivia", "py": "Paraguay", "uy": "Uruguay", "ve": "Venezuela",
    "es": "España", "cr": "Costa Rica", "pa": "Panamá", "gt": "Guatemala", "do": "Rep. Dominicana",
}


def es_correo_corporativo(correo):
    dominio = correo.rsplit("@", 1)[-1].lower()
    return dominio.split(".")[0] not in PROVEEDORES_GRATUITOS


def nivel_cargo(cargo):
    texto = normalizar(cargo)
    for nombre, puntos, patron in NIVELES_CARGO:
        if texto and re.search(patron, texto):
            return nombre, puntos
    return ("Sin informar", 0) if not texto else ("Otro", 5)


def tipo_motivo(motivo):
    texto = normalizar(motivo)
    for nombre, puntos, patron in MOTIVOS:
        if texto and re.search(patron, texto):
            return nombre, puntos
    return "Otro", 5


def pais_de(telefono, correo):
    crudo = (telefono or "").strip()
    digitos = re.sub(r"\D", "", crudo)
    if crudo.startswith("+") or crudo.startswith("00"):
        digitos = digitos[2:] if crudo.startswith("00") else digitos
        for prefijo, pais in PREFIJOS_PAIS:
            if digitos.startswith(prefijo):
                return pais
    elif len(digitos) == 11 and digitos.startswith("569"):
        return "Chile"
    tld = correo.rsplit(".", 1)[-1].lower()
    return TLD_PAIS.get(tld, "Sin identificar")


def nombre_de_pila(nombre):
    partes = (nombre or "").strip().split()
    return partes[0].capitalize() if partes else ""


def puntuar(lead, cfg, ahora=None):
    """Devuelve (score 0-100, segmento, nivel_cargo, país, detalle[(motivo, puntos)])."""
    ahora = ahora or ahora_utc()
    detalle = []

    if es_correo_corporativo(lead["email"]):
        detalle.append(("Correo corporativo", 20))

    nivel, pts = nivel_cargo(lead.get("cargo", ""))
    if pts:
        detalle.append((f"Cargo: {nivel}", pts))

    empresa = normalizar(lead.get("empresa", ""))
    if empresa in EMPRESA_INDEPENDIENTE:
        detalle.append(("Trabaja de forma independiente", 5))
    elif empresa not in EMPRESA_VACIA and len(empresa) >= 3:
        detalle.append(("Informó su empresa", 10))

    motivo, pts = tipo_motivo(lead.get("motivo", ""))
    detalle.append((f"Motivo: {motivo}", pts))

    if re.sub(r"\D", "", lead.get("telefono", "")):
        detalle.append(("Dejó teléfono (WhatsApp)", 5))

    pais = pais_de(lead.get("telefono", ""), lead["email"])
    if pais in cfg.mercado_objetivo:
        detalle.append((f"Mercado objetivo ({pais})", 10))

    if (lead.get("num_descargas") or 1) >= 2:
        detalle.append(("Descargó más de una vez", 10))

    if lead.get("respondio_en"):
        detalle.append(("Respondió un correo", 25))

    ultima = desde_iso(lead.get("ultima_descarga"))
    if ultima and ahora - ultima <= timedelta(days=3):
        detalle.append(("Descarga reciente (≤3 días)", 5))

    score = max(0, min(100, sum(p for _, p in detalle)))
    if score >= cfg.umbral_caliente:
        segmento = "Caliente"
    elif score >= cfg.umbral_tibio:
        segmento = "Tibio"
    else:
        segmento = "Frío"
    return score, segmento, nivel, pais, detalle


def lectura_basica(lead, segmento, nivel, pais):
    """Lectura comercial sin IA, a partir de reglas."""
    partes = []
    quien = lead.get("cargo") or "Cargo no informado"
    if lead.get("empresa"):
        quien += f" en {lead['empresa']}"
    partes.append(f"{quien} ({pais}).")
    motivo, _ = tipo_motivo(lead.get("motivo", ""))
    if motivo == "Necesidad en su empresa":
        partes.append("Declara una necesidad concreta en su empresa: buen candidato para un diagnóstico.")
        angulo = "Ofrecer un diagnóstico de su análisis financiero actual y mostrar 2-3 mejoras rápidas."
    elif motivo == "Consultor / asesor":
        partes.append("Es consultor/asesor: puede ser aliado, referidor o cliente de herramientas.")
        angulo = "Explorar alianza o referidos; mostrar cómo apoyamos a asesores con sus clientes."
    elif motivo == "Aprendizaje":
        partes.append("Descargó para aprender: nutrir con contenido antes de proponer reunión.")
        angulo = "Compartir contenido útil y medir interés antes de invitar a una reunión."
    else:
        partes.append("Motivo poco específico: conviene preguntar qué busca resolver.")
        angulo = "Preguntar qué desafío financiero quiere resolver y ofrecer una conversación breve."
    if nivel == "Decisor":
        partes.append("Tiene poder de decisión.")
    if (lead.get("num_descargas") or 1) >= 2:
        partes.append(f"Descargó {lead['num_descargas']} veces: interés alto.")
    if segmento == "Caliente":
        angulo = "Prioridad alta: contactar hoy. " + angulo
    return " ".join(partes), angulo


def analizar(db, cfg, llm=None):
    """Recalcula score/segmento de todos los leads. Devuelve los leads que cambiaron a Caliente."""
    nuevos_calientes = []
    usados_ia = 0
    for lead in db.leads():
        score, segmento, nivel, pais, detalle = puntuar(lead, cfg)
        campos = {
            "score": score,
            "segmento": segmento,
            "nivel_cargo": nivel,
            "pais": pais,
            "score_detalle": json.dumps(detalle, ensure_ascii=False),
        }
        if not lead.get("lectura_ia"):
            lectura, angulo = lectura_basica(lead, segmento, nivel, pais)
            campos.update(lectura=lectura, angulo=angulo)
            if (
                llm is not None
                and segmento in ("Caliente", "Tibio")
                and usados_ia < cfg.llm_max_leads_por_ciclo
            ):
                resultado = llm.analizar_lead({**lead, **campos}, detalle)
                usados_ia += 1
                if resultado:
                    campos.update(lectura=resultado["lectura"], angulo=resultado["angulo"], lectura_ia=1)
        if segmento == "Caliente" and lead.get("segmento") != "Caliente":
            nuevos_calientes.append(lead["id"])
            db.log(AGENTE, "Lead caliente", f"{lead['nombre']} <{lead['email']}> · score {score}", lead["id"])
        db.actualizar_lead(lead["id"], **campos)
    return nuevos_calientes


def insights(leads, descargas, ahora=None, tz=None):
    """Hallazgos en lenguaje simple para el tablero y el resumen por correo."""
    ahora = ahora or ahora_utc()
    if not leads:
        return ["Aún no hay descargas registradas."]
    hallazgos = []
    fechas = [desde_iso(d["fecha"]) for d in descargas]
    ult7 = sum(1 for f in fechas if f and ahora - f <= timedelta(days=7))
    prev7 = sum(1 for f in fechas if f and timedelta(days=7) < ahora - f <= timedelta(days=14))
    if prev7:
        variacion = (ult7 - prev7) / prev7 * 100
        tendencia = "más" if variacion >= 0 else "menos"
        hallazgos.append(f"{ult7} descargas en los últimos 7 días ({abs(variacion):.0f}% {tendencia} que la semana anterior).")
    else:
        hallazgos.append(f"{ult7} descargas en los últimos 7 días.")

    total = len(leads)
    corporativos = sum(1 for l in leads if es_correo_corporativo(l["email"]))
    hallazgos.append(f"{corporativos / total:.0%} de los leads usa correo corporativo.")

    decisores = sum(1 for l in leads if l.get("nivel_cargo") == "Decisor")
    if decisores:
        hallazgos.append(f"{decisores} leads ({decisores / total:.0%}) tienen cargo de decisión (gerente, socio, fundador…).")

    motivos = Counter(tipo_motivo(l.get("motivo", ""))[0] for l in leads)
    motivo, n = motivos.most_common(1)[0]
    hallazgos.append(f"Motivo más frecuente: «{motivo}» ({n / total:.0%}).")

    paises = Counter(l.get("pais") or "Sin identificar" for l in leads if l.get("pais") != "Sin identificar")
    if paises:
        pais, n = paises.most_common(1)[0]
        hallazgos.append(f"País con más descargas identificadas: {pais} ({n}).")

    calientes = [l for l in leads if l.get("segmento") == "Caliente"]
    sin_contacto = [l for l in calientes if l.get("etapa") == "Nuevo"]
    if sin_contacto:
        hallazgos.append(f"{len(sin_contacto)} leads calientes aún no han sido contactados: priorízalos hoy.")

    horas = Counter(f.astimezone(tz).hour for f in fechas if f)
    if len(fechas) >= 10:
        hora, _ = horas.most_common(1)[0]
        hallazgos.append(f"La franja con más descargas es alrededor de las {hora:02d}:00 h.")
    return hallazgos
