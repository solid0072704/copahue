"""Agente Calificador: compara cada prospecto con tu cliente ideal y le asigna prioridad A, B o C."""

import json
import re

from agentes_ebook.analista import EMPRESA_INDEPENDIENTE, EMPRESA_VACIA, TLD_PAIS
from agentes_ebook.gmail import normalizar

AGENTE = "Calificador"

# El orden importa: la primera coincidencia gana. Los patrones van sin tildes (el texto se normaliza).
NIVELES = [
    (
        "Decisor",
        40,
        r"\b(ceo|gerente general|gerenta general|director[a]? ejecutiv[oa]|director[a]? general|dueno|duena|"
        r"fundador[a]?|co-?fundador[a]?|founder|co-?founder|socio|socia|partner|presidente|presidenta|owner|"
        r"propietari[oa]|country manager|managing director|administrador[a]? general|emprendedor[a]?)\b",
    ),
    (
        "Finanzas",
        25,
        r"\b(cfo|finanzas|financier[oa]|finance|controller|contralor[a]?|tesorer[oa]|tesoreria|contabilidad|contador[a]?)\b",
    ),
    ("Gerencia", 20, r"\b(gerente|gerenta|director[a]?|coo|cto|cmo|vp|vicepresident[ea]|head)\b"),
    ("Jefatura", 10, r"\b(jefe|jefa|subgerente|manager|lider|lead|coordinador[a]?|encargad[oa])\b"),
]

ANGULOS = {
    "Decisor": "Decide directamente. Abre con una pregunta sobre cómo lleva hoy el flujo de caja y los reportes; no vendas en la invitación.",
    "Finanzas": "Ya hay alguien de finanzas: posiciónate como apoyo senior por horas (directorio, bancos, proyecciones), no como reemplazo.",
    "Gerencia": "No decide solo, pero puede abrirte la puerta con la gerencia general: busca la referencia interna.",
    "Jefatura": "Influye poco en la contratación: conecta, aporta contenido y pide que te presente a quien decide.",
    "Otro": "Prioridad baja: conecta solo si su empresa calza con tu cliente ideal.",
    "Sin informar": "Falta su cargo: revisa el perfil antes de escribirle.",
    "Excluido": "Fuera de tu cliente ideal (según tu lista de exclusiones).",
}


def _contiene(texto, termino):
    termino = normalizar(termino)
    return bool(termino) and re.search(rf"(?<![a-z0-9]){re.escape(termino)}(?![a-z0-9])", texto) is not None


def nivel_de(cargo):
    texto = normalizar(cargo)
    if not texto:
        return "Sin informar", 0
    for nombre, puntos, patron in NIVELES:
        if re.search(patron, texto):
            return nombre, puntos
    return "Otro", 5


def en_mercado(p, mercados):
    """¿El prospecto está en tu mercado objetivo? (por ubicación, dominio del correo o subdominio del perfil)."""
    ubicacion = normalizar(p.get("ubicacion"))
    paises = {normalizar(m) for m in mercados}
    if any(_contiene(ubicacion, m) for m in paises):
        return True
    tld = (p.get("email") or "").rsplit(".", 1)[-1].lower()
    if normalizar(TLD_PAIS.get(tld, "")) in paises:
        return True
    sub = re.match(r"https?://([a-z]{2})\.linkedin\.com", (p.get("url") or "").lower())
    return bool(sub) and normalizar(TLD_PAIS.get(sub.group(1), "")) in paises


def puntuar(p, cfgp):
    """Devuelve (score 0-100, prioridad, nivel, detalle[(motivo, puntos)])."""
    cargo = normalizar(p.get("cargo"))
    excluido = next((t for t in cfgp.excluir if _contiene(cargo, t)), None)
    if excluido:
        return 0, "C", "Excluido", [(f"Excluido por «{excluido}»", 0)]

    nivel, puntos = nivel_de(p.get("cargo"))
    detalle = [(f"Cargo: {nivel.lower()}", puntos)]
    if any(_contiene(cargo, c) for c in cfgp.cargos_objetivo):
        detalle.append(("Cargo en tu lista de cliente ideal", 10))
    empresa = normalizar(p.get("empresa"))
    if empresa and empresa not in EMPRESA_VACIA and empresa not in EMPRESA_INDEPENDIENTE:
        detalle.append(("Informa su empresa", 10))
    if en_mercado(p, cfgp.base.mercado_objetivo):
        detalle.append(("Está en tu mercado objetivo", 15))
    contexto = " ".join((normalizar(p.get("industria")), cargo, empresa))
    if any(_contiene(contexto, i) for i in cfgp.industrias_objetivo):
        detalle.append(("Industria objetivo", 10))
    if p.get("grado") == 1:
        detalle.append(("Ya es tu contacto en LinkedIn", 10))
    if p.get("lead_id"):
        detalle.append(("Descargó tu ebook", 15))

    score = min(100, sum(puntos for _, puntos in detalle))
    prioridad = "A" if score >= cfgp.umbral_a else "B" if score >= cfgp.umbral_b else "C"
    return score, prioridad, nivel, detalle


def calificar(db, cfgp):
    """Recalcula score y prioridad de todos los prospectos. Devuelve {prioridad: cantidad}."""
    conteo = {"A": 0, "B": 0, "C": 0}
    cambios = 0
    for p in db.prospectos():
        score, prioridad, nivel, detalle = puntuar(p, cfgp)
        conteo[prioridad] += 1
        nuevo = {
            "score": score,
            "prioridad": prioridad,
            "nivel": nivel,
            "score_detalle": json.dumps(detalle, ensure_ascii=False),
            "angulo": ANGULOS.get(nivel, ""),
        }
        if any(p.get(c) != v for c, v in nuevo.items()):
            db.actualizar_prospecto(p["id"], **nuevo)
            cambios += 1
    if cambios:
        db.log(AGENTE, "Prospectos calificados", f"{cambios} actualizados · A={conteo['A']} B={conteo['B']} C={conteo['C']}")
    return conteo
