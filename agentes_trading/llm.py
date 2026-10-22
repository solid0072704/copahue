"""IA opcional (Claude): da una segunda opinión sobre cada orden y te la explica en simple.

La IA NO puede crear órdenes, cambiar cantidades ni saltarse la Guardia de riesgo:
solo agrega su opinión al mensaje en el que el agente te consulta. Sin ANTHROPIC_API_KEY,
el sistema funciona igual con las explicaciones del Estratega.
"""

import json
import logging

log = logging.getLogger(__name__)

SISTEMA = (
    "Eres el analista de inversiones de un agente que administra la cartera personal de una persona. "
    "El agente propone órdenes con una estrategia de cartera diversificada (ETFs), filtro de tendencia "
    "y rebalanceo, y la persona aprueba o rechaza cada una. Tu trabajo es dar una segunda opinión honesta "
    "y explicar cada orden en español simple, en 2-3 frases, para alguien sin formación financiera. "
    "Usa solo los datos entregados: nunca inventes noticias, precios, cifras ni predicciones, y no prometas "
    "rentabilidades. Si una orden te parece poco conveniente (p. ej. costos frente a un monto muy pequeño, "
    "o va contra la tendencia), dilo con claridad."
)

ESQUEMA = {
    "type": "object",
    "properties": {
        "ordenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "opinion": {"type": "string", "enum": ["a_favor", "neutral", "en_contra"]},
                    "explicacion": {"type": "string", "description": "2-3 frases en español simple."},
                },
                "required": ["id", "opinion", "explicacion"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ordenes"],
    "additionalProperties": False,
}

ETIQUETA_OPINION = {"a_favor": "👍 A favor", "neutral": "🤔 Neutral", "en_contra": "👎 En contra"}


class AnalistaIA:
    def __init__(self, cfg):
        import anthropic

        self.cfg = cfg
        self.anthropic = anthropic
        self.cliente = anthropic.Anthropic(api_key=cfg.anthropic_api_key, max_retries=2, timeout=90)

    def revisar(self, ordenes, valor, indicadores):
        """ordenes: lista de dicts con 'id'. Devuelve {id: {'opinion', 'explicacion'}}."""
        datos = {
            "perfil_riesgo": self.cfg.perfil,
            "cartera": {
                "valor_total_usd": round(valor["total"], 2),
                "efectivo_usd": round(valor["efectivo"], 2),
                "posiciones": {s: round(p["peso"], 1) for s, p in valor["posiciones"].items()},
            },
            "indicadores": indicadores,
            "ordenes": [
                {k: o[k] for k in ("id", "tipo", "simbolo", "lado", "cantidad", "precio_ref", "monto", "motivo")}
                for o in ordenes
            ],
        }
        prompt = (
            "Revisa estas órdenes propuestas y da tu opinión sobre cada una.\n"
            f"<datos>{json.dumps(datos, ensure_ascii=False, default=lambda x: round(x, 2))}</datos>"
        )
        try:
            respuesta = self.cliente.beta.messages.create(
                model=self.cfg.llm_modelo,
                max_tokens=4000,
                system=SISTEMA,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": ESQUEMA}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except self.anthropic.APIStatusError as e:
            log.warning("Claude respondió con error %s: %s", e.status_code, e.message)
            return {}
        except self.anthropic.APIConnectionError as e:
            log.warning("Sin conexión con Claude: %s", e)
            return {}
        if respuesta.stop_reason in ("refusal", "max_tokens"):
            return {}
        texto = next((b.text for b in respuesta.content if b.type == "text"), "")
        try:
            items = json.loads(texto).get("ordenes", [])
        except (json.JSONDecodeError, AttributeError):
            return {}
        ids = {o["id"] for o in ordenes}
        return {i["id"]: i for i in items if i.get("id") in ids and i.get("opinion") in ETIQUETA_OPINION}


def crear_analista(cfg):
    if not cfg.ia_configurada:
        return None
    try:
        return AnalistaIA(cfg)
    except ImportError:
        log.warning("Falta el paquete 'anthropic' (pip install anthropic).")
        return None
