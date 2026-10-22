"""Orquestador: ejecuta a los agentes de trading en orden en cada ciclo.

Ciclo: sincroniza órdenes → lee tus respuestas → vence propuestas viejas → ejecuta lo aprobado
→ valoriza la cartera → revisa stops → propone un plan nuevo si corresponde → te consulta.
"""

import logging
from dataclasses import dataclass, field
from datetime import timedelta

from . import ejecutor, estratega, guardia, notificador
from .brokers import BrokerSimulado, ErrorBroker, crear_broker
from .db import ESTADOS_ABIERTOS, BaseDatos, ahora_utc, desde_iso, iso
from .llm import crear_analista
from .mercado import ErrorMercado, MercadoSintetico, crear_mercado, indicadores

log = logging.getLogger(__name__)
AGENTE = "Orquestador"


@dataclass
class ResultadoCiclo:
    decisiones: list = field(default_factory=list)
    ejecutadas: list = field(default_factory=list)
    vencidas: list = field(default_factory=list)
    nuevas: list = field(default_factory=list)
    plan: str = ""
    valor_total: float = None
    errores: list = field(default_factory=list)

    def texto(self):
        partes = []
        if self.valor_total is not None:
            partes.append(f"cartera US$ {self.valor_total:,.2f}")
        partes += [
            f"{len(self.decisiones)} decisiones tuyas",
            f"{sum(1 for p in self.ejecutadas if p['estado'] == 'ejecutada')} órdenes ejecutadas",
            f"{len(self.nuevas)} propuestas nuevas",
        ]
        if self.vencidas:
            partes.append(f"{len(self.vencidas)} vencidas")
        if self.plan:
            partes.append(self.plan)
        if self.errores:
            partes.append("errores: " + "; ".join(self.errores))
        return " · ".join(partes)


def abrir_db(cfg):
    return BaseDatos(cfg.ruta_db)


def preparar(cfg, db, mercado=None, broker=None, tg=None):
    """Crea (o reutiliza) las conexiones de un ciclo. Una base de demostración siempre usa precios sintéticos."""
    demo = db.get_meta("demo") == "1"
    mercado = mercado or (MercadoSintetico() if demo else crear_mercado(cfg))
    broker = broker or (BrokerSimulado(db, mercado, cfg.comision_simulada_pct) if demo else crear_broker(cfg, db, mercado))
    tg = tg if tg is not None else notificador.crear_telegram(cfg)
    return mercado, broker, tg


def _texto_resultado(prop):
    if prop["estado"] == "ejecutada":
        return (
            f"✅ <b>Ejecutada</b>: {prop['lado']} {prop['cantidad_ejecutada']:.4f} {prop['simbolo']} "
            f"a US$ {prop['precio_ejecucion']:,.2f}"
        )
    if prop["estado"] == "enviada":
        return f"📤 <b>Enviada al broker</b> (se ejecuta cuando abra el mercado): {prop['lado']} {prop['simbolo']}"
    return f"⚠️ <b>No se ejecutó</b> {prop['lado']} {prop['simbolo']}: {prop['error']}"


def procesar_telegram(db, cfg, broker, mercado, tg, r):
    try:
        decisiones, comandos = notificador.leer_respuestas(db, tg)
    except notificador.ErrorTelegram as e:
        r.errores.append(str(e))
        return
    for prop_id, accion, callback_id in decisiones:
        ok, mensaje = ejecutor.decidir(db, prop_id, accion == "si", "Telegram")
        tg.responder_boton(callback_id, mensaje)
        prop = db.propuesta(prop_id)
        if not prop:
            continue
        r.decisiones.append(prop)
        if ok and prop.get("telegram_msg_id"):
            estado = "👍 Aprobaste esta orden. Ejecutando…" if accion == "si" else "❌ Rechazaste esta orden."
            tg.editar(prop["telegram_msg_id"], notificador.texto_propuesta(prop, cfg, estado))
    for comando in comandos:
        if comando in ("/cartera", "/estado", "/start"):
            try:
                valor = estratega.valorizar(broker.cuenta(), mercado)
                notificador.avisar(db, cfg, tg, notificador.texto_cartera(valor, db.total_aportes(), cfg))
            except (ErrorBroker, ErrorMercado) as e:
                notificador.avisar(db, cfg, tg, f"No pude leer tu cartera: {e}")
        elif comando == "/plan":
            db.set_meta("forzar_plan", "1")


def toca_plan(db, cfg, valor, forzar=False, ahora=None):
    """Hay plan nuevo si no hay propuestas abiertas y: lo pediste, pasó el intervalo o llegó dinero nuevo."""
    if db.propuestas(ESTADOS_ABIERTOS):
        return False
    if forzar or db.get_meta("forzar_plan") == "1":
        return True
    ultimo = desde_iso(db.get_meta("ultimo_plan"))
    ahora = ahora or ahora_utc()
    if not ultimo or ahora - ultimo >= timedelta(hours=cfg.plan_cada_horas):
        return True
    efectivo_antes = float(db.get_meta("efectivo_ultimo_plan", 0) or 0)
    return valor["efectivo"] - efectivo_antes >= max(cfg.min_orden_usd, valor["total"] * cfg.umbral_rebalanceo_pct / 100)


def _con_opinion_ia(db, cfg, mercado, analista, props, valor):
    if not analista or not props:
        return props
    inds = {}
    for s in {p["simbolo"] for p in props}:
        ind = indicadores(mercado, s, cfg.media_tendencia_dias)
        inds[s] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in ind.items()}
    opiniones = analista.revisar(props, valor, inds)
    for p in props:
        o = opiniones.get(p["id"])
        if o:
            db.actualizar_propuesta(p["id"], opinion_ia=o["opinion"], explicacion_ia=o["explicacion"])
    return [db.propuesta(p["id"]) for p in props]


def ejecutar_ciclo(db, cfg, mercado=None, broker=None, tg=None, usar_ia=True, forzar_plan=False):
    r = ResultadoCiclo()
    try:
        mercado, broker, tg = preparar(cfg, db, mercado, broker, tg)
    except ErrorBroker as e:
        r.errores.append(str(e))
        db.log(AGENTE, "Broker no disponible", str(e))
        return r

    # 1. Órdenes que quedaron en curso en el broker
    for p in ejecutor.sincronizar_enviadas(db, broker):
        notificador.avisar(db, cfg, tg, _texto_resultado(p))

    # 2. Tus respuestas por Telegram
    if tg:
        procesar_telegram(db, cfg, broker, mercado, tg, r)

    # 3. Propuestas que nadie respondió a tiempo
    r.vencidas = ejecutor.vencer(db)
    for p in r.vencidas:
        if tg and p.get("telegram_msg_id"):
            tg.editar(p["telegram_msg_id"], notificador.texto_propuesta(p, cfg, "⌛ Venció sin respuesta. No se ejecutó."))

    # 4. Ejecutar SOLO lo que aprobaste
    r.ejecutadas = ejecutor.ejecutar_aprobadas(db, cfg, broker, mercado)
    for p in r.ejecutadas:
        notificador.avisar(db, cfg, tg, _texto_resultado(p), asunto=None)

    # 5. Valorizar la cartera
    try:
        mercado.refrescar()
        valor = estratega.valorizar(broker.cuenta(), mercado)
    except (ErrorBroker, ErrorMercado) as e:
        r.errores.append(str(e))
        db.log(AGENTE, "Error leyendo cartera", str(e))
        return r
    aportes = db.total_aportes()
    if aportes > 0 or valor["total"] > 0:
        db.registrar_valorizacion(valor["total"], valor["efectivo"], aportes)
    r.valor_total = valor["total"]

    # 6. Freno de pérdidas (aviso una sola vez cada vez que se activa)
    freno = guardia.freno_activo(db, cfg, valor)
    if freno and db.get_meta("freno_avisado") != "1":
        notificador.avisar(
            db, cfg, tg,
            f"🛑 <b>Freno de pérdidas activado</b>: tu cartera retrocedió {cfg.freno_perdida_pct:g}% o más de lo "
            "depositado desde su máximo. Pausé las compras; las ventas de protección siguen disponibles.",
            asunto="🛑 Freno de pérdidas activado en tu cartera",
        )
        db.log(guardia.AGENTE, "Freno de pérdidas activado", f"Valor US$ {valor['total']:,.2f}")
    db.set_meta("freno_avisado", "1" if freno else "0")

    # 7. Stops de pérdida y plan nuevo
    abiertos = {p["simbolo"] for p in db.propuestas(ESTADOS_ABIERTOS)}
    ahora = ahora_utc()
    # Un stop que rechazaste no se vuelve a proponer hasta el próximo plan, y lo vendido por stop
    # no se recompra durante una semana.
    stop_reciente = db.simbolos_con_stop_desde(iso(ahora - timedelta(hours=cfg.plan_cada_horas)))
    stop_vendido = db.simbolos_con_stop_desde(iso(ahora - timedelta(days=7)), ["ejecutada", "enviada"])
    analista = crear_analista(cfg) if usar_ia else None
    try:
        plan_id = estratega.nuevo_plan_id()
        if toca_plan(db, cfg, valor, forzar_plan):
            ordenes, r.plan = estratega.generar_plan(cfg, mercado, valor, sin_compras=stop_vendido)
            db.set_meta("ultimo_plan", iso(ahora_utc()))
            db.set_meta("efectivo_ultimo_plan", valor["efectivo"])
            db.set_meta("forzar_plan", "0")
            db.log(estratega.AGENTE, "Plan generado", r.plan)
        else:
            ordenes = [
                estratega.orden_stop(p, cfg)
                for p in estratega.stops(valor, cfg)
                if p["simbolo"] not in abiertos | stop_reciente
            ]
            if ordenes:
                r.plan = f"Stop de pérdida en {', '.join(o['simbolo'] for o in ordenes)}."
        if ordenes:
            r.nuevas = ejecutor.guardar_propuestas(db, cfg, plan_id, ordenes, valor)
            r.nuevas = _con_opinion_ia(db, cfg, mercado, analista, r.nuevas, valor)
            if r.nuevas:
                notificador.consultar(db, cfg, tg, r.nuevas, r.plan)
    except ErrorMercado as e:
        r.errores.append(str(e))
        db.log(estratega.AGENTE, "Error de datos de mercado", str(e))

    db.set_meta("ultima_ejecucion", iso(ahora_utc()))
    db.log(AGENTE, "Ciclo completado", r.texto())
    return r


def preparar_demo(db, cfg, monto=10_000):
    """Cartera de práctica con precios sintéticos: deposita y deja al agente proponer (no envía nada)."""
    if db.total_aportes() != 0 and db.get_meta("demo") != "1":
        raise ValueError("Esta base ya tiene depósitos reales o de práctica; usa otra TRADING_DB para la demo.")
    db.set_meta("demo", "1")
    mercado = MercadoSintetico()
    broker = BrokerSimulado(db, mercado, cfg.comision_simulada_pct)
    if db.total_aportes() == 0:
        broker.depositar(monto, "Depósito de demostración")
    return ejecutar_ciclo(db, cfg, mercado=mercado, broker=broker, tg=False, usar_ia=False, forzar_plan=True)
