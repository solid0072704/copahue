"""Agente Ejecutor: registra tus decisiones y ejecuta SOLO las órdenes que aprobaste.

Antes de enviar cada orden al broker vuelve a revisar: que no haya vencido, que el precio
no se haya movido demasiado desde que la viste y que siga cumpliendo los límites de riesgo.
"""

from datetime import datetime, time, timedelta

from . import guardia
from .brokers import ErrorBroker
from .db import ahora_utc, desde_iso, iso
from .estratega import valorizar
from .mercado import ErrorMercado

AGENTE = "Ejecutor"


def guardar_propuestas(db, cfg, plan_id, ordenes, valor, ahora=None):
    """Pasa cada orden por la Guardia y la guarda. Devuelve las que quedan esperando tu decisión."""
    ahora = ahora or ahora_utc()
    vence = iso(ahora + timedelta(hours=cfg.vigencia_propuesta_horas))
    liberado = sum(o["cantidad"] * o["precio_ref"] for o in ordenes if o["lado"] == "venta")
    pendientes = []
    for o in ordenes:
        motivos = guardia.evaluar(o, valor, cfg, db, efectivo_extra=liberado if o["lado"] == "compra" else 0.0)
        estado = "bloqueada" if motivos else "pendiente"
        prop_id = db.crear_propuesta(
            plan_id=plan_id, tipo=o["tipo"], simbolo=o["simbolo"], lado=o["lado"], cantidad=o["cantidad"],
            precio_ref=o["precio_ref"], monto=round(o["cantidad"] * o["precio_ref"], 2), motivo=o["motivo"],
            riesgo=" ".join(motivos), estado=estado, creada_en=iso(ahora), vence_en=vence,
        )
        if motivos:
            db.log(guardia.AGENTE, "Orden bloqueada", f"{o['lado']} {o['simbolo']}: {' '.join(motivos)}", prop_id)
        else:
            pendientes.append(db.propuesta(prop_id))
    return pendientes


def decidir(db, prop_id, aprobar, por):
    """Registra tu decisión. Devuelve (ok, mensaje)."""
    prop = db.propuesta(prop_id)
    if not prop:
        return False, f"La propuesta #{prop_id} no existe."
    if prop["estado"] != "pendiente":
        return False, f"La propuesta #{prop_id} ya no está pendiente ({prop['estado']})."
    if desde_iso(prop["vence_en"]) < ahora_utc():
        db.cambiar_estado(prop_id, "pendiente", "vencida")
        return False, f"La propuesta #{prop_id} venció. El agente te propondrá un plan actualizado."
    hacia = "aprobada" if aprobar else "rechazada"
    if not db.cambiar_estado(prop_id, "pendiente", hacia, decidida_en=iso(ahora_utc()), decidida_por=por):
        return False, f"La propuesta #{prop_id} ya fue decidida."
    db.log(AGENTE, "Aprobada por ti" if aprobar else "Rechazada por ti", f"vía {por}", prop_id)
    return True, f"Propuesta #{prop_id} {'aprobada' if aprobar else 'rechazada'}."


def vencer(db, ahora=None):
    ahora = ahora or ahora_utc()
    vencidas = []
    for p in db.propuestas(["pendiente", "aprobada"]):
        if desde_iso(p["vence_en"]) < ahora and db.cambiar_estado(p["id"], p["estado"], "vencida"):
            vencidas.append(db.propuesta(p["id"]))
            db.log(AGENTE, "Propuesta vencida", f"{p['lado']} {p['simbolo']}", p["id"])
    return vencidas


def _inicio_dia_utc(cfg):
    hoy = datetime.now(cfg.tz).date()
    return iso(datetime.combine(hoy, time.min, tzinfo=cfg.tz))


def ejecutar(db, cfg, broker, mercado, prop_id):
    """Ejecuta una propuesta aprobada. Devuelve la propuesta actualizada."""
    if not db.cambiar_estado(prop_id, "aprobada", "ejecutando", ejecutada_en=iso(ahora_utc())):
        return db.propuesta(prop_id)  # ya la tomó otro proceso o no está aprobada
    prop = db.propuesta(prop_id)

    def fallar(texto, estado="fallida"):
        db.actualizar_propuesta(prop_id, estado=estado, error=texto)
        db.log(AGENTE, "Orden no ejecutada", texto, prop_id)
        return db.propuesta(prop_id)

    if desde_iso(prop["vence_en"]) < ahora_utc():
        return fallar("La propuesta venció antes de ejecutarse.", "vencida")
    try:
        mercado.refrescar()
        precio = mercado.precio(prop["simbolo"])
        valor = valorizar(broker.cuenta(), mercado)
    except (ErrorMercado, ErrorBroker) as e:
        return fallar(str(e))
    ordenes_hoy = db.ejecutadas_desde(_inicio_dia_utc(cfg)) - 1  # sin contar esta
    motivos = guardia.evaluar(prop, valor, cfg, db, ordenes_hoy=ordenes_hoy, precio_actual=precio)
    if motivos:
        return fallar("Bloqueada por la Guardia al ejecutar: " + " ".join(motivos), "bloqueada")
    try:
        res = broker.ejecutar(prop)
    except (ErrorBroker, ErrorMercado) as e:
        return fallar(str(e))
    _aplicar_resultado(db, prop_id, res)
    return db.propuesta(prop_id)


def _aplicar_resultado(db, prop_id, res):
    campos = {"estado": res["estado"], "id_broker": res.get("id_broker", "")}
    if res["estado"] == "ejecutada":
        campos.update(precio_ejecucion=res.get("precio"), cantidad_ejecutada=res.get("cantidad"))
    if res.get("error"):
        campos["error"] = res["error"]
    db.actualizar_propuesta(prop_id, **campos)
    p = db.propuesta(prop_id)
    if res["estado"] == "ejecutada":
        db.log(AGENTE, "Orden ejecutada", f"{p['lado']} {p['cantidad_ejecutada']:.4f} {p['simbolo']} a US$ {p['precio_ejecucion']:,.2f}", prop_id)
    elif res["estado"] == "enviada":
        db.log(AGENTE, "Orden enviada al broker", f"{p['lado']} {p['simbolo']} (id {p['id_broker']})", prop_id)


def ejecutar_aprobadas(db, cfg, broker, mercado):
    # Primero las ventas: liberan el efectivo que usan las compras.
    aprobadas = sorted(db.propuestas(["aprobada"]), key=lambda p: (p["lado"] != "venta", p["id"]))
    return [ejecutar(db, cfg, broker, mercado, p["id"]) for p in aprobadas]


def sincronizar_enviadas(db, broker):
    """Consulta al broker las órdenes que quedaron en curso (p. ej. enviadas con el mercado cerrado)."""
    cambiadas = []
    for p in db.propuestas(["enviada"]):
        try:
            res = broker.estado_orden(p)
        except ErrorBroker as e:
            db.log(AGENTE, "Error consultando orden", str(e), p["id"])
            continue
        if res["estado"] != "enviada":
            _aplicar_resultado(db, p["id"], res)
            cambiadas.append(db.propuesta(p["id"]))
    return cambiadas
