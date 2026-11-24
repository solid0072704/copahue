"""Servicio web que recibe el webhook del workflow de HubSpot y reparte el lead.

Endpoints:
    GET  /salud                      estado del servicio (sin datos de leads)
    POST /webhooks/hubspot/lead      lo llama el workflow "Send webhook" con el contacto calificado
    POST /tareas/sla                 revisión del SLA bajo demanda (para un cron externo)

Ejecuta con un solo proceso (la revisión del SLA corre en un hilo dentro del servicio):
    python run_central.py servir --puerto 8000
"""

import hmac
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from .config import ConfigCentral
from .distribuidor import construir
from .hubspot import ErrorHubSpot
from .sla import revisar_sla

log = logging.getLogger(__name__)


def _valor(x):
    """Las propiedades pueden venir como 'valor' o como {'value': 'valor'} (formato antiguo)."""
    if isinstance(x, dict):
        x = x.get("value")
    return "" if x is None else str(x).strip()


def extraer_contacto_id(cuerpo, query=None):
    query = query or {}
    candidatos = [query.get("contacto"), query.get("contactId")]
    if isinstance(cuerpo, list) and cuerpo:
        cuerpo = cuerpo[0]
    if isinstance(cuerpo, dict):
        for clave in ("objectId", "contactId", "contacto_id", "hs_object_id", "vid", "id"):
            candidatos.append(_valor(cuerpo.get(clave)))
        props = cuerpo.get("properties")
        if isinstance(props, dict):
            candidatos.append(_valor(props.get("hs_object_id")))
    for c in candidatos:
        texto = _valor(c)
        if texto.isdigit():
            return texto
    return ""


def extraer_proyecto(cuerpo, query=None):
    query = query or {}
    if query.get("proyecto"):
        return str(query["proyecto"]).strip()
    if isinstance(cuerpo, list) and cuerpo:
        cuerpo = cuerpo[0]
    if isinstance(cuerpo, dict):
        if cuerpo.get("proyecto"):
            return _valor(cuerpo["proyecto"])
        props = cuerpo.get("properties")
        if isinstance(props, dict) and props.get("proyecto_de_interes"):
            return _valor(props["proyecto_de_interes"])
        if cuerpo.get("proyecto_de_interes"):
            return _valor(cuerpo["proyecto_de_interes"])
    return None


def _bucle_sla(dist, cada_min, parar):
    parar.wait(30)  # deja arrancar el servicio
    while not parar.is_set():
        try:
            revisar_sla(dist)
        except Exception as e:  # el hilo nunca debe morir
            log.warning("SLA: la revisión falló (%s)", type(e).__name__)
        parar.wait(cada_min * 60)


def crear_app(cfg=None, distribuidor=None, iniciar_sla=None):
    cfg = cfg or ConfigCentral.desde_entorno()
    dist = distribuidor or construir(cfg)
    if iniciar_sla is None:
        iniciar_sla = cfg.sla_en_servidor and not cfg.simular

    @asynccontextmanager
    async def ciclo_de_vida(app):
        parar = threading.Event()
        if iniciar_sla:
            threading.Thread(target=_bucle_sla, args=(dist, cfg.sla_cada_min, parar), daemon=True).start()
            log.info("Revisión del SLA activa cada %s minutos", cfg.sla_cada_min)
        yield
        parar.set()

    app = FastAPI(title="Central WhatsApp · distribuidor de leads", lifespan=ciclo_de_vida, docs_url=None, redoc_url=None)
    app.state.distribuidor = dist

    def autorizar(request):
        if not cfg.webhook_secreto:
            raise HTTPException(503, "Falta configurar CENTRAL_WEBHOOK_SECRET en el servicio.")
        enviado = request.headers.get(cfg.webhook_header) or request.query_params.get("token") or ""
        if not hmac.compare_digest(enviado.encode(), cfg.webhook_secreto.encode()):
            raise HTTPException(401, "Token inválido.")

    @app.get("/salud")
    def salud():
        return {
            "ok": True,
            "proyectos": len(dist.tabla.proyectos),
            "sla_minutos": dist.tabla.sla_minutos,
            "simulacion": cfg.simular,
            "webhook_protegido": bool(cfg.webhook_secreto),
        }

    @app.post("/webhooks/hubspot/lead")
    async def webhook_lead(request: Request):
        autorizar(request)
        try:
            cuerpo = await request.json()
        except ValueError:
            cuerpo = {}
        contacto_id = extraer_contacto_id(cuerpo, request.query_params)
        if not contacto_id:
            raise HTTPException(400, "No vino el ID del contacto (objectId / hs_object_id).")
        proyecto = extraer_proyecto(cuerpo, request.query_params)
        try:
            resultado = await run_in_threadpool(dist.procesar, contacto_id, proyecto)
        except ErrorHubSpot as e:
            log.warning("Contacto %s: HubSpot respondió %s (%s)", contacto_id, e.status, e.mensaje)
            if e.reintentable:
                # 503 hace que HubSpot reintente el webhook más tarde
                raise HTTPException(503, "HubSpot no está disponible o el token no tiene permisos; se reintentará.")
            raise HTTPException(404 if e.status == 404 else 422, f"HubSpot rechazó la operación: {e.mensaje}")
        return resultado.como_dict()

    @app.post("/tareas/sla")
    async def tarea_sla(request: Request):
        autorizar(request)
        resultado = await run_in_threadpool(revisar_sla, dist)
        return resultado.como_dict()

    return app
