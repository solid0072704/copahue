"""Agente Desatascador: encuentra negocios estancados y propone (o crea en HubSpot) la siguiente acción.

Un negocio está estancado cuando lleva más días en su etapa que el umbral de esa etapa y no tiene
ninguna actividad agendada a futuro. La acción propuesta depende de en qué parte del embudo está.
En modo `lectura` solo se informa; en modo `tareas` se crea una tarea al propietario del negocio y,
si sigue estancado una semana después, se escala al supervisor.
"""

from dataclasses import dataclass, field
from datetime import timedelta

from agentes_ebook.db import desde_iso

from .embudo import dias_en_etapa, dias_sin_contacto, sin_proxima_actividad

AGENTE = "Desatascador"

NOMBRES_CATEGORIA = {
    "negociacion": "Definir próximo paso",
    "visita": "Visita sin actualizar",
    "primer_contacto": "Primer contacto pendiente",
    "prospeccion": "Reactivar o cerrar",
    "administrativo": "Cierre administrativo",
}
# Orden en que se atienden (y se crean tareas): primero donde hay más valor en juego.
PRIORIDAD_CATEGORIA = {"negociacion": 0, "visita": 0, "primer_contacto": 1, "prospeccion": 2, "administrativo": 3}
ESTADOS = {
    "propuesta": "propuesta",
    "tarea_creada": "tarea creada",
    "tarea_reciente": "ya tiene tarea",
    "escalar": "escalar",
    "escalado": "escalado al supervisor",
}


def categoria_etapa(etapa):
    nombre = etapa["nombre"].lower()
    prob = etapa["probabilidad"]
    if prob >= 0.9:
        return "administrativo"
    if "visita" in nombre and "agend" in nombre:
        return "visita"
    if prob >= 0.3:
        return "negociacion"
    if etapa["orden"] <= 1:
        return "primer_contacto"
    return "prospeccion"


def umbral_dias(etapa, categoria, cfg):
    por_etapa = cfg.dias_por_etapa or {}
    for clave in (etapa["nombre"].lower(), str(etapa["id"]).lower()):
        if clave in por_etapa:
            return por_etapa[clave]
    return {
        "primer_contacto": cfg.dias_primer_contacto,
        "prospeccion": cfg.dias_prospeccion,
        "visita": cfg.dias_visita,
        "negociacion": cfg.dias_negociacion,
        "administrativo": cfg.dias_administrativo,
    }[categoria]


def _dias_txt(d):
    n = int(round(d))
    return f"{n} día{'s' if n != 1 else ''}"


def proponer_accion(negocio, etapa, categoria, dias, dias_sc):
    """Devuelve (acción, detalle, tipo de tarea, prioridad) según la etapa y la historia del negocio."""
    nombre = etapa["nombre"]
    contacto = f" El último contacto fue hace {_dias_txt(dias_sc)}." if dias_sc is not None else ""
    if categoria == "primer_contacto":
        intentos = negocio.get("contactos") or 0
        if intentos >= 3:
            return (
                "Último intento o cerrar como perdido",
                f"Lleva {_dias_txt(dias)} en «{nombre}» con {intentos} intentos de contacto sin respuesta. Haz un último "
                "intento por otro canal (WhatsApp o correo) y, si no responde, cierra el negocio como perdido con la "
                "razón «No responde».",
                "CALL",
                "MEDIUM",
            )
        historial = "sin ningún intento de contacto registrado" if intentos == 0 else f"con solo {intentos} intento(s)"
        return (
            "Llamar hoy: primer contacto",
            f"Lleva {_dias_txt(dias)} en «{nombre}» {historial}. Llama hoy; si no contesta, deja un WhatsApp y agenda "
            "el siguiente intento.",
            "CALL",
            "HIGH",
        )
    if categoria == "prospeccion":
        return (
            "Reactivar con una pregunta concreta o cerrar",
            f"Lleva {_dias_txt(dias)} en «{nombre}» sin próxima actividad.{contacto} Envía un mensaje de reactivación con "
            "una pregunta concreta (fecha de visita, presupuesto, forma de pago) y agenda la siguiente llamada. Si no hay "
            "respuesta en 7 días, cierra como perdido con su razón.",
            "CALL",
            "MEDIUM",
        )
    if categoria == "visita":
        return (
            "Confirmar qué pasó con la visita",
            f"La visita quedó agendada hace {_dias_txt(dias)} y el negocio no se actualizó. Confirma si se realizó: si sí, "
            "mueve la etapa y registra el resultado; si no, reagenda hoy mismo.",
            "TODO",
            "HIGH",
        )
    if categoria == "negociacion":
        paso = negocio.get("proximo_paso")
        paso_txt = f" Próximo paso registrado: «{paso}»." if paso else " No hay próximo paso registrado."
        return (
            "Acordar próximo paso y fecha con el cliente",
            f"Lleva {_dias_txt(dias)} en «{nombre}» sin próxima actividad.{paso_txt}{contacto} Llama al cliente, acuerda "
            "una acción y una fecha concreta, y regístrala en el negocio.",
            "CALL",
            "HIGH",
        )
    # administrativo
    if any(p in nombre.lower() for p in ("pagado", "inscripci", "cbr", "entregad")):
        return (
            "Marcar como ganado o mover a postventa",
            f"El negocio lleva {_dias_txt(dias)} en «{nombre}» y sigue abierto. Márcalo como ganado o muévelo al pipeline "
            "de postventa para que no distorsione el embudo.",
            "TODO",
            "MEDIUM",
        )
    return (
        "Destrabar el cierre legal",
        f"Lleva {_dias_txt(dias)} en «{nombre}» sin próxima actividad.{contacto} Revisa el checklist de cierre (crédito, "
        "notaría, instrucción, pago), contacta al cliente y coordina con el área legal.",
        "TODO",
        "HIGH",
    )


def detectar(negocios, etapas, cfg, ahora):
    """Lista de negocios estancados con su acción propuesta, ordenada por prioridad y gravedad."""
    por_id = {e["id"]: e for e in etapas}
    estancados = []
    for n in negocios:
        etapa = por_id.get(n.get("etapa_id"))
        if not etapa or etapa["probabilidad"] in (0.0, 1.0):  # ganados y perdidos no se destraban
            continue
        if not sin_proxima_actividad(n, ahora):
            continue  # tiene algo agendado a futuro: lo dejamos trabajar
        categoria = categoria_etapa(etapa)
        dias = dias_en_etapa(n, ahora)
        umbral = umbral_dias(etapa, categoria, cfg)
        if dias < umbral:
            continue
        dias_sc = dias_sin_contacto(n, ahora)
        accion, detalle, tipo, prioridad = proponer_accion(n, etapa, categoria, dias, dias_sc)
        estancados.append(
            {
                "negocio": n,
                "etapa": etapa,
                "categoria": categoria,
                "dias": round(dias, 1),
                "dias_sin_contacto": round(dias_sc, 1) if dias_sc is not None else None,
                "umbral": umbral,
                "severidad": round(dias / umbral, 1) if umbral else round(dias, 1),
                "accion": accion,
                "detalle": detalle,
                "tipo": tipo,
                "prioridad": prioridad,
                "estado": "propuesta",
                "tarea_id": "",
            }
        )
    estancados.sort(key=lambda e: (PRIORIDAD_CATEGORIA[e["categoria"]], -e["severidad"], -e["dias"]))
    return estancados


@dataclass
class ResultadoDesatascador:
    estancados: list = field(default_factory=list)
    creadas: int = 0
    escalados: list = field(default_factory=list)
    con_tarea_reciente: int = 0
    errores: list = field(default_factory=list)
    modo: str = "lectura"

    def por_categoria(self):
        grupos = {}
        for e in self.estancados:
            grupos.setdefault(e["categoria"], []).append(e)
        return dict(sorted(grupos.items(), key=lambda kv: PRIORIDAD_CATEGORIA[kv[0]]))


def cuerpo_tarea(est):
    n = est["negocio"]
    lineas = [est["detalle"], "", f"Negocio: {n['nombre']}", f"Etapa: {est['etapa']['nombre']} ({_dias_txt(est['dias'])} en la etapa)"]
    if est["dias_sin_contacto"] is not None:
        lineas.append(f"Último contacto: hace {_dias_txt(est['dias_sin_contacto'])}")
    if n.get("monto"):
        lineas.append(f"Monto: {n['monto']:,.0f}".replace(",", "."))
    lineas += ["", "Tarea creada por el agente Desatascador del embudo."]
    return "\n".join(lineas)


def ejecutar(db, cfg, cliente, estancados, ahora):
    """Decide qué hacer con cada estancado: proponer, crear tarea, respetar una tarea reciente o escalar."""
    r = ResultadoDesatascador(estancados=estancados, modo=cfg.modo)
    vence = ahora + timedelta(days=cfg.tarea_vence_dias)
    for est in estancados:
        n = est["negocio"]
        ultima = db.ultima_tarea(n["id"])
        if ultima:
            edad = (ahora - desde_iso(ultima["creada_en"])).days
            misma_etapa = (ultima["etapa_id"] or "") == (n["etapa_id"] or "")
            if misma_etapa and edad >= cfg.escalar_dias and not ultima["escalada_en"]:
                est["estado"] = "escalar"
                r.escalados.append(est)
                if cfg.escribe_en_hubspot and cfg.supervisor_id:
                    try:
                        asunto = f"Escalado: {n['nombre']} sigue estancado en {est['etapa']['nombre']}"
                        cuerpo = (
                            f"Se creó una tarea hace {edad} días y el negocio sigue en «{est['etapa']['nombre']}» sin "
                            f"próxima actividad.\n\n{cuerpo_tarea(est)}"
                        )
                        cliente.crear_tarea(n["id"], cfg.supervisor_id, asunto, cuerpo, vence, "HIGH", "TODO")
                        db.marcar_escalada(ultima["id"], ahora)
                        est["estado"] = "escalado"
                    except Exception as e:
                        r.errores.append(f"Escalación de {n['nombre']}: {e}")
                continue
            if misma_etapa and edad < cfg.repetir_tarea_dias:
                est["estado"] = "tarea_reciente"
                est["tarea_id"] = ultima["tarea_id"]
                r.con_tarea_reciente += 1
                continue
        if not cfg.escribe_en_hubspot or r.creadas >= cfg.max_tareas_por_ciclo:
            continue
        try:
            tarea_id = cliente.crear_tarea(
                n["id"], n.get("propietario_id") or None, f"{est['accion']}: {n['nombre']}", cuerpo_tarea(est), vence,
                est["prioridad"], est["tipo"],
            )
            db.registrar_tarea(n["id"], n["etapa_id"], est["accion"], tarea_id, n.get("propietario_id"), ahora)
            est["estado"], est["tarea_id"] = "tarea_creada", tarea_id
            r.creadas += 1
        except Exception as e:
            r.errores.append(f"Tarea para {n['nombre']}: {e}")
    db.log(
        AGENTE,
        "Ciclo",
        f"{len(estancados)} estancados · {r.creadas} tareas creadas · {len(r.escalados)} por escalar · modo {cfg.modo}",
    )
    return r
