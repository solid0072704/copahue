"""Crea en HubSpot las propiedades que usan Breeze (calificación) y el distribuidor (estado y SLA).

Es idempotente: si una propiedad ya existe, no la toca. Ajusta las opciones (por ejemplo,
los rangos de presupuesto) antes de ejecutarlo si quieres otros valores.
"""

GRUPO = "central_whatsapp"
ETIQUETA_GRUPO = "Central WhatsApp"


def _enum(nombre, etiqueta, opciones, descripcion):
    return {
        "name": nombre,
        "label": etiqueta,
        "type": "enumeration",
        "fieldType": "select",
        "groupName": GRUPO,
        "description": descripcion,
        "options": [{"label": l, "value": v, "displayOrder": i} for i, (v, l) in enumerate(opciones)],
    }


def _campo(nombre, etiqueta, tipo, tipo_campo, descripcion):
    return {
        "name": nombre,
        "label": etiqueta,
        "type": tipo,
        "fieldType": tipo_campo,
        "groupName": GRUPO,
        "description": descripcion,
    }


CALIFICA_BREEZE = "La completa el agente de WhatsApp (Breeze) al calificar el lead."

PROPIEDADES = {
    "contacts": [
        _enum(
            "central_plazo_compra",
            "Plazo de compra",
            [
                ("inmediato", "Lo antes posible"),
                ("1_3_meses", "En 1 a 3 meses"),
                ("3_6_meses", "En 3 a 6 meses"),
                ("mas_6_meses", "En más de 6 meses"),
                ("solo_cotizando", "Solo está cotizando"),
            ],
            CALIFICA_BREEZE,
        ),
        _enum(
            "central_forma_pago",
            "Forma de pago",
            [
                ("contado", "Contado"),
                ("cuotas", "Crédito directo / cuotas"),
                ("hipotecario", "Crédito hipotecario"),
                ("no_sabe", "Aún no lo sabe"),
            ],
            CALIFICA_BREEZE,
        ),
        _enum(
            "central_presupuesto",
            "Presupuesto",
            [
                ("hasta_15", "Hasta $15 millones"),
                ("15_30", "$15 a $30 millones"),
                ("30_50", "$30 a $50 millones"),
                ("mas_50", "Más de $50 millones"),
                ("no_sabe", "No lo sabe / prefiere no decir"),
            ],
            CALIFICA_BREEZE,
        ),
        _enum(
            "central_quiere_visita",
            "Quiere visitar el proyecto",
            [("si", "Sí, quiere agendar"), ("mas_adelante", "Más adelante"), ("no", "No por ahora")],
            CALIFICA_BREEZE,
        ),
        _campo(
            "central_resumen",
            "Resumen de la conversación (central)",
            "string",
            "textarea",
            "Resumen breve de lo que pidió el cliente. Lo usa la tarea que recibe el vendedor.",
        ),
        {
            "name": "central_pendiente_reparto",
            "label": "Pendiente de reparto (central)",
            "type": "bool",
            "fieldType": "booleancheckbox",
            "groupName": GRUPO,
            "description": "La marca un workflow cuando Breeze califica al cliente; el distribuidor asigna el lead y la desmarca.",
            "options": [{"label": "Sí", "value": "true", "displayOrder": 0}, {"label": "No", "value": "false", "displayOrder": 1}],
        },
        _campo(
            "central_ultimo_ruteo",
            "Último ruteo de la central",
            "datetime",
            "date",
            "Fecha en que la central de WhatsApp asignó o avisó a un vendedor por este contacto.",
        ),
    ],
    "deals": [
        _enum(
            "central_estado",
            "Estado en la central",
            [
                ("asignado", "Asignado"),
                ("reasignado", "Reasignado por SLA"),
                ("atendido", "Atendido"),
                ("escalado", "Escalado a supervisor"),
                ("existente", "Cliente que volvió"),
            ],
            "Seguimiento del lead creado por la central de WhatsApp.",
        ),
        _campo("central_asignado_en", "Asignado por la central", "datetime", "date", "Última asignación (inicio del SLA)."),
        _campo("central_reasignaciones", "Reasignaciones (central)", "number", "number", "Veces que el SLA lo reasignó."),
        _campo(
            "central_vendedores_previos",
            "Vendedores previos (central)",
            "string",
            "text",
            "IDs de propietarios que tuvieron el lead antes (separados por ;).",
        ),
        _campo("central_tarea_id", "Tarea de la central (ID)", "string", "text", "Tarea que debe completar el vendedor."),
        _campo(
            "central_minutos_respuesta",
            "Minutos hábiles hasta la atención",
            "number",
            "number",
            "Tiempo de respuesta del vendedor, en minutos dentro del horario de atención.",
        ),
    ],
}


def instalar(hs):
    """Crea el grupo y las propiedades que falten. Devuelve líneas de resumen."""
    resumen = []
    for objeto, definiciones in PROPIEDADES.items():
        hs.crear_grupo(objeto, GRUPO, ETIQUETA_GRUPO)
        existentes = hs.propiedades(objeto)
        for d in definiciones:
            if d["name"] in existentes:
                resumen.append(f"= {objeto}.{d['name']} ya existía")
                continue
            hs.crear_propiedad(objeto, d)
            resumen.append(f"+ {objeto}.{d['name']} creada")
    return resumen
