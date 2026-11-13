"""Textos para LinkedIn: nota de invitación, secuencia de mensajes y propuesta de reunión.

Reglas de estilo: primera persona (es tu perfil personal), sin vender en la invitación,
una sola pregunta por mensaje y siempre con salida elegante.
"""

from agentes_ebook.analista import nombre_de_pila

NOMBRES_TIPO = {
    "invitacion": "Invitación a conectar",
    "mensaje_1": "Mensaje 1 · abrir conversación",
    "mensaje_2": "Mensaje 2 · aportar valor",
    "mensaje_3": "Mensaje 3 · invitar a conversar",
    "mensaje_4": "Mensaje 4 · seguimiento",
    "reunion": "Propuesta de reunión",
}


def nombre_tipo(tipo):
    return NOMBRES_TIPO.get(tipo, tipo.replace("_", " ").capitalize())


def _variante(p, opciones):
    return opciones[(p.get("id") or 0) % len(opciones)]


def _cta(cfgp):
    if cfgp.base.link_agenda:
        return f"Puedes elegir el horario que te acomode aquí: {cfgp.base.link_agenda}"
    return "Si te interesa, dime qué día te acomoda y lo coordinamos."


def nota_invitacion(p, cfgp):
    """Nota de la invitación, siempre dentro del límite de caracteres de LinkedIn."""
    n = nombre_de_pila(p.get("nombre")) or "qué tal"
    empresa = p.get("empresa") or ""
    servicio = cfgp.servicio
    candidatas = []
    if p.get("lead_id"):
        candidatas.append(
            f"Hola {n}, vi que descargaste {cfgp.base.ebook_nombre} de {cfgp.base.empresa_nombre}. "
            "Me gustaría conectar por aquí y saber qué te pareció. ¡Saludos!"
        )
    if p.get("nivel") == "Finanzas":
        candidatas.append(
            f"Hola {n}, también trabajo en finanzas corporativas, apoyando a empresas como {servicio}. "
            "Me gustaría conectar e intercambiar miradas. ¡Saludos!"
        )
    if empresa:
        candidatas += _variante(
            p,
            [
                [
                    f"Hola {n}, vi tu trabajo en {empresa}. Acompaño a empresas en crecimiento como {servicio} "
                    "y me gustaría tenerte en mi red. ¡Saludos!",
                ],
                [
                    f"Hola {n}, sigo de cerca a quienes lideran empresas como {empresa}. Trabajo en gestión "
                    f"financiera para pymes ({servicio}) y me gustaría conectar contigo.",
                ],
            ],
        )
    candidatas += [
        f"Hola {n}, acompaño a empresas en crecimiento como {servicio} y me gustaría tenerte en mi red. ¡Saludos!",
        f"Hola {n}, me gustaría sumarte a mi red. ¡Saludos!",
    ]
    for texto in candidatas:
        if len(texto) <= cfgp.nota_max:
            return texto
    return candidatas[-1][: cfgp.nota_max]


def _mensaje_1(p, cfgp, n, empresa):
    pregunta = (
        f"¿cómo llevan hoy en {empresa} el control del flujo de caja y los reportes de gestión: "
        "lo ve alguien interno, la contabilidad externa o tú directamente?"
        if empresa
        else "¿cómo llevas hoy el control del flujo de caja y los reportes de gestión: lo ve alguien "
        "interno, la contabilidad externa o tú directamente?"
    )
    if p.get("grado") == 1 and not p.get("invitado_en"):
        # Contacto que ya tenías: retomar el vínculo antes de contar qué haces.
        return (
            f"Hola {n}, ¡tanto tiempo! Espero que todo vaya muy bien"
            + (f" en {empresa}" if empresa else "")
            + f".\n\nTe escribo porque hoy trabajo como {cfgp.servicio}: ayudo a empresas en crecimiento a "
            f"{cfgp.propuesta}, sin que tengan que contratar una gerencia de finanzas a tiempo completo.\n\n"
            f"Por curiosidad, {pregunta}\n\n"
            "Y si conoces a alguien a quien le pueda servir, feliz de que me lo presentes."
        )
    return (
        f"Hola {n}, gracias por conectar.\n\n"
        f"Te cuento en una línea a qué me dedico: ayudo a empresas en crecimiento a {cfgp.propuesta}, "
        f"en modalidad de {cfgp.servicio} (por horas o días al mes, no a tiempo completo).\n\n"
        f"Por curiosidad, {pregunta}"
    )


def _mensaje_2(p, cfgp, n, empresa):
    recurso = (
        f"Te dejo además un material sobre {cfgp.base.ebook_tema} que puede servirte: {cfgp.recurso_link}\n\n"
        if cfgp.recurso_link
        else ""
    )
    return (
        f"Hola {n}, te comparto las tres preguntas que más veo en empresas que están creciendo:\n\n"
        "1. ¿Cuánta caja tengo realmente para los próximos 3 meses?\n"
        "2. ¿Qué línea de negocio deja margen de verdad?\n"
        "3. ¿Puedo financiar el crecimiento sin ahogar la operación?\n\n"
        f"{recurso}"
        "Si alguna te suena, feliz de contarte cómo las abordo."
    )


def _mensaje_3(p, cfgp, n, empresa):
    return (
        f"Hola {n}, último mensaje para no llenarte la bandeja.\n\n"
        f"Si en algún momento quieres una segunda mirada sobre las finanzas"
        + (f" de {empresa}" if empresa else "")
        + f", ofrezco un {cfgp.base.oferta_reunion}. {_cta(cfgp)}\n\n"
        "Y si ahora no es prioridad, todo bien: quedo disponible por aquí."
    )


def _mensaje_extra(p, cfgp, n, empresa):
    return (
        f"Hola {n}, paso a saludar. Si más adelante necesitas apoyo para {cfgp.propuesta}, "
        f"cuenta conmigo. {_cta(cfgp)}"
    )


def _reunion(p, cfgp, n, empresa):
    return (
        f"¡Gracias por responder, {n}! ¿Te parece si lo vemos en una llamada de "
        f"{cfgp.base.reunion_duracion_min} minutos? Así entiendo bien el contexto"
        + (f" de {empresa}" if empresa else "")
        + f" y te digo con franqueza si puedo aportar. {_cta(cfgp)}"
    )


_REDACTORES = {"mensaje_1": _mensaje_1, "mensaje_2": _mensaje_2, "mensaje_3": _mensaje_3, "reunion": _reunion}


def redactar(tipo, p, cfgp):
    """Texto base (sin IA) para una acción sobre un prospecto."""
    if tipo == "invitacion":
        return nota_invitacion(p, cfgp)
    n = nombre_de_pila(p.get("nombre")) or "qué tal"
    return _REDACTORES.get(tipo, _mensaje_extra)(p, cfgp, n, p.get("empresa") or "")
