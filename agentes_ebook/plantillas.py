"""Plantillas de correo y utilidades de enlaces (WhatsApp, mailto, horarios)."""

import re
from urllib.parse import quote

from .analista import nombre_de_pila, tipo_motivo

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]
CODIGO_PAIS = {"Chile": "56", "Perú": "51", "México": "52", "Argentina": "54", "Colombia": "57"}


def fecha_larga(dt):
    return f"{DIAS[dt.weekday()]} {dt.day} de {MESES[dt.month - 1]}, {dt:%H:%M}"


def etiqueta_zona(cfg):
    return "hora de Chile" if cfg.zona_horaria == "America/Santiago" else cfg.zona_horaria


def _cta(cfg, horarios):
    if cfg.link_agenda:
        return f"Puedes elegir el horario que más te acomode aquí: {cfg.link_agenda}"
    if horarios:
        lineas = "\n".join(f"  • {fecha_larga(h)}" for h in horarios)
        return (
            f"¿Te acomoda alguno de estos horarios? ({etiqueta_zona(cfg)})\n{lineas}\n\n"
            "Responde con el que prefieras y te envío la invitación."
        )
    return "Responde este correo y coordinamos el horario que te acomode."


def _pregunta_por_motivo(lead):
    motivo, _ = tipo_motivo(lead.get("motivo", ""))
    if motivo == "Necesidad en su empresa":
        return (
            "Me contaste que buscas mejorar el análisis financiero de tu empresa. "
            "¿Cuál es hoy el mayor desafío: la información llega tarde, cuesta confiar en los números "
            "o cuesta convertirlos en decisiones?"
        )
    if motivo == "Consultor / asesor":
        return (
            "Vi que trabajas como consultor/a o asesor/a. Me encantaría conocer cómo apoyas a tus "
            "clientes; quizás podamos colaborar."
        )
    if motivo == "Aprendizaje":
        return "Si te surgen dudas mientras lo lees, responde este correo y te ayudo."
    return (
        "¿Qué te motivó a descargarlo? Si me cuentas qué te gustaría resolver, "
        "te puedo enviar material más específico."
    )


# Cadencias por segmento: (días desde la descarga, tipo de correo, nombre del paso)
CADENCIAS = {
    "Caliente": [
        (0, "bienvenida_caliente", "Bienvenida + invitación"),
        (2, "horarios", "Propuesta de horarios"),
        (6, "caso", "Contenido de valor"),
        (12, "ultimo", "Último contacto"),
    ],
    "Tibio": [
        (0, "bienvenida", "Bienvenida"),
        (3, "caso", "Contenido de valor"),
        (7, "invitacion", "Invitación a reunión"),
        (14, "ultimo", "Último contacto"),
    ],
    "Frío": [
        (0, "bienvenida", "Bienvenida"),
        (4, "caso", "Contenido de valor"),
        (10, "invitacion", "Invitación a reunión"),
        (20, "ultimo", "Último contacto"),
    ],
}


def redactar(tipo, lead, cfg, horarios=None):
    """Devuelve (asunto, cuerpo) del correo de seguimiento."""
    nombre = nombre_de_pila(lead.get("nombre")) or "hola"
    saludo = f"Hola {nombre}:" if nombre != "hola" else "Hola:"
    empresa_frase = f" en {lead['empresa']}" if (lead.get("empresa") or "").strip() else ""
    firma = f"Saludos,\n{cfg.firma}\n{cfg.empresa_nombre} · {cfg.sitio_web}"
    cta = _cta(cfg, horarios)
    ebook = cfg.ebook_nombre

    if tipo == "bienvenida":
        asunto = f"{nombre.capitalize()}, gracias por descargar {ebook}"
        cuerpo = (
            f"{saludo}\n\nGracias por descargar {ebook}. Espero que te sirva para llevar tu "
            f"{cfg.ebook_tema} al siguiente nivel.\n\n"
            "Un consejo para sacarle provecho: elige un solo indicador que hoy revisas cada mes "
            "(margen, flujo de caja o cuentas por cobrar) y aplícale lo que propone la guía esta semana. "
            "Es la forma más rápida de ver resultados.\n\n"
            f"{_pregunta_por_motivo(lead)}\n\n{firma}"
        )
    elif tipo == "bienvenida_caliente":
        asunto = f"{nombre.capitalize()}, tu ebook y una propuesta"
        cuerpo = (
            f"{saludo}\n\nGracias por descargar {ebook}.\n\n"
            f"Por tu rol{empresa_frase}, creo que te puede servir un {cfg.oferta_reunion}: revisamos "
            "cómo están hoy tus reportes financieros y te llevas 2 o 3 mejoras concretas, sin compromiso.\n\n"
            f"{cta}\n\n{firma}"
        )
    elif tipo == "horarios":
        asunto = f"¿Te acomoda alguno de estos horarios, {nombre.capitalize()}?"
        cuerpo = (
            f"{saludo}\n\nTe escribo para coordinar el {cfg.oferta_reunion} que te mencioné. "
            "En la conversación revisamos tu situación actual y te llevas un plan de acción.\n\n"
            f"{cta}\n\n{firma}"
        )
    elif tipo == "caso":
        asunto = "El error más común al analizar los números de una empresa"
        cuerpo = (
            f"{saludo}\n\nUn error muy común es mirar los resultados del mes sin compararlos con el "
            "presupuesto ni con el flujo de caja. Así, un mes «bueno» en ventas puede esconder problemas "
            "de liquidez.\n\nTres preguntas que te recomiendo hacerte cada mes:\n"
            "  1) ¿Qué indicador se desvió más del presupuesto y por qué?\n"
            "  2) ¿El resultado contable se está convirtiendo en caja?\n"
            "  3) ¿Qué decisión concreta tomaremos con esta información?\n\n"
            f"Si quieres, podemos revisar juntos tus reportes en un {cfg.oferta_reunion}.\n\n"
            f"{cta}\n\n{firma}"
        )
    elif tipo == "invitacion":
        asunto = f"{nombre.capitalize()}, ¿conversamos 30 minutos?"
        cuerpo = (
            f"{saludo}\n\nHace unos días descargaste {ebook}. Quería ofrecerte un {cfg.oferta_reunion}: "
            "revisamos tu situación actual y te llevas un plan de acción concreto, sin compromiso.\n\n"
            f"{cta}\n\n{firma}"
        )
    elif tipo == "reactivacion":
        asunto = f"{nombre.capitalize()}, ¿pudiste revisar {ebook}?"
        cuerpo = (
            f"{saludo}\n\nHace un tiempo descargaste {ebook} y quería saber si te resultó útil.\n\n"
            f"{_pregunta_por_motivo(lead)}\n\n"
            f"Si te sirve, también podemos conversar en un {cfg.oferta_reunion}.\n\n{cta}\n\n{firma}"
        )
    elif tipo == "ultimo":
        asunto = f"¿Cierro tu seguimiento, {nombre.capitalize()}?"
        cuerpo = (
            f"{saludo}\n\nNo quiero llenar tu bandeja de entrada. Si en este momento no es prioridad, "
            "no hay problema: no te volveré a escribir sobre esto.\n\n"
            f"Si más adelante necesitas apoyo con tu {cfg.ebook_tema}, basta con responder este correo.\n\n"
            f"{cta}\n\n{firma}"
        )
    else:
        raise ValueError(f"Tipo de correo desconocido: {tipo}")
    if nombre == "hola":
        asunto = asunto.replace(", Hola?", "?")
    return asunto, cuerpo


def mensaje_whatsapp(lead, cfg):
    nombre = nombre_de_pila(lead.get("nombre"))
    return (
        f"Hola {nombre}, te escribe {cfg.firma} de {cfg.empresa_nombre}. "
        f"Vi que descargaste {cfg.ebook_nombre}. ¿Te puedo ayudar con algo de tu {cfg.ebook_tema}? "
        f"Si te sirve, podemos agendar un {cfg.oferta_reunion}."
    )


def telefono_internacional(telefono, pais):
    """Número en formato internacional sin '+', o None si no se puede deducir."""
    crudo = (telefono or "").strip()
    digitos = re.sub(r"\D", "", crudo)
    if not digitos:
        return None
    if crudo.startswith("+"):
        return digitos
    if crudo.startswith("00"):
        return digitos[2:]
    codigo = CODIGO_PAIS.get(pais)
    if codigo and digitos.startswith(codigo) and len(digitos) > 10:
        return digitos
    if codigo == "56" and len(digitos) == 9 and digitos.startswith("9"):
        return "56" + digitos
    if codigo == "51" and len(digitos) == 9 and digitos.startswith("9"):
        return "51" + digitos
    if codigo == "52" and len(digitos) == 10:
        return "52" + digitos
    return None


def link_whatsapp(lead, cfg, texto=None):
    numero = telefono_internacional(lead.get("telefono"), lead.get("pais"))
    if not numero:
        return None
    return f"https://wa.me/{numero}?text={quote(texto or mensaje_whatsapp(lead, cfg))}"


def link_mailto(correo, asunto, cuerpo):
    return f"mailto:{correo}?subject={quote(asunto)}&body={quote(cuerpo)}"
