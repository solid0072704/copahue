"""Agente LinkedIn: ubica el perfil de cada lead, redacta invitaciones y mensajes, y propone publicaciones.

LinkedIn no permite automatizar invitaciones ni mensajes (ni con su API pública ni con bots:
arriesgas la cuenta). Por eso este agente hace todo lo demás y tú solo pegas y envías:

- Ubica el perfil del lead (enlace directo si lo informó; si no, búsqueda por nombre y empresa).
- Redacta la nota de invitación (máx. 300 caracteres), el mensaje tras conectar y un seguimiento,
  según el segmento y el motivo de descarga, con la misma oferta de reunión que los correos.
- Lleva el registro de en qué paso va cada lead y se detiene apenas responde (por LinkedIn o por correo).
- Propone cada semana una publicación para tu propio LinkedIn que lleve a descargar el ebook.
"""

import re
from datetime import date, timedelta
from urllib.parse import quote, urlencode

from . import reuniones
from .analista import nombre_de_pila, tipo_motivo
from .db import ETAPAS_SIN_SEGUIMIENTO, ahora_utc, desde_iso, iso
from .plantillas import etiqueta_zona, fecha_larga

AGENTE = "LinkedIn"
LIMITE_NOTA = 300  # LinkedIn corta la nota de invitación en 300 caracteres.
LIMITE_MENSAJE = 1900  # LinkedIn admite más, pero nadie lee más que esto en un mensaje directo.
LIMITE_PUBLICACION = 3000  # Límite real de una publicación.
ESTADOS_ABIERTOS = ("pendiente",)
# Estado de la relación en LinkedIn (columna leads.linkedin_estado).
ESTADOS_LEAD = {
    "": "Sin contacto",
    "invitado": "Invitación enviada",
    "conectado": "Conectados",
    "respondió": "Respondió por LinkedIn",
    "sin perfil": "Sin perfil en LinkedIn",
}
LIMITES = {"invitacion": LIMITE_NOTA, "mensaje": LIMITE_MENSAJE, "seguimiento": LIMITE_MENSAJE}
NOMBRES_PASO = {1: "Invitación a conectar", 2: "Mensaje tras conectar", 3: "Seguimiento"}


# --------------------------------------------------------------------------
# Enlaces
# --------------------------------------------------------------------------
def normalizar_url_perfil(url):
    """Acepta 'linkedin.com/in/ana', 'https://cl.linkedin.com/in/ana/' o solo 'ana'. Devuelve '' si no es válida."""
    u = (url or "").strip()
    if not u:
        return ""
    if "linkedin.com/" not in u.lower():
        if re.fullmatch(r"[\w\-]+", u):  # solo el usuario, p. ej. "ana-perez-123"
            u = "https://www.linkedin.com/in/" + u
        else:
            return ""
    if not re.match(r"https?://", u, re.I):
        u = "https://" + u
    m = re.match(r"https?://(?:[a-z]{2,3}\.)?linkedin\.com/(in|company)/([^/?#\s]+)", u, re.I)
    if not m:
        return ""
    return f"https://www.linkedin.com/{m.group(1).lower()}/{m.group(2)}"


def link_perfil(lead):
    return normalizar_url_perfil(lead.get("linkedin_url"))


def link_busqueda(lead):
    """Búsqueda de personas en LinkedIn por nombre y empresa (para encontrar el perfil en un clic)."""
    terminos = " ".join(x for x in ((lead.get("nombre") or "").strip(), (lead.get("empresa") or "").strip()) if x)
    if not terminos:
        return None
    return "https://www.linkedin.com/search/results/people/?" + urlencode({"keywords": terminos}, quote_via=quote)


def link_lead(lead):
    """El perfil si lo conocemos; si no, la búsqueda."""
    return link_perfil(lead) or link_busqueda(lead)


def link_publicar(texto):
    """Abre LinkedIn con el cuadro de publicación ya escrito."""
    return "https://www.linkedin.com/feed/?" + urlencode({"shareActive": "true", "text": texto}, quote_via=quote)


# --------------------------------------------------------------------------
# Textos
# --------------------------------------------------------------------------
def recortar(texto, limite):
    """Corta en un límite de caracteres sin partir palabras (agrega '…' si fue necesario)."""
    texto = texto.strip()
    if len(texto) <= limite:
        return texto
    corte = texto[: limite - 1].rsplit(" ", 1)[0].rstrip(" ,;:.")
    return corte + "…"


def _firma_corta(cfg):
    return cfg.firma.split("\n")[0].strip()


def _gancho(lead):
    motivo, _ = tipo_motivo(lead.get("motivo", ""))
    if motivo == "Necesidad en su empresa":
        return "Si estás revisando cómo mejorar los reportes de tu empresa, con gusto te comparto lo que mejor funciona."
    if motivo == "Consultor / asesor":
        return "Trabajo con asesores y consultores como tú; creo que podemos aportarnos."
    if motivo == "Aprendizaje":
        return "Publico seguido sobre el tema; quizás te sirva."
    return "Me encantaría saber qué te llevó a descargarlo."


def _cta(cfg, horarios):
    if cfg.link_agenda:
        return f"Puedes elegir el horario que más te acomode aquí: {cfg.link_agenda}"
    if horarios:
        lineas = "\n".join(f"• {fecha_larga(h)}" for h in horarios)
        return (
            f"¿Te acomoda alguno de estos horarios? ({etiqueta_zona(cfg)})\n{lineas}\n\n"
            "Respóndeme por aquí y te envío la invitación."
        )
    return "Si te interesa, respóndeme por aquí y coordinamos el horario que te acomode."


def redactar(tipo, lead, cfg, horarios=None):
    """Texto listo para pegar en LinkedIn, dentro del límite de caracteres de cada tipo."""
    nombre = nombre_de_pila(lead.get("nombre"))
    saludo = f"Hola {nombre}" if nombre else "Hola"
    ebook = cfg.ebook_nombre
    firma = _firma_corta(cfg)

    if tipo == "invitacion":
        partes = [
            f"{saludo}, vi que descargaste {ebook} de {cfg.empresa_nombre}.",
            _gancho(lead),
            f"Me gustaría conectar y compartirte más contenido sobre {cfg.ebook_tema}.",
            f"Saludos, {firma}",
        ]
        texto = " ".join(partes)
        if len(texto) > LIMITE_NOTA:  # sin el gancho sigue teniendo sentido
            texto = " ".join(partes[:1] + partes[2:])
        return recortar(texto, LIMITE_NOTA)

    if tipo == "mensaje":
        motivo, _ = tipo_motivo(lead.get("motivo", ""))
        empresa_frase = f" en {lead['empresa']}" if (lead.get("empresa") or "").strip() else ""
        if lead.get("segmento") == "Caliente" or motivo == "Necesidad en su empresa":
            cuerpo = (
                f"Por tu rol{empresa_frase}, creo que te puede servir un {cfg.oferta_reunion}: revisamos cómo "
                "están hoy tus reportes financieros y te llevas 2 o 3 mejoras concretas, sin compromiso."
            )
        elif motivo == "Consultor / asesor":
            cuerpo = (
                "Me gustaría conocer cómo apoyas a tus clientes; quizás podamos colaborar o referirnos casos. "
                f"Si te parece, conversemos en un {cfg.oferta_reunion}."
            )
        else:
            cuerpo = (
                "Un consejo para sacarle provecho al ebook: elige un solo indicador que hoy revisas cada mes "
                "(margen, flujo de caja o cuentas por cobrar) y aplícale lo que propone la guía esta semana.\n\n"
                f"Si quieres, lo revisamos juntos en un {cfg.oferta_reunion}."
            )
        texto = f"{saludo}, gracias por conectar.\n\n{cuerpo}\n\n{_cta(cfg, horarios)}\n\n{firma}"
        return recortar(texto, LIMITE_MENSAJE)

    if tipo == "seguimiento":
        texto = (
            f"{saludo}, te escribí hace unos días por si te interesaba un {cfg.oferta_reunion}. "
            "No quiero insistir: si ahora no es prioridad, no hay problema.\n\n"
            f"Si más adelante quieres revisar tu {cfg.ebook_tema}, me escribes por aquí y lo vemos.\n\n"
            f"¡Buena semana!\n{firma}"
        )
        return recortar(texto, LIMITE_MENSAJE)

    raise ValueError(f"Tipo de mensaje de LinkedIn desconocido: {tipo}")


# --------------------------------------------------------------------------
# Cadencia
# --------------------------------------------------------------------------
def detener(lead):
    """Ya no se preparan más mensajes: respondió, está agendado, es cliente, fue descartado o no tiene perfil."""
    return (
        lead.get("etapa") in ETAPAS_SIN_SEGUIMIENTO
        or bool(lead.get("respondio_en"))
        or (lead.get("linkedin_estado") or "") in ("respondió", "sin perfil")
    )


def siguiente_paso(lead, existentes, cfg, ahora=None):
    """Próxima acción: (paso, tipo, nombre_paso, programado_para) o None.

    1) Invitación a conectar → 2) mensaje cuando tú marcas que aceptó → 3) seguimiento días después.
    """
    ahora = ahora or ahora_utc()
    if detener(lead):
        return None
    if any(a["estado"] in ESTADOS_ABIERTOS for a in existentes):
        return None  # hay un texto esperando que lo envíes: no se acumulan más
    por_paso = {a["paso"]: a for a in existentes}
    estado = lead.get("linkedin_estado") or ""

    if 1 not in por_paso and 2 not in por_paso:
        if (lead.get("segmento") or "") not in cfg.linkedin_segmentos:
            return None
        if estado == "conectado":  # ya eran contactos: se salta la invitación
            return 2, "mensaje", "Mensaje (ya conectados)", ahora
        return 1, "invitacion", NOMBRES_PASO[1], ahora
    if 2 not in por_paso:
        if estado != "conectado":
            return None  # espera a que acepte la invitación
        return 2, "mensaje", NOMBRES_PASO[2], ahora
    if 3 not in por_paso:
        mensaje = por_paso[2]
        if mensaje["estado"] != "enviado":
            return None
        programado = desde_iso(mensaje["actualizado_en"]) + timedelta(days=cfg.linkedin_dias_seguimiento)
        if programado <= ahora:
            return 3, "seguimiento", NOMBRES_PASO[3], programado
    return None


def invitaciones_hoy(db, cfg, ahora=None):
    """Invitaciones preparadas hoy (LinkedIn limita las invitaciones semanales; los mensajes a contactos no)."""
    local = (ahora or ahora_utc()).astimezone(cfg.tz)
    desde = iso(local.replace(hour=0, minute=0, second=0, microsecond=0))
    return sum(
        1 for a in db.acciones_linkedin()
        if a["tipo"] == "invitacion" and a["estado"] != "omitido" and (a["creado_en"] or "") >= desde
    )


def _cerrar_abiertas(db, lead, motivo):
    for a in db.acciones_linkedin(lead_id=lead["id"], estados=ESTADOS_ABIERTOS):
        db.actualizar_accion_linkedin(a["id"], estado="omitido")
        db.log(AGENTE, "Secuencia detenida", f"{lead['email']} · {motivo}", lead["id"])


def ejecutar(db, cfg, asistente=None, ahora=None, solo_lead=None):
    """Prepara los textos de LinkedIn que tocan hoy (quedan pendientes hasta que los envíes). Devuelve sus ids."""
    if not cfg.linkedin_activo:
        return []
    ahora = ahora or ahora_utc()
    cupo = cfg.linkedin_max_dia - invitaciones_hoy(db, cfg, ahora)
    creados = []
    for lead in db.leads():  # ordenados por score: primero los más calientes
        if solo_lead is not None and lead["id"] != solo_lead:
            continue
        existentes = db.acciones_linkedin(lead_id=lead["id"])
        if detener(lead):
            _cerrar_abiertas(db, lead, f"etapa {lead['etapa']}" if not lead.get("linkedin_estado") else lead["linkedin_estado"])
            continue
        plan = siguiente_paso(lead, existentes, cfg, ahora)
        if not plan:
            continue
        paso, tipo, nombre_paso, programado = plan
        if tipo == "invitacion":
            if cupo <= 0:
                continue
            cupo -= 1
        horarios = []
        if tipo == "mensaje" and not cfg.link_agenda:
            horarios = reuniones.horarios_de(db, lead["id"], cfg) or reuniones.proximos_horarios(
                cfg, db.horarios_ocupados(), rotacion=lead["id"]
            )
        texto = redactar(tipo, lead, cfg, horarios)
        ia = False
        if asistente is not None and lead.get("segmento") in ("Caliente", "Tibio"):
            personalizado = asistente.personalizar_linkedin(lead, tipo, texto, LIMITES[tipo])
            if personalizado:
                texto, ia = personalizado, True
        accion_id = db.crear_accion_linkedin(lead["id"], paso, tipo, nombre_paso, iso(programado), texto, ia)
        if accion_id is None:
            continue
        db.log(AGENTE, f"Paso {paso}: {nombre_paso}", f"{lead['email']} · listo para copiar en el tablero", lead["id"])
        creados.append(accion_id)
    return creados


# --------------------------------------------------------------------------
# Acciones desde el tablero
# --------------------------------------------------------------------------
def marcar(db, accion_id, estado):
    """Marca un texto como enviado u omitido. Si era la invitación, el lead pasa a 'invitado'."""
    db.actualizar_accion_linkedin(accion_id, estado=estado)
    a = db.accion_linkedin(accion_id)
    if not a:
        return
    lead = db.lead(a["lead_id"])
    if estado == "enviado":
        campos = {}
        if a["paso"] == 1 and not lead.get("linkedin_estado"):
            campos["linkedin_estado"] = "invitado"
        if lead["etapa"] == "Nuevo":
            campos["etapa"] = "En seguimiento"
        if campos:
            db.actualizar_lead(lead["id"], **campos)
    db.log(AGENTE, f"Marcado como {estado}", f"{a['email']} · paso {a['paso']}", a["lead_id"])


def marcar_conectado(db, cfg, lead_id, asistente=None):
    """El lead aceptó la conexión: se prepara de inmediato el mensaje de bienvenida."""
    lead = db.lead(lead_id)
    for a in db.acciones_linkedin(lead_id=lead_id, estados=ESTADOS_ABIERTOS):
        if a["paso"] == 1:
            db.actualizar_accion_linkedin(a["id"], estado="enviado")
    db.actualizar_lead(lead_id, linkedin_estado="conectado", etapa="En seguimiento" if lead["etapa"] == "Nuevo" else lead["etapa"])
    db.log(AGENTE, "Conexión aceptada", lead["email"], lead_id)
    return ejecutar(db, cfg, asistente, solo_lead=lead_id)


def marcar_respondio(db, lead_id):
    """Respondió por LinkedIn: se detienen los mensajes (también los correos) y pasa a 'Respondió'."""
    lead = db.lead(lead_id)
    etapa = lead["etapa"] if lead["etapa"] in ("Reunión agendada", "Cliente") else "Respondió"
    db.actualizar_lead(
        lead_id,
        linkedin_estado="respondió",
        etapa=etapa,
        respondio_en=lead.get("respondio_en") or iso(ahora_utc()),
    )
    _cerrar_abiertas(db, {**lead, "linkedin_estado": "respondió"}, "respondió por LinkedIn")
    db.log(AGENTE, "Respuesta por LinkedIn", f"{lead['nombre']} <{lead['email']}> te respondió", lead_id)


def marcar_sin_perfil(db, lead_id):
    lead = db.lead(lead_id)
    db.actualizar_lead(lead_id, linkedin_estado="sin perfil")
    _cerrar_abiertas(db, {**lead, "linkedin_estado": "sin perfil"}, "sin perfil en LinkedIn")
    db.log(AGENTE, "Sin perfil", lead["email"], lead_id)


def guardar_perfil(db, lead_id, url):
    """Guarda el perfil del lead (acepta la URL completa o solo el usuario). Devuelve la URL normalizada o ''."""
    normalizada = normalizar_url_perfil(url)
    if normalizada or not (url or "").strip():
        db.actualizar_lead(lead_id, linkedin_url=normalizada)
        if normalizada:
            db.log(AGENTE, "Perfil guardado", normalizada, lead_id)
    return normalizada


# --------------------------------------------------------------------------
# Publicaciones para tu propio LinkedIn
# --------------------------------------------------------------------------
HASHTAGS = "#finanzas #pymes #controldegestion #flujodecaja"


def _publicaciones(cfg, hallazgo):
    tema, ebook, link = cfg.ebook_tema, cfg.ebook_nombre, cfg.link_descarga
    return [
        (
            "El error más común",
            "Un mes «bueno» en ventas puede esconder un problema de caja.\n\n"
            "Es el error más común que vemos al revisar los números de una empresa: mirar el resultado del mes "
            "sin compararlo con el presupuesto ni con el flujo de caja.\n\n"
            "Tres preguntas que vale la pena hacerse cada mes:\n"
            "1) ¿Qué indicador se desvió más del presupuesto y por qué?\n"
            "2) ¿El resultado contable se está convirtiendo en caja?\n"
            "3) ¿Qué decisión concreta tomaremos con esta información?\n\n"
            f"Si quieres ordenar tu {tema} con un método simple, en {ebook} lo explicamos paso a paso. "
            f"Descárgalo gratis aquí: {link}\n\n{HASHTAGS}",
        ),
        (
            "Un indicador a la vez",
            "No necesitas 40 indicadores. Necesitas uno que de verdad mires.\n\n"
            "Elige un solo número que hoy revisas cada mes (margen, flujo de caja o cuentas por cobrar) y "
            "pregúntate: ¿qué decisión cambiaría si este número fuera distinto?\n\n"
            "Si la respuesta es «ninguna», ese indicador no te está sirviendo. Si es «varias», ahí está tu "
            "tablero.\n\n"
            f"En {ebook} proponemos cómo elegirlos y cómo leerlos. Está disponible gratis aquí: {link}\n\n{HASHTAGS}",
        ),
        (
            "Pregunta a la comunidad",
            "¿Cuántos días después de cerrar el mes llegan los números a tu escritorio?\n\n"
            "En muchas pymes la respuesta es «20 o más». Y con 20 días de atraso, los reportes sirven para "
            "explicar el pasado, no para decidir el futuro.\n\n"
            "Cuéntame en los comentarios cuántos días tardan en tu empresa y qué es lo que más los retrasa.\n\n"
            f"Y si quieres acortar ese plazo, en {ebook} hay una ruta práctica para lograrlo: {link}\n\n{HASHTAGS}",
        ),
        (
            "Qué buscan quienes descargan el ebook",
            f"Hace unas semanas publicamos {ebook}, una guía práctica de {tema}.\n\n"
            f"{hallazgo}\n\n"
            "Eso confirma algo que vemos todos los días: los números existen, lo que falta es convertirlos "
            "en decisiones a tiempo.\n\n"
            f"Si aún no lo tienes, puedes descargarlo gratis aquí: {link}\n\n{HASHTAGS}",
        ),
    ]


def hallazgo_para_publicar(leads):
    """Una frase agregada (sin datos personales) sobre quienes descargaron el ebook."""
    reales = [l for l in leads if l.get("fuente") != "demo"] or leads
    if len(reales) < 5:
        return "Lo más interesante no es cuántas personas lo descargaron, sino para qué: la mayoría busca mejorar el análisis financiero de su propia empresa."
    from collections import Counter

    motivos = Counter(tipo_motivo(l.get("motivo", ""))[0] for l in reales)
    motivo, n = motivos.most_common(1)[0]
    frase = {
        "Necesidad en su empresa": "quiere mejorar el análisis financiero de su propia empresa",
        "Consultor / asesor": "es consultor o asesor y lo usa con sus clientes",
        "Aprendizaje": "lo descargó para aprender",
    }.get(motivo, "no detalló el motivo")
    decisores = sum(1 for l in reales if l.get("nivel_cargo") == "Decisor")
    extra = f" Y {decisores / len(reales):.0%} tiene un cargo de decisión (gerencia, socios, fundadores)." if decisores else ""
    return f"Lo más interesante no es cuántas personas lo descargaron, sino para qué: {n / len(reales):.0%} {frase}.{extra}"


def redactar_publicacion(cfg, leads=(), semana=None):
    """Publicación sugerida para la semana (rota entre 4 enfoques). Devuelve (título, texto)."""
    semana = semana if semana is not None else date.today().isocalendar()[1]
    opciones = _publicaciones(cfg, hallazgo_para_publicar(list(leads)))
    titulo, texto = opciones[semana % len(opciones)]
    return titulo, recortar(texto, LIMITE_PUBLICACION)


# --------------------------------------------------------------------------
# Salida por terminal (python run_agentes.py --linkedin [--lead correo] [--publicacion])
# --------------------------------------------------------------------------
def imprimir(db, cfg, correo=None, publicacion=False, escribir=print):
    """Muestra los textos pendientes (y, si se pide, la publicación de la semana) para copiarlos desde la terminal."""
    if publicacion:
        titulo, texto = redactar_publicacion(cfg, db.leads())
        escribir(f"=== Publicación sugerida: {titulo} ===\n{texto}\n")
    pendientes = db.acciones_linkedin(estados=ESTADOS_ABIERTOS)
    if correo:
        pendientes = [a for a in pendientes if a["email"] == correo.strip().lower()]
    if not pendientes and not publicacion:
        escribir("No hay textos de LinkedIn pendientes.")
    for a in pendientes:
        escribir(f"=== {a['nombre'] or a['email']} · {a['cargo']} · {a['empresa']} · paso {a['paso']}: {a['nombre_paso']} ===")
        escribir(f"Perfil: {link_lead(a)}")
        escribir(a["texto"] + "\n")
    return pendientes
