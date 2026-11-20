"""Suscripción, planes y cuota mensual de tokens de IA por usuario.

Modelo de cobro: cada estudio paga un precio mensual por usuario según su plan. Cada usuario
trae una cuota de tokens de IA al mes; si se agota, el estudio puede comprar paquetes que van a
una bolsa común. Sin cuota disponible, Tu Cartera sigue funcionando con plantillas y reglas.

Los tokens se cuentan como todo lo que la IA lee y escribe (entrada + caché + salida). El costo
real se calcula aparte con los precios del modelo, para medir el margen de cada plan.
"""

from datetime import date, timedelta

from . import calendario as cal
from .db import fila, filas, insertar

# Modelo único de la aplicación y su precio de lista (US$ por millón de tokens).
MODELO = "claude-opus-5-5"
PRECIO_MTOK_USD = {"entrada": 4.00, "salida": 20.00, "cache_lectura": 0.20, "cache_escritura": 5.00}

# Precios en pesos chilenos por usuario al mes, IVA no incluido. `None` = sin límite.
PLANES = {
    "esencial": {
        "nombre": "Esencial", "precio_usuario": 19_990, "max_usuarios": 2, "max_clientes": 40,
        "tokens_usuario": 300_000,
        "para": "Contadores independientes que parten con su cartera.",
    },
    "profesional": {
        "nombre": "Profesional", "precio_usuario": 34_990, "max_usuarios": 10, "max_clientes": 200,
        "tokens_usuario": 1_000_000,
        "para": "Estudios con equipo y cartera en crecimiento.",
    },
    "firma": {
        "nombre": "Firma", "precio_usuario": 54_990, "max_usuarios": None, "max_clientes": None,
        "tokens_usuario": 2_000_000,
        "para": "Firmas con varias áreas y cientos de clientes.",
    },
}
PRUEBA_DIAS = 14
PRUEBA_PLAN = "profesional"
PRUEBA_TOKENS_USUARIO = 200_000
PAQUETE_TOKENS = 1_000_000
PAQUETE_PRECIO = 14_990
DIAS_GRACIA = 3  # días tras el vencimiento del pago antes de suspender

# Tokens aproximados por acción, para mostrar equivalencias comprensibles.
TOKENS_POR_INSTRUCCION = 16_000
TOKENS_POR_CORREO = 1_400


def periodo(hoy):
    return cal.periodo_de(hoy)


def suscripcion(estudio, hoy):
    """Estado comercial del estudio y los límites que le aplican hoy."""
    clave = estudio.get("plan") if estudio.get("plan") in PLANES else PRUEBA_PLAN
    plan = PLANES[clave]
    estado = estudio.get("estado_suscripcion") or "prueba"
    hasta = None
    if estado == "prueba":
        hasta = date.fromisoformat(estudio["prueba_hasta"]) if estudio.get("prueba_hasta") else hoy
        vigente = hoy <= hasta
    elif estado == "activa":
        hasta = date.fromisoformat(estudio["pagado_hasta"]) if estudio.get("pagado_hasta") else None
        vigente = hasta is None or hoy <= hasta + timedelta(days=DIAS_GRACIA)
    else:
        vigente = False
    if not vigente and estado in ("prueba", "activa"):
        estado = "vencida"
    return {
        "plan": clave, "nombre": plan["nombre"], "estado": estado, "vigente": vigente,
        "hasta": hasta.isoformat() if hasta else None,
        "dias_restantes": (hasta - hoy).days if hasta and vigente else None,
        "precio_usuario": plan["precio_usuario"],
        "max_usuarios": plan["max_usuarios"], "max_clientes": plan["max_clientes"],
        "tokens_usuario": PRUEBA_TOKENS_USUARIO if estado == "prueba" else plan["tokens_usuario"],
    }


def costo_usd(entrada, salida, cache_lectura=0, cache_escritura=0):
    p = PRECIO_MTOK_USD
    return (entrada * p["entrada"] + salida * p["salida"] + cache_lectura * p["cache_lectura"]
            + cache_escritura * p["cache_escritura"]) / 1_000_000


def registrar_uso(conn, estudio_id, usuario_id, tipo, uso, fecha):
    """Guarda el consumo de una llamada. `uso` trae entrada, salida, cache_lectura y cache_escritura."""
    e, s = int(uso.get("entrada", 0)), int(uso.get("salida", 0))
    cl, ce = int(uso.get("cache_lectura", 0)), int(uso.get("cache_escritura", 0))
    insertar(conn, "uso_ia", {
        "estudio_id": estudio_id, "usuario_id": usuario_id, "periodo": fecha[:7], "fecha": fecha, "tipo": tipo,
        "entrada": e, "salida": s, "cache_lectura": cl, "cache_escritura": ce,
        "tokens": e + s + cl + ce, "costo_usd": round(costo_usd(e, s, cl, ce), 6),
    })


def uso_del_mes(conn, estudio_id, hoy):
    por_usuario = {r["usuario_id"]: r["tokens"] for r in filas(conn, """
        SELECT usuario_id, SUM(tokens) AS tokens FROM uso_ia WHERE estudio_id = ? AND periodo = ?
        GROUP BY usuario_id""", estudio_id, periodo(hoy))}
    extras = fila(conn, "SELECT COALESCE(SUM(tokens), 0) AS t FROM paquetes_ia WHERE estudio_id = ? AND periodo = ?",
                  estudio_id, periodo(hoy))["t"]
    return por_usuario, extras


def cupo(conn, estudio, usuario_id, hoy):
    """¿Puede este usuario usar la IA ahora? Su cuota personal primero, luego la bolsa de paquetes."""
    s = suscripcion(estudio, hoy)
    por_usuario, extras = uso_del_mes(conn, estudio["id"], hoy)
    limite = s["tokens_usuario"]
    usado = por_usuario.get(usuario_id, 0)
    excedido = sum(max(0, t - limite) for t in por_usuario.values())
    bolsa = max(0, extras - excedido)
    if not s["vigente"]:
        motivo = "suscripcion"
    elif usado >= limite and bolsa <= 0:
        motivo = "limite"
    else:
        motivo = ""
    return {"permitido": not motivo, "motivo": motivo, "usado": usado, "limite": limite,
            "bolsa": bolsa, "extras": extras, "porcentaje": min(100, round(usado * 100 / limite)) if limite else 100}


def mensaje_sin_cupo(motivo):
    if motivo == "suscripcion":
        return "Tu suscripción no está activa, así que usé el modo básico. Revisa Plan y uso."
    return "Llegaste a tu cuota mensual de IA, así que usé el modo básico. Puedes sumar un paquete en Plan y uso."


def total_mensual(estudio, usuarios, hoy):
    s = suscripcion(estudio, hoy)
    return s["precio_usuario"] * max(1, usuarios)


def iniciar_prueba(hoy):
    return {"plan": PRUEBA_PLAN, "estado_suscripcion": "prueba",
            "prueba_hasta": (hoy + timedelta(days=PRUEBA_DIAS)).isoformat(), "pagado_hasta": None}
