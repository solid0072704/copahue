"""Solicitud de documentos: pide al cliente lo que falta para cerrar el mes y recuerda si no llega."""

from datetime import date, timedelta

from .. import calendario as cal
from ..db import filas
from .base import Agente, firma, primer_nombre, proponer

DIAS_ENTRE_RECORDATORIOS = 3


def asegurar_documentos(conn, estudio_id, cliente, periodo):
    for tipo in cal.documentos_requeridos(cliente["servicios"], cliente["trabajadores"]):
        conn.execute("""
            INSERT OR IGNORE INTO documentos (estudio_id, cliente_id, periodo, tipo, estado)
            VALUES (?, ?, ?, ?, 'pendiente')""", (estudio_id, cliente["id"], periodo, tipo))


class Documentos(Agente):
    clave = "documentos"
    nombre = "Solicitud de documentos"
    descripcion = "Pide a cada cliente lo que falta para cerrar el mes y recuerda si no llega."
    detalle = ("Al comenzar el mes arma la lista de documentos por cliente (cartolas, gastos, novedades "
               "de remuneraciones) y redacta la solicitud con la fecha del próximo vencimiento. "
               f"Si no hay respuesta en {DIAS_ENTRE_RECORDATORIOS} días, prepara un recordatorio.")
    icono = "documento"

    def ejecutar(self, ctx):
        periodo = cal.periodo_cerrado(ctx.hoy)
        nombre_mes = cal.nombre_periodo(periodo)
        for cliente in ctx.clientes():
            asegurar_documentos(ctx.conn, ctx.estudio_id, cliente, periodo)
            faltan = filas(ctx.conn, """
                SELECT * FROM documentos WHERE cliente_id = ? AND periodo = ? AND estado != 'recibido'
                ORDER BY id""", cliente["id"], periodo)
            if not faltan or not cliente["contacto_email"]:
                continue
            obligaciones = filas(ctx.conn, """
                SELECT * FROM obligaciones WHERE cliente_id = ? AND periodo = ? AND estado != 'presentada'
                ORDER BY vence""", cliente["id"], periodo)
            if not obligaciones:
                continue  # ya se declaró todo el mes: no tiene sentido seguir pidiendo
            ultima = max((d["solicitado_el"] for d in faltan if d["solicitado_el"]), default=None)
            if ultima and ctx.hoy - date.fromisoformat(ultima) < timedelta(days=DIAS_ENTRE_RECORDATORIOS):
                continue
            envio = max(d["recordatorios"] for d in faltan) + 1
            proxima = obligaciones[0]
            vence = date.fromisoformat(proxima["vence"])
            tipo = cal.TIPOS[proxima["tipo"]]["nombre"]
            nombre_cliente = cliente["nombre_corto"] or cliente["razon_social"]
            lista = "\n".join(f"• {d['tipo']} de {nombre_mes}" for d in faltan)
            if envio == 1:
                asunto = f"Documentos de {nombre_mes} · {nombre_cliente}"
                apertura = (f"Estamos preparando el cierre de {nombre_mes} de {nombre_cliente}. Para presentar "
                            f"el {tipo} a tiempo (vence el {cal.fecha_larga(vence)}) necesitamos:")
            else:
                asunto = f"Recordatorio: documentos de {nombre_mes} · {nombre_cliente}"
                apertura = (f"Te escribimos de nuevo porque aún nos faltan estos documentos de {nombre_mes}. "
                            f"El {tipo} vence el {cal.fecha_larga(vence)}:")
            cuerpo = (
                f"Hola {primer_nombre(cliente['contacto_nombre'])},\n\n{apertura}\n\n{lista}\n\n"
                "Puedes responder este correo con los archivos adjuntos. Si alguno no aplica este mes, "
                "avísanos y lo marcamos como recibido.\n\n"
                f"Gracias,\n{firma(ctx, cliente)}"
            )
            habiles = cal.habiles_hasta(ctx.hoy, vence)
            motivos = [d["tipo"] for d in faltan]
            motivos.append("Primer aviso" if envio == 1 else
                           f"Recordatorio n.º {envio - 1}: el último aviso fue el {cal.fecha_larga(ultima, con_dia=False)}")
            proponer(
                ctx,
                clave=f"docs:{cliente['id']}:{periodo}:{envio}",
                tipo="correo",
                cliente_id=cliente["id"],
                titulo=("Pedir" if envio == 1 else "Recordar") + f" documentos de {nombre_mes}",
                resumen=f"{nombre_cliente} · faltan {len(faltan)} · {tipo} vence el {cal.fecha_larga(vence)}",
                motivos=motivos,
                prioridad=1 if habiles <= 3 else 2,
                datos={
                    "para": cliente["contacto_email"],
                    "responder_a": cliente.get("responsable_email") or "",
                    "asunto": asunto,
                    "cuerpo": cuerpo,
                    "efectos": [{"tipo": "documentos_solicitados", "ids": [d["id"] for d in faltan]}],
                },
            )
