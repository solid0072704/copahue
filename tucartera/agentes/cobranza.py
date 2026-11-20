"""Cobranza de honorarios: recuerda con tacto las facturas vencidas y escala las antiguas."""

from collections import defaultdict
from datetime import date

from .. import calendario as cal
from ..db import filas
from .base import Agente, clp, firma, primer_nombre, proponer


class Cobranza(Agente):
    clave = "cobranza"
    nombre = "Cobranza de honorarios"
    descripcion = "Recuerda las facturas vencidas del estudio con un tono cordial y escala las antiguas."
    detalle = ("Hasta 14 días de atraso prepara un recordatorio amable; entre 15 y 44 días, un segundo "
               "aviso; desde 45 días propone que el socio llame al cliente. Nunca envía más de un "
               "recordatorio por semana al mismo cliente.")
    icono = "tarjeta"

    def ejecutar(self, ctx):
        clientes = {c["id"]: c for c in ctx.clientes()}
        vencidas = defaultdict(list)
        for f in filas(ctx.conn, """
            SELECT * FROM facturas WHERE estudio_id = ? AND pagada_el IS NULL AND vence < ? ORDER BY vence""",
            ctx.estudio_id, ctx.hoy.isoformat()):
            if f["cliente_id"] in clientes:
                vencidas[f["cliente_id"]].append(f)

        for cid, facturas in vencidas.items():
            cliente = clientes[cid]
            nombre_cliente = cliente["nombre_corto"] or cliente["razon_social"]
            dias = max((ctx.hoy - date.fromisoformat(f["vence"])).days for f in facturas)
            total = sum(f["monto"] for f in facturas)
            etapa = 1 if dias < 15 else 2 if dias < 45 else 3
            detalle_facturas = "\n".join(
                f"• Factura N.º {f['folio']} · {cal.nombre_periodo(f['periodo'], True)} · {clp(f['monto'])} "
                f"· venció el {cal.fecha_larga(f['vence'], con_dia=False)}" for f in facturas)
            motivos = [f"Factura {f['folio']}: {clp(f['monto'])}, {(ctx.hoy - date.fromisoformat(f['vence'])).days} días"
                       for f in facturas]
            efectos = [{"tipo": "factura_recordada", "id": f["id"]} for f in facturas]
            plural = len(facturas) > 1

            if etapa == 3:
                socio = ctx.socio()
                proponer(
                    ctx,
                    clave=f"cobro:{cid}:3",
                    tipo="tarea",
                    cliente_id=cid,
                    titulo=f"Llamar por honorarios impagos ({dias} días)",
                    resumen=f"{nombre_cliente} · {clp(total)} adeudado",
                    motivos=motivos + ["Ya recibió recordatorios por correo" if any(f["recordatorios"] for f in facturas)
                                       else "Aún no recibe recordatorios"],
                    prioridad=1,
                    enfriamiento_dias=14,
                    datos={
                        "tarea": {
                            "titulo": f"Llamar a {cliente['contacto_nombre'] or nombre_cliente} por honorarios",
                            "detalle": (f"{nombre_cliente} adeuda {clp(total)} con {dias} días de atraso. "
                                        "Acordar fecha de pago o evaluar pausar el servicio."
                                        + (f" Teléfono: {cliente['contacto_telefono']}." if cliente["contacto_telefono"] else "")),
                            "vence": cal.sumar_habiles(ctx.hoy, 1).isoformat(),
                            "responsable_id": socio["id"] if socio else cliente["responsable_id"],
                        },
                        "efectos": efectos,
                    },
                )
                continue

            if not cliente["contacto_email"]:
                continue
            if etapa == 1:
                asunto = f"Recordatorio de pago · {nombre_cliente}"
                apertura = (f"Te escribimos para recordarte que {'estas facturas vencieron' if plural else 'esta factura venció'} "
                            "hace algunos días:")
            else:
                asunto = f"Segundo aviso: honorarios pendientes · {nombre_cliente}"
                apertura = (f"Aún {'figuran pendientes estas facturas' if plural else 'figura pendiente esta factura'} "
                            "de nuestros servicios:")
            cuerpo = (
                f"Hola {primer_nombre(cliente['contacto_nombre'])},\n\n{apertura}\n\n{detalle_facturas}\n\n"
                f"Total pendiente: {clp(total)}.\n\n"
                "Si ya realizaste el pago, envíanos el comprobante respondiendo este correo y lo registramos. "
                "Si necesitas coordinar otra fecha, cuéntanos y lo vemos.\n\n"
                f"Saludos,\n{firma(ctx, cliente)}"
            )
            proponer(
                ctx,
                clave=f"cobro:{cid}:{etapa}",
                tipo="correo",
                cliente_id=cid,
                titulo="Recordar pago de honorarios" if etapa == 1 else "Segundo aviso de pago",
                resumen=f"{nombre_cliente} · {clp(total)} · {dias} días de atraso",
                motivos=motivos,
                prioridad=2 if etapa == 1 else 1,
                enfriamiento_dias=7,
                datos={
                    "para": cliente["contacto_email"],
                    "responder_a": ctx.estudio.get("email") or "",
                    "asunto": asunto,
                    "cuerpo": cuerpo,
                    "efectos": efectos,
                },
            )
