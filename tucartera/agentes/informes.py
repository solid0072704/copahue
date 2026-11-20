"""Informe mensual: cuando todo el mes quedó declarado, prepara el resumen para el cliente."""

from datetime import date

from .. import calendario as cal
from ..db import fila, filas
from .base import Agente, clp, firma, porcentaje, primer_nombre, proponer


class Informes(Agente):
    clave = "informes"
    nombre = "Informe mensual al cliente"
    descripcion = "Cuando el mes quedó declarado, prepara el resumen para el cliente."
    detalle = ("Apenas se presentan todas las declaraciones del mes, redacta un resumen con lo pagado, "
               "la variación de ventas y los próximos vencimientos. El cliente sabe en qué está sin "
               "tener que preguntar.")
    icono = "informe"

    def ejecutar(self, ctx):
        periodo = cal.periodo_cerrado(ctx.hoy)
        mes = cal.nombre_periodo(periodo)
        for cliente in ctx.clientes():
            if not cliente["contacto_email"]:
                continue
            obligaciones = filas(ctx.conn, """
                SELECT * FROM obligaciones WHERE cliente_id = ? AND periodo = ? ORDER BY vence""",
                cliente["id"], periodo)
            if not obligaciones or any(o["estado"] != "presentada" for o in obligaciones):
                continue
            nombre_cliente = cliente["nombre_corto"] or cliente["razon_social"]
            lineas = []
            for o in obligaciones:
                tipo = cal.TIPOS[o["tipo"]]
                texto = f"• {tipo['nombre']} ({tipo['descripcion'].lower()}): presentado el " \
                        f"{cal.fecha_larga(o['presentada_el'] or o['vence'], con_dia=False)}"
                if o["tipo"] == "F29" and o["monto"] is not None:
                    texto += f" · IVA a pagar {clp(o['monto'])}" if o["monto"] else " · sin IVA a pagar"
                lineas.append(texto)

            actual = fila(ctx.conn, "SELECT * FROM indicadores WHERE cliente_id = ? AND periodo = ?",
                          cliente["id"], periodo)
            previo = fila(ctx.conn, "SELECT * FROM indicadores WHERE cliente_id = ? AND periodo = ?",
                          cliente["id"], cal.mover_periodo(periodo, -1))
            cifras = ""
            if actual:
                cifras = f"\nVentas netas del mes: {clp(actual['ventas'])}"
                if previo and previo["ventas"]:
                    var = porcentaje(actual["ventas"], previo["ventas"])
                    cifras += f" ({'+' if var >= 0 else ''}{var}% respecto de {cal.nombre_periodo(previo['periodo'])})"
                cifras += f".\nCompras netas: {clp(actual['compras'])}.\n"

            proximas = filas(ctx.conn, """
                SELECT * FROM obligaciones WHERE cliente_id = ? AND estado != 'presentada' AND vence >= ?
                ORDER BY vence LIMIT 3""", cliente["id"], ctx.hoy.isoformat())
            agenda = ""
            if proximas:
                agenda = "\nPróximos vencimientos:\n" + "\n".join(
                    f"• {cal.TIPOS[p['tipo']]['nombre']} de {cal.nombre_periodo(p['periodo'])}: "
                    f"{cal.fecha_larga(date.fromisoformat(p['vence']))}" for p in proximas) + "\n"

            cuerpo = (
                f"Hola {primer_nombre(cliente['contacto_nombre'])},\n\n"
                f"Ya quedó todo {mes} de {nombre_cliente} al día. Este es el resumen:\n\n"
                + "\n".join(lineas) + "\n" + cifras + agenda +
                "\nCualquier duda, respondemos por aquí.\n\n"
                f"Saludos,\n{firma(ctx, cliente)}"
            )
            proponer(
                ctx,
                clave=f"informe:{cliente['id']}:{periodo}",
                tipo="correo",
                cliente_id=cliente["id"],
                titulo=f"Enviar resumen de {mes}",
                resumen=f"{nombre_cliente} · {len(obligaciones)} declaraci{'ones' if len(obligaciones) > 1 else 'ón'} presentada{'s' if len(obligaciones) > 1 else ''}",
                motivos=[cal.TIPOS[o["tipo"]]["nombre"] + " presentado" for o in obligaciones],
                prioridad=3,
                enfriamiento_dias=365,
                datos={
                    "para": cliente["contacto_email"],
                    "responder_a": cliente.get("responsable_email") or "",
                    "asunto": f"Resumen de {mes} · {nombre_cliente}",
                    "cuerpo": cuerpo,
                },
            )
