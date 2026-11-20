"""Revisión de cifras: compara el mes con el historial del cliente y marca lo que no calza."""

from .. import calendario as cal
from ..db import filas
from .base import Agente, clp, firma, porcentaje, primer_nombre, proponer

MESES_HISTORIA = 6


class Control(Agente):
    clave = "control"
    nombre = "Revisión de cifras"
    descripcion = "Compara ventas, compras, IVA y remuneraciones del mes con el historial de cada cliente."
    detalle = ("Antes de declarar, contrasta el mes con el promedio de los últimos seis. Si las ventas o "
               "compras se mueven fuera de lo normal, redacta la consulta al cliente; si el crédito fiscal "
               "se acumula o las remuneraciones saltan, propone una revisión interna.")
    icono = "grafico"

    def ejecutar(self, ctx):
        periodo = cal.periodo_cerrado(ctx.hoy)
        mes = cal.nombre_periodo(periodo)
        desde = cal.mover_periodo(periodo, -MESES_HISTORIA)
        for cliente in ctx.clientes():
            serie = filas(ctx.conn, """
                SELECT * FROM indicadores WHERE cliente_id = ? AND periodo BETWEEN ? AND ? ORDER BY periodo""",
                cliente["id"], desde, periodo)
            if len(serie) < 4 or serie[-1]["periodo"] != periodo:
                continue
            actual, historia = serie[-1], serie[:-1]
            nombre_cliente = cliente["nombre_corto"] or cliente["razon_social"]

            def promedio(campo):
                return sum(h[campo] for h in historia) / len(historia)

            hallazgos = []
            var_ventas = porcentaje(actual["ventas"], promedio("ventas"))
            if var_ventas >= 50 or var_ventas <= -40:
                hallazgos.append(
                    f"las ventas netas de {mes} fueron {clp(actual['ventas'])}, un {abs(var_ventas)}% "
                    f"{'sobre' if var_ventas > 0 else 'bajo'} el promedio de los últimos meses ({clp(promedio('ventas'))})")
            var_compras = porcentaje(actual["compras"], promedio("compras"))
            if var_compras >= 60:
                hallazgos.append(
                    f"las compras llegaron a {clp(actual['compras'])}, un {var_compras}% sobre el promedio "
                    f"({clp(promedio('compras'))})")
            if hallazgos:
                cuerpo = (
                    f"Hola {primer_nombre(cliente['contacto_nombre'])},\n\n"
                    f"Revisando la información de {mes} de {nombre_cliente} notamos que "
                    + " y ".join(hallazgos) + ".\n\n"
                    "¿Nos puedes contar a qué se debe? Por ejemplo, una venta o compra puntual, mercadería de "
                    "temporada o la compra de un activo fijo. Nos ayuda a registrarlo correctamente antes de "
                    "presentar la declaración.\n\n"
                    f"Gracias,\n{firma(ctx, cliente)}"
                )
                motivos = []
                if var_ventas >= 50 or var_ventas <= -40:
                    motivos.append(f"Ventas {'+' if var_ventas > 0 else ''}{var_ventas}% vs. promedio de 6 meses")
                if var_compras >= 60:
                    motivos.append(f"Compras +{var_compras}% vs. promedio de 6 meses")
                proponer(
                    ctx,
                    clave=f"ctrl:{cliente['id']}:{periodo}:variacion",
                    tipo="correo",
                    cliente_id=cliente["id"],
                    titulo=f"Consultar variación de {mes}",
                    resumen=f"{nombre_cliente} · {motivos[0]}",
                    motivos=motivos,
                    prioridad=1,
                    enfriamiento_dias=60,
                    datos={
                        "para": cliente["contacto_email"],
                        "responder_a": cliente.get("responsable_email") or "",
                        "asunto": f"Consulta sobre las cifras de {mes} · {nombre_cliente}",
                        "cuerpo": cuerpo,
                    },
                )

            ultimos = serie[-3:]
            if len(ultimos) == 3 and all(m["iva_credito"] > m["iva_debito"] for m in ultimos):
                acumulado = sum(m["iva_credito"] - m["iva_debito"] for m in ultimos)
                proponer(
                    ctx,
                    clave=f"ctrl:{cliente['id']}:{periodo}:remanente",
                    tipo="tarea",
                    cliente_id=cliente["id"],
                    titulo="Revisar remanente de crédito fiscal",
                    resumen=f"{nombre_cliente} · 3 meses con crédito mayor al débito ({clp(acumulado)})",
                    motivos=[
                        f"Crédito fiscal mayor al débito desde {cal.nombre_periodo(ultimos[0]['periodo'])}",
                        f"Remanente acumulado en el trimestre: {clp(acumulado)}",
                        "Evaluar si corresponde imputación o solicitud de devolución",
                    ],
                    prioridad=2,
                    enfriamiento_dias=90,
                    datos={"tarea": {
                        "titulo": f"Revisar remanente de IVA · {nombre_cliente}",
                        "detalle": (f"El crédito fiscal supera al débito hace tres meses ({clp(acumulado)} acumulado). "
                                    "Verificar el registro de compras y evaluar alternativas con el cliente."),
                        "vence": cal.sumar_habiles(ctx.hoy, 5).isoformat(),
                        "responsable_id": cliente["responsable_id"],
                    }},
                )

            if "remuneraciones" in cliente["servicios"] and actual["remuneraciones"]:
                var_rem = porcentaje(actual["remuneraciones"], promedio("remuneraciones"))
                if var_rem >= 25 and actual["trabajadores"] <= historia[-1]["trabajadores"]:
                    proponer(
                        ctx,
                        clave=f"ctrl:{cliente['id']}:{periodo}:remuneraciones",
                        tipo="tarea",
                        cliente_id=cliente["id"],
                        titulo="Verificar alza de remuneraciones",
                        resumen=f"{nombre_cliente} · +{var_rem}% con la misma dotación",
                        motivos=[
                            f"Remuneraciones de {mes}: {clp(actual['remuneraciones'])} (+{var_rem}%)",
                            f"Trabajadores: {actual['trabajadores']}, sin cambios",
                            "Revisar horas extra, bonos o finiquitos antes de pagar Previred",
                        ],
                        prioridad=2,
                        enfriamiento_dias=60,
                        datos={"tarea": {
                            "titulo": f"Verificar liquidaciones de {mes} · {nombre_cliente}",
                            "detalle": f"Las remuneraciones subieron {var_rem}% sin cambios en la dotación.",
                            "vence": cal.sumar_habiles(ctx.hoy, 2).isoformat(),
                            "responsable_id": cliente["responsable_id"],
                        }},
                    )
