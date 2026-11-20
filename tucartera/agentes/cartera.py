"""Salud de la cartera: detecta clientes en riesgo y oportunidades de servicio u honorario."""

from datetime import date

from .. import calendario as cal
from ..db import filas
from ..salud import evaluar_cartera
from .base import Agente, clp, firma, porcentaje, primer_nombre, proponer


class Cartera(Agente):
    clave = "cartera"
    nombre = "Salud de la cartera"
    descripcion = "Mide qué tan al día está cada cliente y detecta riesgos y oportunidades."
    detalle = ("Combina declaraciones atrasadas, facturas impagas y documentos sin respuesta en un puntaje "
               "de 0 a 100. Si un cliente cae a riesgo, propone una conversación. También detecta "
               "clientes con trabajadores sin servicio de remuneraciones y honorarios que quedaron atrás.")
    icono = "pulso"

    def ejecutar(self, ctx):
        salud = evaluar_cartera(ctx.conn, ctx.estudio_id, ctx.hoy)
        periodo = cal.periodo_de(ctx.hoy)
        for cliente in ctx.clientes():
            nombre_cliente = cliente["nombre_corto"] or cliente["razon_social"]
            estado = salud.get(cliente["id"])
            if estado and estado["nivel"] == "riesgo":
                proponer(
                    ctx,
                    clave=f"salud:{cliente['id']}:{periodo}",
                    tipo="tarea",
                    cliente_id=cliente["id"],
                    titulo="Conversar con el cliente",
                    resumen=f"{nombre_cliente} · salud {estado['puntaje']}/100",
                    motivos=estado["motivos"],
                    prioridad=1,
                    enfriamiento_dias=14,
                    datos={"tarea": {
                        "titulo": f"Llamar a {cliente['contacto_nombre'] or nombre_cliente} ({nombre_cliente})",
                        "detalle": "Puntos a conversar: " + "; ".join(estado["motivos"]) + ".",
                        "vence": cal.sumar_habiles(ctx.hoy, 2).isoformat(),
                        "responsable_id": cliente["responsable_id"],
                    }},
                )

            if cliente["trabajadores"] > 0 and "remuneraciones" not in cliente["servicios"] and cliente["contacto_email"]:
                trabajadores = cliente["trabajadores"]
                proponer(
                    ctx,
                    clave=f"oferta:remuneraciones:{cliente['id']}",
                    tipo="correo",
                    cliente_id=cliente["id"],
                    titulo="Ofrecer servicio de remuneraciones",
                    resumen=f"{nombre_cliente} · {trabajadores} trabajadores, remuneraciones fuera del estudio",
                    motivos=[f"{trabajadores} trabajadores registrados",
                             "Hoy no contrata remuneraciones con el estudio"],
                    prioridad=3,
                    enfriamiento_dias=120,
                    datos={
                        "para": cliente["contacto_email"],
                        "responder_a": cliente.get("responsable_email") or "",
                        "asunto": f"Remuneraciones de {nombre_cliente}",
                        "cuerpo": (
                            f"Hola {primer_nombre(cliente['contacto_nombre'])},\n\n"
                            f"Como ya llevamos la contabilidad de {nombre_cliente}, quería comentarte que también "
                            f"podemos encargarnos de las remuneraciones de tus {trabajadores} trabajadores: "
                            "liquidaciones de sueldo, pago de cotizaciones en Previred, Libro de Remuneraciones "
                            "Electrónico y contratos.\n\n"
                            "Al estar todo en un mismo lugar, el cierre mensual es más rápido y evitamos "
                            "diferencias entre la contabilidad y las remuneraciones. ¿Te parece si lo "
                            "conversamos 15 minutos esta semana?\n\n"
                            f"Saludos,\n{firma(ctx, cliente)}"
                        ),
                    },
                )

            self._revisar_honorario(ctx, cliente, nombre_cliente)

    def _revisar_honorario(self, ctx, cliente, nombre_cliente):
        if not cliente["honorario"] or not cliente["honorario_desde"]:
            return
        desde = date.fromisoformat(cliente["honorario_desde"])
        if (ctx.hoy - desde).days < 365:
            return
        cerrado = cal.periodo_cerrado(ctx.hoy)
        serie = {i["periodo"]: i["ventas"] for i in filas(ctx.conn, """
            SELECT periodo, ventas FROM indicadores WHERE cliente_id = ?""", cliente["id"])}
        recientes = [serie.get(cal.mover_periodo(cerrado, -i)) for i in range(6)]
        anteriores = [serie.get(cal.mover_periodo(cerrado, -12 - i)) for i in range(6)]
        if None in recientes or None in anteriores:
            return
        variacion = porcentaje(sum(recientes), sum(anteriores))
        if variacion < 35:
            return
        socio = ctx.socio()
        proponer(
            ctx,
            clave=f"honorario:{cliente['id']}:{ctx.hoy.year}",
            tipo="tarea",
            cliente_id=cliente["id"],
            titulo="Revisar honorario mensual",
            resumen=f"{nombre_cliente} · ventas +{variacion}% en 12 meses",
            motivos=[
                f"Ventas de los últimos 6 meses: +{variacion}% vs. el mismo período del año anterior",
                f"Honorario actual: {clp(cliente['honorario'])}, sin cambios desde {cal.nombre_periodo(cliente['honorario_desde'][:7], True)}",
                "Más volumen suele implicar más horas de registro y revisión",
            ],
            prioridad=3,
            enfriamiento_dias=180,
            datos={"tarea": {
                "titulo": f"Revisar honorario de {nombre_cliente}",
                "detalle": f"El negocio creció {variacion}% y el honorario no se ajusta hace más de un año.",
                "vence": cal.sumar_habiles(ctx.hoy, 10).isoformat(),
                "responsable_id": socio["id"] if socio else cliente["responsable_id"],
            }},
        )
