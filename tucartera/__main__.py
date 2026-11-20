"""Ejecuta Tu Cartera y administra las suscripciones.

    python -m tucartera                                  # servidor web en http://localhost:8000
    python -m tucartera servir --host 0.0.0.0 --puerto 8080
    python -m tucartera automatizaciones                 # corre las automatizaciones de todos los estudios (cron)
    python -m tucartera plan socio@estudio.cl profesional --meses 1   # activa o renueva un plan pagado
    python -m tucartera paquete socio@estudio.cl         # suma un paquete de tokens este mes
    python -m tucartera uso                              # consumo, costo y margen por estudio del mes
"""

import argparse
import logging
from datetime import date, datetime


def _hoy(cfg):
    return date.fromisoformat(cfg.hoy_fijo) if cfg.hoy_fijo else datetime.now(cfg.tz).date()


def _sumar_meses(d, meses):
    total = d.year * 12 + d.month - 1 + meses
    anio, mes = divmod(total, 12)
    dia = min(d.day, [31, 29 if anio % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][mes])
    return date(anio, mes + 1, dia)


def _estudio_por_correo(conn, correo):
    from .db import fila

    e = fila(conn, """SELECT e.* FROM estudios e LEFT JOIN usuarios u ON u.estudio_id = e.id
                      WHERE lower(e.email) = lower(?) OR lower(u.email) = lower(?) LIMIT 1""", correo, correo)
    if not e:
        raise SystemExit(f"No hay un estudio con el correo {correo}.")
    return e


def main():
    parser = argparse.ArgumentParser(prog="tucartera", description="Gestión de clientes para estudios contables.")
    sub = parser.add_subparsers(dest="comando")
    servir = sub.add_parser("servir", help="Inicia el servidor web (por defecto).")
    servir.add_argument("--host", default="127.0.0.1")
    servir.add_argument("--puerto", type=int, default=8000)
    sub.add_parser("automatizaciones", help="Corre una vez las automatizaciones de todos los estudios.")
    plan = sub.add_parser("plan", help="Activa o renueva el plan pagado de un estudio.")
    plan.add_argument("correo", help="Correo del estudio o de alguien de su equipo.")
    plan.add_argument("plan", help="esencial, profesional o firma")
    plan.add_argument("--meses", type=int, default=1, help="Meses pagados desde hoy (o desde el vencimiento vigente).")
    paquete = sub.add_parser("paquete", help="Suma tokens extra al estudio por el mes en curso.")
    paquete.add_argument("correo")
    paquete.add_argument("--tokens", type=int, default=None)
    cancelar = sub.add_parser("cancelar", help="Cancela la suscripción de un estudio.")
    cancelar.add_argument("correo")
    uso = sub.add_parser("uso", help="Consumo de IA, costo y margen por estudio.")
    uso.add_argument("--mes", help="AAAA-MM (por defecto, el mes actual)")
    # Compatibilidad con la forma anterior: python -m tucartera --puerto 8080
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--puerto", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from . import calendario as cal
    from . import planes
    from .config import cargar
    from .db import conectar, fila, filas, inicializar, insertar

    cfg = cargar()
    inicializar(cfg.db)
    hoy = _hoy(cfg)

    if args.comando == "automatizaciones":
        from . import llm
        from .api import ejecutar_todos

        for estudio, resultados in ejecutar_todos(cfg, hoy, llm.crear(cfg)).items():
            print(estudio)
            for r in resultados or []:
                print(f"  {r['nombre']}: {r['texto']}")
        return

    if args.comando == "plan":
        if args.plan not in planes.PLANES:
            raise SystemExit(f"Plan desconocido. Opciones: {', '.join(planes.PLANES)}")
        with conectar(cfg.db) as conn:
            e = _estudio_por_correo(conn, args.correo)
            desde = hoy
            if e.get("estado_suscripcion") == "activa" and e.get("pagado_hasta") and date.fromisoformat(e["pagado_hasta"]) > hoy:
                desde = date.fromisoformat(e["pagado_hasta"])
            hasta = _sumar_meses(desde, args.meses)
            conn.execute("UPDATE estudios SET plan = ?, estado_suscripcion = 'activa', pagado_hasta = ? WHERE id = ?",
                         (args.plan, hasta.isoformat(), e["id"]))
        print(f"{e['nombre']}: plan {planes.PLANES[args.plan]['nombre']} activo hasta {hasta.isoformat()}.")
        return

    if args.comando == "paquete":
        tokens = args.tokens or planes.PAQUETE_TOKENS
        with conectar(cfg.db) as conn:
            e = _estudio_por_correo(conn, args.correo)
            insertar(conn, "paquetes_ia", {"estudio_id": e["id"], "periodo": planes.periodo(hoy), "tokens": tokens,
                                           "fecha": cal.ahora(cfg.tz), "nota": "Paquete adicional"})
        print(f"{e['nombre']}: +{tokens:,} tokens para {cal.nombre_periodo(planes.periodo(hoy), True)}.".replace(",", "."))
        return

    if args.comando == "cancelar":
        with conectar(cfg.db) as conn:
            e = _estudio_por_correo(conn, args.correo)
            conn.execute("UPDATE estudios SET estado_suscripcion = 'cancelada' WHERE id = ?", (e["id"],))
        print(f"{e['nombre']}: suscripción cancelada. Sus datos se conservan.")
        return

    if args.comando == "uso":
        periodo = args.mes or planes.periodo(hoy)
        with conectar(cfg.db) as conn:
            estudios = filas(conn, "SELECT * FROM estudios WHERE demo = 0 ORDER BY nombre")
            print(f"Uso de IA · {cal.nombre_periodo(periodo, True)} · modelo {planes.MODELO} · dólar {cfg.dolar_clp} CLP\n")
            print(f"{'Estudio':32} {'Plan':12} {'Estado':10} {'Usuarios':>8} {'Tokens':>12} {'Costo IA':>12} {'Ingreso':>12} {'Margen IA':>9}")
            totales = [0, 0.0, 0]
            for e in estudios:
                s = planes.suscripcion(e, hoy)
                usuarios = fila(conn, "SELECT COUNT(*) AS n FROM usuarios WHERE estudio_id = ?", e["id"])["n"]
                u = fila(conn, "SELECT COALESCE(SUM(tokens), 0) AS t, COALESCE(SUM(costo_usd), 0) AS c FROM uso_ia "
                               "WHERE estudio_id = ? AND periodo = ?", e["id"], periodo)
                costo = u["c"] * cfg.dolar_clp
                ingreso = s["precio_usuario"] * usuarios if s["estado"] == "activa" else 0
                margen = f"{(1 - costo / ingreso) * 100:.0f}%" if ingreso else "—"
                print(f"{e['nombre'][:32]:32} {s['nombre']:12} {s['estado']:10} {usuarios:>8} {u['t']:>12,} "
                      f"{'$' + format(round(costo), ','):>12} {'$' + format(ingreso, ','):>12} {margen:>9}".replace(",", "."))
                totales[0] += u["t"]
                totales[1] += costo
                totales[2] += ingreso
            if totales[0]:
                print(f"\nCosto real por millón de tokens: ${totales[1] / totales[0] * 1e6:,.0f} CLP".replace(",", "."))
            print(f"Total: {totales[0]:,} tokens · costo ${round(totales[1]):,} · ingreso ${totales[2]:,} CLP".replace(",", "."))
        return

    import uvicorn

    from .api import crear_app

    host = getattr(args, "host", "127.0.0.1")
    puerto = getattr(args, "puerto", 8000)
    uvicorn.run(crear_app(cfg), host=host, port=puerto, proxy_headers=True)


if __name__ == "__main__":
    main()
