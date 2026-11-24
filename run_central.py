"""Central de WhatsApp: Breeze (HubSpot) conversa con los clientes y este servicio reparte los leads.

Puesta en marcha (ver central_whatsapp/README.md):
    python run_central.py instalar                    # 1. crea las propiedades en HubSpot (una vez)
    python run_central.py sugerir-ruteo               # 2. arma central_whatsapp/ruteo.yaml con tus datos reales
    python run_central.py validar                     # 3. revisa el ruteo contra los vendedores activos
    python run_central.py simular "Raíces de Curicó"  #    cómo se repartirían 20 leads (sin tocar HubSpot)
    python run_central.py asignar 123456 --simular    # 4. prueba con un contacto real sin escribir nada
    python run_central.py trabajar                    # 5. reparte los pendientes cada minuto y revisa el SLA

Otros:
    python run_central.py consultar                   # una sola vuelta de reparto (para cron / GitHub Actions)
    python run_central.py sla [--loop 5]              # solo la revisión del SLA
    python run_central.py servir --puerto 8000        # modo webhook (requiere Data Hub Professional en HubSpot)
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta

from central_whatsapp.config import ConfigCentral
from central_whatsapp.ruteo import ErrorRuteo, cargar_tabla, elegir_vendedor


def _cliente(cfg):
    from central_whatsapp.hubspot import ClienteHubSpot

    if not cfg.hubspot_configurado:
        sys.exit("Falta HUBSPOT_TOKEN (token de la app privada de HubSpot). Revisa tu .env.")
    return ClienteHubSpot(cfg.hubspot_token, cfg.hubspot_portal_id)


def cmd_instalar(cfg, args):
    from central_whatsapp.instalar import instalar

    for linea in instalar(_cliente(cfg)):
        print(linea)
    print("Listo. Las propiedades quedaron en el grupo «Central WhatsApp» de contactos y negocios.")


def cmd_sugerir(cfg, args):
    from central_whatsapp.sugerir import a_yaml, sugerir_ruteo

    datos = sugerir_ruteo(_cliente(cfg), dias=args.dias, participacion_minima=args.minimo / 100, supervisor=args.supervisor)
    if os.path.exists(args.salida) and not args.sobrescribir:
        sys.exit(f"{args.salida} ya existe. Usa --sobrescribir o --salida otro_archivo.yaml")
    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    with open(args.salida, "w", encoding="utf-8") as f:
        f.write(a_yaml(datos, dias=args.dias))
    print(f"{len(datos['proyectos'])} proyectos con vendedores → {args.salida}")
    if not datos["sla"]["supervisor"]:
        print("Falta definir sla.supervisor (quién recibe los leads que nadie atendió).")


def cmd_validar(cfg, args):
    tabla = cargar_tabla(cfg)
    propietarios = _cliente(cfg).propietarios() if cfg.hubspot_configurado else None
    avisos = tabla.validar(propietarios)
    print(f"{len(tabla.proyectos)} proyectos · {len(tabla.todos_los_vendedores())} vendedores · SLA {tabla.sla_minutos} min")
    for a in avisos:
        print("⚠️ ", a)
    if not avisos:
        print("Sin observaciones.")
    return 1 if avisos and propietarios is not None else 0


def cmd_simular(cfg, args):
    """Reparte N leads ficticios con las reglas del archivo (sin HubSpot) para revisar los pesos."""
    tabla = cargar_tabla(cfg)
    ahora = datetime.fromisoformat(args.hora).replace(tzinfo=tabla.tz) if args.hora else datetime.now(tabla.tz)
    candidatos, origen, proyecto = tabla.candidatos(args.proyecto)
    cargas_p, cargas_t = Counter(), Counter()
    for _ in range(args.leads):
        v = elegir_vendedor(tabla, candidatos, ahora, cargas_p, cargas_t)
        if not v:
            print("Nadie disponible: el lead iría al supervisor", tabla.supervisor or "(no configurado)")
            break
        cargas_p[v.id] += 1
        cargas_t[v.id] += 1
        ahora += timedelta(minutes=7)
    nombre = proyecto.nombre if proyecto else (args.proyecto or "sin proyecto")
    print(f"{nombre} · origen de los candidatos: {origen}")
    for v in candidatos:
        print(f"  {cargas_p[v.id]:>3} leads  peso {v.peso:<2} {v.nombre or v.id}")


def cmd_asignar(cfg, args):
    from central_whatsapp.distribuidor import construir

    if not cfg.hubspot_configurado:
        sys.exit("Falta HUBSPOT_TOKEN.")
    dist = construir(cfg, simular=args.simular)
    r = dist.procesar(args.contacto, proyecto=args.proyecto)
    print(json.dumps(r.como_dict(), ensure_ascii=False, indent=2))
    if args.simular:
        print("\nModo simulación: esto es lo que se habría escrito en HubSpot:")
        for e in dist.hs.escrituras:
            if e[0] == "crear":
                print(f"  crear {e[1]}: {json.dumps(e[2], ensure_ascii=False)[:400]}")
            elif e[0] == "actualizar":
                print(f"  actualizar {e[1]} {e[2]}: {json.dumps(e[3], ensure_ascii=False)[:400]}")


def cmd_servir(cfg, args):
    import uvicorn

    from central_whatsapp.servicio import crear_app

    if not cfg.hubspot_configurado:
        sys.exit("Falta HUBSPOT_TOKEN.")
    if not cfg.webhook_secreto:
        sys.exit("Falta CENTRAL_WEBHOOK_SECRET (el secreto que enviará el workflow de HubSpot).")
    uvicorn.run(crear_app(cfg), host=args.host, port=args.puerto, workers=1, log_level="info")


def cmd_consultar(cfg, args):
    from central_whatsapp.consulta import consultar_pendientes
    from central_whatsapp.distribuidor import construir

    if not cfg.hubspot_configurado:
        sys.exit("Falta HUBSPOT_TOKEN.")
    r = consultar_pendientes(construir(cfg))
    print(time.strftime("%Y-%m-%d %H:%M"), "· reparto:", r.texto(), flush=True)  # solo cifras
    return 1 if r.errores else 0


def cmd_trabajar(cfg, args):
    """Proceso permanente: reparte pendientes cada CENTRAL_CONSULTA_CADA_SEG y revisa el SLA cada CENTRAL_SLA_CADA_MIN."""
    from central_whatsapp.consulta import consultar_pendientes
    from central_whatsapp.distribuidor import construir
    from central_whatsapp.sla import revisar_sla

    if not cfg.hubspot_configurado:
        sys.exit("Falta HUBSPOT_TOKEN.")
    dist = construir(cfg)
    modo = " (SIMULACIÓN: no escribe en HubSpot)" if cfg.simular else ""
    print(f"Central trabajando{modo}: reparto cada {cfg.consulta_cada_seg} s, SLA cada {cfg.sla_cada_min} min", flush=True)
    ultimo_sla = 0.0
    while True:
        try:
            r = consultar_pendientes(dist)
            if r.pendientes or r.errores:
                print(time.strftime("%Y-%m-%d %H:%M:%S"), "· reparto:", r.texto(), flush=True)
            if time.monotonic() - ultimo_sla >= cfg.sla_cada_min * 60:
                ultimo_sla = time.monotonic()
                s = revisar_sla(dist)
                if s.reasignados or s.escalados or s.errores:
                    print(time.strftime("%Y-%m-%d %H:%M:%S"), "· SLA:", s.texto(), flush=True)
        except Exception as e:  # el proceso nunca debe caerse por un error pasajero
            logging.getLogger("central").warning("Vuelta fallida (%s): %s", type(e).__name__, e)
        if args.una_vez:
            return 0
        time.sleep(cfg.consulta_cada_seg)


def cmd_sla(cfg, args):
    from central_whatsapp.distribuidor import construir
    from central_whatsapp.sla import revisar_sla

    if not cfg.hubspot_configurado:
        sys.exit("Falta HUBSPOT_TOKEN.")
    dist = construir(cfg)
    while True:
        r = revisar_sla(dist)
        print(time.strftime("%Y-%m-%d %H:%M"), "·", r.texto(), flush=True)  # solo cifras: sin datos de clientes
        if not args.loop:
            return 1 if r.errores else 0
        time.sleep(args.loop * 60)


def main():
    parser = argparse.ArgumentParser(description="Central de WhatsApp: reparto de leads de Breeze a los vendedores")
    sub = parser.add_subparsers(dest="comando", required=True)

    sub.add_parser("instalar", help="crea las propiedades de la central en HubSpot")

    s = sub.add_parser("sugerir-ruteo", help="arma el archivo de ruteo con los negocios de los últimos días")
    s.add_argument("--dias", type=int, default=90)
    s.add_argument("--minimo", type=float, default=8, help="participación mínima (%%) para quedar en un proyecto")
    s.add_argument("--supervisor", help="ID de propietario que recibe los leads sin atender")
    s.add_argument("--salida", default="central_whatsapp/ruteo.yaml")
    s.add_argument("--sobrescribir", action="store_true")

    sub.add_parser("validar", help="revisa el archivo de ruteo (y los vendedores activos si hay token)")

    s = sub.add_parser("simular", help="reparte leads ficticios para revisar los pesos (sin HubSpot)")
    s.add_argument("proyecto", nargs="?", default="")
    s.add_argument("--leads", type=int, default=20)
    s.add_argument("--hora", help="fecha y hora local, p. ej. 2026-11-25T10:00")

    s = sub.add_parser("asignar", help="rutea un contacto real (usa --simular para no escribir en HubSpot)")
    s.add_argument("contacto")
    s.add_argument("--proyecto")
    s.add_argument("--simular", action="store_true")

    s = sub.add_parser("trabajar", help="reparte los pendientes cada minuto y revisa el SLA (recomendado)")
    s.add_argument("--una-vez", action="store_true", help="una sola vuelta y salir (para probar)")

    sub.add_parser("consultar", help="una vuelta de reparto de los contactos pendientes")

    s = sub.add_parser("servir", help="modo webhook: servicio web que recibe el aviso de HubSpot")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--puerto", type=int, default=int(os.environ.get("PORT", "8000")))

    s = sub.add_parser("sla", help="reasigna los leads que nadie atendió a tiempo")
    s.add_argument("--loop", type=int, metavar="MIN", help="repetir cada MIN minutos")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = ConfigCentral.desde_entorno()
    comandos = {
        "instalar": cmd_instalar,
        "sugerir-ruteo": cmd_sugerir,
        "validar": cmd_validar,
        "simular": cmd_simular,
        "asignar": cmd_asignar,
        "trabajar": cmd_trabajar,
        "consultar": cmd_consultar,
        "servir": cmd_servir,
        "sla": cmd_sla,
    }
    try:
        return comandos[args.comando](cfg, args) or 0
    except ErrorRuteo as e:
        sys.exit(f"Error en el archivo de ruteo: {e}")


if __name__ == "__main__":
    sys.exit(main())
