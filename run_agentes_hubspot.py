"""Ejecuta a los agentes del embudo de HubSpot desde la terminal, cron o GitHub Actions.

Ejemplos:
    python run_agentes_hubspot.py                 # diagnóstico + desatascador; envía el informe si hay Gmail
    python run_agentes_hubspot.py --demo          # datos ficticios: no toca HubSpot ni envía nada
    python run_agentes_hubspot.py --sin-correo    # solo guarda el informe en data/informes
    python run_agentes_hubspot.py --imprimir      # muestra el informe completo (Markdown) en pantalla
    python run_agentes_hubspot.py --modo tareas   # crea tareas en HubSpot en esta ejecución
"""

import argparse
import dataclasses
import logging
import sys

from agentes_hubspot import ConfigHubSpot, abrir_db, ejecutar_ciclo, preparar_demo
from agentes_hubspot.config import MODOS


def main():
    parser = argparse.ArgumentParser(description="Agentes del embudo de HubSpot: Analista de embudo y Desatascador")
    parser.add_argument("--demo", action="store_true", help="usar datos ficticios (no se conecta a HubSpot)")
    parser.add_argument("--modo", choices=MODOS, help="lectura (solo informa) o tareas (crea tareas en HubSpot)")
    parser.add_argument("--sin-correo", action="store_true", help="no enviar el informe por correo")
    parser.add_argument("--sin-ia", action="store_true", help="no usar Claude aunque haya API key")
    parser.add_argument("--imprimir", action="store_true", help="mostrar el informe completo en pantalla")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = ConfigHubSpot.desde_entorno()
    if args.modo:
        cfg = dataclasses.replace(cfg, modo=args.modo)
    if args.demo:
        cfg = dataclasses.replace(cfg, ruta_db=cfg.ruta_db.replace(".db", "_demo.db"))
    elif not cfg.hubspot_configurado:
        print(
            "Falta HUBSPOT_TOKEN (token de una app privada de HubSpot). "
            "Prueba con --demo para ver el informe con datos ficticios."
        )
        return 2

    db = abrir_db(cfg)
    try:
        if args.demo:
            resultado, _ = preparar_demo(cfg, db)
        else:
            resultado = ejecutar_ciclo(cfg, db, enviar=not args.sin_correo, usar_ia=not args.sin_ia)
    finally:
        db.cerrar()  # deja todo escrito en el archivo .db (importante en GitHub Actions)
    print(resultado.texto(), flush=True)
    if args.imprimir and resultado.markdown:
        print("\n" + resultado.markdown)
    return 1 if resultado.errores else 0


if __name__ == "__main__":
    sys.exit(main())
