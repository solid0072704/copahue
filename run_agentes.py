"""Ejecuta a los agentes del ebook desde la terminal, cron o GitHub Actions.

Ejemplos:
    python run_agentes.py                 # un ciclo completo
    python run_agentes.py --loop 15       # un ciclo cada 15 minutos (deja tu PC o servidor trabajando)
    python run_agentes.py --resumen       # fuerza el envío del resumen diario
    python run_agentes.py --demo          # carga datos ficticios para probar (no envía nada)
"""

import argparse
import logging
import sys
import time

from agentes_ebook import Config, abrir_db, ejecutar_ciclo, preparar_demo


def main():
    parser = argparse.ArgumentParser(description="Agentes de seguimiento de descargas del ebook")
    parser.add_argument("--loop", type=int, metavar="MIN", help="repetir cada MIN minutos")
    parser.add_argument("--resumen", action="store_true", help="forzar el resumen diario por correo")
    parser.add_argument("--sin-notificar", action="store_true", help="no enviar alertas ni resumen")
    parser.add_argument("--sin-ia", action="store_true", help="no usar Claude aunque haya API key")
    parser.add_argument("--demo", action="store_true", help="cargar datos de demostración")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config.desde_entorno()
    db = abrir_db(cfg)

    try:
        if args.demo:
            print("Demo:", preparar_demo(db, cfg).texto())
            return 0
        if not cfg.gmail_configurado:
            print(
                "Gmail no está configurado (GMAIL_USER / GMAIL_APP_PASSWORD). "
                "Los agentes solo analizarán los datos ya guardados."
            )
        while True:
            resultado = ejecutar_ciclo(
                db, cfg, notificar=not args.sin_notificar, forzar_resumen=args.resumen, usar_ia=not args.sin_ia
            )
            print(time.strftime("%Y-%m-%d %H:%M"), "·", resultado.texto(), flush=True)
            if not args.loop:
                return 1 if resultado.errores else 0
            args.resumen = False
            time.sleep(args.loop * 60)
    finally:
        db.cerrar()  # deja todo escrito en el archivo .db (importante en GitHub Actions)


if __name__ == "__main__":
    sys.exit(main())
