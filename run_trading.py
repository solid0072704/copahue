"""Ejecuta a los agentes de trading desde la terminal, cron o GitHub Actions.

Ejemplos:
    python run_trading.py                     # un ciclo completo
    python run_trading.py --loop 10           # un ciclo cada 10 minutos (deja tu PC o servidor trabajando)
    python run_trading.py --depositar 1000    # deposita US$ 1.000 en la cartera simulada
    python run_trading.py --plan              # pide un plan nuevo ahora
    python run_trading.py --demo              # cartera de práctica con precios sintéticos (no envía nada)
"""

import argparse
import logging
import sys
import time

from agentes_trading import Config, abrir_db, ejecutar_ciclo, preparar, preparar_demo
from agentes_trading.brokers import ErrorBroker


def main():
    parser = argparse.ArgumentParser(description="Agentes de trading con tu aprobación")
    parser.add_argument("--loop", type=int, metavar="MIN", help="repetir cada MIN minutos")
    parser.add_argument("--depositar", type=float, metavar="USD", help="depositar en la cartera simulada")
    parser.add_argument("--plan", action="store_true", help="generar un plan nuevo ahora")
    parser.add_argument("--sin-ia", action="store_true", help="no usar Claude aunque haya API key")
    parser.add_argument("--demo", action="store_true", help="cartera de demostración con precios sintéticos")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config.desde_entorno()
    db = abrir_db(cfg)

    try:
        if args.demo:
            print("Demo:", preparar_demo(db, cfg).texto())
            return 0
        listo, estado = cfg.broker_listo
        print("Broker:", estado)
        if not listo:
            return 1
        if args.depositar:
            _, broker, _ = preparar(cfg, db, tg=False)
            try:
                broker.depositar(args.depositar)
            except ErrorBroker as e:
                print(e)
                return 1
            print(f"Depositaste US$ {args.depositar:,.2f}.")
        if not cfg.telegram_configurado:
            print("Telegram no está configurado: aprueba o rechaza las propuestas en el tablero (streamlit run tablero_trading.py).")
        while True:
            resultado = ejecutar_ciclo(db, cfg, usar_ia=not args.sin_ia, forzar_plan=args.plan)
            print(time.strftime("%Y-%m-%d %H:%M"), "·", resultado.texto(), flush=True)
            if not args.loop:
                return 1 if resultado.errores else 0
            args.plan = False
            time.sleep(args.loop * 60)
    finally:
        db.cerrar()  # deja todo escrito en el archivo .db (importante en GitHub Actions)


if __name__ == "__main__":
    sys.exit(main())
