"""Ejecuta el agente de compras desde la terminal o cron.

Ejemplos:
    python run_super.py                 # un ciclo: cotiza las compras que se acercan, busca ofertas y avisa
    python run_super.py --loop 360      # un ciclo cada 6 horas (deja tu PC o servidor trabajando)
    python run_super.py --lista 1       # cotiza ahora la lista 1 y muestra el resultado
    python run_super.py --demo          # supermercados ficticios para probar sin internet
"""

import argparse
import logging
import sys
import time

from agentes_super import ConfigSuper, abrir_db, cotizar_lista, ejecutar_ciclo, preparar_demo
from agentes_super.notificador import texto_compra


def main():
    parser = argparse.ArgumentParser(description="Agente de compras de supermercado")
    parser.add_argument("--loop", type=int, metavar="MIN", help="repetir cada MIN minutos")
    parser.add_argument("--lista", type=int, metavar="ID", help="cotizar ahora una lista")
    parser.add_argument("--sin-notificar", action="store_true", help="no enviar correos")
    parser.add_argument("--sin-ia", action="store_true", help="no usar Claude aunque haya API key")
    parser.add_argument("--demo", action="store_true", help="cargar datos de demostración")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = ConfigSuper.desde_entorno()
    db = abrir_db(cfg)
    try:
        if args.demo:
            print("Demo:", preparar_demo(db, cfg).texto())
            return 0
        if args.lista:
            compra_id, cot = cotizar_lista(db, cfg, args.lista)
            for e in cot.errores:
                print("⚠️ ", e)
            if not compra_id:
                print("No se encontraron productos.")
                return 1
            compra = db.compra(compra_id)
            print(texto_compra(db.lista(args.lista), compra))
            return 0
        while True:
            resultado = ejecutar_ciclo(db, cfg, notificar=not args.sin_notificar, usar_ia=not args.sin_ia)
            print(time.strftime("%Y-%m-%d %H:%M"), "·", resultado.texto(), flush=True)
            for e in resultado.errores:
                print("⚠️ ", e)
            if not args.loop:
                return 1 if resultado.errores and not resultado.cotizadas else 0
            time.sleep(args.loop * 60)
    finally:
        db.cerrar()


if __name__ == "__main__":
    sys.exit(main())
