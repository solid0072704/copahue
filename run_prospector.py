"""Ejecuta el prospector de LinkedIn desde la terminal o cron.

El prospector NO entra a LinkedIn ni envía nada allí: califica tus prospectos, prepara los textos,
lee en tu Gmail los avisos de LinkedIn (quién aceptó, quién respondió) y te manda tu cola del día.

Ejemplos:
    python run_prospector.py                          # un ciclo
    python run_prospector.py --loop 30                # un ciclo cada 30 minutos
    python run_prospector.py --importar Connections.csv   # tus contactos exportados de LinkedIn
    python run_prospector.py --importar lista.xlsx    # una lista de prospectos (aún no contactos)
    python run_prospector.py --ebook                  # suma los leads calientes y tibios del ebook
    python run_prospector.py --cola                   # fuerza el correo con la cola de hoy
    python run_prospector.py --demo                   # carga prospectos ficticios para probar
"""

import argparse
import logging
import sys
import time

from prospector_linkedin import ConfigProspector, abrir_db, ejecutar_ciclo, fuentes, preparar_demo, secuencia
from prospector_linkedin.mensajes import nombre_tipo


def main():
    parser = argparse.ArgumentParser(description="Prospector de LinkedIn (asistido, sin automatizar tu cuenta)")
    parser.add_argument("--loop", type=int, metavar="MIN", help="repetir cada MIN minutos")
    parser.add_argument("--importar", metavar="ARCHIVO", help="CSV/Excel de prospectos o Connections.csv de LinkedIn")
    parser.add_argument("--contactos", action="store_true", help="los del archivo ya son tus contactos de 1er grado")
    parser.add_argument("--ebook", action="store_true", help="sumar los leads calientes y tibios del ebook")
    parser.add_argument("--cola", action="store_true", help="forzar el correo con la cola de hoy")
    parser.add_argument("--ver", action="store_true", help="mostrar la cola de hoy en la terminal")
    parser.add_argument("--sin-notificar", action="store_true", help="no enviar correos")
    parser.add_argument("--sin-ia", action="store_true", help="no usar Claude aunque haya API key")
    parser.add_argument("--demo", action="store_true", help="cargar datos de demostración")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfgp = ConfigProspector.desde_entorno()
    db = abrir_db(cfgp)

    try:
        if args.demo:
            print("Demo:", preparar_demo(db, cfgp).texto())
            return 0
        if args.importar:
            with open(args.importar, "rb") as f:
                nuevos, existentes = fuentes.desde_archivo(db, args.importar, f.read(), grado=1 if args.contactos else 2)
            print(f"Importados: {nuevos} prospectos nuevos ({existentes} ya estaban).")
        if args.ebook:
            nuevos, existentes = fuentes.desde_leads_ebook(db)
            print(f"Desde el ebook: {nuevos} prospectos nuevos ({existentes} ya estaban).")
        while True:
            resultado = ejecutar_ciclo(
                db, cfgp, notificar=not args.sin_notificar, usar_ia=not args.sin_ia, forzar_cola=args.cola
            )
            print(time.strftime("%Y-%m-%d %H:%M"), "·", resultado.texto(), flush=True)
            if args.ver:
                for a in secuencia.cola_de_hoy(db, cfgp):
                    print(f"\n[{nombre_tipo(a['tipo'])}] {a['nombre']} · {a['cargo']} · {a['empresa']}")
                    print(fuentes.link_perfil(a))
                    print(a["texto"])
            if not args.loop:
                return 1 if resultado.errores else 0
            args.cola = False
            time.sleep(args.loop * 60)
    finally:
        db.cerrar()


if __name__ == "__main__":
    sys.exit(main())
