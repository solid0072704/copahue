"""Ejecuta Copahue.

    python -m copahue                      # servidor web en http://localhost:8000
    python -m copahue --puerto 8080
    python -m copahue --automatizaciones   # corre las automatizaciones de todos los estudios (para cron)
"""

import argparse
import logging


def main():
    parser = argparse.ArgumentParser(prog="copahue", description="Gestión de clientes para estudios contables.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--puerto", type=int, default=8000)
    parser.add_argument("--automatizaciones", action="store_true",
                        help="Corre una vez las automatizaciones de todos los estudios y termina.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from .config import cargar

    cfg = cargar()
    if args.automatizaciones:
        from datetime import date, datetime

        from . import llm
        from .api import ejecutar_todos
        from .db import inicializar

        inicializar(cfg.db)
        hoy = date.fromisoformat(cfg.hoy_fijo) if cfg.hoy_fijo else datetime.now(cfg.tz).date()
        for estudio, resultados in ejecutar_todos(cfg, hoy, llm.crear(cfg)).items():
            print(estudio)
            for r in resultados:
                print(f"  {r['nombre']}: {r['texto']}")
        return

    import uvicorn

    from .api import crear_app

    uvicorn.run(crear_app(cfg), host=args.host, port=args.puerto, proxy_headers=True)


if __name__ == "__main__":
    main()
