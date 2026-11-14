"""Ejecuta a los agentes de la tienda desde la terminal, cron o GitHub Actions.

Ejemplos:
    python run_tienda.py                      # ciclo completo: buscar → posicionar → construir el sitio
    python run_tienda.py --demo               # tienda de ejemplo, sin claves ni internet
    python run_tienda.py --servir             # vista previa en http://localhost:8000
    python run_tienda.py --construir          # solo regenera el sitio con los datos guardados
    python run_tienda.py --importar productos.csv --nicho "Hogar y cocina" [--moneda CLP]
    python run_tienda.py --indexnow           # avisa a Bing/IndexNow (después de publicar)
    python run_tienda.py --borrar-demo        # elimina los productos de ejemplo
"""

import argparse
import dataclasses
import functools
import http.server
import logging
import os
import sys

from agentes_tienda import Config, abrir, avisar_buscadores, ejecutar_ciclo, preparar_demo
from agentes_tienda import buscador, divisas
from agentes_tienda.fuentes import archivo


def main():
    parser = argparse.ArgumentParser(description="Agentes de la tienda: buscador de productos y posicionamiento SEO")
    parser.add_argument("--demo", action="store_true", help="cargar el catálogo de ejemplo y construir el sitio")
    parser.add_argument("--borrar-demo", action="store_true", help="eliminar los productos de ejemplo")
    parser.add_argument("--construir", action="store_true", help="solo regenerar el sitio")
    parser.add_argument("--sin-busqueda", action="store_true", help="no consultar proveedores (solo SEO y sitio)")
    parser.add_argument("--sin-ia", action="store_true", help="no usar Claude aunque haya API key")
    parser.add_argument("--importar", metavar="ARCHIVO", help="importar productos desde un CSV o Excel")
    parser.add_argument("--nicho", help="categoría de los productos importados")
    parser.add_argument("--moneda", default="USD", help="moneda de los costos del archivo (USD o CLP)")
    parser.add_argument("--indexnow", action="store_true", help="avisar a IndexNow las URLs nuevas o cambiadas")
    parser.add_argument("--servir", nargs="?", const=8000, type=int, metavar="PUERTO", help="vista previa local")
    parser.add_argument("--resumen", metavar="ARCHIVO", help="escribir el resumen del ciclo en Markdown (p. ej. $GITHUB_STEP_SUMMARY)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config.desde_entorno()
    if args.servir:
        cfg = dataclasses.replace(cfg, url=f"http://localhost:{args.servir}", carpeta_sitio=cfg.carpeta_sitio)
    almacen = abrir(cfg)

    if args.indexnow:
        enviadas, estado = avisar_buscadores(almacen, cfg)
        print(f"IndexNow: {enviadas} URLs · {estado}")
        return 0
    if args.borrar_demo:
        print(f"Productos de ejemplo eliminados: {almacen.borrar_demo()}")
        almacen.guardar()
        return 0
    if args.demo:
        resultado = preparar_demo(almacen, cfg)
    elif args.importar:
        with open(args.importar, "rb") as f:
            tabla = archivo.leer(f.read(), args.importar)
        tc = divisas.tipo_cambio(almacen, cfg)
        crudos = archivo.productos_desde_tabla(tabla, args.moneda, tc, args.nicho)
        nuevos = buscador.incorporar(almacen, cfg, crudos, tc, args.nicho, agente="Importación")
        print(f"Importados {len(crudos)} productos ({len(nuevos)} nuevos)")
        resultado = ejecutar_ciclo(almacen, cfg, buscar=False, usar_ia=not args.sin_ia)
    else:
        solo_sitio = args.construir
        resultado = ejecutar_ciclo(
            almacen, cfg, buscar=not (solo_sitio or args.sin_busqueda), posicionar=not solo_sitio, usar_ia=not args.sin_ia
        )
    print(resultado.texto(), flush=True)
    if args.resumen:
        with open(args.resumen, "a", encoding="utf-8") as f:
            f.write(resultado.markdown())

    if args.servir:
        manejador = functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.abspath(cfg.carpeta_sitio))
        print(f"Vista previa en http://localhost:{args.servir}  (Ctrl+C para salir)")
        with http.server.ThreadingHTTPServer(("", args.servir), manejador) as servidor:
            try:
                servidor.serve_forever()
            except KeyboardInterrupt:
                pass
    sin_fuentes = not (cfg.aliexpress_configurado or cfg.cj_configurado)
    if sin_fuentes and not (args.demo or args.importar or args.construir) and not almacen.publicados():
        print("Aviso: no hay proveedores configurados (ALIEXPRESS_APP_KEY/SECRET o CJ_API_KEY). Usa --demo para ver un ejemplo.")
    return 1 if resultado.errores and not resultado.en_vitrina else 0


if __name__ == "__main__":
    sys.exit(main())
