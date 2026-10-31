"""Ejecuta al agente de caja desde la terminal, cron o GitHub Actions.

Ejemplos:
    python run_agente_caja.py --demo --imprimir      # datos ficticios, muestra el informe completo
    python run_agente_caja.py --plantillas           # crea data/cobros.csv y data/egresos.csv con filas de ejemplo
    python run_agente_caja.py --archivo cobros.xlsx  # usa tu planilla de cuentas por cobrar (se puede repetir)
    python run_agente_caja.py                        # lee COBROS_ARCHIVOS (y Forpay/HubSpot si están configurados)
    python run_agente_caja.py --sin-correo           # solo guarda el informe en data/informes
"""

import argparse
import dataclasses
import logging
import sys

from agentes_caja import ConfigCaja, abrir_db, ejecutar_ciclo, preparar_demo
from agentes_caja.fuentes import escribir_plantilla_cobros, escribir_plantilla_egresos


def main():
    parser = argparse.ArgumentParser(description="Agente de caja: cobranza, proyección semanal y alertas de liquidez")
    parser.add_argument("--demo", action="store_true", help="usar datos ficticios (no lee ningún archivo ni sistema)")
    parser.add_argument("--archivo", action="append", default=[], metavar="RUTA", help="CSV/Excel de cuentas por cobrar (se puede repetir)")
    parser.add_argument("--plantillas", "--plantilla-egresos", action="store_true", help="crear data/cobros.csv y el archivo de egresos con filas de ejemplo, y salir")
    parser.add_argument("--sin-correo", action="store_true", help="no enviar el informe por correo")
    parser.add_argument("--sin-ia", action="store_true", help="no usar Claude aunque haya API key")
    parser.add_argument("--imprimir", action="store_true", help="mostrar el informe completo en pantalla")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = ConfigCaja.desde_entorno()
    if args.plantillas:
        ruta_cobros = escribir_plantilla_cobros(cfg.cobros_archivos[0] if cfg.cobros_archivos else "data/cobros.csv")
        ruta_egresos = escribir_plantilla_egresos(cfg.egresos_archivo)
        print(f"Plantillas creadas: {ruta_cobros} (cuentas por cobrar) y {ruta_egresos} (egresos). Reemplaza las filas de ejemplo por tus datos.")
        return 0
    if args.archivo:
        cfg = dataclasses.replace(cfg, cobros_archivos=cfg.cobros_archivos + args.archivo)
    if args.demo:
        cfg = dataclasses.replace(cfg, ruta_db=cfg.ruta_db.replace(".db", "_demo.db"))
    elif not cfg.cobros_configurado and not cfg.incluir_pipeline:
        print(
            "Falta la fuente de cuentas por cobrar: pon la ruta de tu CSV/Excel en COBROS_ARCHIVOS (o pásala con --archivo), "
            "o activa CAJA_INCLUIR_PIPELINE=true para proyectar con las ventas de HubSpot. "
            "Con --plantillas creas los archivos de ejemplo; con --demo ves el informe con datos ficticios."
        )
        return 2

    db = abrir_db(cfg)
    try:
        if args.demo:
            resultado, _ = preparar_demo(cfg, db)
        else:
            resultado = ejecutar_ciclo(cfg, db, enviar=not args.sin_correo, usar_ia=not args.sin_ia)
    finally:
        db.cerrar()
    print(resultado.texto(), flush=True)
    if args.imprimir and resultado.markdown:
        print("\n" + resultado.markdown)
    return 1 if resultado.errores else 0


if __name__ == "__main__":
    sys.exit(main())
