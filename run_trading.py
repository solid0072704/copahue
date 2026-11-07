"""Ejecuta el agente de trading desde la terminal, cron o GitHub Actions.

Ejemplos:
    python run_trading.py                  # un ciclo: descarga precios, decide y guarda la cartera
    python run_trading.py --backtest       # prueba histórica de la estrategia con precios reales
    python run_trading.py --demo           # ciclo sobre precios ficticios (sin internet ni broker)
    python run_trading.py --informe        # imprime el estado de la cartera sin operar
"""

import argparse
import dataclasses
import logging
import sys

from agentes_trading import ConfigTrading, ejecutar_ciclo, preparar_demo
from agentes_trading.backtest import ejecutar_backtest
from agentes_trading.cartera import Cartera
from agentes_trading.datos import cargar_universo, precios_sinteticos, tabla
from agentes_trading.informe import AVISO, construir_informe


def imprimir_backtest(r, cfg):
    m, ref = r.metricas, r.metricas_referencia
    print(f"Periodo: {r.curva.index[0].date()} → {r.curva.index[-1].date()} ({m['años']:.1f} años) · capital {cfg.capital_inicial:,.0f} USD")
    print(f"{'':22}{'Estrategia':>12}{cfg.referencia + ' (comprar y mantener)':>28}")
    for clave, nombre, fmt in (
        ("rentabilidad_total", "Rentabilidad total", "{:+.1%}"),
        ("cagr", "Rentabilidad anual", "{:+.1%}"),
        ("volatilidad", "Volatilidad anual", "{:.1%}"),
        ("sharpe", "Sharpe", "{:.2f}"),
        ("caida_maxima", "Caída máxima", "{:.1%}"),
        ("meses_positivos", "Meses positivos", "{:.0%}"),
        ("operaciones", "Operaciones", "{}"),
    ):
        print(f"{nombre:22}{fmt.format(m[clave]):>12}{fmt.format(ref[clave]):>28}")
    print()
    print(f"Meta de {cfg.meta_usd:,.0f} USD con {cfg.capital_inicial:,.0f} USD:")
    print(f"  · En {r.probabilidad_meta:.0%} de los periodos históricos de 12 meses se habría alcanzado.")
    if r.meses_para_meta:
        print(f"  · Si se repitiera la rentabilidad anual histórica, tardaría ~{r.meses_para_meta:.0f} meses.")
    else:
        print("  · Con la rentabilidad histórica no se alcanzaría.")
    print()
    print(AVISO)


def main():
    parser = argparse.ArgumentParser(description="Agente de trading automático")
    parser.add_argument("--backtest", action="store_true", help="prueba histórica con precios reales")
    parser.add_argument("--demo", action="store_true", help="ciclo con precios ficticios, sin internet")
    parser.add_argument("--informe", action="store_true", help="muestra la cartera actual sin operar")
    parser.add_argument("--sin-avisar", action="store_true", help="no enviar el informe por correo")
    parser.add_argument("--desde", help="fecha inicial del backtest (AAAA-MM-DD)")
    parser.add_argument("--capital", type=float, help="capital inicial para el backtest")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = ConfigTrading.desde_entorno()
    if args.capital:
        cfg = dataclasses.replace(cfg, capital_inicial=args.capital)
    for problema in cfg.validar():
        print("Aviso:", problema)

    if args.demo:
        cfg = dataclasses.replace(cfg, modo="simulado", ruta_datos=cfg.ruta_datos + "/demo")
        print("Backtest con precios ficticios:")
        imprimir_backtest(ejecutar_backtest(precios_sinteticos(cfg.universo), cfg), cfg)
        print("\nCiclo del agente sobre los mismos precios ficticios:")
        resultado = preparar_demo(cfg)
        print(resultado.informe)
        return 0

    if args.backtest:
        precios, fallidos = cargar_universo(cfg.universo, cfg.ruta_precios, cfg.años_historia)
        if fallidos:
            print("Sin precios para:", ", ".join(fallidos))
        if not precios:
            return 1
        imprimir_backtest(ejecutar_backtest(precios, cfg, desde=args.desde), cfg)
        return 0

    if args.informe:
        cartera = Cartera.abrir(cfg)
        precios, _ = cargar_universo(cfg.universo, cfg.ruta_precios, cfg.años_historia)
        cierres = tabla(precios, "Close")
        ultimos = {t: float(v) for t, v in cierres.iloc[-1].dropna().items()} if len(cierres) else {}
        print(construir_informe(cartera, ultimos, cfg, [], cartera.pendiente))
        return 0

    resultado = ejecutar_ciclo(cfg, notificar=not args.sin_avisar)
    print(resultado.informe)
    print("\nResumen:", resultado.texto())
    return 1 if resultado.errores else 0


if __name__ == "__main__":
    sys.exit(main())
