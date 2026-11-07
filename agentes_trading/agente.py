"""Ciclo diario del agente: datos → motor → broker → informe."""

import logging
from dataclasses import dataclass, field

from .cartera import Cartera
from .datos import cargar_universo, precios_sinteticos, tabla
from .informe import asunto_informe, construir_informe, enviar_informe
from .motor import decidir, paso

log = logging.getLogger("trading.agente")


@dataclass
class ResultadoCiclo:
    fecha: str = ""
    patrimonio: float = 0.0
    ganancia: float = 0.0
    operaciones: list = field(default_factory=list)
    pendiente: dict = field(default_factory=dict)
    informe: str = ""
    aviso: str = ""
    errores: list = field(default_factory=list)

    def texto(self):
        partes = [f"datos al {self.fecha or '-'}", f"patrimonio {self.patrimonio:,.2f} USD", f"ganancia {self.ganancia:+,.2f} USD", f"{len(self.operaciones)} operaciones"]
        if self.pendiente:
            partes.append("órdenes pendientes para la apertura")
        if self.aviso:
            partes.append(f"aviso {self.aviso}")
        if self.errores:
            partes.append(f"{len(self.errores)} error(es)")
        return " · ".join(partes)


def _broker(cfg):
    if cfg.modo == "simulado":
        return None
    from .broker import BrokerAlpaca

    return BrokerAlpaca(cfg)


def ejecutar_ciclo(cfg, precios=None, notificar=True, hoy=None):
    """Procesa todas las ruedas nuevas desde la última ejecución y guarda la cartera."""
    resultado = ResultadoCiclo()
    for problema in cfg.validar():
        resultado.errores.append(problema)
    cartera = Cartera.abrir(cfg)
    fallidos = []
    if precios is None:
        precios, fallidos = cargar_universo(cfg.universo, cfg.ruta_precios, cfg.años_historia, hoy=hoy)
        if fallidos:
            resultado.errores.append(f"Sin precios para: {', '.join(fallidos)}")
    if not precios:
        resultado.errores.append("No hay datos de mercado: no se puede operar.")
        return resultado
    cierres, aperturas = tabla(precios, "Close"), tabla(precios, "Open")
    if hoy is not None:
        cierres, aperturas = cierres.loc[:str(hoy)], aperturas.loc[:str(hoy)]

    broker = None
    try:
        broker = _broker(cfg)
    except Exception as e:
        resultado.errores.append(str(e))
        return resultado

    operaciones_hoy = []
    # Una cartera nueva arranca hoy: decide con el último cierre y opera en la próxima apertura.
    nuevas = cierres.index[cierres.index > cartera.ultima_fecha] if cartera.ultima_fecha else cierres.index[-1:]
    try:
        if broker is None:
            for fecha in nuevas:
                hechas, _ = paso(cartera, aperturas, cierres, fecha, cfg)
                operaciones_hoy.extend(hechas)
        else:
            broker.sincronizar(cartera)
            if len(nuevas):
                fecha = nuevas[-1]
                pendiente = decidir(cartera, cierres, fecha, cfg)
                precios_hoy = {t: float(v) for t, v in cierres.loc[fecha].dropna().items()}
                operaciones_hoy = broker.enviar(cartera, pendiente, precios_hoy)
                cartera.pendiente = {}
    except Exception as e:
        log.exception("Error al operar")
        resultado.errores.append(f"Error al operar: {e}")
    finally:
        cartera.guardar(cfg.ruta_cartera)

    ultimo = cierres.index[-1]
    precios_hoy = {t: float(v) for t, v in cierres.loc[ultimo].dropna().items()}
    resultado.fecha = cartera.ultima_fecha
    resultado.patrimonio = cartera.patrimonio(precios_hoy)
    resultado.ganancia = resultado.patrimonio - cartera.capital_inicial
    resultado.operaciones = operaciones_hoy
    resultado.pendiente = cartera.pendiente
    nombre = broker.nombre if broker else "cartera simulada (dinero ficticio)"
    resultado.informe = construir_informe(cartera, precios_hoy, cfg, operaciones_hoy, cartera.pendiente, fallidos, nombre)
    if resultado.errores:
        resultado.informe += "\n\nErrores:\n" + "\n".join(f"  · {e}" for e in resultado.errores)

    if notificar and cfg.gmail_configurado and (operaciones_hoy or cartera.pendiente or resultado.errores or not cfg.avisar_solo_con_operaciones):
        try:
            resultado.aviso = enviar_informe(cfg, asunto_informe(cartera, precios_hoy, operaciones_hoy), resultado.informe)
        except Exception as e:
            resultado.errores.append(f"No se pudo enviar el informe: {e}")
    return resultado


def preparar_demo(cfg, dias=900):
    """Corre el agente sobre precios ficticios (sin internet ni broker) para ver cómo funciona."""
    precios = precios_sinteticos(cfg.universo, dias=dias)
    return ejecutar_ciclo(cfg, precios=precios, notificar=False)
