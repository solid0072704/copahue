"""Agente de trading automático con cartera simulada (paper trading) y broker opcional.

Nada aquí garantiza ganancias: los mercados pueden subir o bajar y el agente puede
perder dinero. Por defecto opera con dinero FICTICIO; solo pasa a dinero real cuando
el dueño de la cuenta lo configura explícitamente (ver config.py).
"""

from .agente import ResultadoCiclo, ejecutar_ciclo, preparar_demo
from .config import ConfigTrading

__all__ = ["ConfigTrading", "ResultadoCiclo", "ejecutar_ciclo", "preparar_demo"]
