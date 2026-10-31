"""Agente de caja para la gerencia de finanzas: cobranza, proyección semanal y alertas de liquidez.

Lee las cuotas por cobrar (Forpay, por API o archivo), los egresos (CSV) y, opcionalmente, las ventas
comprometidas de HubSpot. Entrega un informe con hallazgos, acciones y la lista de cobranza prioritaria.
"""

from .config import ConfigCaja
from .orquestador import ResultadoCiclo, abrir_db, ejecutar_ciclo, preparar_demo

__all__ = ["ConfigCaja", "ResultadoCiclo", "abrir_db", "ejecutar_ciclo", "preparar_demo"]
