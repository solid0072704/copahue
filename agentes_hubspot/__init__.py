"""Agentes que destraban cuellos de botella en el embudo de ventas de HubSpot.

- Analista de embudo: mide dónde se acumulan y se pierden los negocios, y recomienda qué hacer.
- Desatascador: encuentra negocios estancados y propone (o crea en HubSpot) la siguiente acción.
"""

from .config import ConfigHubSpot
from .db import MemoriaEmbudo
from .orquestador import ResultadoCiclo, abrir_db, ejecutar_ciclo, preparar_demo

__all__ = ["ConfigHubSpot", "MemoriaEmbudo", "ResultadoCiclo", "abrir_db", "ejecutar_ciclo", "preparar_demo"]
