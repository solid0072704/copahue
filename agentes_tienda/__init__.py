"""Tienda tipo marketplace con agentes: un Buscador de productos ganadores y un Posicionador SEO."""

from .config import Config
from .orquestador import ResultadoCiclo, abrir, avisar_buscadores, ejecutar_ciclo, preparar_demo

__all__ = ["Config", "ResultadoCiclo", "abrir", "avisar_buscadores", "ejecutar_ciclo", "preparar_demo"]
