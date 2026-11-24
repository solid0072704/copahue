"""Central de WhatsApp: Breeze (HubSpot) conversa con los clientes y este paquete reparte los leads."""

from .config import ConfigCentral
from .distribuidor import Distribuidor, Resultado, construir
from .ruteo import ErrorRuteo, TablaRuteo, cargar_tabla, elegir_vendedor
from .sla import ResultadoSLA, revisar_sla

__all__ = [
    "ConfigCentral",
    "Distribuidor",
    "Resultado",
    "construir",
    "ErrorRuteo",
    "TablaRuteo",
    "cargar_tabla",
    "elegir_vendedor",
    "ResultadoSLA",
    "revisar_sla",
]
