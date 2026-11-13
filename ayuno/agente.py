"""El agente: revisa qué aviso toca, lo redacta y te lo envía."""

import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta

from . import plan
from .canales import crear_canales, enviar_a_todos
from .estado import Estado
from .mensajes import crear_redactor, redactar

log = logging.getLogger(__name__)


@dataclass
class Resultado:
    enviados: list = field(default_factory=list)  # tipos de aviso enviados
    fallidos: list = field(default_factory=list)  # tipos que ningún canal pudo entregar
    proximo: object = None

    def texto(self, cfg, detalle=True):
        partes = []
        if self.enviados:
            partes.append(f"enviado: {', '.join(self.enviados)}")
        if self.fallidos:
            partes.append(f"sin entregar: {', '.join(self.fallidos)}")
        if not partes:
            partes.append("nada que enviar")
        if self.proximo and detalle:
            partes.append(f"próximo: {self.proximo.tipo} a las {self.proximo.cuando.astimezone(cfg.tz):%H:%M}")
        return " · ".join(partes)


def ejecutar(
    cfg,
    esperar_max_min=0,
    usar_ia=True,
    canales=None,
    redactor=None,
    estado=None,
    reloj=plan.ahora_utc,
    dormir=time.sleep,
):
    """Envía los avisos que correspondan ahora.

    Con `esperar_max_min`, si el próximo aviso llega dentro de ese plazo, espera
    hasta su hora exacta y lo envía (así GitHub Actions no lo atrasa).
    """
    canales = crear_canales(cfg) if canales is None else canales
    if redactor is None and usar_ia:
        redactor = crear_redactor(cfg)
    estado = estado or Estado(cfg.ruta_estado)
    resultado = Resultado()
    limite = reloj() + timedelta(minutes=esperar_max_min)

    while True:
        ahora = reloj()
        vigentes = plan.pendientes(cfg, ahora, estado.ya_enviado)
        for aviso in vigentes[-1:]:  # si hubiera varios atrasados, solo el más reciente tiene sentido
            texto = redactar(aviso, cfg, ahora, redactor)
            if enviar_a_todos(canales, texto):
                estado.marcar(aviso.clave, ahora)
                resultado.enviados.append(aviso.tipo)
            else:
                resultado.fallidos.append(aviso.tipo)
        for aviso in vigentes[:-1]:  # los más antiguos ya no aplican: no insistir con ellos
            estado.marcar(aviso.clave, ahora)

        siguiente = plan.proximo(cfg, ahora)
        resultado.proximo = siguiente
        if not siguiente or siguiente.cuando > limite or resultado.fallidos:
            break
        dormir(max(0.0, (siguiente.cuando - reloj()).total_seconds()) + 1)

    estado.guardar(reloj())
    return resultado, estado
