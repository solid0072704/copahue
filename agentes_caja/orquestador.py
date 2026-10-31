"""Orquestador del agente de caja: lee las fuentes, analiza la cobranza, proyecta la caja e informa."""

import logging
from dataclasses import dataclass, field
from datetime import timedelta

from agentes_ebook.db import ahora_utc, iso
from agentes_hubspot.db import MemoriaEmbudo

from . import analista, cobranza, informe, proyeccion
from .fuentes import FuentesReales, expandir_egresos
from .llm import crear_asistente

log = logging.getLogger(__name__)
CLAVE_FOTO = "caja"


@dataclass
class ResultadoCiclo:
    diagnostico: dict = None
    informe_ruta: str = ""
    correo: str = None
    markdown: str = ""
    errores: list = field(default_factory=list)

    def texto(self):
        if not self.diagnostico:
            return "sin diagnóstico · errores: " + "; ".join(self.errores)
        d = self.diagnostico
        cb, py = d["cobranza"], d["proyeccion"]
        partes = [
            f"{d['cuotas_total']} cuotas de {len(d['empresas'])} sociedades",
            f"pendiente {analista.fmt_monto(cb['total_pendiente'], d['moneda'])}",
            f"moroso {analista.fmt_monto(cb['total_moroso'], d['moneda'])} ({cb['pct_mora']:.0f}%)",
            f"{cb['compromisos_morosos']} compromisos en mora",
        ]
        if py["alerta"]:
            partes.append(f"ALERTA: caja bajo el mínimo la semana del {analista.fecha_corta(py['alerta']['inicio'])}")
        if self.informe_ruta:
            partes.append(f"informe: {self.informe_ruta}")
        if self.correo:
            partes.append(f"correo: {self.correo}")
        if self.errores:
            partes.append("errores: " + "; ".join(self.errores))
        return " · ".join(partes)


def abrir_db(cfg):
    return MemoriaEmbudo(cfg.ruta_db)


def ejecutar_ciclo(cfg, db, fuentes=None, ahora=None, enviar=True, guardar=True, usar_ia=True):
    """Un ciclo completo: fuentes → cobranza → proyección → hallazgos → informe."""
    ahora = ahora or ahora_utc()
    local = ahora.astimezone(cfg.tz)
    hoy = local.date()
    fuentes = fuentes or FuentesReales(cfg)
    r = ResultadoCiclo()
    try:
        cuotas, errores = fuentes.cuotas(hoy)
    except Exception as e:
        r.errores.append(f"Lectura de cuotas: {e}")
        db.log("Orquestador caja", "Error leyendo cuotas", str(e))
        return r
    errores = list(errores)
    egresos, err = fuentes.egresos()
    errores += err
    ventas, err = fuentes.ventas(hoy)
    errores += err
    if not cuotas and not ventas:
        r.errores.extend(errores or ["No se obtuvo ninguna cuenta por cobrar: revisa COBROS_ARCHIVOS (o FORPAY_EMPRESAS) y CAJA_INCLUIR_PIPELINE."])
        db.log("Orquestador caja", "Sin cuentas por cobrar", "; ".join(r.errores))
        return r

    cb = cobranza.analizar(cuotas, hoy, cfg.top_cobranza)
    fin_horizonte = hoy + timedelta(days=7 * cfg.horizonte_semanas - 1)
    eventos_egresos = expandir_egresos(egresos, hoy, fin_horizonte)
    py = proyeccion.proyectar(cuotas, eventos_egresos, ventas, cfg, hoy, cb["tasa_cobro_historica"])
    anterior = db.foto_anterior(CLAVE_FOTO, 7, ahora)
    diag = analista.diagnosticar(cuotas, cb, py, eventos_egresos, cfg, ahora, hoy, anterior, errores)
    db.guardar_foto(diag, CLAVE_FOTO)
    db.log(analista.AGENTE, "Diagnóstico", f"pendiente {cb['total_pendiente']} · moroso {cb['total_moroso']} · alerta {bool(py['alerta'])}")

    lectura = None
    if usar_ia:
        asistente = crear_asistente(cfg)
        if asistente:
            lectura = asistente.lectura(diag)

    titulo, md, html, texto = informe.construir(diag, cfg, lectura, local)
    r.diagnostico, r.markdown = diag, md
    if guardar:
        try:
            r.informe_ruta = informe.guardar(md, cfg, local)
        except OSError as e:
            r.errores.append(f"No se pudo guardar el informe: {e}")
    if enviar:
        r.correo = informe.enviar(cfg, titulo, texto, html)
        if r.correo.startswith("error"):
            r.errores.append(f"Correo: {r.correo}")
    db.set_meta("ultima_ejecucion_caja", iso(ahora))
    db.log("Orquestador caja", "Ciclo completado", r.texto())
    return r


def preparar_demo(cfg, db, ahora=None):
    """Corre un ciclo con datos ficticios (sin Forpay, sin HubSpot, sin correo, sin IA). Devuelve (resultado, fuentes)."""
    import dataclasses

    from .demo import FuentesDemo

    ahora = ahora or ahora_utc()
    hoy = ahora.astimezone(cfg.tz).date()
    fuentes = FuentesDemo(hoy)
    if not cfg.saldo_inicial_total and not cfg.caja_minima:
        cfg = dataclasses.replace(cfg, saldo_inicial=fuentes.saldo_inicial, caja_minima=fuentes.caja_minima)
    return ejecutar_ciclo(cfg, db, fuentes, ahora, enviar=False, guardar=True, usar_ia=False), fuentes
