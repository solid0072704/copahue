"""Orquestador: un ciclo completo de la tienda.

    Buscador  →  busca en proveedores, calcula precios, puntúa y elige qué publicar
    Posicionador → palabras clave, textos SEO, categorías, guías y aprendizaje desde Google
    Pagos (dropshipping) → links de Mercado Pago
    Constructor → genera el sitio, sitemap, datos estructurados y auditoría SEO
"""

import json
import logging
import os
from dataclasses import dataclass, field

from . import buscador, divisas, pagos, seo, sitio
from .almacen import Almacen, ahora
from .indexacion import avisar_indexnow
from .llm import crear_redactor

log = logging.getLogger(__name__)


@dataclass
class ResultadoCiclo:
    tipo_cambio: float = 0
    nuevos: list = field(default_factory=list)
    publicados: list = field(default_factory=list)
    pausados: list = field(default_factory=list)
    textos: int = 0
    textos_ia: int = 0
    guias: list = field(default_factory=list)
    reoptimizados: list = field(default_factory=list)
    links_pago: int = 0
    paginas: int = 0
    en_vitrina: int = 0
    cambiadas: list = field(default_factory=list)
    auditoria: float = 0
    errores: list = field(default_factory=list)

    def texto(self):
        partes = [
            f"dólar ${self.tipo_cambio:,.0f}".replace(",", "."),
            f"{len(self.nuevos)} productos encontrados",
            f"{len(self.publicados)} publicados",
            f"{len(self.pausados)} pausados",
            f"{self.textos} textos SEO ({self.textos_ia} con IA)",
            f"{len(self.guias)} guías",
            f"{len(self.reoptimizados)} reoptimizados con Google",
            f"{self.en_vitrina} en vitrina · {self.paginas} páginas · SEO {self.auditoria}/100",
            f"{len(self.cambiadas)} URLs nuevas o cambiadas",
        ]
        if self.links_pago:
            partes.append(f"{self.links_pago} links de pago")
        if self.errores:
            partes.append("errores: " + "; ".join(self.errores[:5]))
        return " · ".join(partes)

    def markdown(self):
        filas = [
            ("Dólar observado", f"${self.tipo_cambio:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")),
            ("Productos nuevos encontrados", len(self.nuevos)),
            ("Publicados hoy", len(self.publicados)),
            ("Pausados hoy", len(self.pausados)),
            ("Textos SEO escritos (con IA)", f"{self.textos} ({self.textos_ia})"),
            ("Guías creadas o actualizadas", len(self.guias)),
            ("Páginas reoptimizadas con Search Console", len(self.reoptimizados)),
            ("Productos en vitrina", self.en_vitrina),
            ("Páginas del sitio", self.paginas),
            ("Puntaje SEO promedio", f"{self.auditoria}/100"),
            ("URLs nuevas o cambiadas", len(self.cambiadas)),
        ]
        md = "## 🛍️ Ciclo de la tienda\n\n| | |\n|---|---|\n" + "".join(f"| {a} | {b} |\n" for a, b in filas)
        if self.errores:
            md += "\n**Avisos:**\n" + "".join(f"- {e}\n" for e in self.errores)
        return md


def abrir(cfg):
    return Almacen(cfg.carpeta_datos)


def ruta_pendientes(cfg):
    return os.path.normpath(cfg.carpeta_sitio).rstrip(os.sep) + "_indexnow.json"


def ejecutar_ciclo(almacen, cfg, buscar=True, posicionar=True, construir=True, usar_ia=True, usar_red=True, clientes=None):
    r = ResultadoCiclo()
    r.tipo_cambio = divisas.tipo_cambio(almacen, cfg, usar_red)

    if buscar:
        r.nuevos, errores = buscador.buscar(almacen, cfg, r.tipo_cambio, clientes=clientes, usar_red=usar_red)
        r.errores += errores
    r.publicados, r.pausados = buscador.seleccionar(almacen, cfg)

    if posicionar:
        redactor = crear_redactor(cfg) if usar_ia else None
        res = seo.posicionar(almacen, cfg, redactor, usar_red=usar_red)
        r.textos, r.textos_ia, r.guias, r.reoptimizados = len(res["textos"]), res["ia"], res["guias"], res["reoptimizados"]
        r.errores += res["errores"]

    r.links_pago = len(pagos.asegurar_links(almacen, cfg, usar_red))

    if construir:
        res = sitio.construir(almacen, cfg)
        r.paginas, r.en_vitrina, r.cambiadas, r.auditoria = res["paginas"], res["productos"], res["cambiadas"], res["auditoria"]
        with open(ruta_pendientes(cfg), "w", encoding="utf-8") as f:
            json.dump(r.cambiadas, f)

    almacen.estado["ultimo_ciclo"] = ahora()
    almacen.log("Orquestador", "Ciclo completo", r.texto())
    almacen.guardar()
    return r


def avisar_buscadores(almacen, cfg):
    """Se llama después de publicar el sitio: avisa a IndexNow de las URLs nuevas o cambiadas."""
    try:
        with open(ruta_pendientes(cfg), encoding="utf-8") as f:
            urls = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        urls = []
    enviadas, estado = avisar_indexnow(cfg, urls)
    almacen.log("Posicionador", "Aviso a IndexNow", f"{enviadas} URLs · {estado}")
    almacen.guardar()
    return enviadas, estado


def preparar_demo(almacen, cfg):
    """Carga el catálogo de ejemplo y arma el sitio completo sin conexión."""
    import dataclasses

    buscador.cargar_demo(almacen, cfg)
    cfg = dataclasses.replace(cfg, max_guias_por_ciclo=20)
    return ejecutar_ciclo(almacen, cfg, buscar=False, usar_ia=False, usar_red=False)
