"""Configuración del prospector de LinkedIn.

Reutiliza la configuración de los agentes del ebook (Gmail, marca, IA, zona horaria)
y agrega lo propio de la prospección: a quién buscas, qué ofreces y cuánto contactas por día.
"""

from dataclasses import dataclass, field

from agentes_ebook.config import Config, _env, _env_int, _env_lista

PRIORIDADES = ("A", "B", "C")

CARGOS_OBJETIVO = (
    "gerente general, CEO, fundador, fundadora, cofundador, cofundadora, founder, co-founder, socio, socia, "
    "dueño, dueña, director ejecutivo, directora ejecutiva, country manager, managing director, "
    "gerente de administración y finanzas, COO, gerente de operaciones"
)
INDUSTRIAS_OBJETIVO = (
    "tecnología, software, saas, startup, fintech, servicios, construcción, inmobiliaria, retail, "
    "ecommerce, logística, transporte, salud, educación, agro, alimentos, manufactura, energía, turismo"
)
EXCLUIR = (
    "estudiante, practicante, tesista, en búsqueda, busca empleo, open to work, jubilado, "
    "cfo fraccional, fractional cfo, cfo externo, cfo as a service, headhunter, reclutador, recruiter"
)


@dataclass
class ConfigProspector:
    base: Config = field(default_factory=Config)

    # --- Qué ofreces ---
    servicio: str = "CFO fraccional"
    propuesta: str = "ordenar sus finanzas, proyectar el flujo de caja y tener reportes de gestión confiables"
    recurso_link: str = ""

    # --- A quién buscas (cliente ideal) ---
    cargos_objetivo: list = field(default_factory=lambda: _lista(CARGOS_OBJETIVO))
    industrias_objetivo: list = field(default_factory=lambda: _lista(INDUSTRIAS_OBJETIVO))
    excluir: list = field(default_factory=lambda: _lista(EXCLUIR))
    umbral_a: int = 70
    umbral_b: int = 45
    prioridad_minima: str = "B"

    # --- Ritmo (límites conservadores para cuidar tu cuenta) ---
    invitaciones_dia: int = 15
    invitaciones_semana: int = 80
    mensajes_dia: int = 20
    nota_max: int = 200

    # --- Cadencia de mensajes ---
    dias_seguimiento: list = field(default_factory=lambda: [4, 8])
    dias_cierre: int = 10
    dias_retirar: int = 21

    # --- Dónde abres el tablero (para el botón del correo diario) ---
    tablero_url: str = "http://localhost:8501"

    @classmethod
    def desde_entorno(cls, base=None):
        base = base or Config.desde_entorno()
        prioridad = _env("LI_PRIORIDAD_MINIMA", "B").upper()
        dias = [int(d) for d in _env_lista("LI_DIAS_SEGUIMIENTO", "4,8") if d.isdigit() and int(d) > 0]
        return cls(
            base=base,
            servicio=_env("LI_SERVICIO", "CFO fraccional"),
            propuesta=_env("LI_PROPUESTA", cls.propuesta),
            recurso_link=_env("LI_RECURSO_LINK", base.sitio_web),
            cargos_objetivo=_env_lista("LI_CARGOS_OBJETIVO", CARGOS_OBJETIVO),
            industrias_objetivo=_env_lista("LI_INDUSTRIAS_OBJETIVO", INDUSTRIAS_OBJETIVO),
            excluir=_env_lista("LI_EXCLUIR", EXCLUIR),
            umbral_a=_env_int("LI_UMBRAL_A", 70),
            umbral_b=_env_int("LI_UMBRAL_B", 45),
            prioridad_minima=prioridad if prioridad in PRIORIDADES else "B",
            invitaciones_dia=max(0, _env_int("LI_INVITACIONES_DIA", 15)),
            invitaciones_semana=max(0, _env_int("LI_INVITACIONES_SEMANA", 80)),
            mensajes_dia=max(0, _env_int("LI_MENSAJES_DIA", 20)),
            nota_max=max(60, _env_int("LI_NOTA_MAX", 200)),
            dias_seguimiento=dias or [4, 8],
            dias_cierre=_env_int("LI_DIAS_CIERRE", 10),
            dias_retirar=_env_int("LI_DIAS_RETIRAR", 21),
            tablero_url=_env("LI_TABLERO_URL", "http://localhost:8501"),
        )

    @property
    def tz(self):
        return self.base.tz

    @property
    def prioridades_activas(self):
        """Prioridades que entran a la cola (A; A y B; o todas)."""
        return PRIORIDADES[: PRIORIDADES.index(self.prioridad_minima) + 1]

    @property
    def total_mensajes(self):
        return 1 + len(self.dias_seguimiento)


def _lista(texto):
    return [x.strip() for x in texto.split(",") if x.strip()]
