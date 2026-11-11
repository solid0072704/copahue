"""Apariencia de la plataforma: tema oscuro, tarjetas y navegación pensada primero para el celular.

Este módulo solo arma texto (CSS y HTML). Todo dato que venga del usuario pasa por `e()`
antes de entrar al HTML.
"""

import html

VERDE = "#21d566"
ROJO = "#f15e6c"
AMBAR = "#ffa42b"
AMARILLO = "#f2c94c"
AZUL = "#4da3ff"
GRIS = "#8a8a8a"

# Series de los gráficos (pareja validada para fondo oscuro) y tinta de ejes.
SERIE_1 = "#3987e5"
SERIE_2 = "#d95926"
TINTA = "#b3b3b3"
REJILLA = "#2a2a2a"

COLOR_SEMAFORO = {"Atrasada": ROJO, "Vence hoy": AMBAR, "Próxima": AMARILLO, "Al día": VERDE, "Cerrada": GRIS}
COLOR_ESTADO_CLIENTE = {"Activo": VERDE, "Prospecto": AZUL, "Suspendido": AMBAR, "De baja": GRIS}
_AVATARES = ["#1e3264", "#8d67ab", "#148a08", "#e8115b", "#bc5900", "#477d95", "#af2896", "#27856a", "#503750", "#777777"]

CSS = """
<style>
:root {
  --fondo: #121212; --sup: #181818; --sup2: #242424; --borde: #2a2a2a;
  --txt: #ffffff; --txt2: #b3b3b3; --txt3: #8a8a8a;
  --verde: #21d566; --rojo: #f15e6c; --ambar: #ffa42b; --nav: 232px;
}

/* ---------- estructura ---------- */
[data-testid="stHeader"] { background: transparent; height: 0; min-height: 0; }
[data-testid="stMainBlockContainer"] {
  padding: 2.2rem 2.5rem 4rem calc(var(--nav) + 2.5rem); max-width: 1500px;
}
h1, h2, h3 { font-weight: 800 !important; letter-spacing: -0.02em; }
h1 { font-size: 2.1rem !important; padding-top: 0 !important; }
h2 { font-size: 1.35rem !important; }
h3 { font-size: 1.1rem !important; }
[data-testid="stHeaderActionElements"] { display: none; }

/* ---------- navegación: columna a la izquierda en PC ---------- */
.st-key-nav {
  position: fixed; left: 0; top: 0; bottom: 0; width: var(--nav); z-index: 999;
  background: #000; padding: 1.4rem 0.75rem; gap: 0.2rem;
}
.st-key-nav .marca { font-size: 1.15rem; font-weight: 800; padding: 0 0.6rem 1.1rem; letter-spacing: -0.02em; }
.st-key-nav .marca small { display: block; font-size: 0.75rem; font-weight: 400; color: var(--txt3); margin-top: 0.15rem; letter-spacing: 0; }
.st-key-nav button {
  width: 100%; justify-content: flex-start; background: transparent !important; border: 0 !important;
  color: var(--txt2) !important; padding: 0.6rem 0.75rem; border-radius: 8px; min-height: 0;
}
.st-key-nav button > div { justify-content: flex-start; width: 100%; }
.st-key-nav button p { font-weight: 700; font-size: 0.95rem; }
.st-key-nav button:hover { color: var(--txt) !important; }
.st-key-nav button[kind="primary"] { color: var(--txt) !important; background: var(--sup2) !important; }
.st-key-nav button[kind="primary"] [data-testid="stIconMaterial"] { color: var(--verde); }
.st-key-nav [data-testid="stIconMaterial"] { font-size: 1.4rem; }

/* ---------- botones ---------- */
button[kind^="primary"], a[kind="primary"] { color: #000 !important; font-weight: 700; }
button[kind^="primary"] p, a[kind="primary"] p { font-weight: 700; }
span[data-baseweb="tag"] { color: #000; font-weight: 600; }
span[data-baseweb="tag"] svg { fill: #000; }
button[kind="primary"]:hover { filter: brightness(1.08); transform: scale(1.02); }
button[kind="secondary"], a[kind="secondary"] { background: transparent; border-color: #727272; }
button[kind="secondary"]:hover, a[kind="secondary"]:hover { border-color: #fff; color: #fff; }
button[kind="secondary"] p, a[kind="secondary"] p { font-weight: 600; }

/* ---------- tarjetas de lista ---------- */
[class*="st-key-fila-"] {
  position: relative; background: var(--sup); border-radius: 10px; padding: 0.65rem 0.8rem;
  flex-wrap: nowrap; gap: 0.6rem; transition: background 0.15s;
}
[class*="st-key-fila-"]:hover { background: var(--sup2); }
[class*="st-key-fila-"] > div:has(> [data-testid="stHtml"]) { flex: 1 1 auto; min-width: 0; width: auto; }
[class*="st-key-fila-"] [class*="st-key-accion-"] { position: relative; z-index: 2; flex: 0 0 auto; width: auto; }
/* El botón «Abrir» cubre toda la tarjeta: se toca en cualquier parte. */
[class*="st-key-abrir-"] { position: absolute !important; inset: 0; z-index: 1; width: auto !important; margin: 0; }
[class*="st-key-abrir-"] > div, [class*="st-key-abrir-"] button { width: 100%; height: 100%; }
[class*="st-key-abrir-"] button { opacity: 0; cursor: pointer; }
[class*="st-key-fila-"]:has([class*="st-key-abrir-"] button:focus-visible) { outline: 2px solid #fff; }

.fila { display: flex; align-items: center; gap: 0.75rem; min-width: 0; }
.fila .cuerpo { flex: 1 1 auto; min-width: 0; }
.fila .titulo { font-weight: 700; font-size: 0.98rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fila .sub { color: var(--txt2); font-size: 0.82rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fila .lado { flex: 0 0 auto; text-align: right; font-size: 0.82rem; color: var(--txt2); }
.fila .lado b { display: block; font-size: 0.98rem; color: var(--txt); }
.avatar {
  flex: 0 0 auto; width: 44px; height: 44px; border-radius: 8px; display: flex; align-items: center;
  justify-content: center; font-weight: 800; font-size: 0.95rem; color: #fff;
}
.avatar.grande { width: 88px; height: 88px; font-size: 1.9rem; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,.5); }
.punto { flex: 0 0 auto; width: 10px; height: 10px; border-radius: 50%; margin: 0 0.35rem 0 0.2rem; }
.chip {
  display: inline-block; padding: 0.12rem 0.55rem; border-radius: 500px; font-size: 0.72rem; font-weight: 700;
  background: var(--sup2); color: var(--txt2); white-space: nowrap; margin: 0.1rem 0.25rem 0.1rem 0;
}
.chip.color { background: color-mix(in srgb, var(--c) 18%, transparent); color: var(--c); }

/* ---------- indicadores ---------- */
.st-key-kpis { display: grid !important; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.6rem; }
[class*="st-key-fila-kpi-"] { padding: 0.9rem 1rem; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.6rem; }
.kpi { background: var(--sup); border-radius: 10px; padding: 0.9rem 1rem; min-width: 0; }
[class*="st-key-fila-kpi-"] .kpi { background: transparent; padding: 0; }
.kpi .v { font-size: 1.75rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; white-space: nowrap; }
.kpi .l { color: var(--txt2); font-size: 0.82rem; font-weight: 600; }
.kpi .d { color: var(--txt3); font-size: 0.75rem; margin-top: 0.1rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ---------- textos ---------- */
.saludo { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.15; }
.saludo small { display: block; font-size: 0.9rem; font-weight: 400; color: var(--txt2); letter-spacing: 0; margin-top: 0.25rem; }
.heroe { display: flex; align-items: flex-end; gap: 1.1rem; padding: 0.4rem 0 0.2rem; }
.heroe .nombre { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; }
.heroe .sub { color: var(--txt2); font-size: 0.88rem; margin-top: 0.3rem; }
.vacio { background: var(--sup); border-radius: 10px; padding: 1.4rem 1.2rem; color: var(--txt2); text-align: center; }
.vacio b { display: block; color: var(--txt); font-size: 1.05rem; margin-bottom: 0.2rem; }
.aviso-demo { display: inline-block; font-size: 0.75rem; font-weight: 700; color: var(--ambar);
  background: color-mix(in srgb, var(--ambar) 16%, transparent); border-radius: 500px; padding: 0.15rem 0.7rem; }
.datos { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.2rem 1.5rem; }
.datos div { padding: 0.55rem 0; border-bottom: 1px solid var(--borde); min-width: 0; }
.datos span { display: block; color: var(--txt3); font-size: 0.74rem; font-weight: 600; }
.datos b { font-weight: 500; overflow-wrap: anywhere; }
.nota { padding: 0.65rem 0; border-bottom: 1px solid var(--borde); }
.nota span { color: var(--txt3); font-size: 0.76rem; font-weight: 600; }
.mensaje { background: var(--sup); border-radius: 10px; padding: 0.9rem 1rem; white-space: pre-wrap; font-size: 0.88rem; color: var(--txt2); }

/* ---------- varios ---------- */
[data-testid="stExpander"] details { background: var(--sup); border: 0; border-radius: 10px; }
[data-testid="stExpander"] summary p { font-weight: 700; }
div[role="dialog"] { background: #1c1c1c; border-radius: 14px; }
[data-testid="stMetric"] { background: var(--sup); border-radius: 10px; padding: 0.8rem 1rem; }

/* ---------- celular ---------- */
@media (max-width: 760px) {
  [data-testid="stMainBlockContainer"] { padding: 1.1rem 0.9rem 6.5rem 0.9rem; }
  h1 { font-size: 1.6rem !important; }
  .saludo, .heroe .nombre { font-size: 1.55rem; }
  .avatar.grande { width: 68px; height: 68px; font-size: 1.5rem; }
  .heroe { align-items: center; gap: 0.9rem; }

  /* La navegación baja al borde inferior, al alcance del pulgar. */
  .st-key-nav {
    top: auto; right: 0; width: auto; flex-direction: row; align-items: stretch; gap: 0;
    padding: 0.3rem 0.2rem calc(0.3rem + env(safe-area-inset-bottom)); background: #0a0a0a;
    border-top: 1px solid var(--borde);
  }
  .st-key-nav > div { flex: 1 1 0; min-width: 0; width: auto; }
  .st-key-nav > div:has(.marca) { display: none; }
  .st-key-nav button, .st-key-nav button[kind="primary"] { padding: 0.3rem 0; background: transparent !important; justify-content: center; }
  .st-key-nav button > div, .st-key-nav button > div > span { flex-direction: column; align-items: center; justify-content: center; gap: 1px; }
  .st-key-nav button p { font-size: 0.66rem; font-weight: 600; white-space: nowrap; }
  .st-key-nav [data-testid="stIconMaterial"] { font-size: 1.55rem; }

  .st-key-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .kpi .v { font-size: 1.45rem; }
  /* Tarjetas: el texto ocupa todo el ancho y el plazo o el monto baja a una segunda línea. */
  [class*="st-key-fila-"] { padding: 0.65rem 0.7rem; gap: 0.5rem; }
  .fila { display: grid; grid-template-columns: auto minmax(0, 1fr); column-gap: 0.65rem; row-gap: 0.1rem; }
  .fila .titulo { white-space: normal; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
  .fila .sub { white-space: normal; }
  .fila .lado { grid-column: 2; text-align: left; display: flex; gap: 0.5rem; align-items: baseline; }
  .fila .lado b { display: inline; font-size: 0.85rem; }
  /* Botones cómodos para el dedo; las acciones de las tarjetas quedan como un círculo con su ícono. */
  button[kind^="primary"], button[kind^="secondary"] { min-height: 2.6rem; }
  [class*="st-key-accion-"] button { width: 2.75rem; height: 2.75rem; min-height: 0; padding: 0; }
  [class*="st-key-accion-"] button [data-testid="stMarkdownContainer"] {
    position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%);
  }
}
</style>
"""


def e(texto):
    """Escapa texto del usuario para insertarlo en HTML."""
    return html.escape(str(texto if texto is not None else ""))


def iniciales(nombre):
    palabras = [p for p in str(nombre or "").replace(".", " ").split() if p[0].isalnum()]
    return "".join(p[0] for p in palabras[:2]).upper() or "?"


def color_avatar(nombre):
    return _AVATARES[sum(ord(ch) for ch in str(nombre or "")) % len(_AVATARES)]


def avatar(nombre, grande=False):
    return f'<div class="avatar{" grande" if grande else ""}" style="background:{color_avatar(nombre)}">{e(iniciales(nombre))}</div>'


def chip(texto, color=None):
    if color:
        return f'<span class="chip color" style="--c:{color}">{e(texto)}</span>'
    return f'<span class="chip">{e(texto)}</span>'


def fila(titulo, sub="", lado="", lado_sub="", icono="", color_lado=None):
    """Contenido de una tarjeta de lista. `icono` y los chips de `sub` ya vienen como HTML seguro."""
    estilo = f' style="color:{color_lado}"' if color_lado else ""
    fuerte = f"<b{estilo}>{e(lado)}</b>" if lado else ""
    derecha = f'<div class="lado">{fuerte}{e(lado_sub)}</div>' if lado or lado_sub else ""
    return (
        f'<div class="fila">{icono}<div class="cuerpo"><div class="titulo">{e(titulo)}</div>'
        f'<div class="sub">{sub}</div></div>{derecha}</div>'
    )


def punto(color):
    return f'<span class="punto" style="background:{color}"></span>'


def kpi(valor, etiqueta, detalle="", color=None):
    estilo = f' style="color:{color}"' if color else ""
    return (
        f'<div class="kpi"><div class="v"{estilo}>{e(valor)}</div><div class="l">{e(etiqueta)}</div>'
        + (f'<div class="d">{e(detalle)}</div>' if detalle else "")
        + "</div>"
    )


def kpis(*tarjetas):
    return f'<div class="kpis">{"".join(tarjetas)}</div>'


def vacio(titulo, texto=""):
    return f'<div class="vacio"><b>{e(titulo)}</b>{e(texto)}</div>'


def datos(pares):
    """Lista de «etiqueta: valor» en grilla; omite los valores vacíos."""
    celdas = "".join(f"<div><span>{e(k)}</span><b>{e(v)}</b></div>" for k, v in pares if v not in (None, "", []))
    return f'<div class="datos">{celdas}</div>'
