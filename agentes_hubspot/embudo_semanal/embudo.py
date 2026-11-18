#!/usr/bin/env python3
"""Agente de embudo semanal · Tu Proyecto de Vida.

Cada lunes compara el embudo real de cada proyecto en HubSpot con el Escenario Pesimista
de la planilla «05 Performance Embudo» y arma el correo con los puntos críticos, el plan
de acción y su responsable. Solo usa la biblioteca estándar de Python.

Uso:
  python3 embudo.py consultas --hoy AAAA-MM-DD [--config config.json]
      Imprime en JSON las consultas SQL de HubSpot (query_crm_data) y el archivo donde
      guardar cada resultado.

  python3 embudo.py build --hoy AAAA-MM-DD --plan plan.json --entrada entrada/ \
      [--historial historial.json] [--config config.json] [--salida salida/]
      Lee la planilla y los conteos de HubSpot, y deja en salida/: correo.html,
      correo.txt, pagina.html, historial.json y resumen.json. Imprime las líneas
      ASUNTO=, RESUMEN= y, si corresponde, ADVERTENCIA: …

Códigos de salida: 0 bien · 2 la planilla no se pudo leer · 3 los conteos de HubSpot son
incoherentes (revisar que se copiaron tal cual) · 4 falta un archivo de entrada.
"""

import argparse
import json
import re
import sys
import unicodedata
from datetime import date, timedelta
from html import escape
from pathlib import Path

AQUI = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------------------
# Etapas del «Pipeline de ventas» (pipeline = 'default') de HubSpot.
# Un negocio «llegó» a una etapa si entró a ella o a cualquier etapa posterior: así se
# cuenta a los que se la saltaron (p. ej. una reserva sin visita registrada) y a los
# perdidos después de pasar por ella, como en un embudo acumulado de HubSpot.
# ---------------------------------------------------------------------------------------
PIPELINE = "default"
V_IDS = ["1026814100", "100760847", "189519373", "1049776740", "189519372", "closedwon"]
# «CV Firmada con Instrucción» y «CV Firmada con Crédito Hipotecario» no tienen fecha de
# entrada calculada en HubSpot: se cuentan por la etapa actual.
V_ACTUAL = ["1433853017", "1433840938"]
PR_IDS = ["decisionmakerboughtin", "contractsent", "100768101", "100779324"] + V_IDS
VR_IDS = ["1036111516"] + PR_IDS
VA_IDS = ["189519371"] + VR_IDS
IC_IDS = ["232657274", "232653831", "232659764", "qualifiedtobuy"] + VA_IDS

ORDEN = ["io", "ic", "va", "vr", "pr", "v"]
ETAPAS = {
    "io": {"sigla": "IO", "nombre": "Identificación de Oportunidad", "ids": None},
    "ic": {"sigla": "IC", "nombre": "Interés Confirmado", "ids": IC_IDS},
    "va": {"sigla": "VA", "nombre": "Visita Agendada", "ids": VA_IDS},
    "vr": {"sigla": "VR", "nombre": "Visita Realizada", "ids": VR_IDS},
    "pr": {"sigla": "PR", "nombre": "Parcela Reservada", "ids": PR_IDS},
    "v": {"sigla": "V", "nombre": "Venta (promesa o compraventa firmada)", "ids": V_IDS},
}

# Tramos del embudo, con el nombre de la columna en la planilla.
TASAS = [
    {"clave": "mql", "desde": "io", "hasta": "ic", "tramo": "IO → IC", "columna": "%MQL",
     "frase": "de las oportunidades llega a Interés Confirmado"},
    {"clave": "sql", "desde": "ic", "hasta": "va", "tramo": "IC → VA", "columna": "%SQL",
     "frase": "de los interesados confirmados agenda visita"},
    {"clave": "opp", "desde": "va", "hasta": "vr", "tramo": "VA → VR", "columna": "%Opportunity",
     "frase": "de las visitas agendadas se realiza"},
    {"clave": "closed", "desde": "vr", "hasta": "pr", "tramo": "VR → PR", "columna": "%Closed",
     "frase": "de las visitas realizadas termina en reserva"},
    {"clave": "won", "desde": "pr", "hasta": "v", "tramo": "PR → V", "columna": "%WON",
     "frase": "de las reservas llega a promesa o compraventa"},
]
TASA = {t["clave"]: t for t in TASAS}

# Qué hacer cuando un tramo (o el ritmo de oportunidades) está bajo el plan.
ACCIONES = {
    "ritmo": "Revisar las campañas de {proyectos} (inversión ejecutada, costo por lead y anuncios activos) "
             "y proponer el ajuste de inversión o de anuncios para volver al ritmo de oportunidades del plan.",
    "mql": "Revisar con ventas una muestra de 20 oportunidades de {proyectos} que no llegaron a Interés "
           "Confirmado: separar lead de mala calidad (segmentación, formulario, fuente) de lead no "
           "contactado, y corregir la campaña o el formulario con peor conversión.",
    "sql": "En cada llamada a interesados de {proyectos}, ofrecer dos horarios concretos de visita; "
           "cadencia de 3 intentos en 5 días para los Interés Confirmado sin visita, y revisar el "
           "guion de agendamiento con los ejecutivos de esos proyectos.",
    "opp": "Confirmar cada visita de {proyectos} por WhatsApp 24 h y 2 h antes (con ubicación y cómo "
           "llegar), reagendar dentro de 48 h toda inasistencia y registrar en HubSpot el resultado "
           "de cada visita.",
    "closed": "Seguimiento dentro de 24 h a cada visita realizada de {proyectos} con propuesta escrita "
              "(precio, forma de pago, vigencia de la oferta); el jefe de cada proyecto revisa las "
              "objeciones de las visitas que no reservaron y acompaña las visitas de la semana.",
    "won": "Revisar una a una las reservas abiertas de {proyectos} con Escrituración (documentos, "
           "financiamiento y fecha de firma), fijar fecha de promesa para cada una y registrar el "
           "motivo de las reservas caídas.",
}

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
ICONO = {"rojo": "🔴", "amarillo": "🟡", "verde": "🟢"}

# Colores del correo (estilos en línea: los clientes de correo no leen hojas de estilo).
C_TEXTO, C_SECUNDARIO, C_BORDE, C_FONDO, C_ACENTO = "#1d211e", "#5b615c", "#e1e4de", "#f6f7f4", "#2f5d46"
C_ESTADO = {
    "rojo": ("#a8231a", "#fdecea"),
    "amarillo": ("#7a4b00", "#fff3d6"),
    "verde": ("#1a6b34", "#e6f4ea"),
    "muestra": ("#6b706b", "#f1f2ef"),
    "sin_datos": ("#6b706b", "#f1f2ef"),
}
FUENTE = "font-family:Arial,Helvetica,sans-serif;"

# Clases del correo. Gmail respeta un <style> en <head> con selectores de clase simples; así el
# correo pesa una fracción de lo que pesaría con estilos en línea (Gmail recorta sobre 102 KB).
CORREO_CSS = (
    f".e{{max-width:680px;margin:0 auto;padding:20px 14px;background:{C_FONDO};color:{C_TEXTO};{FUENTE}}}"
    f".eceja{{margin:0 0 4px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:{C_ACENTO};font-weight:bold}}"
    f".eh1{{margin:0 0 6px;font-size:22px;line-height:1.3;color:{C_TEXTO}}}"
    f".eh2{{font-size:17px;line-height:1.3;margin:28px 0 10px;color:{C_TEXTO}}}"
    f".ep{{font-size:14px;line-height:1.5;margin:0 0 8px;color:{C_TEXTO}}}"
    f".es{{font-size:12px;color:{C_SECUNDARIO}}}"
    f".exs{{font-size:11px;color:{C_SECUNDARIO}}}"
    f".e13{{font-size:13px}}"
    f".etx{{color:{C_TEXTO}}}"
    f".eul{{font-size:14px;line-height:1.5;margin:0;padding-left:20px;color:{C_TEXTO}}}"
    f".eli{{margin:0 0 6px}}"
    f".eox{{overflow-x:auto}}"
    f".etab{{width:100%;border-collapse:collapse}}"
    f".emin{{min-width:600px}}"
    f".eth{{font-size:11px;font-weight:bold;color:{C_SECUNDARIO};text-align:right;padding:6px;"
    f"border-bottom:1px solid {C_BORDE};white-space:nowrap}}"
    f".etd{{font-size:13px;color:{C_TEXTO};text-align:right;padding:6px;border-bottom:1px solid {C_BORDE};vertical-align:top}}"
    f".el{{text-align:left}}"
    f".e12{{font-size:12px}}"
    f".etasa{{white-space:nowrap;padding:5px}}"
    f".etr{{font-weight:bold;font-size:13px}}"
    f".etp{{font-size:10px;color:{C_SECUNDARIO}}}"
    f".echip{{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px;font-weight:bold}}"
    f".ecard{{border:1px solid {C_BORDE};border-radius:6px;padding:12px 14px;margin:0 0 12px;background:#ffffff}}"
    f".ecardt{{margin:0 0 2px;font-size:15px;line-height:1.35;color:{C_TEXTO}}}"
    f".ecards{{margin:0 0 8px;font-size:13px;color:{C_SECUNDARIO}}}"
    f".epie{{margin:28px 0 0;font-size:11px;line-height:1.5;color:{C_SECUNDARIO}}}"
    f".ea{{color:{C_ACENTO}}}"
    + "".join(f".e-{k}{{color:{c};background:{f}}}.ec-{k}{{color:{c}}}.eb-{k}{{border-left:4px solid {c}}}"
              for k, (c, f) in C_ESTADO.items())
)


class ErrorEntrada(Exception):
    def __init__(self, mensaje, codigo):
        super().__init__(mensaje)
        self.codigo = codigo


# ---------------------------------------------------------------------------------------
# Utilidades de formato (formato chileno: 1.234 y 12,3 %)
# ---------------------------------------------------------------------------------------
def normalizar(texto):
    sin_tildes = unicodedata.normalize("NFKD", str(texto or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", sin_tildes).strip().lower()


def f_ent(n):
    return f"{int(round(n)):,}".replace(",", ".")


def f_dec(x, dec=1):
    return f"{x:.{dec}f}".replace(".", ",")


def f_pct(x, dec=1):
    return "—" if x is None else f"{x * 100:.{dec}f}".replace(".", ",") + "%"


def f_signo(x, dec=1):
    if x is None:
        return "—"
    if round(x, dec) == 0:
        return f_dec(0, dec)
    return ("+" if x > 0 else "−") + f_dec(abs(x), dec)


def f_brecha(x):
    return "—" if x is None else f_signo(x * 100, 0) + "%"


def f_pp(x):
    """Diferencia en puntos porcentuales."""
    return "—" if x is None else f_signo(x * 100) + " pp"


def f_fecha(d):
    return d.strftime("%d-%m-%Y")


def f_corta(d):
    return d.strftime("%d-%m")


def f_larga(d):
    return f"{DIAS[d.weekday()]} {f_fecha(d)}"


def f_dia(d):
    return f"{DIAS[d.weekday()]} {f_corta(d)}"


# ---------------------------------------------------------------------------------------
# Configuración, fechas y consultas
# ---------------------------------------------------------------------------------------
def cargar_config(ruta=None):
    ruta = Path(ruta) if ruta else AQUI / "config.json"
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def ventanas(hoy, cfg):
    coh = cfg.get("cohorte", {})
    fin = hoy - timedelta(weeks=coh.get("desfase_semanas", 6))
    inicio = fin - timedelta(weeks=coh.get("semanas", 52))
    dias = cfg.get("ritmo_dias", 28)
    return {
        "cohorte_desde": inicio,
        "cohorte_hasta": fin,  # exclusivo
        "ritmo_desde": hoy - timedelta(days=dias),
        "ritmo_hasta": hoy,  # exclusivo
        "ritmo_dias": dias,
    }


def plazos(hoy):
    viernes = hoy + timedelta(days=(4 - hoy.weekday()) % 7)
    revision = hoy + timedelta(days=(7 - hoy.weekday()) % 7 or 7)
    return viernes, revision


def sql_conteo(etapa, desde, hasta):
    condicion = ""
    ids = ETAPAS[etapa]["ids"]
    if ids:
        partes = [f"hs_v2_date_entered_{i} IS NOT NULL" for i in ids]
        partes += [f"dealstage = '{i}'" for i in V_ACTUAL]
        condicion = " AND (" + " OR ".join(partes) + ")"
    return (
        f"SELECT proyecto_de_interes, COUNT(*) FROM DEAL WHERE pipeline = '{PIPELINE}' "
        f"AND createdate >= '{desde.isoformat()}' AND createdate < '{hasta.isoformat()}'{condicion} "
        "GROUP BY proyecto_de_interes ORDER BY COUNT(*) DESC LIMIT 200"
    )


def consultas(hoy, cfg):
    v = ventanas(hoy, cfg)
    lista = [
        {"archivo": f"cohorte_{e}.tsv", "etapa": e, "sql": sql_conteo(e, v["cohorte_desde"], v["cohorte_hasta"])}
        for e in ORDEN
    ]
    lista.append({"archivo": "ritmo_io.tsv", "etapa": "io", "sql": sql_conteo("io", v["ritmo_desde"], v["ritmo_hasta"])})
    return {
        "hoy": hoy.isoformat(),
        "cohorte": {"desde": v["cohorte_desde"].isoformat(), "hasta_exclusivo": v["cohorte_hasta"].isoformat()},
        "ritmo": {"desde": v["ritmo_desde"].isoformat(), "hasta_exclusivo": v["ritmo_hasta"].isoformat()},
        "consultas": lista,
    }


# ---------------------------------------------------------------------------------------
# Lectura de la planilla (resultado de Google Sheets get_values)
# ---------------------------------------------------------------------------------------
def _num(texto):
    t = str(texto or "").strip().replace("$", "").replace(" ", "")
    if not t or t in ("-", "—"):
        return None
    try:
        return float(t.replace(",", ""))
    except ValueError:
        digitos = re.sub(r"[^\d]", "", t)
        return float(digitos) if digitos else None


def _pct(texto):
    t = str(texto or "").strip()
    if not t:
        return None
    t = t.replace("%", "").replace(" ", "")
    if "," in t and "." not in t:
        t = t.replace(",", ".")
    try:
        valor = float(t)
    except ValueError:
        return None
    return valor / 100 if "%" in str(texto) or valor > 1 else valor


def _valores_planilla(datos):
    if isinstance(datos, dict):
        if "values" in datos:
            return datos["values"]
        for clave in ("result", "content", "data"):
            if clave in datos:
                return _valores_planilla(datos[clave])
    if isinstance(datos, list):
        return datos
    raise ErrorEntrada("La planilla no trae la lista «values» de Google Sheets.", 2)


def leer_plan(ruta):
    try:
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)
    except FileNotFoundError:
        raise ErrorEntrada(f"No existe el archivo de la planilla: {ruta}", 4)
    except json.JSONDecodeError as e:
        raise ErrorEntrada(f"La planilla no es un JSON válido ({e}).", 2)
    filas = _valores_planilla(datos)

    buscadas = {"proyecto": "proyecto", "io total": "io", "venta cohorte potencial": "potencial", "venta total": "venta_total"}
    buscadas.update({normalizar(t["columna"]): t["clave"] for t in TASAS})
    encabezado, columnas = None, {}
    for i, fila in enumerate(filas):
        normal = [normalizar(c) for c in fila]
        if "proyecto" in normal and "io total" in normal:
            encabezado = i
            columnas = {buscadas[n]: j for j, n in enumerate(normal) if n in buscadas}
            break
    faltan = [c for c in ["proyecto", "io"] + [t["clave"] for t in TASAS] if c not in columnas]
    if encabezado is None or faltan:
        raise ErrorEntrada(
            "No encontré el encabezado de la hoja (Proyecto, IO TOTAL, %MQL, %SQL, %Opportunity, %Closed, %WON)"
            + (f"; faltan: {', '.join(faltan)}" if faltan else "") + ".", 2)

    def celda(fila, clave):
        j = columnas.get(clave)
        return fila[j] if j is not None and j < len(fila) else ""

    proyectos, advertencias = [], []
    for fila in filas[encabezado + 1:]:
        nombre = str(celda(fila, "proyecto")).strip()
        if not nombre:
            continue
        if normalizar(nombre) == "total":
            break
        tasas = {t["clave"]: _pct(celda(fila, t["clave"])) for t in TASAS}
        if any(v is None for v in tasas.values()):
            advertencias.append(f"{nombre}: la planilla tiene tasas vacías; queda fuera del análisis.")
            continue
        io = _num(celda(fila, "io")) or 0.0
        potencial = io
        for valor in tasas.values():
            potencial *= valor
        p = {
            "nombre": nombre,
            "io_plan": io,
            "tasas_plan": tasas,
            "pot_plan": potencial,
            "pot_hoja": _num(celda(fila, "potencial")),
            "venta_total": _num(celda(fila, "venta_total")),
        }
        if p["pot_hoja"] is not None and abs(p["pot_hoja"] - potencial) > 1.01:
            advertencias.append(
                f"{nombre}: la venta potencial de la planilla ({f_dec(p['pot_hoja'], 0)}) no calza con IO × tasas "
                f"({f_dec(potencial)}). Se usa IO × tasas.")
        proyectos.append(p)
    if len(proyectos) < 3:
        raise ErrorEntrada(f"La hoja trae solo {len(proyectos)} proyectos con datos completos.", 2)
    return proyectos, advertencias


# ---------------------------------------------------------------------------------------
# Lectura de los conteos de HubSpot (resultado de query_crm_data)
# ---------------------------------------------------------------------------------------
VACIOS = {"", "(no value)", "(sin valor)", "null", "none", "-", "(empty)"}


def _texto_tsv(contenido):
    """Acepta el TSV tal cual o el JSON completo que devuelve query_crm_data."""
    contenido = contenido.strip()
    if contenido[:1] in "{[":
        try:
            datos = json.loads(contenido)
        except json.JSONDecodeError:
            datos = None
        if datos is not None:
            textos = []

            def juntar(x):
                if isinstance(x, dict):
                    for v in x.values():
                        juntar(v)
                elif isinstance(x, list):
                    for v in x:
                        juntar(v)
                elif isinstance(x, str):
                    textos.append(x)

            juntar(datos)
            contenido = "\n".join(textos)
    if "Dataset TSV:" in contenido:
        contenido = contenido.split("Dataset TSV:", 1)[1]
    return contenido


def leer_conteos(ruta):
    try:
        contenido = Path(ruta).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ErrorEntrada(f"No existe el archivo de HubSpot: {ruta}", 4)
    conteos = {}
    for linea in _texto_tsv(contenido).splitlines():
        if "\t" not in linea:
            continue
        nombre, valor = linea.rsplit("\t", 1)
        nombre = nombre.strip()
        if "[proyecto_de_interes]" in nombre or "[count]" in valor:
            continue
        try:
            n = int(round(float(valor.strip().replace(".", "").replace(",", ""))))
        except ValueError:
            continue
        clave = "" if normalizar(nombre) in VACIOS else nombre
        conteos[clave] = conteos.get(clave, 0) + n
    return conteos


def claves_hubspot(nombre):
    """«Etiqueta (valor)» → {etiqueta, valor, texto completo}, normalizados."""
    claves = {normalizar(nombre)}
    m = re.match(r"^(.*)\s\(([^()]*)\)$", nombre.strip())
    if m:
        claves.update({normalizar(m.group(1)), normalizar(m.group(2))})
    return claves


# ---------------------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------------------
def _z_tasa(real, plan, d):
    """Desviación de la tasa real respecto del plan, en errores estándar (binomial)."""
    if real is None or not d or plan is None or not 0 < plan < 1:
        return None
    return (real - plan) / (plan * (1 - plan) / d) ** 0.5


def _z_conteo(real, esperado):
    """Desviación de un conteo respecto de lo esperado, en errores estándar (Poisson)."""
    return (real - esperado) / esperado ** 0.5 if esperado else None


def _estado(brecha, z, umbrales):
    """rojo / amarillo solo si la brecha supera el umbral y además sale del margen normal (95%)."""
    if brecha is None:
        return "muestra"
    if brecha > umbrales["amarillo"]:
        return "verde"
    if z is not None and z > -umbrales.get("z_minimo", 1.645):
        return "muestra"  # bajo el plan, pero dentro de la variación normal
    return "rojo" if brecha <= umbrales["rojo"] else "amarillo"


def _mapa_proyectos(plan, cfg):
    """Nombre en la planilla → conjunto de claves de HubSpot."""
    por_config = {normalizar(p["plan"]): p.get("hubspot", []) for p in cfg.get("proyectos", [])}
    mapa = {}
    for p in plan:
        valores = por_config.get(normalizar(p["nombre"]))
        if valores is None:
            valores = [p["nombre"]]  # sin configuración: se busca por el mismo nombre
        claves = set()
        for v in valores:
            claves |= claves_hubspot(v)
        mapa[p["nombre"]] = claves
    return mapa


def _semanas_previas(historial, hoy):
    """Fotos anteriores, de la más reciente a la más antigua, solo mientras sean semanales."""
    fotos = sorted(
        (s for s in historial.get("semanas", []) if s.get("fecha", "") < hoy.isoformat()),
        key=lambda s: s["fecha"], reverse=True)
    cadena, esperado = [], hoy
    for s in fotos:
        dias = (esperado - date.fromisoformat(s["fecha"])).days
        if not 5 <= dias <= 9:
            break
        cadena.append(s)
        esperado = date.fromisoformat(s["fecha"])
    return cadena


def analizar(plan, conteos, ritmo, cfg, hoy, historial=None):
    umbrales = cfg["umbrales"]
    n_min = umbrales["n_minimo"]
    v = ventanas(hoy, cfg)
    responsables = cfg["responsables"]
    advertencias = []
    mapa = _mapa_proyectos(plan, cfg)

    # Asignar cada nombre de HubSpot a un proyecto del plan.
    usados = set()

    def contar(tabla, claves):
        total = 0
        for nombre, n in tabla.items():
            if nombre and claves_hubspot(nombre) & claves:
                total += n
                usados.add(nombre)
        return total

    nombres_config = {normalizar(x["plan"]): x["plan"] for x in cfg.get("proyectos", [])}
    proyectos = []
    for p in plan:
        claves = mapa[p["nombre"]]
        # Se muestra el nombre de la configuración (corrige erratas de la planilla, p. ej. «RaÍces»).
        p = {**p, "nombre": nombres_config.get(normalizar(p["nombre"]), p["nombre"])}
        c = {e: contar(conteos[e], claves) for e in ORDEN}
        io4 = contar(ritmo, claves)
        for a, b in zip(ORDEN, ORDEN[1:]):
            if c[b] > c[a]:
                raise ErrorEntrada(
                    f"{p['nombre']}: {ETAPAS[b]['sigla']} ({c[b]}) es mayor que {ETAPAS[a]['sigla']} ({c[a]}). "
                    "Los conteos de HubSpot deben bajar etapa a etapa: revisa que cada resultado se copió tal "
                    "cual en su archivo.", 3)
        tiene_datos = c["io"] > 0
        tramos = {}
        pot_real = p["io_plan"]
        for t in TASAS:
            d, n = c[t["desde"]], c[t["hasta"]]
            real = n / d if d else None
            plan_t = p["tasas_plan"][t["clave"]]
            muestra_ok = d >= n_min and real is not None
            brecha = (real / plan_t - 1) if (muestra_ok and plan_t) else None
            z = _z_tasa(real, plan_t, d) if muestra_ok else None
            if not tiene_datos:
                estado = "sin_datos"
            elif not muestra_ok:
                estado = "muestra"
            else:
                estado = _estado(brecha, z, umbrales)
            impacto = p["pot_plan"] * brecha if brecha is not None else None
            pot_real *= real if muestra_ok else plan_t
            tramos[t["clave"]] = {
                "real": real, "plan": plan_t, "n": n, "d": d, "brecha": brecha, "z": z,
                "estado": estado, "impacto": impacto,
            }
        plan_periodo = p["io_plan"] * v["ritmo_dias"] / 365
        r = io4 / plan_periodo if plan_periodo else None
        ritmo_ok = tiene_datos and r is not None and plan_periodo >= n_min
        z = _z_conteo(io4, plan_periodo) if ritmo_ok else None
        if not tiene_datos:
            estado_ritmo = "sin_datos"
        elif not ritmo_ok:
            estado_ritmo = "muestra"
        else:
            estado_ritmo = _estado(r - 1, z, umbrales)
        tramos["ritmo"] = {
            "real": io4, "plan": plan_periodo, "razon": r, "brecha": (r - 1) if ritmo_ok else None, "z": z,
            "estado": estado_ritmo, "impacto": p["pot_plan"] * (r - 1) if ritmo_ok else None,
        }
        proyectos.append({
            **p, "c": c, "io4": io4, "tiene_datos": tiene_datos, "tramos": tramos,
            "pot_real": pot_real if tiene_datos else None,
        })

    fuera = {}
    for tabla in (conteos["io"],):
        for nombre, n in tabla.items():
            if nombre not in usados:
                fuera[nombre or "(sin proyecto)"] = fuera.get(nombre or "(sin proyecto)", 0) + n

    # Portafolio: proyectos con plan y con datos en HubSpot.
    validos = [p for p in proyectos if p["tiene_datos"] and p["pot_plan"] > 0]
    pot_plan = sum(p["pot_plan"] for p in validos)
    pot_real = sum(p["pot_real"] for p in validos)
    razon = pot_real / pot_plan if pot_plan else None
    portafolio = {}
    for t in TASAS:
        d = sum(p["c"][t["desde"]] for p in validos)
        n = sum(p["c"][t["hasta"]] for p in validos)

        def flujo(p, hasta):
            valor = p["io_plan"]
            for tt in TASAS:
                if ORDEN.index(tt["hasta"]) > ORDEN.index(hasta):
                    break
                valor *= p["tasas_plan"][tt["clave"]]
            return valor

        plan_d = sum(flujo(p, t["desde"]) for p in validos)
        plan_n = sum(flujo(p, t["hasta"]) for p in validos)
        real = n / d if d else None
        plan_t = plan_n / plan_d if plan_d else None
        impacto = sum(p["tramos"][t["clave"]]["impacto"] or 0 for p in validos)
        brecha = (real / plan_t - 1) if (real is not None and plan_t) else None
        z = _z_tasa(real, plan_t, d)
        portafolio[t["clave"]] = {"real": real, "plan": plan_t, "n": n, "d": d, "impacto": impacto,
                                  "brecha": brecha, "z": z, "estado": _estado(brecha, z, umbrales)}
    io4 = sum(p["io4"] for p in validos)
    io4_plan = sum(p["tramos"]["ritmo"]["plan"] for p in validos)
    brecha = (io4 / io4_plan - 1) if io4_plan else None
    z = _z_conteo(io4, io4_plan)
    portafolio["ritmo"] = {
        "real": io4, "plan": io4_plan, "razon": io4 / io4_plan if io4_plan else None, "brecha": brecha, "z": z,
        "estado": _estado(brecha, z, umbrales), "impacto": sum(p["tramos"]["ritmo"]["impacto"] or 0 for p in validos),
    }

    # Puntos críticos: tramos en rojo o amarillo con efecto relevante en ventas.
    previas = _semanas_previas(historial or {}, hoy)
    puntos = []
    for p in proyectos:
        for clave, tr in p["tramos"].items():
            if tr["estado"] not in ("rojo", "amarillo") or tr["impacto"] is None:
                continue
            if tr["impacto"] > -umbrales["impacto_minimo"]:
                continue
            semanas_rojo = 0
            if tr["estado"] == "rojo":
                semanas_rojo = 1
                for s in previas:
                    if s.get("proyectos", {}).get(p["nombre"], {}).get("estados", {}).get(clave) == "rojo":
                        semanas_rojo += 1
                    else:
                        break
            anterior = previas[0]["proyectos"].get(p["nombre"], {}).get("estados", {}).get(clave) if previas else None
            puntos.append({
                "proyecto": p["nombre"], "clave": clave, "estado": tr["estado"], "impacto": tr["impacto"],
                "semanas_rojo": semanas_rojo, "anterior": anterior,
                "nuevo": bool(previas) and anterior not in ("rojo", "amarillo"),
                "escalar": semanas_rojo >= umbrales["semanas_escalar"],
                "responsable": responsables["io" if clave == "ritmo" else clave],
            })
    puntos.sort(key=lambda x: x["impacto"])  # prioridad = efecto en ventas (más negativo primero)

    # Focos: los puntos agrupados por tramo, porque cada tramo tiene un responsable.
    focos = {}
    for x in puntos:
        f = focos.setdefault(x["clave"], {"clave": x["clave"], "responsable": x["responsable"], "puntos": []})
        f["puntos"].append(x)
    for f in focos.values():
        f["impacto"] = sum(x["impacto"] for x in f["puntos"])
        f["estado"] = "rojo" if any(x["estado"] == "rojo" for x in f["puntos"]) else "amarillo"
        f["escalar"] = [x for x in f["puntos"] if x["escalar"]]
    focos = sorted(focos.values(), key=lambda f: f["impacto"])

    salieron = []
    if previas:
        actuales = {(x["proyecto"], x["clave"]) for x in puntos}
        for nombre, datos in previas[0].get("proyectos", {}).items():
            for clave, estado in datos.get("estados", {}).items():
                if estado == "rojo" and (nombre, clave) not in actuales:
                    p = next((q for q in proyectos if q["nombre"] == nombre), None)
                    if p and p["tramos"].get(clave, {}).get("estado") in ("verde", "amarillo"):
                        salieron.append({"proyecto": nombre, "clave": clave, "estado": p["tramos"][clave]["estado"]})

    rojos = sum(1 for x in puntos if x["estado"] == "rojo")
    if razon is None:
        semaforo = "amarillo"
    elif razon >= umbrales["semaforo_verde"] and rojos == 0:
        semaforo = "verde"
    elif razon >= umbrales["semaforo_amarillo"]:
        semaforo = "amarillo"
    else:
        semaforo = "rojo"

    anterior_total = previas[0].get("total") if previas else None
    return {
        "hoy": hoy, "ventanas": v, "proyectos": proyectos, "portafolio": portafolio,
        "total": {"pot_plan": pot_plan, "pot_real": pot_real, "razon": razon,
                  "io_plan": sum(p["io_plan"] for p in validos), "io_real": sum(p["c"]["io"] for p in validos)},
        "puntos": puntos, "focos": focos, "rojos": rojos, "semaforo": semaforo, "fuera": fuera, "salieron": salieron,
        "anterior": anterior_total, "advertencias": advertencias, "cfg": cfg,
    }


def foto_historial(res):
    return {
        "fecha": res["hoy"].isoformat(),
        "cohorte": [res["ventanas"]["cohorte_desde"].isoformat(), res["ventanas"]["cohorte_hasta"].isoformat()],
        "total": {
            "pot_plan": round(res["total"]["pot_plan"], 3),
            "pot_real": round(res["total"]["pot_real"], 3),
            "razon": round(res["total"]["razon"], 5) if res["total"]["razon"] is not None else None,
            "tasas": {k: (round(x["real"], 5) if x.get("real") is not None and k != "ritmo" else x.get("real"))
                      for k, x in res["portafolio"].items()},
        },
        "proyectos": {
            p["nombre"]: {
                "c": p["c"], "io4": p["io4"],
                "pot_real": round(p["pot_real"], 3) if p["pot_real"] is not None else None,
                "estados": {k: tr["estado"] for k, tr in p["tramos"].items()},
            }
            for p in res["proyectos"]
        },
    }


def actualizar_historial(historial, res, maximo=104):
    semanas = [s for s in historial.get("semanas", []) if s.get("fecha") != res["hoy"].isoformat()]
    semanas.append(foto_historial(res))
    semanas.sort(key=lambda s: s["fecha"])
    return {"version": 1, "semanas": semanas[-maximo:]}


# ---------------------------------------------------------------------------------------
# Textos
# ---------------------------------------------------------------------------------------
def nombre_tramo(clave):
    if clave == "ritmo":
        return "Ritmo de IO"
    return TASA[clave]["tramo"]


def lista_nombres(nombres, maximo=3):
    nombres = list(nombres)
    if len(nombres) <= 1:
        return "".join(nombres)
    primeros = nombres[:maximo]
    texto = ", ".join(primeros[:-1]) + " y " + primeros[-1]
    return texto + (f" (y {len(nombres) - maximo} más)" if len(nombres) > maximo else "")


def que_hacer(foco):
    return ACCIONES[foco["clave"]].format(proyectos=lista_nombres(x["proyecto"] for x in foco["puntos"]))


def titulo_foco(clave):
    if clave == "ritmo":
        return "Volumen de oportunidades nuevas (IO)"
    t = TASA[clave]
    return f"{t['tramo']} · {ETAPAS[t['desde']]['nombre']} a {ETAPAS[t['hasta']]['nombre'].split(' (')[0]}"


def con_cargo(persona, extra=""):
    return " · ".join(x for x in (persona.get("cargo", ""), extra) if x)


def etiqueta_persistencia(x):
    if x["escalar"]:
        return f"{x['semanas_rojo']} semanas seguidas en rojo"
    if x["semanas_rojo"] >= 2:
        return f"{x['semanas_rojo']}ª semana en rojo"
    if x["nuevo"]:
        return "nuevo esta semana"
    return ""


def linea_resumen(res):
    t = res["total"]
    focos = "; ".join(f"{nombre_tramo(f['clave'])} {f_signo(f['impacto'])} ({len(f['puntos'])} proy., "
                      f"{f['responsable']['nombre']})" for f in res["focos"])
    return (f"{res['semaforo']} · ventas proyectadas {f_dec(t['pot_real'])} vs {f_dec(t['pot_plan'])} del plan "
            f"({f_pct(t['razon'], 0)}) · {res['rojos']} puntos en rojo y "
            f"{sum(1 for x in res['puntos'] if x['estado'] == 'amarillo')} en amarillo · focos: {focos or 'ninguno'}")


def asunto(res):
    texto = (f"Embudo semanal {f_fecha(res['hoy'])}: {ICONO[res['semaforo']]} ventas al "
             f"{f_pct(res['total']['razon'], 0)} del plan")
    if res["focos"]:
        texto += f" · foco en {nombre_tramo(res['focos'][0]['clave'])}"
    return texto


# ---------------------------------------------------------------------------------------
# Correo HTML
# ---------------------------------------------------------------------------------------
def _h2(texto):
    return f'<h2 class="eh2">{texto}</h2>'


def _p(texto, clase=""):
    return f'<p class="ep{" " + clase if clase else ""}">{texto}</p>'


def _chip(estado, texto):
    estado = estado if estado in C_ESTADO else "muestra"
    return f'<span class="echip e-{estado}">{escape(texto)}</span>'


def _th(texto, alin="right"):
    return f'<th class="eth{" el" if alin == "left" else ""}">{texto}</th>'


def _td(contenido, alin="right", clase=""):
    clases = "etd" + (" el" if alin == "left" else "") + (" " + clase if clase else "")
    return f'<td class="{clases}">{contenido}</td>'


def _sub(texto):
    return f'<br><span class="exs">{texto}</span>'


def _celda_tasa(tr):
    estado = tr["estado"] if tr["estado"] in C_ESTADO else "muestra"
    real = f_pct(tr["real"]) if tr["real"] is not None else "—"
    return (f'<td class="etd etasa e-{estado}"><span class="etr">{real}</span>'
            f'<br><span class="etp">{f_pct(tr["plan"])}</span></td>')


def _tabla(encabezado, filas, clase=""):
    return (f'<div class="eox"><table role="presentation" class="etab{" " + clase if clase else ""}">'
            f'<tr>{encabezado}</tr>{"".join(filas)}</table></div>')


def html_correo(res):
    cfg = res["cfg"]
    hoy = res["hoy"]
    v = res["ventanas"]
    viernes, revision = plazos(hoy)
    t = res["total"]
    pe = res["portafolio"]
    planilla = cfg.get("planilla", {})
    partes = []

    # Encabezado
    partes.append(
        f'<p class="eceja">Tu Proyecto de Vida · Embudo semanal</p>'
        f'<h1 class="eh1">{ICONO[res["semaforo"]]} Embudo comercial al {escape(f_larga(hoy))}</h1>'
        f'<p class="ep es">'
        f'Real: negocios creados en HubSpot del {f_fecha(v["cohorte_desde"])} al '
        f'{f_fecha(v["cohorte_hasta"] - timedelta(days=1))} (52 semanas que terminan hace 6, para que alcancen a avanzar). '
        f'Plan: Escenario Pesimista, hoja «{escape(planilla.get("hoja", "05 Performance Embudo"))}».</p>'
        f'<p class="ep es">Plazo de las acciones: <b>{escape(f_dia(viernes))}</b> · revisión en el informe del {escape(f_dia(revision))}.</p>'
    )

    # Resumen
    resumen = []
    variacion = ""
    if res["anterior"] and res["anterior"].get("pot_real") is not None:
        delta = t["pot_real"] - res["anterior"]["pot_real"]
        variacion = f" ({f_signo(delta)} vs la semana anterior)"
    resumen.append(
        f"<b>{ICONO[res['semaforo']]} Con las tasas reales, los {f_ent(t['io_plan'])} IO del plan darían "
        f"{f_dec(t['pot_real'])} ventas al año vs {f_dec(t['pot_plan'])} del Escenario Pesimista "
        f"({f_pct(t['razon'], 0)})</b>{variacion}.")
    amarillos = sum(1 for x in res["puntos"] if x["estado"] == "amarillo")
    if res["focos"]:
        f0 = res["focos"][0]
        resumen.append(
            f"<b>Foco principal: {nombre_tramo(f0['clave'])}</b>, bajo el plan en {len(f0['puntos'])} "
            f"proyecto{'s' if len(f0['puntos']) != 1 else ''} ({f_signo(f0['impacto'])} ventas al año). "
            f"Responsable: {escape(f0['responsable']['nombre'])}.")
        resumen.append(
            f"{res['rojos']} puntos en rojo y {amarillos} en amarillo, agrupados en {len(res['focos'])} "
            f"tramo{'s' if len(res['focos']) != 1 else ''}: " + "; ".join(
                f"{nombre_tramo(f['clave'])} {f_signo(f['impacto'])} ({escape(f['responsable']['nombre'])})"
                for f in res["focos"]) + ".")
    else:
        resumen.append("Ningún tramo de ningún proyecto está bajo el Escenario Pesimista fuera del margen normal.")
    bien = [tt for tt in TASAS if pe[tt["clave"]]["estado"] == "verde" and (pe[tt["clave"]]["brecha"] or 0) > 0.05
            and (pe[tt["clave"]]["z"] or 0) >= cfg["umbrales"].get("z_minimo", 1.645)]
    if bien:
        resumen.append("Bien: " + "; ".join(
            f"{tt['tramo']} {f_pct(pe[tt['clave']]['real'])} vs {f_pct(pe[tt['clave']]['plan'])} del plan "
            f"({f_signo(pe[tt['clave']]['impacto'])} ventas)" for tt in bien) + ".")
    ri = pe["ritmo"]
    resumen.append(
        f"Oportunidades nuevas de los últimos {v['ritmo_dias']} días: {f_ent(ri['real'])} vs {f_ent(ri['plan'])} "
        f"del ritmo del plan ({f_pct(ri['razon'], 0)}).")
    escalar = [x for x in res["puntos"] if x["escalar"]]
    if escalar:
        resumen.append(
            f"<b>{len(escalar)} punto{'s' if len(escalar) != 1 else ''} con {cfg['umbrales']['semanas_escalar']}+ "
            f"semanas en rojo</b>: se escala{'n' if len(escalar) != 1 else ''} a "
            f"{escape(cfg['escalamiento']['nombre'])}.")
    partes.append(_h2("Resumen"))
    partes.append(
        '<ul class="eul">' + "".join(f'<li class="eli">{r}</li>' for r in resumen) + "</ul>")

    # Embudo del portafolio
    filas = []
    ri_estado = ri["estado"]
    filas.append(
        "<tr>" + _td("<b>Ritmo IO</b>" + _sub(f"últimos {v['ritmo_dias']} días"), "left")
        + _td(f_ent(ri["real"])) + _td(f_ent(ri["plan"])) + _td(_chip(ri_estado, f_brecha(ri["brecha"])))
        + _td(f_signo(ri["impacto"])) + _td(escape(cfg["responsables"]["io"]["nombre"]), "left") + "</tr>")
    for tt in TASAS:
        x = pe[tt["clave"]]
        estado = x["estado"]
        variacion = ""
        if res["anterior"] and res["anterior"].get("tasas", {}).get(tt["clave"]) is not None and x["real"] is not None:
            dif = x["real"] - res["anterior"]["tasas"][tt["clave"]]
            variacion = _sub(f"{f_pp(dif)} vs sem. ant.")
        filas.append(
            "<tr>" + _td(f"<b>{tt['tramo']}</b>" + _sub(tt["columna"]), "left")
            + _td(f_pct(x["real"]) + variacion) + _td(f_pct(x["plan"]))
            + _td(_chip(estado, f_brecha(x["brecha"])))
            + _td(f_signo(x["impacto"])) + _td(escape(cfg["responsables"][tt["clave"]]["nombre"]), "left") + "</tr>")
    partes.append(_h2("Embudo del portafolio"))
    partes.append(_p(
        "Tasa real de todos los proyectos con plan, contra la tasa del plan ponderada por sus IO. "
        "«Efecto» = ventas al año que se pierden (o ganan) si ese tramo sigue como hoy en cada proyecto.", "es"))
    partes.append(_tabla(_th("Tramo", "left") + _th("Real") + _th("Plan") + _th("Brecha") + _th("Efecto")
                         + _th("Responsable", "left"), filas))

    # Puntos críticos: un bloque por tramo (cada tramo tiene un responsable)
    maximo = cfg["umbrales"]["max_puntos"]
    partes.append(_h2(f"Puntos críticos y plan de acción · plazo {escape(f_dia(viernes))}"))
    if not res["focos"]:
        partes.append(_p("Esta semana no hay tramos bajo el Escenario Pesimista fuera del margen normal y con un efecto "
                         f"de al menos {f_dec(cfg['umbrales']['impacto_minimo'])} ventas al año."))
    proyectos = {p["nombre"]: p for p in res["proyectos"]}
    for i, f in enumerate(res["focos"], 1):
        x_pe = pe[f["clave"]]
        if f["clave"] == "ritmo":
            portafolio_txt = (f"Portafolio: {f_ent(x_pe['real'])} oportunidades en {v['ritmo_dias']} días vs "
                              f"{f_ent(x_pe['plan'])} del ritmo del plan ({f_pct(x_pe['razon'], 0)}).")
        else:
            portafolio_txt = (f"Portafolio: {f_pct(x_pe['real'])} {TASA[f['clave']]['frase']} vs "
                              f"{f_pct(x_pe['plan'])} del plan.")
        filas = []
        for x in f["puntos"][:maximo]:
            tr = proyectos[x["proyecto"]]["tramos"][x["clave"]]
            if x["clave"] == "ritmo":
                real, plan_txt, base = f_ent(tr["real"]), f_ent(tr["plan"]), f"{f_pct(tr['razon'], 0)} del ritmo"
            else:
                real, plan_txt, base = f_pct(tr["real"]), f_pct(tr["plan"]), f"{f_ent(tr['n'])} de {f_ent(tr['d'])}"
            nota = etiqueta_persistencia(x)
            filas.append(
                "<tr>" + _td(f"{ICONO[x['estado']]} {escape(x['proyecto'])}", "left")
                + _td(f"<b>{real}</b>" + _sub(base))
                + _td(plan_txt) + _td(f"<b>{f_signo(x['impacto'])}</b>")
                + _td(_chip(x["estado"], nota) if nota else "", "left") + "</tr>")
        resto = f["puntos"][maximo:]
        if resto:
            filas.append(
                "<tr>" + '<td colspan="5" class="etd el es">'
                f"Además: {escape(lista_nombres([x['proyecto'] for x in resto], 99))} "
                f"({f_signo(sum(x['impacto'] for x in resto))} ventas).</td></tr>")
        escalamiento = ""
        if f["escalar"]:
            escalamiento = _p(
                f"<b>Escalamiento:</b> {escape(lista_nombres([x['proyecto'] for x in f['escalar']], 99))} "
                f"lleva{'n' if len(f['escalar']) != 1 else ''} {cfg['umbrales']['semanas_escalar']} o más semanas en rojo: "
                f"{escape(cfg['escalamiento']['nombre'])} define esta semana si se reasigna presupuesto o foco comercial.",
                "ec-rojo")
        persona = f["responsable"]
        partes.append(
            f'<div class="ecard eb-{f["estado"]}">'
            f'<p class="ecardt"><b>{i}. {ICONO[f["estado"]]} {escape(titulo_foco(f["clave"]))}</b></p>'
            f'<p class="ecards">Responsable: <b class="etx">{escape(persona["nombre"])}</b>'
            f'{(" (" + escape(persona["cargo"]) + ")") if persona.get("cargo") else ""} · '
            f'efecto <b class="ec-{f["estado"]}">{f_signo(f["impacto"])} ventas al año</b> en '
            f'{len(f["puntos"])} proyecto{"s" if len(f["puntos"]) != 1 else ""}</p>'
            + _p(escape(portafolio_txt), "e13")
            + _tabla(_th("Proyecto", "left") + _th("Real") + _th("Plan") + _th("Efecto") + _th("", "left"), filas)
            + _p(f"<b>Qué hacer antes del {escape(f_dia(viernes))}:</b> {escape(que_hacer(f))}")
            + escalamiento + "</div>")

    # Plan por responsable
    por_persona = {}
    for x in res["puntos"]:
        nombre = x["responsable"]["nombre"]
        por_persona.setdefault(nombre, {"persona": x["responsable"], "puntos": []})["puntos"].append(x)
    filas = []
    a_cargo = {}  # persona → tramos que le tocan, en el orden del embudo
    for clave in ["io", "mql", "sql", "opp", "closed", "won"]:
        persona = cfg["responsables"][clave]
        etiqueta = "Volumen de IO" if clave == "io" else TASA[clave]["tramo"]
        a_cargo.setdefault(persona["nombre"], {"persona": persona, "tramos": []})["tramos"].append(etiqueta)
    for nombre, datos in a_cargo.items():
        persona, tramos = datos["persona"], datos["tramos"]
        suyos = por_persona.get(nombre, {"puntos": []})["puntos"]
        focos_suyos = [f for f in res["focos"] if f["responsable"]["nombre"] == nombre]
        if focos_suyos:
            lista = "; ".join(
                f"{ICONO[f['estado']]} {nombre_tramo(f['clave'])} en {len(f['puntos'])} "
                f"proyecto{'s' if len(f['puntos']) != 1 else ''} ({f_signo(f['impacto'])})" for f in focos_suyos)
            lista += f". Ver focos {', '.join(str(res['focos'].index(f) + 1) for f in focos_suyos)}."
            efecto = f_signo(sum(x["impacto"] for x in suyos))
        else:
            lista, efecto = "Sin puntos críticos esta semana.", "0"
        filas.append(
            "<tr>" + _td(f"<b>{escape(nombre)}</b>" + _sub(escape(con_cargo(persona, ' · '.join(tramos)))), "left")
            + _td(str(len(suyos))) + _td(efecto) + _td(lista, "left", "e12") + "</tr>")
    esc = cfg["escalamiento"]
    seg = cfg["seguimiento"]
    filas.append(
        "<tr>" + _td(f"<b>{escape(esc['nombre'])}</b>" + _sub(escape(con_cargo(esc, 'escalamiento'))), "left")
        + _td(str(len(escalar))) + _td(f_signo(sum(x['impacto'] for x in escalar)) if escalar else "0")
        + _td("Define reasignación de presupuesto o foco en los puntos con "
              f"{cfg['umbrales']['semanas_escalar']}+ semanas en rojo." if escalar else
              f"Nada que escalar (se escala tras {cfg['umbrales']['semanas_escalar']} semanas seguidas en rojo).",
              "left", "e12") + "</tr>")
    filas.append(
        "<tr>" + _td(f"<b>{escape(seg['nombre'])}</b>" + _sub(escape(con_cargo(seg, 'seguimiento'))), "left")
        + _td("—") + _td("—")
        + _td(f"Revisa el avance de cada acción en el informe del {escape(f_dia(revision))}.", "left", "e12")
        + "</tr>")
    partes.append(_h2("Plan de acción por responsable"))
    partes.append(_tabla(_th("Responsable", "left") + _th("Puntos") + _th("Efecto") + _th("Tramos críticos", "left"), filas))

    # Seguimiento
    if res["salieron"]:
        partes.append(_h2("Seguimiento de la semana anterior"))
        partes.append(_p("<b>Salieron de rojo:</b> " + "; ".join(
            f"{ICONO[x['estado']]} {escape(x['proyecto'])} {nombre_tramo(x['clave'])}" for x in res["salieron"]) + "."))

    # Detalle por proyecto (mismas columnas que la planilla)
    filas = []
    for p in res["proyectos"]:
        if not p["tiene_datos"]:
            filas.append(
                "<tr>" + _td(escape(p["nombre"]), "left") + _td("—")
                + '<td colspan="5" class="etd el es">Sin negocios en HubSpot para este proyecto.</td>'
                + _td(f_dec(p["pot_plan"])) + _td("—") + _td("—") + "</tr>")
            continue
        brecha = p["pot_real"] - p["pot_plan"]
        clase = "ec-rojo" if brecha < -cfg["umbrales"]["impacto_minimo"] else (
            "ec-verde" if brecha > cfg["umbrales"]["impacto_minimo"] else "etx")
        filas.append(
            "<tr>" + _td(escape(p["nombre"]), "left") + _td(f_ent(p["c"]["io"]))
            + "".join(_celda_tasa(p["tramos"][tt["clave"]]) for tt in TASAS)
            + _td(f_dec(p["pot_plan"])) + _td(f_dec(p["pot_real"]))
            + _td(f'<b class="{clase}">{f_signo(brecha)}</b>') + "</tr>")
    filas.append(
        "<tr>" + _td("<b>Total con plan</b>", "left") + _td(f"<b>{f_ent(t['io_real'])}</b>")
        + "".join(_td(f"<b>{f_pct(pe[tt['clave']]['real'])}</b>") for tt in TASAS)
        + _td(f"<b>{f_dec(t['pot_plan'])}</b>") + _td(f"<b>{f_dec(t['pot_real'])}</b>")
        + _td(f"<b>{f_signo(t['pot_real'] - t['pot_plan'])}</b>") + "</tr>")
    partes.append(_h2("Detalle por proyecto"))
    partes.append(_p(
        "Arriba la tasa real y abajo, en gris, la del Escenario Pesimista. Color: 🔴 20% o más bajo · 🟡 10–20% "
        "bajo · 🟢 sobre el 90% del plan. En gris: diferencia dentro del margen normal, o menos de "
        f"{cfg['umbrales']['n_minimo']} negocios (ahí la proyección usa la tasa del plan). "
        "«Pot. real» = IO del plan × tasas reales.", "es"))
    partes.append(_tabla(_th("Proyecto", "left") + _th("IO real") + "".join(_th(tt["columna"]) for tt in TASAS)
                         + _th("Pot. plan") + _th("Pot. real") + _th("Brecha"), filas, "emin"))

    # Fuera del plan
    notas = []
    sin_plan = [p for p in res["proyectos"] if p["pot_plan"] <= 0 and p["tiene_datos"]]
    if sin_plan:
        notas.append("Con negocios en HubSpot pero con IO = 0 en el plan: " + ", ".join(
            f"{escape(p['nombre'])} ({f_ent(p['c']['io'])} IO)" for p in sin_plan) + ".")
    grandes = sorted(((n, c) for n, c in res["fuera"].items() if c >= 20), key=lambda kv: -kv[1])
    if grandes:
        notas.append("En HubSpot pero no en la planilla: " + ", ".join(
            f"{escape(n)} ({f_ent(c)} IO)" for n, c in grandes) + ".")
    if notas:
        partes.append(_h2("Proyectos fuera del plan"))
        partes.extend(_p(n, "e13") for n in notas)

    # Cómo se calcula
    partes.append(_h2("Cómo se calcula"))
    metodo = [
        "Etapas del «Pipeline de ventas» de HubSpot: IO Identificación de Oportunidad (negocio creado) · IC Interés "
        "Confirmado · VA Visita Agendada · VR Visita Realizada · PR Parcela Reservada · V Promesa o compraventa firmada.",
        "Un negocio cuenta en una etapa si llegó a ella o a una posterior (también si después se perdió).",
        "Cohorte: negocios creados en 52 semanas que terminan 6 semanas antes de hoy; así las tasas no castigan a "
        "los negocios recién creados.",
        "Efecto en ventas = venta potencial del plan × (tasa real ÷ tasa del plan − 1). Prioriza los puntos.",
        "Un tramo es punto crítico si está 10% o más bajo el plan, la diferencia sale del margen normal de "
        f"variación (95%) y su efecto es de al menos {f_dec(cfg['umbrales']['impacto_minimo'])} venta al año.",
        f"Semáforo general: 🟢 ventas proyectadas ≥ plan y sin rojos · 🟡 ≥ {f_pct(cfg['umbrales']['semaforo_amarillo'], 0)} "
        f"del plan · 🔴 bajo eso.",
    ]
    partes.append(
        '<ul class="eul es">' + "".join(f'<li class="eli">{escape(m)}</li>' for m in metodo) + "</ul>")
    if res["advertencias"]:
        partes.append(_h2("Notas de datos"))
        partes.extend(_p(escape(a), "es") for a in res["advertencias"])

    enlace = planilla.get("url", "")
    pie = ("Informe automático del agente «Análisis semanal del embudo» a partir de HubSpot (solo lectura) y la planilla "
           f"«{escape(planilla.get('nombre', ''))}»")
    if enlace:
        pie += f' (<a href="{escape(enlace)}" class="ea">abrir</a>)'
    partes.append(f'<p class="epie">{pie}.</p>')

    cuerpo = "".join(partes)
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{CORREO_CSS}</style></head>'
            f'<body style="margin:0;padding:0;background:{C_FONDO};"><div class="e">{cuerpo}</div></body></html>'), cuerpo


def texto_correo(res):
    t = res["total"]
    viernes, revision = plazos(res["hoy"])
    lineas = [
        asunto(res),
        "",
        f"Con las tasas reales, los IO del plan darían {f_dec(t['pot_real'])} ventas al año vs {f_dec(t['pot_plan'])} "
        f"del Escenario Pesimista ({f_pct(t['razon'], 0)}).",
        "",
        f"Puntos críticos y plan de acción (plazo {f_dia(viernes)}, revisión {f_dia(revision)}):",
    ]
    for i, f in enumerate(res["focos"], 1):
        lineas.append(f"{i}. {titulo_foco(f['clave'])}: {f_signo(f['impacto'])} ventas al año. "
                      f"Responsable: {f['responsable']['nombre']}. Proyectos: "
                      f"{lista_nombres([x['proyecto'] for x in f['puntos']], 6)}.")
    if not res["focos"]:
        lineas.append("Sin puntos críticos esta semana.")
    lineas += ["", "El detalle por proyecto está en la versión HTML de este correo."]
    return "\n".join(lineas)


# ---------------------------------------------------------------------------------------
# Página del agente (Artifact): última versión del informe
# ---------------------------------------------------------------------------------------
PAGINA_CSS = """
/* Concepto: ficha técnica del agente arriba y, debajo, el último informe tal como llega por correo. */
:root {
  --fondo: #f3f4ef; --superficie: #ffffff; --texto: #1d211e; --tenue: #5b615c;
  --linea: #dcdfd6; --bosque: #2f5d46; --musgo: #e3ebe4;
  --f-titulo: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --f-texto: "Inter", "Helvetica Neue", Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --fondo: #141815; --superficie: #1c211d; --texto: #e6e9e3; --tenue: #a3aaa2;
    --linea: #2f3630; --bosque: #8cc4a2; --musgo: #22312a; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --fondo: #141815; --superficie: #1c211d; --texto: #e6e9e3; --tenue: #a3aaa2;
  --linea: #2f3630; --bosque: #8cc4a2; --musgo: #22312a; color-scheme: dark;
}
body { background: var(--fondo); color: var(--texto); font-family: var(--f-texto); line-height: 1.5; }
.marco { max-width: 760px; margin: 0 auto; padding-block: 28px 48px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 24px; }
.marco > * { min-width: 0; }
.ceja { margin: 0; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--bosque); font-weight: 600; }
h1 { margin: 4px 0 6px; font-family: var(--f-titulo); font-size: clamp(24px, 4vw, 32px); line-height: 1.15; text-wrap: balance; }
.bajada { margin: 0; color: var(--tenue); max-width: 62ch; }
.ficha { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
.dato { background: var(--superficie); border: 1px solid var(--linea); border-radius: 8px; padding: 12px 14px; min-width: 0; }
.dato dt { font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--tenue); margin: 0 0 4px; }
.dato dd { margin: 0; font-size: 14px; }
h2 { margin: 0 0 10px; font-family: var(--f-titulo); font-size: 18px; }
.correo { border-radius: 10px; overflow: hidden; border: 1px solid var(--linea); }
.correo-barra { background: var(--musgo); color: var(--texto); font-size: 13px; padding: 8px 14px; }
.correo-cuerpo { background: #f6f7f4; color: #1d211e; overflow-x: auto; }
.pie { font-size: 12px; color: var(--tenue); margin: 0; }
code { font-size: 12px; }
"""


def html_pagina(res, cuerpo_correo):
    cfg = res["cfg"]
    responsables = cfg["responsables"]
    tramos = []
    for clave in ["io", "mql", "sql", "opp", "closed", "won"]:
        etiqueta = "Volumen IO" if clave == "io" else TASA[clave]["tramo"]
        tramos.append(f"{etiqueta}: {responsables[clave]['nombre']}")
    destinatarios = ", ".join(d["nombre"] for d in cfg.get("destinatarios", []))
    return f"""<title>Embudo semanal TPDV</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=Inter:wght@400;600&display=swap">
<style>{PAGINA_CSS}{CORREO_CSS}</style>
<main class="marco">
  <header>
    <p class="ceja">Tu Proyecto de Vida · agente programado</p>
    <h1>Análisis semanal del embudo</h1>
    <p class="bajada">Cada lunes antes de las 8:00 compara el embudo real de cada proyecto en HubSpot con el Escenario
    Pesimista de la planilla de marketing y envía por correo los puntos críticos, el plan de acción y su responsable.</p>
  </header>
  <dl class="ficha">
    <div class="dato"><dt>Último informe</dt><dd>{escape(f_larga(res['hoy']))}</dd></div>
    <div class="dato"><dt>Destinatarios</dt><dd>{escape(destinatarios)}</dd></div>
    <div class="dato"><dt>Responsables por tramo</dt><dd>{escape(' · '.join(tramos))}</dd></div>
    <div class="dato"><dt>Escalamiento y seguimiento</dt><dd>{escape(cfg['escalamiento']['nombre'])} (escala tras
      {cfg['umbrales']['semanas_escalar']} semanas en rojo) · {escape(cfg['seguimiento']['nombre'])} (seguimiento)</dd></div>
  </dl>
  <section>
    <h2>Último informe</h2>
    <div class="correo">
      <div class="correo-barra">{escape(asunto(res))}</div>
      <div class="correo-cuerpo"><div class="e">{cuerpo_correo}</div></div>
    </div>
  </section>
  <p class="pie">Archivos que usa la ejecución de cada lunes: <code>embudo.py.txt</code> (cálculo y correo),
  <code>config.json</code> (proyectos, responsables y umbrales) e <code>historial.json</code> (una foto por semana).</p>
</main>
"""


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def construir(hoy, ruta_plan, dir_entrada, ruta_historial, cfg, dir_salida):
    plan, adv_plan = leer_plan(ruta_plan)
    dir_entrada = Path(dir_entrada)
    conteos = {e: leer_conteos(dir_entrada / f"cohorte_{e}.tsv") for e in ORDEN}
    ritmo = leer_conteos(dir_entrada / "ritmo_io.tsv")
    if not conteos["io"]:
        raise ErrorEntrada("cohorte_io.tsv no trae conteos: revisa que guardaste el resultado de HubSpot.", 3)
    historial = {"version": 1, "semanas": []}
    if ruta_historial and Path(ruta_historial).exists():
        texto = Path(ruta_historial).read_text(encoding="utf-8").strip()
        if texto:
            historial = json.loads(texto)
    res = analizar(plan, conteos, ritmo, cfg, hoy, historial)
    res["advertencias"] = adv_plan + res["advertencias"]

    salida = Path(dir_salida)
    salida.mkdir(parents=True, exist_ok=True)
    correo, cuerpo = html_correo(res)
    (salida / "correo.html").write_text(correo, encoding="utf-8")
    (salida / "correo.txt").write_text(texto_correo(res), encoding="utf-8")
    (salida / "pagina.html").write_text(html_pagina(res, cuerpo), encoding="utf-8")
    nuevo = actualizar_historial(historial, res)
    (salida / "historial.json").write_text(json.dumps(nuevo, ensure_ascii=False, indent=1), encoding="utf-8")
    resumen = {
        "asunto": asunto(res), "semaforo": res["semaforo"], "rojos": res["rojos"],
        "puntos": [{"proyecto": x["proyecto"], "tramo": nombre_tramo(x["clave"]), "estado": x["estado"],
                    "efecto": round(x["impacto"], 2), "responsable": x["responsable"]["nombre"],
                    "semanas_rojo": x["semanas_rojo"]} for x in res["puntos"]],
        "focos": [{"tramo": nombre_tramo(f["clave"]), "efecto": round(f["impacto"], 2),
                   "responsable": f["responsable"]["nombre"], "proyectos": [x["proyecto"] for x in f["puntos"]]}
                  for f in res["focos"]],
        "pot_plan": round(res["total"]["pot_plan"], 2), "pot_real": round(res["total"]["pot_real"], 2),
        "advertencias": res["advertencias"],
    }
    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def main(argv=None):
    parser = argparse.ArgumentParser(description="Agente de embudo semanal (HubSpot vs. Escenario Pesimista)")
    sub = parser.add_subparsers(dest="comando", required=True)
    c = sub.add_parser("consultas", help="imprime las consultas SQL para HubSpot")
    c.add_argument("--hoy", required=True)
    c.add_argument("--config")
    b = sub.add_parser("build", help="arma el correo")
    b.add_argument("--hoy", required=True)
    b.add_argument("--plan", required=True)
    b.add_argument("--entrada", required=True)
    b.add_argument("--historial")
    b.add_argument("--config")
    b.add_argument("--salida", default="salida")
    args = parser.parse_args(argv)

    hoy = date.fromisoformat(args.hoy)
    cfg = cargar_config(args.config)
    if args.comando == "consultas":
        print(json.dumps(consultas(hoy, cfg), ensure_ascii=False, indent=1))
        return 0
    try:
        res = construir(hoy, args.plan, args.entrada, args.historial, cfg, args.salida)
    except ErrorEntrada as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return e.codigo
    print("ASUNTO=" + asunto(res))
    print("RESUMEN=" + linea_resumen(res))
    for a in res["advertencias"]:
        print("ADVERTENCIA: " + a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
