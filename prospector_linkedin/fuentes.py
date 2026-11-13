"""Agente de Fuentes: trae prospectos al embudo sin tocar LinkedIn de forma automática.

Fuentes admitidas:
- La exportación oficial de tus contactos de LinkedIn (Connections.csv).
- Cualquier CSV/Excel con prospectos (Sales Navigator, Apollo, una lista propia…).
- Los leads calientes y tibios del tablero del ebook.
- Texto copiado de una búsqueda de LinkedIn (requiere IA).
- Alta manual desde el tablero.
"""

import io
import re
from datetime import datetime, timezone
from urllib.parse import quote, unquote

import pandas as pd

from agentes_ebook.db import iso
from agentes_ebook.gmail import normalizar

AGENTE = "Fuentes"
BUSQUEDA_PERSONAS = "https://www.linkedin.com/search/results/people/?keywords="

ALIAS = {
    "nombre": ("nombre", "name", "full_name", "fullname", "nombre_completo", "contacto"),
    "nombre_pila": ("first_name", "firstname", "nombre_de_pila", "nombres"),
    "apellido": ("last_name", "lastname", "apellido", "apellidos"),
    "cargo": ("cargo", "title", "position", "headline", "puesto", "job_title", "titular", "occupation"),
    "empresa": ("empresa", "company", "company_name", "organizacion", "organization", "compania", "account_name"),
    "url": (
        "url", "profile_url", "profileurl", "linkedin", "linkedin_url", "linkedin_profile", "perfil",
        "person_linkedin_url", "public_profile_url", "linkedin_profile_url",
    ),
    "ubicacion": ("ubicacion", "location", "ciudad", "city", "pais", "country", "geography", "region"),
    "industria": ("industria", "industry", "rubro", "sector"),
    "email": ("email", "correo", "email_address", "e-mail", "mail"),
    "grado": ("grado", "degree", "connection_degree"),
    "conectado": ("connected_on", "conectado_el"),
}


def clave_de(url, nombre, empresa):
    """Identificador estable para no duplicar: el perfil de LinkedIn o, si falta, nombre + empresa."""
    m = re.search(r"linkedin\.com/(in|pub|sales/lead|sales/people)/([^/?#,\s]+)", url or "", re.I)
    if m:
        return f"{m.group(1).lower()}/{unquote(m.group(2)).lower()}"
    return f"{normalizar(nombre)}|{normalizar(empresa)}"


def link_busqueda(nombre, empresa=""):
    return BUSQUEDA_PERSONAS + quote(" ".join(x for x in (nombre, empresa) if x))


def link_perfil(p):
    """El perfil si lo tenemos; si no, una búsqueda de LinkedIn con su nombre y empresa."""
    return p.get("url") or link_busqueda(p.get("nombre", ""), p.get("empresa", ""))


def _limpio(valor):
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return ""
    texto = str(valor).strip()
    return "" if texto.lower() in ("nan", "none", "null") else texto


def _separar_titular(cargo, empresa):
    """«Gerente General en Acme» → («Gerente General», «Acme») cuando no viene la empresa aparte."""
    if empresa or not cargo:
        return cargo, empresa
    m = re.match(r"^(.{2,80}?)\s+(?:en|at|@)\s+(.{2,80})$", cargo.split("|")[0].strip(), re.I)
    return (m.group(1).strip(), m.group(2).strip()) if m else (cargo, empresa)


def normalizar_registro(fila, fuente, grado=2):
    """Convierte una fila con nombres de columna libres en un prospecto. None si no tiene nombre."""
    datos = {normalizar(str(k)).replace(" ", "_"): _limpio(v) for k, v in fila.items()}

    def campo(nombre):
        return next((datos[a] for a in ALIAS[nombre] if datos.get(a)), "")

    nombre = campo("nombre") or " ".join(x for x in (campo("nombre_pila"), campo("apellido")) if x)
    nombre = re.sub(r"\s+", " ", nombre).strip()
    if not nombre:
        return None
    cargo, empresa = _separar_titular(campo("cargo"), campo("empresa"))
    url = campo("url")
    if url and not url.lower().startswith("http"):
        url = "https://" + url.lstrip("/") if "linkedin.com" in url.lower() else ""
    grado_txt = campo("grado")
    conectado = None
    if campo("conectado"):
        for formato in ("%d %b %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                conectado = iso(datetime.strptime(campo("conectado"), formato).replace(tzinfo=timezone.utc))
                break
            except ValueError:
                continue
    return {
        "clave": clave_de(url, nombre, empresa),
        "nombre": nombre,
        "cargo": cargo,
        "empresa": empresa,
        "ubicacion": campo("ubicacion"),
        "industria": campo("industria"),
        "url": url,
        "email": campo("email").lower(),
        "fuente": fuente,
        "grado": 1 if grado == 1 or grado_txt.startswith("1") else 2,
        "conectado_en": conectado,
    }


def registrar(db, registros, origen):
    """Guarda prospectos sin duplicar. Devuelve (nuevos, ya_existentes)."""
    nuevos = existentes = 0
    for reg in registros:
        if not reg:
            continue
        _, es_nuevo = db.guardar_prospecto(reg)
        nuevos += es_nuevo
        existentes += not es_nuevo
    if nuevos:
        db.log(AGENTE, "Prospectos agregados", f"{nuevos} nuevos desde {origen} ({existentes} ya estaban)")
    return nuevos, existentes


def leer_archivo(nombre, contenido):
    """Lee un CSV/Excel. Devuelve (DataFrame, es_export_de_contactos_de_linkedin)."""
    if nombre.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(contenido)), False
    texto = contenido.decode("utf-8-sig", errors="replace")
    lineas = texto.splitlines()
    # La exportación de LinkedIn trae unas líneas de «Notes:» antes del encabezado real.
    inicio = next((i for i, l in enumerate(lineas[:15]) if l.replace('"', "").startswith("First Name,Last Name")), None)
    if inicio is not None:
        return pd.read_csv(io.StringIO("\n".join(lineas[inicio:])), dtype=str), True
    return pd.read_csv(io.StringIO(texto), dtype=str, sep=None, engine="python"), False


def desde_tabla(db, df, grado=2, fuente="csv"):
    registros = [normalizar_registro(fila, fuente, grado) for fila in df.to_dict("records")]
    return registrar(db, registros, fuente)


def desde_archivo(db, nombre, contenido, grado=2):
    df, es_contactos = leer_archivo(nombre, contenido)
    if es_contactos:
        return desde_tabla(db, df, grado=1, fuente="contactos")
    return desde_tabla(db, df, grado=grado, fuente="csv")


def desde_leads_ebook(db, segmentos=("Caliente", "Tibio")):
    """Suma al embudo de LinkedIn a quienes descargaron tu ebook (ya conocen tu contenido)."""
    registros = []
    for lead in db.leads():
        if lead.get("segmento") not in segmentos or lead.get("etapa") in ("Cliente", "Descartado"):
            continue
        if lead.get("fuente") == "demo" or len((lead.get("nombre") or "").split()) < 2:
            continue
        registros.append(
            {
                "clave": clave_de("", lead["nombre"], lead.get("empresa")),
                "nombre": lead["nombre"],
                "cargo": lead.get("cargo", ""),
                "empresa": lead.get("empresa", ""),
                "ubicacion": lead.get("pais", "") if lead.get("pais") != "Sin identificar" else "",
                "email": lead.get("email", ""),
                "fuente": "ebook",
                "grado": 2,
                "lead_id": lead["id"],
            }
        )
    return registrar(db, registros, "ebook")


def agregar_manual(db, nombre, cargo="", empresa="", url="", ubicacion="", grado=2):
    reg = normalizar_registro(
        {"nombre": nombre, "cargo": cargo, "empresa": empresa, "url": url, "ubicacion": ubicacion}, "manual", grado
    )
    return registrar(db, [reg], "alta manual")


def desde_texto(db, texto, asistente):
    """Extrae prospectos de texto copiado de una búsqueda de LinkedIn (usa IA)."""
    if not asistente:
        raise RuntimeError("Esta opción necesita ANTHROPIC_API_KEY (IA).")
    filas = asistente.extraer_prospectos(texto)
    registros = [normalizar_registro(f, "busqueda", 1 if str(f.get("grado", "")).startswith("1") else 2) for f in filas]
    return registrar(db, registros, "búsqueda pegada")


def busquedas_sugeridas(cfgp, maximo=8):
    """Búsquedas de LinkedIn listas para abrir: tus cargos objetivo en tu mercado, contactos de 2.º grado."""
    mercado = (cfgp.base.mercado_objetivo or [""])[0]
    vistos, resultado = set(), []
    for cargo in cfgp.cargos_objetivo:
        # «fundador» y «fundadora» (o «socio» y «socia») son la misma búsqueda: se deja una.
        base = " ".join(re.sub(r"[oa]$", "", palabra) for palabra in normalizar(cargo).replace("-", "").split())
        if base in vistos or len(resultado) >= maximo:
            continue
        vistos.add(base)
        consulta = f'"{cargo}" {mercado}'.strip()
        resultado.append((f"{cargo} · {mercado}" if mercado else cargo, BUSQUEDA_PERSONAS + quote(consulta) + "&network=%5B%22S%22%5D"))
    return resultado


# --------------------------------------------------------------------------
# Datos de demostración (ficticios)
# --------------------------------------------------------------------------
DEMO = [
    ("Camila Rojas Demo", "Gerente General", "Constructora Andes Demo", "Santiago, Chile", "Construcción", 2),
    ("Matías Fuentes Demo", "Fundador y CEO", "Logística Sur Demo", "Concepción, Chile", "Logística", 2),
    ("Valentina Soto Demo", "Cofundadora", "PagoFácil Demo", "Santiago, Chile", "Fintech", 2),
    ("Ignacio Vera Demo", "Socio", "Agrícola Los Robles Demo", "Rancagua, Chile", "Agro", 1),
    ("Fernanda Muñoz Demo", "Gerente de Administración y Finanzas", "Clínica Norte Demo", "Antofagasta, Chile", "Salud", 2),
    ("Tomás Herrera Demo", "Country Manager", "SoftCloud Demo", "Santiago, Chile", "Software", 2),
    ("Daniela Paz Demo", "Dueña", "Tiendas Paz Demo", "Viña del Mar, Chile", "Retail", 1),
    ("Rodrigo Leiva Demo", "Director Ejecutivo", "Energía Limpia Demo", "Santiago, Chile", "Energía", 2),
    ("Paula Araya Demo", "Jefa de Contabilidad", "Transportes Araya Demo", "Temuco, Chile", "Transporte", 2),
    ("Sebastián Ortiz Demo", "Gerente Comercial", "Alimentos del Valle Demo", "Talca, Chile", "Alimentos", 2),
    ("Javiera Campos Demo", "Estudiante de Ingeniería Comercial", "", "Santiago, Chile", "", 2),
    ("Andrés Molina Demo", "CFO fraccional", "Independiente", "Santiago, Chile", "", 2),
    ("Lucía Bravo Demo", "CEO", "EduTech Demo", "Lima, Perú", "Educación", 2),
    ("Felipe Navarro Demo", "Analista de marketing", "Agencia Navarro Demo", "Santiago, Chile", "", 2),
]


def cargar_demo(db):
    registros = []
    for nombre, cargo, empresa, ubicacion, industria, grado in DEMO:
        url = "https://www.linkedin.com/in/" + normalizar(nombre).replace(" ", "-")
        registros.append(
            {
                "clave": clave_de(url, nombre, empresa), "nombre": nombre, "cargo": cargo, "empresa": empresa,
                "ubicacion": ubicacion, "industria": industria, "url": url, "fuente": "demo", "grado": grado,
            }
        )
    return registrar(db, registros, "demo")
