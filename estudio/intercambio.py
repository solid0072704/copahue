"""Entrada y salida de datos: importar clientes (CSV/Excel), exportar a Excel y traer clientes del ebook."""

import io
import os
import sqlite3
import unicodedata

import pandas as pd

from . import calendario
from . import rut as rutlib
from .config import a_numero
from .db import ESTADOS_CLIENTE, MONEDAS, ErrorValidacion, estado_cobro

# Nombres de columna aceptados al importar → campo del cliente.
ALIAS = {
    "rut": "rut",
    "razon social": "razon_social", "razon_social": "razon_social", "nombre": "razon_social", "cliente": "razon_social",
    "empresa": "razon_social",
    "nombre fantasia": "nombre_fantasia", "nombre_fantasia": "nombre_fantasia", "fantasia": "nombre_fantasia",
    "tipo": "tipo",
    "giro": "giro", "actividad": "giro",
    "regimen": "regimen", "regimen tributario": "regimen",
    "direccion": "direccion", "comuna": "comuna",
    "representante": "representante", "representante legal": "representante",
    "contacto": "contacto",
    "email": "email", "correo": "email", "mail": "email", "e-mail": "email",
    "telefono": "telefono", "celular": "telefono", "fono": "telefono",
    "responsable": "responsable", "contador": "responsable", "encargado": "responsable",
    "estado": "estado",
    "fecha inicio": "fecha_inicio", "fecha_inicio": "fecha_inicio", "inicio": "fecha_inicio",
    "servicios": "servicios",
    "honorario": "honorario", "honorarios": "honorario", "honorario mensual": "honorario",
    "moneda": "moneda",
    "trabajadores": "trabajadores", "n trabajadores": "trabajadores",
    "notas": "notas", "observaciones": "notas",
}


def _clave(texto):
    sin_tildes = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return " ".join(sin_tildes.lower().replace("°", "").replace(".", "").split())


def _texto(valor):
    if valor is None or (isinstance(valor, float) and valor != valor) or valor is pd.NaT:
        return ""
    return str(valor).strip()


def _servicios(texto):
    """'IVA; Remuneraciones' → servicios del catálogo (coincidencia flexible, sin tildes ni mayúsculas)."""
    encontrados = []
    for parte in _texto(texto).replace("|", ";").replace(",", ";").split(";"):
        buscado = _clave(parte)
        if not buscado:
            continue
        for servicio in calendario.SERVICIOS:
            if buscado in _clave(servicio) and servicio not in encontrados:
                encontrados.append(servicio)
                break
    return encontrados


def importar_clientes(db, df, fuente="importado"):
    """Crea clientes desde una planilla. Los que ya existen se omiten (no se sobrescribe nada).

    Devuelve (creados, existentes, errores) donde errores es una lista de textos «Fila N: motivo».
    """
    columnas = {c: ALIAS[_clave(c)] for c in df.columns if _clave(c) in ALIAS}
    if "razon_social" not in columnas.values():
        raise ErrorValidacion("La planilla necesita al menos una columna «razon_social» (o «nombre») y, ojalá, «rut».")
    creados, existentes, errores = 0, 0, []
    nombres = {c["razon_social"].casefold() for c in db.clientes()}
    for n, (_, fila) in enumerate(df.iterrows(), start=2):  # la fila 1 es el encabezado
        datos = {}
        for columna, campo in columnas.items():
            datos.setdefault(campo, _texto(fila[columna]))
        if not datos.get("razon_social") and not datos.get("rut"):
            continue  # fila vacía
        try:
            # Con RUT se reconoce por RUT; sin RUT (prospectos), por el nombre exacto.
            if db.cliente_por_rut(datos["rut"]) if datos.get("rut") else datos["razon_social"].casefold() in nombres:
                existentes += 1
                continue
            if "servicios" in datos:
                datos["servicios"] = _servicios(datos["servicios"])
            if "honorario" in datos:
                datos["honorario"] = a_numero(datos["honorario"])
            if "trabajadores" in datos:
                datos["trabajadores"] = int(a_numero(datos["trabajadores"]))
            moneda = datos.get("moneda", "").upper()
            datos["moneda"] = moneda if moneda in MONEDAS else "CLP"
            estado = datos.get("estado", "").capitalize()
            datos["estado"] = estado if estado in ESTADOS_CLIENTE else ("Activo" if datos.get("rut") else "Prospecto")
            if datos.get("fecha_inicio"):
                fecha = pd.to_datetime(datos["fecha_inicio"], errors="coerce", dayfirst="/" in datos["fecha_inicio"])
                datos["fecha_inicio"] = None if pd.isna(fecha) else fecha.date()
            db.crear_cliente(fuente=fuente, **datos)
            nombres.add(datos["razon_social"].casefold())
            creados += 1
        except (ErrorValidacion, ValueError) as e:
            errores.append(f"Fila {n}: {e}")
    return creados, existentes, errores


def plantilla_importacion():
    """Planilla de ejemplo para importar clientes (CSV)."""
    df = pd.DataFrame(
        [
            {
                "rut": "55.555.001-4", "razon_social": "Ejemplo SpA", "giro": "Comercio", "regimen": "Pro Pyme General (14 D N°3)",
                "contacto": "Nombre Apellido", "email": "contacto@ejemplo.example", "telefono": "+56 9 5550 0000",
                "responsable": "", "servicios": "Contabilidad mensual; IVA mensual; Remuneraciones; Renta anual",
                "honorario": "150000", "moneda": "CLP", "trabajadores": 3, "fecha_inicio": "2026-01-01", "notas": "",
            }
        ]
    )
    return df.to_csv(index=False).encode("utf-8-sig")


def exportar_excel(db, hoy, dias_alerta=7):
    """Libro Excel con clientes, vencimientos, honorarios y pagos."""
    clientes = pd.DataFrame(
        [
            {
                "RUT": rutlib.formatear(c["rut"]) if c["rut"] else "", "Razón social": c["razon_social"],
                "Nombre de fantasía": c["nombre_fantasia"], "Tipo": c["tipo"], "Giro": c["giro"], "Régimen": c["regimen"],
                "Estado": c["estado"], "Responsable": c["responsable"], "Representante legal": c["representante"],
                "Contacto": c["contacto"], "Correo": c["email"], "Teléfono": c["telefono"], "Dirección": c["direccion"],
                "Comuna": c["comuna"], "Servicios": "; ".join(c["servicios"]), "Honorario": c["honorario"],
                "Moneda": c["moneda"], "Trabajadores": c["trabajadores"], "Cliente desde": c["fecha_inicio"] or "",
                "Notas": c["notas"],
            }
            for c in db.clientes()
        ]
    )
    vencimientos = pd.DataFrame(
        [
            {
                "Vence": o["vence"], "Situación": calendario.semaforo(o, hoy, dias_alerta), "Cliente": o["razon_social"],
                "RUT": rutlib.formatear(o["rut"]) if o["rut"] else "", "Obligación": o["nombre"], "Período": o["periodo"],
                "Estado": o["estado"], "Responsable": o["responsable"], "Notas": o["notas"],
            }
            for o in db.obligaciones()
        ]
    )
    cobros = pd.DataFrame(
        [
            {
                "Cliente": c["razon_social"], "RUT": rutlib.formatear(c["rut"]) if c["rut"] else "", "Concepto": c["concepto"],
                "Período": c["periodo"], "Emitido": c["emitido"], "Vence": c["vence"], "Monto": c["monto"],
                "Pagado": c["pagado"], "Saldo": 0 if c["anulado"] else c["saldo"], "Estado": estado_cobro(c),
                "Días de mora": max(0, (hoy - pd.Timestamp(c["vence"]).date()).days) if estado_cobro(c) == "Pendiente" else 0,
                "Documento": c["documento"], "Notas": c["notas"],
            }
            for c in db.cobros(incluir_anulados=True)
        ]
    )
    pagos = pd.DataFrame(
        [
            {
                "Fecha": p["fecha"], "Cliente": p["razon_social"], "Concepto": p["concepto"], "Período": p["periodo"],
                "Monto": p["monto"], "Medio": p["medio"], "Nota": p["nota"],
            }
            for p in db.pagos()
        ]
    )
    salida = io.BytesIO()
    with pd.ExcelWriter(salida, engine="xlsxwriter") as writer:
        for nombre, df in (("Clientes", clientes), ("Vencimientos", vencimientos), ("Honorarios", cobros), ("Pagos", pagos)):
            df.to_excel(writer, sheet_name=nombre, index=False)
            hoja = writer.sheets[nombre]
            hoja.freeze_panes(1, 0)
            for i, columna in enumerate(df.columns):
                ancho = max([len(str(columna))] + [len(str(v)) for v in df[columna].head(200)]) + 2
                hoja.set_column(i, i, min(ancho, 50))
    return salida.getvalue()


def importar_leads_ebook(db, ruta_db_ebook):
    """Trae como prospectos los leads que el tablero del ebook marcó en la etapa «Cliente».

    Lee la base del tablero en modo solo lectura. Devuelve cuántos prospectos nuevos se crearon.
    """
    if not ruta_db_ebook or not os.path.exists(ruta_db_ebook):
        return 0
    con = sqlite3.connect(f"file:{os.path.abspath(ruta_db_ebook)}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        leads = [dict(f) for f in con.execute("SELECT * FROM leads WHERE etapa='Cliente' AND fuente<>'demo'")]
    except sqlite3.Error:
        return 0
    finally:
        con.close()
    conocidos = {c["email"] for c in db.clientes() if c["email"]}
    creados = 0
    for lead in leads:
        email = (lead.get("email") or "").strip().lower()
        if not email or email in conocidos:
            continue
        empresa = (lead.get("empresa") or "").strip()
        db.crear_cliente(
            razon_social=empresa if len(empresa) > 1 else (lead.get("nombre") or email),
            contacto=lead.get("nombre") or "",
            email=email,
            telefono=lead.get("telefono") or "",
            estado="Prospecto",
            fuente="ebook",
            notas=f"Llegó por el ebook. Cargo: {lead.get('cargo') or '—'}. Motivo: {lead.get('motivo') or '—'}.",
        )
        conocidos.add(email)
        creados += 1
    return creados
