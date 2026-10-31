"""Fuentes del flujo de caja: cuentas por cobrar (CSV/Excel de cualquier sistema, o la API de Forpay), egresos (CSV/Excel)
y ventas comprometidas de HubSpot.

Los archivos de cobros pueden venir de una planilla, de la contabilidad o del banco, así que cada campo se detecta por una
lista de nombres candidatos (y se puede fijar con COBROS_CAMPOS). Toda cuota o factura termina normalizada en:
    empresa, compromiso, cliente, proyecto, numero, monto, monto_pagado, vence, pagado_en,
    estado (moroso | por_vencer | pagada | en_cobro | repactada | anulada), estado_original, dias_mora
"""

import csv
import logging
import os
import re
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone

import requests

log = logging.getLogger(__name__)

CANDIDATOS_CUOTA = {
    "monto": ("monto_cuota", "valor_cuota", "monto", "valor", "importe", "total_cuota"),
    "monto_pagado": ("monto_pagado", "monto_abonado", "pagado", "abono"),
    "vencimiento": ("fecha_vencimiento", "fecha_de_vencimiento", "vencimiento", "fecha_venc", "fecha_vcto", "vence"),
    "pago": ("fecha_pago", "fecha_de_pago", "fecha_pagada", "fecha_abono", "pagado_en"),
    "estado": ("estado_cuota", "estado"),
    "numero": ("numero_cuota", "nro_cuota", "num_cuota", "n_cuota", "numero", "cuota"),
    "dias_mora": ("dias_mora", "dias_de_mora", "dias_atraso", "mora"),
}
CANDIDATOS_COMPROMISO = {
    "compromiso": ("id_compromiso", "compromiso_id", "compromiso", "folio", "codigo", "id"),
    "cliente": ("nombre_cliente", "razon_social", "cliente", "deudor", "rut_cliente", "nombre", "rut"),
    "empresa": ("empresa", "nombre_empresa", "acreedor"),
    "proyecto": ("proyecto", "producto", "parcela", "lote", "glosa", "descripcion"),
    "total": ("monto_total", "valor_total", "monto_compromiso", "total"),
}
ESTADOS = ("moroso", "por_vencer", "pagada", "en_cobro", "repactada", "anulada")
FORMATOS_FECHA = (
    "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d",
    "%d-%m-%Y", "%d/%m/%Y", "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d-%m-%y", "%d/%m/%y",
)


# --------------------------------------------------------------------------- utilidades de parseo
def normalizar_clave(texto):
    sin_tildes = unicodedata.normalize("NFKD", str(texto or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", sin_tildes.lower()).strip("_")


def parsear_monto(valor):
    """'$ 1.250.000' -> 1250000.0 · '1.250,50' -> 1250.5 · '1250.50' -> 1250.5 · None/'-' -> 0.0"""
    if valor is None:
        return 0.0
    if isinstance(valor, (int, float)):
        return float(valor)
    s = re.sub(r"[^\d,.\-]", "", str(valor))
    if not s or s in ("-", ".", ","):
        return 0.0
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        entero, _, decimal = s.rpartition(",")
        s = f"{entero}.{decimal}" if len(decimal) in (1, 2) and "," not in entero else s.replace(",", "")
    elif s.count(".") > 1 or (s.count(".") == 1 and len(s.rsplit(".", 1)[1]) == 3):
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def parsear_fecha(valor):
    """Devuelve una `date` o None. Acepta texto en varios formatos, epoch (s o ms), date y datetime."""
    if valor in (None, "", "-"):
        return None
    if isinstance(valor, datetime):
        return valor.date()
    if isinstance(valor, date):
        return valor
    if isinstance(valor, (int, float)):
        try:
            segundos = valor / 1000 if valor > 1e11 else valor
            return datetime.fromtimestamp(segundos, tz=timezone.utc).date()
        except (OSError, OverflowError, ValueError):
            return None
    texto = str(valor).strip()
    if texto.lstrip("-").isdigit():
        return parsear_fecha(int(texto))
    for formato in FORMATOS_FECHA:
        try:
            return datetime.strptime(texto[: len(datetime.now().strftime(formato))], formato).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(texto.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _buscar_columna(claves, candidatos, usadas, fijada=None):
    """Elige la columna real para un campo: la fijada por configuración, un candidato exacto o uno contenido."""
    if fijada:
        for original in claves.values():
            if normalizar_clave(original) == normalizar_clave(fijada):
                return original
    for c in candidatos:
        if c in claves and claves[c] not in usadas:
            return claves[c]
    for c in candidatos:
        if len(c) >= 5:
            for norm, original in claves.items():
                if c in norm and original not in usadas:
                    return original
    return None


def detectar_campos(claves_cuota, claves_compromiso=(), fijados=None):
    """Mapa {campo interno: columna real} para una cuota (y su compromiso)."""
    fijados = fijados or {}
    mapa, usadas = {}, set()
    normalizadas = {normalizar_clave(k): k for k in claves_cuota}
    for campo, candidatos in CANDIDATOS_CUOTA.items():
        col = _buscar_columna(normalizadas, candidatos, usadas, fijados.get(campo))
        if col:
            mapa[campo], _ = col, usadas.add(col)
    normalizadas_comp = {normalizar_clave(k): k for k in (claves_compromiso or claves_cuota)}
    usadas_comp = set(usadas) if not claves_compromiso else set()
    for campo, candidatos in CANDIDATOS_COMPROMISO.items():
        col = _buscar_columna(normalizadas_comp, candidatos, usadas_comp, fijados.get(campo))
        if col:
            mapa[campo], _ = col, usadas_comp.add(col)
    return mapa


def clasificar_estado(texto, vence, pagado_en, hoy):
    """Lleva el estado de Forpay (o lo que haya) a uno de los estados internos."""
    t = normalizar_clave(texto)
    if "anulad" in t or "revers" in t:
        return "anulada"
    if "repact" in t:
        return "repactada"
    if "proceso" in t or "cobro" in t or "judicial" in t:
        return "en_cobro"
    if "pagad" in t or "finaliz" in t or "cancelad" in t:
        return "pagada"
    if "no_moros" in t or "no_morosa" in t or t.endswith("_hoy"):
        return "por_vencer"
    if "moros" in t or "vencid" in t or "atras" in t:
        return "moroso"
    if pagado_en:
        return "pagada"
    if vence and vence < hoy:
        return "moroso"
    return "por_vencer"


def normalizar_cuota(cuota, compromiso, mapa, hoy, empresa="", estado_consultado=""):
    def valor(campo, origen):
        col = mapa.get(campo)
        return origen.get(col) if col else None

    vence = parsear_fecha(valor("vencimiento", cuota))
    pagado_en = parsear_fecha(valor("pago", cuota))
    estado_txt = valor("estado", cuota) or estado_consultado
    estado = clasificar_estado(estado_txt, vence, pagado_en, hoy)
    dias_mora = 0
    if estado in ("moroso", "en_cobro") and vence:
        dias_mora = max(0, (hoy - vence).days)
    elif valor("dias_mora", cuota) not in (None, "", "-"):
        try:
            dias_mora = int(float(valor("dias_mora", cuota)))
        except (TypeError, ValueError):
            dias_mora = 0
    numero = valor("numero", cuota)
    try:
        numero = int(float(numero)) if numero not in (None, "", "-") else None
    except (TypeError, ValueError):
        numero = str(numero)
    return {
        "empresa": str(valor("empresa", compromiso) or empresa or "").strip() or empresa,
        "compromiso": str(valor("compromiso", compromiso) or "").strip(),
        "cliente": str(valor("cliente", compromiso) or "").strip(),
        "proyecto": str(valor("proyecto", compromiso) or "").strip(),
        "numero": numero,
        "monto": parsear_monto(valor("monto", cuota)),
        "monto_pagado": parsear_monto(valor("monto_pagado", cuota)),
        "vence": vence,
        "pagado_en": pagado_en,
        "estado": estado,
        "estado_original": str(estado_txt or ""),
        "dias_mora": dias_mora,
    }


def _clave_cuota(c):
    return (c["empresa"], c["compromiso"], c["numero"], c["vence"])


def deduplicar(cuotas):
    """Si una cuota aparece en dos consultas (p. ej. 'Pendiente' y 'Repactada'), gana la más específica."""
    prioridad = {"anulada": 5, "pagada": 4, "en_cobro": 3, "repactada": 3, "moroso": 2, "por_vencer": 1}
    mejores = {}
    for c in cuotas:
        k = _clave_cuota(c)
        if k not in mejores or prioridad[c["estado"]] > prioridad[mejores[k]["estado"]]:
            mejores[k] = c
    return list(mejores.values())


# --------------------------------------------------------------------------- Forpay
def aplanar_forpay(datos, hoy, empresa="", estado_consultado="", fijados=None):
    """Convierte la respuesta de la API (o un JSON equivalente) en cuotas normalizadas."""
    registros = datos.get("data") if isinstance(datos, dict) else datos
    if isinstance(registros, dict):
        registros = [registros]
    if not isinstance(registros, list):
        return []
    cuotas = []
    for reg in registros:
        if not isinstance(reg, dict):
            continue
        interno = reg.get("data") if isinstance(reg.get("data"), dict) else {}
        compromiso = {k: v for k, v in reg.items() if k not in ("data", "cuotas")}
        compromiso.update({k: v for k, v in interno.items() if k != "cuotas"})
        lista = reg.get("cuotas") or interno.get("cuotas") or []
        if not isinstance(lista, list) or not lista:
            lista = [compromiso]  # el registro ya es una cuota plana
        mapa = None
        for cuota in lista:
            if not isinstance(cuota, dict):
                continue
            if mapa is None:
                mapa = detectar_campos(cuota.keys(), compromiso.keys(), fijados)
            cuotas.append(normalizar_cuota(cuota, compromiso, mapa, hoy, empresa, estado_consultado))
    return cuotas


class ClienteForpay:
    """Lee las cuotas de cada sociedad desde la API de Forpay."""

    def __init__(self, cfg, sesion=None):
        self.cfg = cfg
        self.sesion = sesion or requests.Session()

    def cuotas(self, hoy):
        """Devuelve (cuotas, errores). Un error en una sociedad no detiene a las demás."""
        resultado, errores = [], []
        for empresa, token in self.cfg.forpay_empresas.items():
            for estado in self.cfg.forpay_estados:
                try:
                    r = self.sesion.get(
                        self.cfg.forpay_url,
                        params={"token": token, "estado_cuota": estado, "estado_compromiso": "Activa"},
                        timeout=60,
                    )
                    r.raise_for_status()
                    datos = r.json()
                except requests.RequestException as e:
                    errores.append(f"Forpay {empresa} / {estado}: {e}")
                    continue
                except ValueError:
                    errores.append(f"Forpay {empresa} / {estado}: la respuesta no es JSON")
                    continue
                resultado.extend(aplanar_forpay(datos, hoy, empresa, estado, self.cfg.cobros_campos))
                time.sleep(0.2)
        return deduplicar(resultado), errores


def leer_archivo_cuotas(ruta, hoy, empresa="", fijados=None):
    """Lee un CSV/Excel de cuentas por cobrar: una fila por cuota o factura (sirve el exportado de Forpay o una planilla propia).

    Si el archivo no trae columna de sociedad, se usa `empresa` o el nombre del archivo."""
    if ruta.lower().endswith((".xlsx", ".xls")):
        import pandas as pd

        filas = pd.read_excel(ruta).fillna("").to_dict("records")
    else:
        with open(ruta, encoding="utf-8-sig", newline="") as f:
            filas = list(csv.DictReader(f))
    if not filas:
        return []
    nombre = empresa or re.sub(r"(?i)^(morosos forpay|cobros|cuentas por cobrar)\s*", "", os.path.splitext(os.path.basename(ruta))[0]).strip()
    mapa = detectar_campos(filas[0].keys(), fijados=fijados)
    return [normalizar_cuota(f, f, mapa, hoy, nombre) for f in filas]


# --------------------------------------------------------------------------- plantillas y egresos
PLANTILLA_COBROS = [
    ("empresa", "cliente", "compromiso", "proyecto", "numero_cuota", "monto", "fecha_vencimiento", "estado", "fecha_pago"),
    ("Sociedad Ejemplo SpA", "Cliente Uno", "CV-0001", "Parcela 12", "7", "850000", "2026-10-05", "Pendiente", ""),
    ("Sociedad Ejemplo SpA", "Cliente Uno", "CV-0001", "Parcela 12", "8", "850000", "2026-11-05", "Pendiente", ""),
    ("Sociedad Ejemplo SpA", "Cliente Dos", "CV-0002", "Parcela 31", "3", "1200000", "2026-09-20", "Pagada", "2026-09-22"),
    ("Sociedad Ejemplo SpA", "Cliente Dos", "CV-0002", "Parcela 31", "4", "1200000", "2026-10-20", "Pendiente", ""),
    ("Otra Sociedad SpA", "Cliente Tres", "F-1045", "", "1", "4500000", "2026-11-15", "Pendiente", ""),
]


def escribir_plantilla_cobros(ruta):
    os.makedirs(os.path.dirname(os.path.abspath(ruta)) or ".", exist_ok=True)
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(PLANTILLA_COBROS)
    return ruta


PLANTILLA_EGRESOS = [
    ("fecha", "concepto", "categoria", "empresa", "monto", "recurrencia", "estado", "hasta"),
    ("2026-11-05", "Remuneraciones", "Personal", "", "18500000", "mensual", "programado", ""),
    ("2026-11-12", "PPM e IVA", "Impuestos", "", "6200000", "mensual", "programado", ""),
    ("2026-11-15", "Cuota crédito bancario", "Financiamiento", "Boulevard Macul SpA", "9800000", "mensual", "programado", "2027-06-15"),
    ("2026-11-20", "Movimiento de tierras lote 4", "Obras", "LOS BARBECHOS SpA", "24000000", "unica", "programado", ""),
    ("2026-11-28", "Honorarios notaría y CBR", "Legal", "", "1500000", "mensual", "estimado", ""),
]
RECURRENCIAS = {"unica": None, "semanal": 7, "quincenal": 14, "mensual": "mes"}


def escribir_plantilla_egresos(ruta):
    os.makedirs(os.path.dirname(os.path.abspath(ruta)) or ".", exist_ok=True)
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(PLANTILLA_EGRESOS)
    return ruta


def leer_egresos(ruta):
    """Lee el CSV/Excel de egresos. Devuelve (egresos, errores)."""
    if not ruta or not os.path.exists(ruta):
        return [], [f"No existe el archivo de egresos '{ruta}' (créalo con --plantilla-egresos)."] if ruta else []
    if ruta.lower().endswith((".xlsx", ".xls")):
        import pandas as pd

        filas = pd.read_excel(ruta).fillna("").to_dict("records")
    else:
        with open(ruta, encoding="utf-8-sig", newline="") as f:
            filas = list(csv.DictReader(f))
    egresos, errores = [], []
    for i, fila in enumerate(filas, 2):
        f = {normalizar_clave(k): v for k, v in fila.items()}
        fecha = parsear_fecha(f.get("fecha"))
        monto = parsear_monto(f.get("monto"))
        if not fecha or monto <= 0:
            errores.append(f"Egresos fila {i}: fecha o monto inválidos")
            continue
        recurrencia = normalizar_clave(f.get("recurrencia") or "unica") or "unica"
        if recurrencia not in RECURRENCIAS:
            errores.append(f"Egresos fila {i}: recurrencia '{recurrencia}' no reconocida (unica, semanal, quincenal, mensual)")
            recurrencia = "unica"
        egresos.append(
            {
                "fecha": fecha,
                "concepto": str(f.get("concepto") or "Egreso").strip(),
                "categoria": str(f.get("categoria") or "Otros").strip(),
                "empresa": str(f.get("empresa") or "").strip(),
                "monto": monto,
                "recurrencia": recurrencia,
                "estado": normalizar_clave(f.get("estado") or "programado") or "programado",
                "hasta": parsear_fecha(f.get("hasta")),
            }
        )
    return egresos, errores


def _sumar_mes(fecha, meses):
    mes = fecha.month - 1 + meses
    anio = fecha.year + mes // 12
    mes = mes % 12 + 1
    ultimo = (date(anio + (mes // 12), mes % 12 + 1, 1) - timedelta(days=1)).day
    return date(anio, mes, min(fecha.day, ultimo))


def expandir_egresos(egresos, desde, hasta):
    """Ocurrencias de cada egreso dentro de [desde, hasta], expandiendo las recurrencias. Omite los ya pagados."""
    eventos = []
    for e in egresos:
        if e["estado"] == "pagado":
            continue
        fin = min(hasta, e["hasta"]) if e.get("hasta") else hasta
        paso = RECURRENCIAS[e["recurrencia"]]
        fecha, i = e["fecha"], 0
        while fecha <= fin and i < 400:
            if fecha >= desde:
                eventos.append({"fecha": fecha, "monto": e["monto"], "concepto": e["concepto"], "categoria": e["categoria"], "empresa": e["empresa"]})
            if paso is None:
                break
            i += 1
            fecha = _sumar_mes(e["fecha"], i) if paso == "mes" else e["fecha"] + timedelta(days=paso * i)
    return sorted(eventos, key=lambda x: x["fecha"])


# --------------------------------------------------------------------------- ventas comprometidas (HubSpot)
def ventas_comprometidas(cfg_caja, hoy, cliente_hubspot=None):
    """Pie esperado de los negocios avanzados de HubSpot, ponderado por la probabilidad de su etapa.

    Devuelve (eventos, errores). Cada evento: {fecha, monto, concepto}. Requiere HUBSPOT_TOKEN.
    """
    try:
        from agentes_hubspot.api import ClienteHubSpot
        from agentes_hubspot.config import ConfigHubSpot
    except ImportError as e:  # pragma: no cover
        return [], [f"HubSpot no disponible: {e}"]
    try:
        cliente = cliente_hubspot
        if cliente is None:
            cfg_h = ConfigHubSpot.desde_entorno()
            if not cfg_h.hubspot_configurado:
                return [], ["HubSpot: falta HUBSPOT_TOKEN para sumar las ventas comprometidas."]
            cliente = ClienteHubSpot(cfg_h)
        etapas, pipeline_id, _ = cliente.etapas()
        negocios = cliente.negocios_abiertos(pipeline_id, etapas)
    except Exception as e:
        return [], [f"HubSpot: {e}"]
    prob = {e["id"]: e["probabilidad"] for e in etapas}
    eventos = []
    for n in negocios:
        p = prob.get(n["etapa_id"], 0)
        if p < cfg_caja.pipeline_prob_minima or p >= 1 or not n.get("monto"):
            continue
        fecha = n["cerrado"].date() if n.get("cerrado") else hoy + timedelta(days=7)
        if fecha < hoy:
            fecha = hoy + timedelta(days=7)
        eventos.append(
            {"fecha": fecha, "monto": n["monto"] * cfg_caja.pie_porcentaje / 100 * p, "concepto": f"Pie {n['nombre']} ({int(p * 100)}%)"}
        )
    return eventos, []


# --------------------------------------------------------------------------- fuentes reales combinadas
class FuentesReales:
    """Reúne los archivos de cuentas por cobrar (y la API de Forpay si está configurada), el CSV de egresos y, si se pide, HubSpot."""

    def __init__(self, cfg):
        self.cfg = cfg

    def cuotas(self, hoy):
        cuotas, errores = [], []
        if self.cfg.forpay_empresas:
            c, e = ClienteForpay(self.cfg).cuotas(hoy)
            cuotas.extend(c)
            errores.extend(e)
        for ruta in self.cfg.cobros_archivos:
            try:
                cuotas.extend(leer_archivo_cuotas(ruta, hoy, fijados=self.cfg.cobros_campos))
            except Exception as e:
                errores.append(f"Archivo {ruta}: {e}")
        return deduplicar(cuotas), errores

    def egresos(self):
        return leer_egresos(self.cfg.egresos_archivo)

    def ventas(self, hoy):
        if not self.cfg.incluir_pipeline:
            return [], []
        return ventas_comprometidas(self.cfg, hoy)
