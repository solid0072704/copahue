"""Honorarios del estudio: emisión mensual, pagos, morosidad y mensajes de cobranza."""

import calendar
import re
from datetime import date
from urllib.parse import quote

from . import calendario
from .config import a_numero

META_UF = "valor_uf"
CONCEPTO_HONORARIO = "Honorario mensual"
TRAMOS_MORA = ["Por vencer", "1 a 30 días", "31 a 60 días", "61 a 90 días", "Más de 90 días"]


def formato_clp(monto):
    return "$" + f"{int(round(monto or 0)):,}".replace(",", ".")


def formato_corto(monto):
    """Monto abreviado para indicadores: $1,7 M · $850 mil · $900."""
    monto = int(round(monto or 0))
    if abs(monto) >= 1_000_000:
        return f"${monto / 1e6:.1f} M".replace(".", ",")
    if abs(monto) >= 10_000:
        return f"${monto / 1e3:.0f} mil"
    return formato_clp(monto)


def formato_honorario(cliente):
    if not cliente.get("honorario"):
        return "—"
    if cliente.get("moneda") == "UF":
        return f"{cliente['honorario']:g} UF".replace(".", ",")
    return formato_clp(cliente["honorario"])


def valor_uf(db, cfg):
    """Valor de la UF que se usa para convertir honorarios pactados en UF (0 si no está definido)."""
    return a_numero(db.get_meta(META_UF), cfg.valor_uf) or 0.0


def monto_honorario(cliente, uf):
    """Honorario mensual en pesos, o None si está en UF y no hay valor de UF."""
    honorario = cliente.get("honorario") or 0
    if honorario <= 0:
        return 0
    if cliente.get("moneda") == "UF":
        return int(round(honorario * uf)) if uf > 0 else None
    return int(round(honorario))


def emitir_honorarios(db, cfg, periodo, hoy=None):
    """Crea el cobro de honorarios del período para cada cliente activo. No duplica los ya emitidos.

    Devuelve (creados, avisos): avisos lista los clientes que no se pudieron cobrar y por qué.
    """
    anio, mes = calendario.partes_periodo(periodo)
    if mes is None:
        raise ValueError("Los honorarios se emiten por mes: usa un período AAAA-MM.")
    hoy = hoy or cfg.hoy()
    uf = valor_uf(db, cfg)
    anio_v, mes_v = calendario.mes_siguiente(anio, mes)
    vence = date(anio_v, mes_v, min(cfg.honorarios_dia_vence, calendar.monthrange(anio_v, mes_v)[1]))
    creados, avisos = 0, []
    for cliente in db.clientes(estados=["Activo"]):
        monto = monto_honorario(cliente, uf)
        if monto is None:
            avisos.append(f"{cliente['razon_social']}: honorario en UF, pero falta definir el valor de la UF.")
            continue
        if monto <= 0:
            continue
        nuevo = db.crear_cobro(cliente["id"], CONCEPTO_HONORARIO, monto, hoy, vence, periodo=periodo)
        creados += bool(nuevo)
    return creados, avisos


def tramo_mora(dias):
    if dias <= 0:
        return TRAMOS_MORA[0]
    if dias <= 30:
        return TRAMOS_MORA[1]
    if dias <= 60:
        return TRAMOS_MORA[2]
    if dias <= 90:
        return TRAMOS_MORA[3]
    return TRAMOS_MORA[4]


def cartera(db, hoy, cliente_id=None):
    """Cobros con saldo pendiente, con sus días de mora y tramo."""
    filas = db.cobros(cliente_id=cliente_id, pendientes=True)
    for f in filas:
        f["dias_mora"] = max(0, (hoy - date.fromisoformat(f["vence"])).days)
        f["tramo"] = tramo_mora(f["dias_mora"])
    return filas


def resumen(db, hoy):
    pendientes = cartera(db, hoy)
    inicio_mes = hoy.replace(day=1)
    return {
        "por_cobrar": sum(c["saldo"] for c in pendientes),
        "vencido": sum(c["saldo"] for c in pendientes if c["dias_mora"] > 0),
        "clientes_morosos": len({c["cliente_id"] for c in pendientes if c["dias_mora"] > 0}),
        "cobrado_mes": sum(p["monto"] for p in db.pagos(desde=inicio_mes, hasta=hoy)),
    }


def por_tramo(pendientes):
    totales = dict.fromkeys(TRAMOS_MORA, 0)
    for c in pendientes:
        totales[c["tramo"]] += c["saldo"]
    return totales


def serie_mensual(db, hoy, meses=6):
    """Emitido (por mes de emisión) y cobrado (por mes de pago) de los últimos meses."""
    claves = []
    for i in range(meses - 1, -1, -1):
        anio, mes = calendario.mes_siguiente(hoy.year, hoy.month, -i)
        claves.append(f"{anio:04d}-{mes:02d}")
    emitido, cobrado = dict.fromkeys(claves, 0), dict.fromkeys(claves, 0)
    for c in db.cobros():
        if c["emitido"][:7] in emitido:
            emitido[c["emitido"][:7]] += c["monto"]
    for p in db.pagos():
        if p["fecha"][:7] in cobrado:
            cobrado[p["fecha"][:7]] += p["monto"]
    return [{"mes": k, "emitido": emitido[k], "cobrado": cobrado[k]} for k in claves]


# ---------------- mensajes ----------------
def _saludo(cliente):
    nombre = (cliente.get("contacto") or "").strip().split(" ")[0]
    return f"Hola {nombre}:" if nombre else "Estimados:"


def _fecha(texto):
    return date.fromisoformat(texto).strftime("%d/%m/%Y")


def mensaje_cobranza(cliente, pendientes, cfg):
    """(asunto, cuerpo) del recordatorio de pago por los cobros pendientes de un cliente."""
    total = sum(c["saldo"] for c in pendientes)
    vencidos = any(c.get("dias_mora", 0) > 0 for c in pendientes)
    lineas = [
        f"- {c['concepto']}"
        + (f" {calendario.nombre_periodo(c['periodo'])}" if c["periodo"] else "")
        + f": {formato_clp(c['saldo'])} (vence el {_fecha(c['vence'])})"
        for c in pendientes
    ]
    cuerpo = [
        _saludo(cliente),
        "",
        "Junto con saludar, le recordamos que "
        + ("mantiene pendiente de pago lo siguiente:" if vencidos else "está próximo a vencer lo siguiente:"),
        "",
        *lineas,
        "",
        f"Total: {formato_clp(total)}",
    ]
    if cfg.datos_pago:
        cuerpo += ["", "Datos para el pago:", cfg.datos_pago]
    cuerpo += ["", "Si ya realizó el pago, por favor envíenos el comprobante y omita este mensaje.", "", "Saludos,", cfg.firma, cfg.estudio_nombre]
    asunto = f"{'Recordatorio de pago' if vencidos else 'Aviso de vencimiento'} · {cliente['razon_social']} · {cfg.estudio_nombre}"
    return asunto, "\n".join(cuerpo)


def mensaje_documentos(cliente, obligaciones, cfg):
    """(asunto, cuerpo) para pedir al cliente la información que falta para sus declaraciones."""
    lineas = [
        f"- {o['nombre']}"
        + (f" ({calendario.nombre_periodo(o['periodo'])})" if o["periodo"] else "")
        + f": vence el {_fecha(o['vence'])}"
        for o in obligaciones
    ]
    cuerpo = [
        _saludo(cliente),
        "",
        "Para presentar a tiempo sus declaraciones necesitamos que nos envíe la información pendiente de:",
        "",
        *lineas,
        "",
        "Mientras antes la recibamos, antes podremos revisarla y evitar multas o recargos.",
        "",
        "Saludos,",
        cfg.firma,
        cfg.estudio_nombre,
    ]
    return f"Información pendiente · {cliente['razon_social']} · {cfg.estudio_nombre}", "\n".join(cuerpo)


def link_mailto(correo, asunto, cuerpo):
    return f"mailto:{quote(correo or '')}?subject={quote(asunto)}&body={quote(cuerpo)}"


def telefono_whatsapp(telefono):
    """Número en formato internacional sin '+' (asume Chile si viene un celular de 9 dígitos)."""
    crudo = (telefono or "").strip()
    digitos = re.sub(r"\D", "", crudo)
    if crudo.startswith("+") and len(digitos) >= 10:
        return digitos
    if len(digitos) == 9 and digitos.startswith("9"):
        return "56" + digitos
    if len(digitos) == 11 and digitos.startswith("569"):
        return digitos
    return None


def link_whatsapp(telefono, texto):
    numero = telefono_whatsapp(telefono)
    return f"https://wa.me/{numero}?text={quote(texto)}" if numero else None
