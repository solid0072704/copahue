"""Caso práctico del curso: Distribuidora Alerce SpA (empresa ficticia).

Todas las cifras están en millones de pesos (MM$). Este módulo es la única fuente
de los números que aparecen en las diapositivas y en el guion: los totales, los
ratios y las variaciones se calculan aquí y `verificar()` comprueba que los estados
cuadren (se ejecuta al importar el módulo).

Uso:
    python -m curso_youtube.caso            # imprime la ficha de cifras del caso
    python -m curso_youtube.caso --excel    # genera material/caso_distribuidora_alerce.xlsx
"""

from __future__ import annotations

import sys
from pathlib import Path

EMPRESA = "Distribuidora Alerce SpA"
GIRO = "Distribuidora mayorista de alimentos y abarrotes"
MONEDA = "millones de pesos"
TASA_IMPUESTO = 0.27

# --- Estado de resultados -------------------------------------------------
EERR = {
    "2024": dict(ventas=10000, costo=7400, remuneraciones=900, distribucion=430, otros_gav=370,
                 depreciacion=150, gastos_fin=110, impuesto=173),
    "2025": dict(ventas=11800, costo=8850, remuneraciones=1010, distribucion=560, otros_gav=370,
                 depreciacion=170, gastos_fin=175, impuesto=180),
    "ppto": dict(ventas=11200, costo=8288, remuneraciones=990, distribucion=470, otros_gav=390,
                 depreciacion=165, gastos_fin=120, impuesto=210),
}

# --- Balance --------------------------------------------------------------
BALANCE = {
    "2024": dict(efectivo=620, cxc=1370, inventarios=1115, otros_ac=95, ppe=1800,
                 proveedores=1014, deuda_cp=400, otros_pc=186, deuda_lp=900,
                 capital=1200, utilidades_acum=1300),
    "2025": dict(efectivo=65, cxc=2330, inventarios=1650, otros_ac=110, ppe=1950,
                 proveedores=1165, deuda_cp=1150, otros_pc=205, deuda_lp=800,
                 capital=1200, utilidades_acum=1585),
}

# --- Otros datos del caso -------------------------------------------------
CAPEX = {"2024": 200, "2025": 320}
DIVIDENDOS = {"2024": 150, "2025": 200}
# Flujo 2024 (no hay balance 2023 en el caso; se entrega como dato)
FLUJO_2024 = dict(utilidad=467, depreciacion=150, var_cxc=-110, var_inventarios=-75, var_otros_ac=-5,
                  var_proveedores=70, var_otros_pc=13, capex=-200, var_deuda=-100, dividendos=-150,
                  efectivo_inicial=560)
# Antigüedad de las cuentas por cobrar
ANTIGUEDAD = {
    "2024": {"Por vencer": 1000, "1 a 30 días": 250, "31 a 60 días": 80, "61 a 90 días": 30, "Más de 90 días": 10},
    "2025": {"Por vencer": 1280, "1 a 30 días": 520, "31 a 60 días": 290, "61 a 90 días": 150, "Más de 90 días": 90},
}
# Segmentos 2025: dos cadenas nuevas frente al resto de los clientes
SEGMENTOS_2025 = {
    "Cadenas nuevas (2 clientes)": dict(ventas=2360, margen_pct=0.19, cxc=650),
    "Clientes tradicionales": dict(ventas=9440, margen_pct=0.265, cxc=1680),
}

ETIQUETAS = {
    "ventas": "Ventas", "costo": "Costo de ventas", "margen_bruto": "Margen bruto",
    "remuneraciones": "Remuneraciones", "distribucion": "Distribución y fletes",
    "otros_gav": "Arriendos y otros gastos", "gav": "Gastos de administración y ventas",
    "depreciacion": "Depreciación", "ebit": "Resultado operacional", "ebitda": "EBITDA",
    "gastos_fin": "Gastos financieros", "rai": "Resultado antes de impuestos",
    "impuesto": "Impuesto a la renta", "utilidad": "Utilidad neta",
    "efectivo": "Efectivo", "cxc": "Cuentas por cobrar", "inventarios": "Inventarios",
    "otros_ac": "Otros activos corrientes", "activo_corriente": "Total activos corrientes",
    "ppe": "Propiedades, planta y equipo", "activos": "Total activos",
    "proveedores": "Proveedores", "deuda_cp": "Deuda bancaria de corto plazo",
    "otros_pc": "Otros pasivos corrientes", "pasivo_corriente": "Total pasivos corrientes",
    "deuda_lp": "Deuda bancaria de largo plazo", "pasivos": "Total pasivos",
    "capital": "Capital", "utilidades_acum": "Utilidades acumuladas", "patrimonio": "Total patrimonio",
    "pasivo_patrimonio": "Total pasivos y patrimonio",
}


# --- Cálculos -------------------------------------------------------------
def eerr(periodo: str) -> dict:
    d = dict(EERR[periodo])
    d["margen_bruto"] = d["ventas"] - d["costo"]
    d["gav"] = d["remuneraciones"] + d["distribucion"] + d["otros_gav"]
    d["ebit"] = d["margen_bruto"] - d["gav"] - d["depreciacion"]
    d["ebitda"] = d["ebit"] + d["depreciacion"]
    d["rai"] = d["ebit"] - d["gastos_fin"]
    d["utilidad"] = d["rai"] - d["impuesto"]
    return d


def balance(periodo: str) -> dict:
    d = dict(BALANCE[periodo])
    d["activo_corriente"] = d["efectivo"] + d["cxc"] + d["inventarios"] + d["otros_ac"]
    d["activos"] = d["activo_corriente"] + d["ppe"]
    d["pasivo_corriente"] = d["proveedores"] + d["deuda_cp"] + d["otros_pc"]
    d["pasivos"] = d["pasivo_corriente"] + d["deuda_lp"]
    d["patrimonio"] = d["capital"] + d["utilidades_acum"]
    d["pasivo_patrimonio"] = d["pasivos"] + d["patrimonio"]
    d["deuda_fin"] = d["deuda_cp"] + d["deuda_lp"]
    d["deuda_fin_neta"] = d["deuda_fin"] - d["efectivo"]
    d["capital_trabajo_op"] = d["cxc"] + d["inventarios"] - d["proveedores"]
    return d


def flujo_2025() -> dict:
    a, b, r = balance("2024"), balance("2025"), eerr("2025")
    f = dict(
        utilidad=r["utilidad"],
        depreciacion=r["depreciacion"],
        var_cxc=-(b["cxc"] - a["cxc"]),
        var_inventarios=-(b["inventarios"] - a["inventarios"]),
        var_otros_ac=-(b["otros_ac"] - a["otros_ac"]),
        var_proveedores=b["proveedores"] - a["proveedores"],
        var_otros_pc=b["otros_pc"] - a["otros_pc"],
        capex=-CAPEX["2025"],
        var_deuda_cp=b["deuda_cp"] - a["deuda_cp"],
        var_deuda_lp=b["deuda_lp"] - a["deuda_lp"],
        dividendos=-DIVIDENDOS["2025"],
        efectivo_inicial=a["efectivo"],
    )
    f["var_capital_trabajo"] = (f["var_cxc"] + f["var_inventarios"] + f["var_otros_ac"]
                                + f["var_proveedores"] + f["var_otros_pc"])
    f["operacional"] = f["utilidad"] + f["depreciacion"] + f["var_capital_trabajo"]
    f["inversion"] = f["capex"]
    f["financiamiento"] = f["var_deuda_cp"] + f["var_deuda_lp"] + f["dividendos"]
    f["var_neta"] = f["operacional"] + f["inversion"] + f["financiamiento"]
    f["efectivo_final"] = f["efectivo_inicial"] + f["var_neta"]
    f["flujo_libre"] = f["operacional"] + f["inversion"]
    return f


def flujo_2024() -> dict:
    f = dict(FLUJO_2024)
    f["var_capital_trabajo"] = (f["var_cxc"] + f["var_inventarios"] + f["var_otros_ac"]
                                + f["var_proveedores"] + f["var_otros_pc"])
    f["operacional"] = f["utilidad"] + f["depreciacion"] + f["var_capital_trabajo"]
    f["inversion"] = f["capex"]
    f["financiamiento"] = f["var_deuda"] + f["dividendos"]
    f["var_neta"] = f["operacional"] + f["inversion"] + f["financiamiento"]
    f["efectivo_final"] = f["efectivo_inicial"] + f["var_neta"]
    f["flujo_libre"] = f["operacional"] + f["inversion"]
    return f


def ratios(periodo: str) -> dict:
    r, b = eerr(periodo), balance(periodo)
    x = {}
    # Liquidez
    x["liquidez_corriente"] = b["activo_corriente"] / b["pasivo_corriente"]
    x["prueba_acida"] = (b["activo_corriente"] - b["inventarios"]) / b["pasivo_corriente"]
    x["razon_caja"] = b["efectivo"] / b["pasivo_corriente"]
    # Endeudamiento
    x["endeudamiento"] = b["pasivos"] / b["patrimonio"]
    x["dfn_ebitda"] = b["deuda_fin_neta"] / r["ebitda"]
    x["cobertura_intereses"] = r["ebit"] / r["gastos_fin"]
    x["deuda_cp_pct"] = b["deuda_cp"] / b["deuda_fin"]
    # Rentabilidad
    x["margen_bruto"] = r["margen_bruto"] / r["ventas"]
    x["margen_ebitda"] = r["ebitda"] / r["ventas"]
    x["margen_operacional"] = r["ebit"] / r["ventas"]
    x["margen_neto"] = r["utilidad"] / r["ventas"]
    x["roe"] = r["utilidad"] / b["patrimonio"]
    x["roa"] = r["utilidad"] / b["activos"]
    x["rotacion_activos"] = r["ventas"] / b["activos"]
    x["multiplicador"] = b["activos"] / b["patrimonio"]
    # Eficiencia (días)
    x["dias_cobro"] = b["cxc"] / r["ventas"] * 365
    x["dias_inventario"] = b["inventarios"] / r["costo"] * 365
    x["dias_pago"] = b["proveedores"] / r["costo"] * 365
    x["ciclo_caja"] = x["dias_cobro"] + x["dias_inventario"] - x["dias_pago"]
    return x


def variacion(clave: str, base: str = "2024", actual: str = "2025") -> dict:
    """Variación absoluta y porcentual de una partida del estado de resultados o del balance."""
    fuente = eerr if clave in eerr(base) else balance
    a, b = fuente(base)[clave], fuente(actual)[clave]
    return dict(base=a, actual=b, abs=b - a, pct=(b - a) / a)


def desviaciones() -> dict:
    """Real 2025 contra presupuesto 2025, con el efecto volumen y el efecto margen."""
    real, ppto = eerr("2025"), eerr("ppto")
    d = {k: dict(real=real[k], ppto=ppto[k], abs=real[k] - ppto[k], pct=(real[k] - ppto[k]) / ppto[k])
         for k in real}
    tasa_ppto = ppto["margen_bruto"] / ppto["ventas"]
    tasa_real = real["margen_bruto"] / real["ventas"]
    d["efecto_volumen"] = (real["ventas"] - ppto["ventas"]) * tasa_ppto
    d["efecto_margen"] = (tasa_real - tasa_ppto) * real["ventas"]
    return d


def escenario() -> dict:
    """Palancas del plan de acción: caja que se libera y resultado que se recupera."""
    r, b, x = eerr("2025"), balance("2025"), ratios("2025")
    e = {}
    e["cxc_objetivo"] = round(r["ventas"] * 60 / 365)
    e["libera_cobranza"] = b["cxc"] - e["cxc_objetivo"]
    e["inv_objetivo"] = round(r["costo"] * 60 / 365)
    e["libera_inventario"] = b["inventarios"] - e["inv_objetivo"]
    e["caja_liberada"] = e["libera_cobranza"] + e["libera_inventario"]
    deuda_prom = (balance("2024")["deuda_fin"] + b["deuda_fin"]) / 2
    e["tasa_deuda"] = r["gastos_fin"] / deuda_prom
    e["ahorro_intereses"] = e["caja_liberada"] * e["tasa_deuda"]
    e["deuda_cp_despues"] = b["deuda_cp"] - e["caja_liberada"]
    e["punto_margen"] = r["ventas"] * 0.01
    e["fletes_pct_2024"] = eerr("2024")["distribucion"] / eerr("2024")["ventas"]
    e["ahorro_fletes"] = r["distribucion"] - e["fletes_pct_2024"] * r["ventas"]
    e["ciclo_despues"] = 60 + 60 - x["dias_pago"]
    return e


def valor_de_un_dia() -> dict:
    """Cuánta caja representa un día de cobro, de inventario y de pago (2025)."""
    r = eerr("2025")
    return dict(cobro=r["ventas"] / 365, inventario=r["costo"] / 365, pago=r["costo"] / 365)


ESCENARIOS = {"Conservador": (66, 64), "Base": (60, 60), "Optimista": (55, 55)}  # (días de cobro, días de inventario)


def escenarios() -> dict:
    """Caja liberada, deuda de corto plazo y ciclo de efectivo en tres escenarios."""
    r, b, x = eerr("2025"), balance("2025"), ratios("2025")
    out = {}
    for nombre, (dc, di) in ESCENARIOS.items():
        cxc, inv = round(r["ventas"] * dc / 365), round(r["costo"] * di / 365)
        libera = (b["cxc"] - cxc) + (b["inventarios"] - inv)
        out[nombre] = dict(dias_cobro=dc, dias_inventario=di, caja_liberada=libera,
                           deuda_cp=b["deuda_cp"] - libera, ciclo=dc + di - round(x["dias_pago"]))
    return out


def calidad_utilidad() -> dict:
    """Flujo de la operación dividido por la utilidad neta."""
    return {"2024": flujo_2024()["operacional"] / eerr("2024")["utilidad"],
            "2025": flujo_2025()["operacional"] / eerr("2025")["utilidad"]}


def costo_de_crecer() -> dict:
    """Capital de trabajo adicional que exigió cada peso de venta nueva."""
    var_kt = balance("2025")["capital_trabajo_op"] - balance("2024")["capital_trabajo_op"]
    var_ventas = eerr("2025")["ventas"] - eerr("2024")["ventas"]
    por_100 = var_kt / var_ventas * 100
    return dict(var_kt=var_kt, var_ventas=var_ventas, por_cada_100=por_100,
                veces_utilidad=por_100 / (ratios("2025")["margen_neto"] * 100))


def antiguedad() -> dict:
    """Antigüedad de las cuentas por cobrar con totales de cartera vencida."""
    out = {}
    for p in ("2024", "2025"):
        total = BALANCE[p]["cxc"]
        vencido = total - ANTIGUEDAD[p]["Por vencer"]
        mas_60 = ANTIGUEDAD[p]["61 a 90 días"] + ANTIGUEDAD[p]["Más de 90 días"]
        out[p] = dict(tramos=ANTIGUEDAD[p], total=total, vencido=vencido, pct_vencido=vencido / total,
                      pct_al_dia=ANTIGUEDAD[p]["Por vencer"] / total, mas_60=mas_60)
    return out


SEMANAS_SUELDOS = (4, 8, 13)            # fin de cada mes
SEMANAS_IMPUESTOS_INTERESES = (3, 7, 11)


def proyeccion_13_semanas() -> dict:
    """Proyección ilustrativa de caja a 13 semanas: sin medidas y con el plan de acción.

    Supuestos: cobranzas, pagos a proveedores y gastos al ritmo semanal de 2025; sueldos a fin de mes;
    impuestos e intereses una vez al mes. El plan lleva la cobranza y el inventario a 60 días en 26 semanas
    y empieza a notarse desde la semana 2.
    """
    r, f, e = eerr("2025"), flujo_2025(), escenario()
    cobranza = (r["ventas"] + f["var_cxc"]) / 52
    proveedores = (r["costo"] - f["var_inventarios"] - f["var_proveedores"]) / 52
    otros = (r["distribucion"] + r["otros_gav"]) / 52
    sueldos = r["remuneraciones"] / 12
    imp_int = (r["impuesto"] + r["gastos_fin"]) / 12
    mejora = (e["libera_cobranza"] + e["libera_inventario"]) / 26
    base, plan = [float(balance("2025")["efectivo"])], [float(balance("2025")["efectivo"])]
    for s in range(1, 14):
        neto = cobranza - proveedores - otros
        neto -= sueldos if s in SEMANAS_SUELDOS else 0
        neto -= imp_int if s in SEMANAS_IMPUESTOS_INTERESES else 0
        base.append(base[-1] + neto)
        plan.append(plan[-1] + neto + (mejora if s >= 2 else 0))
    primera_negativa = next((s for s in range(1, 14) if base[s] < 0), None)
    return dict(base=base, plan=plan, cobranza=cobranza, proveedores=proveedores, otros=otros, sueldos=sueldos,
                imp_int=imp_int, mejora=mejora, primera_negativa=primera_negativa,
                minimo_base=min(base[1:]), minimo_plan=min(plan[1:]), semana_minimo_plan=plan.index(min(plan[1:])))


def segmentos() -> dict:
    s = {}
    for nombre, d in SEGMENTOS_2025.items():
        s[nombre] = dict(d, margen=d["ventas"] * d["margen_pct"], dias_cobro=d["cxc"] / d["ventas"] * 365,
                         pct_ventas=d["ventas"] / EERR["2025"]["ventas"], pct_cxc=d["cxc"] / BALANCE["2025"]["cxc"])
    return s


def verificar() -> None:
    """Comprueba las cuadraturas del caso. Lanza AssertionError si algo no cuadra."""
    for p in ("2024", "2025"):
        b = balance(p)
        assert b["activos"] == b["pasivo_patrimonio"], f"El balance {p} no cuadra"
        assert sum(ANTIGUEDAD[p].values()) == b["cxc"], f"La antigüedad {p} no suma las cuentas por cobrar"
    for p in ("2024", "2025", "ppto"):
        r = eerr(p)
        assert abs(r["impuesto"] - r["rai"] * TASA_IMPUESTO) < 1, f"Impuesto {p} fuera de la tasa"
    f25, f24 = flujo_2025(), flujo_2024()
    assert f25["efectivo_final"] == balance("2025")["efectivo"], "El flujo 2025 no llega al efectivo del balance"
    assert f24["efectivo_final"] == balance("2024")["efectivo"], "El flujo 2024 no llega al efectivo del balance"
    assert f24["utilidad"] == eerr("2024")["utilidad"] and f24["depreciacion"] == eerr("2024")["depreciacion"]
    a, b = balance("2024"), balance("2025")
    assert b["ppe"] == a["ppe"] + CAPEX["2025"] - eerr("2025")["depreciacion"], "El activo fijo no concilia"
    assert b["utilidades_acum"] == a["utilidades_acum"] + eerr("2025")["utilidad"] - DIVIDENDOS["2025"], \
        "El patrimonio no concilia con la utilidad y los dividendos"
    s = segmentos()
    assert sum(d["ventas"] for d in s.values()) == EERR["2025"]["ventas"]
    assert sum(d["cxc"] for d in s.values()) == BALANCE["2025"]["cxc"]
    assert abs(sum(d["margen"] for d in s.values()) - eerr("2025")["margen_bruto"]) < 0.5
    d = desviaciones()
    assert abs(d["efecto_volumen"] + d["efecto_margen"] - d["margen_bruto"]["abs"]) < 0.5
    for p in ("2024", "2025"):
        x = ratios(p)
        assert abs(x["margen_neto"] * x["rotacion_activos"] * x["multiplicador"] - x["roe"]) < 1e-9


verificar()


# --- Formato --------------------------------------------------------------
def miles(n: float, dec: int = 0) -> str:
    """1344 -> '1.344'; -685 -> '-685'; 1.65 con dec=2 -> '1,65' (formato chileno)."""
    s = f"{abs(n):,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return ("-" if round(n, dec) < 0 else "") + s


def parentesis(n: float) -> str:
    """Formato contable: los negativos entre paréntesis."""
    return f"({miles(abs(n))})" if n < 0 else miles(n)


def pct(x: float, dec: int = 1, signo: bool = False) -> str:
    s = miles(x * 100, dec)
    if signo and x > 0:
        s = "+" + s
    return s + " %"


def veces(x: float, dec: int = 2) -> str:
    return miles(x, dec) + "x"


def ficha() -> str:
    """Ficha con todas las cifras del caso, para escribir y revisar el guion."""
    L = [f"{EMPRESA} — {GIRO} (cifras en {MONEDA})", ""]
    r24, r25, rp = eerr("2024"), eerr("2025"), eerr("ppto")
    L.append("ESTADO DE RESULTADOS        2024     2025    var %   | Ppto 25   desv   desv %")
    d = desviaciones()
    for k in ("ventas", "costo", "margen_bruto", "remuneraciones", "distribucion", "otros_gav", "gav",
              "depreciacion", "ebit", "ebitda", "gastos_fin", "rai", "impuesto", "utilidad"):
        v = (r25[k] - r24[k]) / r24[k]
        L.append(f"  {ETIQUETAS[k]:<34}{r24[k]:>7}{r25[k]:>9}{pct(v, 1, True):>9}  |{rp[k]:>8}{d[k]['abs']:>7}{pct(d[k]['pct'], 1, True):>9}")
    L.append(f"  Efecto volumen {d['efecto_volumen']:.0f} · efecto margen {d['efecto_margen']:.0f}")
    L.append("")
    b24, b25 = balance("2024"), balance("2025")
    L.append("BALANCE                               2024     2025    var     var %")
    for k in ("efectivo", "cxc", "inventarios", "otros_ac", "activo_corriente", "ppe", "activos", "proveedores",
              "deuda_cp", "otros_pc", "pasivo_corriente", "deuda_lp", "pasivos", "capital", "utilidades_acum",
              "patrimonio", "pasivo_patrimonio"):
        v = (b25[k] - b24[k]) / b24[k]
        L.append(f"  {ETIQUETAS[k]:<34}{b24[k]:>7}{b25[k]:>9}{b25[k] - b24[k]:>8}{pct(v, 1, True):>9}")
    L.append(f"  Deuda financiera {b24['deuda_fin']} -> {b25['deuda_fin']} · neta {b24['deuda_fin_neta']} -> {b25['deuda_fin_neta']}"
             f" · capital de trabajo operativo {b24['capital_trabajo_op']} -> {b25['capital_trabajo_op']}")
    L.append("")
    for nombre, f in (("FLUJO 2024", flujo_2024()), ("FLUJO 2025", flujo_2025())):
        L.append(nombre + ": " + ", ".join(f"{k}={v}" for k, v in f.items()))
    L.append("")
    x24, x25 = ratios("2024"), ratios("2025")
    L.append("RATIOS                         2024      2025")
    for k in x24:
        L.append(f"  {k:<26}{x24[k]:>9.3f}{x25[k]:>10.3f}")
    L.append("")
    L.append("SEGMENTOS 2025: " + "; ".join(
        f"{n}: ventas {s['ventas']} ({pct(s['pct_ventas'])}), margen {pct(s['margen_pct'])} = {s['margen']:.0f}, "
        f"CxC {s['cxc']} ({pct(s['pct_cxc'])}), {s['dias_cobro']:.0f} días" for n, s in segmentos().items()))
    for p in ("2024", "2025"):
        venc = BALANCE[p]["cxc"] - ANTIGUEDAD[p]["Por vencer"]
        L.append(f"ANTIGÜEDAD {p}: {ANTIGUEDAD[p]} · vencido {venc} ({pct(venc / BALANCE[p]['cxc'])})")
    L.append("ESCENARIO: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in escenario().items()))
    L.append("ESCENARIOS: " + "; ".join(f"{n}: {d}" for n, d in escenarios().items()))
    L.append("VALOR DE UN DÍA: " + ", ".join(f"{k}={v:.1f}" for k, v in valor_de_un_dia().items()))
    L.append("CALIDAD DE LA UTILIDAD: " + ", ".join(f"{k}={v:.2f}" for k, v in calidad_utilidad().items()))
    L.append("COSTO DE CRECER: " + ", ".join(f"{k}={v:.1f}" for k, v in costo_de_crecer().items()))
    pr = proyeccion_13_semanas()
    L.append("PROYECCIÓN 13 SEMANAS sin medidas: " + " ".join(f"{v:.0f}" for v in pr["base"]))
    L.append("PROYECCIÓN 13 SEMANAS con plan:    " + " ".join(f"{v:.0f}" for v in pr["plan"]))
    L.append(f"  primera semana negativa {pr['primera_negativa']}, mínimo con plan {pr['minimo_plan']:.0f} "
             f"(semana {pr['semana_minimo_plan']}), sueldos {pr['sueldos']:.0f}, mejora semanal {pr['mejora']:.1f}")
    L.append("ANTIGÜEDAD: " + "; ".join(f"{k}: vencido {v['vencido']} ({pct(v['pct_vencido'])}), al día {pct(v['pct_al_dia'])}, "
                                        f"más de 60 días {v['mas_60']}" for k, v in antiguedad().items()))
    return "\n".join(L)


def exportar_excel(ruta: Path) -> Path:
    """Genera el archivo del caso para que cada persona repita los prompts del curso."""
    import xlsxwriter

    ruta.parent.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(str(ruta))
    tit = wb.add_format({"bold": True, "font_size": 14})
    enc = wb.add_format({"bold": True, "bg_color": "#0B1B2B", "font_color": "#FFFFFF", "align": "center"})
    txt = wb.add_format({})
    neg = wb.add_format({"bold": True})
    num = wb.add_format({"num_format": "#,##0"})
    numb = wb.add_format({"num_format": "#,##0", "bold": True, "top": 1})
    nota = wb.add_format({"italic": True, "font_color": "#555555", "text_wrap": True, "valign": "top"})

    def hoja(nombre, titulo, columnas, filas, totales=()):
        ws = wb.add_worksheet(nombre)
        ws.set_column(0, 0, 40)
        ws.set_column(1, len(columnas), 16)
        ws.write(0, 0, f"{EMPRESA} — {titulo}", tit)
        ws.write(1, 0, f"Cifras en {MONEDA}. Empresa ficticia con fines educativos.", nota)
        ws.write(3, 0, "", enc)
        for j, c in enumerate(columnas, start=1):
            ws.write(3, j, c, enc)
        for i, (etq, valores) in enumerate(filas, start=4):
            es_total = etq in totales
            ws.write(i, 0, etq, neg if es_total else txt)
            for j, v in enumerate(valores, start=1):
                ws.write_number(i, j, v, numb if es_total else num)
        return ws

    r24, r25, rp = eerr("2024"), eerr("2025"), eerr("ppto")
    orden = [("ventas", 1), ("costo", -1), ("margen_bruto", 1), ("remuneraciones", -1), ("distribucion", -1),
             ("otros_gav", -1), ("depreciacion", -1), ("ebit", 1), ("gastos_fin", -1), ("rai", 1),
             ("impuesto", -1), ("utilidad", 1)]
    tot_r = {ETIQUETAS[k] for k in ("margen_bruto", "ebit", "rai", "utilidad")}
    hoja("Estado de resultados", "Estado de resultados", ["2024", "2025"],
         [(ETIQUETAS[k], [s * r24[k], s * r25[k]]) for k, s in orden], tot_r)
    hoja("Presupuesto 2025", "Presupuesto 2025 y resultado real", ["Presupuesto 2025", "Real 2025"],
         [(ETIQUETAS[k], [s * rp[k], s * r25[k]]) for k, s in orden], tot_r)
    b24, b25 = balance("2024"), balance("2025")
    claves_b = ["efectivo", "cxc", "inventarios", "otros_ac", "activo_corriente", "ppe", "activos",
                "proveedores", "deuda_cp", "otros_pc", "pasivo_corriente", "deuda_lp", "pasivos",
                "capital", "utilidades_acum", "patrimonio", "pasivo_patrimonio"]
    tot_b = {ETIQUETAS[k] for k in ("activo_corriente", "activos", "pasivo_corriente", "pasivos", "patrimonio",
                                    "pasivo_patrimonio")}
    hoja("Balance", "Balance general al 31 de diciembre", ["2024", "2025"],
         [(ETIQUETAS[k], [b24[k], b25[k]]) for k in claves_b], tot_b)
    f24, f25 = flujo_2024(), flujo_2025()
    filas_f = [
        ("Utilidad neta", [f24["utilidad"], f25["utilidad"]]),
        ("Depreciación", [f24["depreciacion"], f25["depreciacion"]]),
        ("Variación de cuentas por cobrar", [f24["var_cxc"], f25["var_cxc"]]),
        ("Variación de inventarios", [f24["var_inventarios"], f25["var_inventarios"]]),
        ("Variación de otros activos corrientes", [f24["var_otros_ac"], f25["var_otros_ac"]]),
        ("Variación de proveedores", [f24["var_proveedores"], f25["var_proveedores"]]),
        ("Variación de otros pasivos corrientes", [f24["var_otros_pc"], f25["var_otros_pc"]]),
        ("Flujo de la operación", [f24["operacional"], f25["operacional"]]),
        ("Inversión en activo fijo", [f24["capex"], f25["capex"]]),
        ("Flujo de inversión", [f24["inversion"], f25["inversion"]]),
        ("Variación de deuda bancaria", [f24["var_deuda"], f25["var_deuda_cp"] + f25["var_deuda_lp"]]),
        ("Dividendos pagados", [f24["dividendos"], f25["dividendos"]]),
        ("Flujo de financiamiento", [f24["financiamiento"], f25["financiamiento"]]),
        ("Variación neta del efectivo", [f24["var_neta"], f25["var_neta"]]),
        ("Efectivo inicial", [f24["efectivo_inicial"], f25["efectivo_inicial"]]),
        ("Efectivo final", [f24["efectivo_final"], f25["efectivo_final"]]),
    ]
    hoja("Flujo de efectivo", "Estado de flujo de efectivo (método indirecto)", ["2024", "2025"], filas_f,
         {"Flujo de la operación", "Flujo de inversión", "Flujo de financiamiento", "Variación neta del efectivo",
          "Efectivo final"})
    hoja("Cuentas por cobrar", "Antigüedad de las cuentas por cobrar", ["2024", "2025"],
         [(k, [ANTIGUEDAD["2024"][k], ANTIGUEDAD["2025"][k]]) for k in ANTIGUEDAD["2025"]]
         + [("Total cuentas por cobrar", [b24["cxc"], b25["cxc"]])], {"Total cuentas por cobrar"})
    ws = wb.add_worksheet("Notas")
    ws.set_column(0, 0, 110)
    notas = [
        f"{EMPRESA} es una empresa ficticia creada para este curso. Cualquier parecido con una empresa real es coincidencia.",
        f"Giro: {GIRO.lower()} para almacenes, restaurantes y hoteles. Cifras en {MONEDA}.",
        "En 2025 la empresa incorporó dos cadenas como clientes: representan el 20 % de las ventas, "
        "compran con descuento (margen bruto de 19 %) y pagan en torno a 100 días.",
        "El resto de los clientes mantiene un margen bruto de 26,5 %.",
        "Para atender a las cadenas se amplió el surtido y el inventario, y aumentó el gasto en distribución y fletes.",
        "El crecimiento se financió con una línea de crédito bancaria de corto plazo.",
        "No existe provisión por deudores incobrables.",
        "Inversión en activo fijo: 200 en 2024 y 320 en 2025. Dividendos pagados: 150 en 2024 y 200 en 2025.",
        "Tasa de impuesto a la renta: 27 %.",
    ]
    ws.write(0, 0, f"{EMPRESA} — Notas del caso", tit)
    for i, n in enumerate(notas, start=2):
        ws.write(i, 0, n, nota)
    wb.close()
    return ruta


if __name__ == "__main__":
    if "--excel" in sys.argv:
        destino = Path(__file__).parent / "material" / "caso_distribuidora_alerce.xlsx"
        print("Archivo generado:", exportar_excel(destino))
    else:
        print(ficha())
