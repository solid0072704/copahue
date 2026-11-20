"""Paso 3: análisis vertical y horizontal."""

from .. import caso
from ._util import m, pc, rank, var

R24, R25 = caso.eerr("2024"), caso.eerr("2025")
B24, B25 = caso.balance("2024"), caso.balance("2025")


def _er(clave, etq=None, cls="", p=None, tono=None):
    a, b = R24[clave], R25[clave]
    v = var((b - a) / a)
    return dict(c=[etq or caso.ETIQUETAS[clave], m(a), pc(a / R24["ventas"]), m(b), pc(b / R25["ventas"]),
                   (v, tono) if tono else v], cls=cls, p=p)


def _bal(clave, etq=None, cls="", p=None, tono=None):
    a, b = B24[clave], B25[clave]
    v = var((b - a) / a)
    return dict(c=[etq or caso.ETIQUETAS[clave], m(a), pc(a / B24["activos"]), m(b), pc(b / B25["activos"]),
                   (v, tono) if tono else v], cls=cls, p=p)


LAMINAS = [
    dict(
        tipo="seccion", paso=3, etiqueta="Paso 3", nombre="Entiende la estructura y la tendencia",
        bajada="Análisis vertical y horizontal, en un solo prompt",
        pasos=[
            "Paso 3: entender la estructura y la tendencia. Son las dos lecturas clásicas del análisis financiero: "
            "el análisis vertical y el análisis horizontal. Con Claude, las tienes en un solo prompt.",
        ],
    ),
    dict(
        tipo="dos", titulo="Dos lecturas que se complementan", une=True,
        paneles=[
            dict(p=0, clase="azul", icono="capas", titulo="Vertical: la estructura", icono_item="flecha", items=[
                dict(t="Estado de resultados", d="Cada línea como % de las ventas"),
                dict(t="Balance", d="Cada cuenta como % del total de activos")]),
            dict(p=1, clase="oro", icono="sube", titulo="Horizontal: la tendencia", icono_item="flecha", items=[
                dict(t="De un período a otro", d="Cuánto cambió cada cuenta"),
                dict(t="En monto y en porcentaje", d="Para ver tamaño y velocidad")]),
        ],
        pie=dict(p=2, t="La pregunta clave: ==¿qué creció más rápido que las ventas?=="),
        pasos=[
            "Un repaso rápido. El análisis vertical mira la estructura. En el estado de resultados, expresa cada "
            "línea como porcentaje de las ventas: de cada 100 pesos que vendo, cuánto se va en costo, cuánto en "
            "gastos y cuánto queda. En el balance, expresa cada cuenta como porcentaje del total de activos.",
            "El análisis horizontal mira la tendencia: cuánto cambió cada cuenta de un período a otro, en monto y "
            "en porcentaje.",
            "Por separado dicen poco. Juntos responden una pregunta muy útil: qué creció más rápido que las ventas. "
            "Porque lo que crece más rápido que las ventas se está comiendo el margen o la caja.",
        ],
    ),
    dict(
        tipo="prompt", titulo="Prompt 4: análisis vertical y horizontal", numero=4, nombre="Estructura y tendencia",
        segs=[
            ("Haz un análisis vertical y horizontal de los dos períodos.\n\n", None),
            ("1. Estado de resultados: cada línea como % de las ventas,\n"
             "   y su variación en monto y en %.\n"
             "2. Balance: cada cuenta como % del total de activos,\n"
             "   y su variación en monto y en %.\n", 0),
            ("3. Marca las partidas que crecieron MÁS que las ventas\n"
             "   y explica qué implica cada una.\n", 1),
            ("4. Termina con los 3 hallazgos más relevantes, ordenados\n"
             "   por impacto en pesos.\n", 2),
            ("\nEntrega tablas y calcula con código.", None),
        ],
        notas=[
            dict(p=0, t="Las dos lecturas", d="Para el estado de resultados y para el balance"),
            dict(p=1, t="Más rápido que las ventas", d="Ahí se pierde margen o caja"),
            dict(p=2, t="3 hallazgos, por impacto", d="Evita la lista de 20 observaciones"),
        ],
        pasos=[
            "El prompt pide las dos lecturas, para el estado de resultados y para el balance.",
            "Pero lo valioso está en los puntos 3 y 4. En el 3 le pido que marque las partidas que crecieron más "
            "que las ventas, y que explique qué implica cada una.",
            "Y en el 4, que termine con los tres hallazgos más relevantes, ordenados por impacto en pesos. Esa "
            "instrucción evita el problema típico de estos análisis: una lista de 20 observaciones donde todo "
            "parece igual de importante.",
        ],
    ),
    dict(
        tipo="tabla", titulo="Resultados: vende más, pero gana menos por cada peso", compacta=True, flex=2.35,
        alto=True,
        columnas=["Millones de pesos", "2024", "% vtas.", "2025", "% vtas.", "Var."],
        clases_col=["", "", "pct", "", "pct", ""],
        filas=[
            _er("ventas", p=0),
            _er("costo", p=1, tono="neg"),
            _er("margen_bruto", cls="total", p=1),
            _er("gav", etq="Gastos de adm. y ventas", p=2),
            _er("distribucion", etq="Distribución y fletes", cls="sub", p=2, tono="neg"),
            _er("depreciacion"),
            _er("ebit", cls="total", p=4),
            _er("gastos_fin", p=3, tono="neg"),
            _er("impuesto"),
            _er("utilidad", cls="total", p=4, tono="amb"),
        ],
        notas=[
            dict(p=1, gr="−1 punto", tono="neg", t="de margen bruto", d="118 millones sobre las ventas de 2025"),
            dict(p=2, gr="+30 %", tono="neg", t="Distribución y fletes"),
            dict(p=3, gr="+59 %", tono="neg", t="Gastos financieros", d="La partida que más creció"),
            dict(p=4, gr="4,7 % → 4,1 %", t="Margen neto"),
        ],
        pasos=[
            "Veamos el estado de resultados de Alerce. Las ventas crecieron 18 %.",
            "Pero el costo de ventas creció 19,6 %. Por eso el margen bruto bajó de 26 a 25 % de las ventas. Un "
            "punto parece poco. Sobre 11.800 millones de ventas, son 118 millones.",
            "Los gastos de administración y ventas crecieron menos que las ventas, lo que es bueno. Salvo una "
            "línea: distribución y fletes, que subió 30 %.",
            "Y más abajo, los gastos financieros crecieron 59 %. Es la partida que más creció de todo el estado.",
            "Resultado: las ventas subieron 18 %, el resultado operacional 12 %, y la utilidad neta apenas 3,9 %. "
            "La empresa vende mucho más, pero en cada peso de venta gana menos: el margen neto bajó de 4,7 a 4,1 %.",
        ],
    ),
    dict(
        tipo="respuesta", titulo="El hábito más rentable: repreguntar", etiqueta="Respuesta de Claude",
        previo=dict(p=0, t="El margen bruto bajó 1 punto. Dame las 3 hipótesis más probables y, para cada una, "
                           "el dato exacto que necesitas para confirmarla o descartarla."),
        bloques=[
            dict(p=1, h="1 · Mezcla de clientes", html="<p>Los clientes nuevos compran con descuento. "
                                                       "<b>Dato:</b> margen por cliente.</p>"),
            dict(p=2, h="2 · Costos no traspasados a precios", html="<p>El costo subió más que el precio de venta. "
                                                                    "<b>Dato:</b> costo y precio promedio por familia de productos.</p>"),
            dict(p=3, h="3 · Mezcla de productos", html="<p>Se vende más de las líneas de menor margen. "
                                                        "<b>Dato:</b> ventas y margen por línea.</p>"),
        ],
        notas=[
            dict(p=0, t="La tabla dice qué pasó", d="No dice por qué"),
            dict(p=4, t="No adivina la causa", d="Te entrega una lista ordenada de qué ir a buscar"),
        ],
        pasos=[
            "Y aquí viene un hábito que quiero que te lleves: repreguntar. La tabla dice que el margen bajó un "
            "punto, pero no dice por qué. Entonces le pregunto: dame las tres hipótesis más probables y, para cada "
            "una, el dato exacto que necesitas para confirmarla o descartarla.",
            "Primera hipótesis: la mezcla de clientes. Los clientes nuevos compran con descuento. Dato necesario: "
            "el margen por cliente.",
            "Segunda: un aumento de costos que no se traspasó a precios. Dato: la evolución del costo y del precio "
            "promedio, por familia de productos.",
            "Tercera: la mezcla de productos. Se está vendiendo más de las líneas de menor margen. Dato: ventas y "
            "margen por línea.",
            "Fíjate que Claude no adivina la causa. Te entrega una lista ordenada de qué ir a buscar. En el caso de "
            "Alerce, como sabemos por el contexto que entraron dos cadenas con descuento, la primera hipótesis es "
            "la candidata. La vamos a confirmar en el paso 5.",
        ],
    ),
    dict(
        tipo="tabla", titulo="Balance: aquí está la historia", compacta=True, flex=2.35, alto=True,
        columnas=["Millones de pesos", "2024", "% act.", "2025", "% act.", "Var."],
        clases_col=["", "", "pct", "", "pct", ""],
        filas=[
            _bal("efectivo", p=4, tono="neg"),
            _bal("cxc", p=1, tono="neg"),
            _bal("inventarios", p=2, tono="neg"),
            _bal("otros_ac"),
            _bal("ppe", etq="Activo fijo"),
            _bal("activos", cls="total"),
            _bal("proveedores", p=3),
            _bal("deuda_cp", etq="Deuda bancaria de corto plazo", p=3, tono="neg"),
            _bal("otros_pc"),
            _bal("deuda_lp", etq="Deuda bancaria de largo plazo"),
            _bal("patrimonio", etq="Patrimonio", p=5),
        ],
        notas=[
            dict(p=1, gr="+70 %", tono="neg", t="Cuentas por cobrar", d="Hoy son el 38 % de los activos"),
            dict(p=2, gr="+48 %", tono="neg", t="Inventarios"),
            dict(p=3, gr="×2,9", tono="neg", t="Deuda de corto plazo", d="Los proveedores, solo +15 %"),
            dict(p=4, gr="12 % → 1 %", tono="neg", t="Caja sobre activos"),
        ],
        pasos=[
            "Ahora el balance, que es donde está la historia. Recuerda: las ventas crecieron 18 %.",
            "Las cuentas por cobrar crecieron 70 %. Pasaron de 1.370 a 2.330 millones, y hoy son el 38 % de todos "
            "los activos.",
            "Los inventarios crecieron 48 %.",
            "¿Y cómo se financió ese crecimiento? Los proveedores aumentaron solo 15 %. La diferencia la puso el "
            "banco: la deuda de corto plazo pasó de 400 a 1.150 millones, casi el triple.",
            "Y la caja pagó el resto: de ser el 12 % de los activos, hoy es el 1 %.",
            "Mira también la última fila. El patrimonio pasó de financiar el 50 % de los activos a financiar el "
            "46 %. La empresa depende cada vez más de recursos de terceros.",
        ],
    ),
    dict(
        tipo="html", titulo="Los 3 hallazgos, ordenados por impacto",
        html=rank([
            dict(p=0, n=1, t="Cuentas por cobrar e inventarios", d="Capital de trabajo inmovilizado: +960 y +535", v="1.495", unidad="millones"),
            dict(p=1, n=2, t="Financiamiento", d="Deuda de corto plazo +750 · gastos financieros +59 %", v="750", unidad="millones"),
            dict(p=2, n=3, t="Margen y fletes", d="1 punto menos de margen bruto (118) · fletes +130", v="248", unidad="millones"),
        ]),
        pie=dict(p=3, t="Hipótesis: creció dando más plazo, cargando más inventario y cediendo margen."),
        pasos=[
            "Estos son los tres hallazgos que entrega Claude, ordenados por impacto. Uno: el capital de trabajo. "
            "Entre cuentas por cobrar e inventarios, la empresa inmovilizó casi 1.500 millones de pesos adicionales.",
            "Dos: el financiamiento. Ese crecimiento se pagó con 750 millones de deuda de corto plazo, y por eso los "
            "gastos financieros subieron 59 %.",
            "Y tres: el margen. Un punto menos de margen bruto equivale a 118 millones, y los fletes subieron "
            "130 millones más.",
            "Ya tenemos una hipótesis sólida: la empresa creció dando más plazo, cargando más inventario y cediendo "
            "margen. Ahora hay que medirlo.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Paso 3 en resumen", estilo="check",
        items=[
            dict(p=0, t="Vertical: la estructura. Horizontal: la tendencia", d="Las dos lecturas, en un solo prompt"),
            dict(p=1, t="Busca lo que crece más rápido que las ventas", d="Ahí se pierde margen o caja"),
            dict(p=2, t="Pide 3 hallazgos por impacto, y repregunta", d="Hipótesis, con el dato para confirmarlas"),
        ],
        pasos=[
            "El paso 3, en resumen. El análisis vertical muestra la estructura, y el horizontal, la tendencia.",
            "Busca siempre lo que crece más rápido que las ventas.",
            "Y pide tres hallazgos ordenados por impacto. Después, repregunta por las causas.",
        ],
    ),
]
