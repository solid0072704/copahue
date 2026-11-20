"""Paso 5: flujo de efectivo y capital de trabajo."""

from .. import caso, graficos
from ._util import m, pc

SEG = caso.segmentos()
CAD, TRA = SEG["Cadenas nuevas (2 clientes)"], SEG["Clientes tradicionales"]
R25, B25, X25 = caso.eerr("2025"), caso.balance("2025"), caso.ratios("2025")
CC = caso.costo_de_crecer()
F25 = caso.flujo_2025()
A24, A25 = caso.antiguedad()["2024"], caso.antiguedad()["2025"]

LAMINAS = [
    dict(
        tipo="seccion", paso=5, etiqueta="Paso 5", nombre="Sigue la caja",
        bajada="Una empresa sobrevive años con poca utilidad, pero ni un mes sin caja",
        pasos=[
            "Paso 5: seguir la caja. Si tuviera que quedarme con un solo paso del método, sería este. Porque una "
            "empresa puede sobrevivir varios años con poca utilidad, pero no sobrevive ni un mes sin caja.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Por qué la utilidad no es caja", une=True,
        items=[
            dict(p=0, t="Vendes hoy, cobras después", d="La venta ya está en el resultado; la plata aún no está en el banco"),
            dict(p=1, t="Compras lo que todavía no vendes", d="Lo pagaste, pero no es costo hasta que lo vendes"),
            dict(p=2, t="Hay pagos que no pasan por el resultado", d="Inversiones, cuotas de créditos y dividendos"),
        ],
        lateral=[dict(p=3, clase="tinta", eti="El flujo de efectivo lo ordena en",
                      ps=["**1.** La operación", "**2.** La inversión", "**3.** El financiamiento"], ancho=500)],
        pasos=[
            "La utilidad y la caja son cosas distintas, por tres razones principales. Una: vendes hoy, pero cobras "
            "después. La venta ya está en el estado de resultados; la plata todavía no está en el banco.",
            "Dos: compras mercadería que todavía no vendes. La pagaste, pero no es costo hasta que la vendes.",
            "Y tres: hay pagos que no pasan por el resultado, como las inversiones, las cuotas de los créditos y "
            "los dividendos.",
            "El estado de flujo de efectivo ordena todo esto en tres bloques: la operación, la inversión y el "
            "financiamiento.",
        ],
    ),
    dict(
        tipo="prompt", titulo="Prompt 6: qué pasó con la caja", numero=6, nombre="Flujo de efectivo y capital de trabajo",
        alto=True,
        segs=[
            ("Explícame qué pasó con la caja.\n\n", None),
            ("1. Construye el flujo de efectivo del año por el método\n"
             "   indirecto, partiendo de la utilidad neta. Si te entregué\n"
             "   el flujo, concílialo con el balance.\n", 1),
            ("2. Muestra un puente: utilidad neta → flujo de la\n"
             "   operación → variación del efectivo.\n", 2),
            ("3. Cuantifica cuánta caja absorbió cada componente del\n"
             "   capital de trabajo.\n", 3),
            ("4. Calcula el flujo de caja libre.\n", 4),
            ("5. Responde en 5 líneas: ¿por qué la utilidad no se\n"
             "   transformó en caja?", 5),
        ],
        notas=[
            dict(p=1, t="¿No tienes flujo de efectivo?", d="Claude lo arma con dos balances y un estado de resultados"),
            dict(p=2, t="Un puente", d="De la utilidad a la caja, paso a paso"),
            dict(p=3, t="Capital de trabajo", d="Clientes, inventario y proveedores"),
            dict(p=5, t="La respuesta en 5 líneas", d="Directa y sin rodeos"),
        ],
        pasos=[
            "El prompt de la caja.",
            "Primero le pido que construya el flujo de efectivo por el método indirecto, partiendo de la utilidad. "
            "Si no tienes flujo de efectivo, y muchas empresas pequeñas no lo preparan, Claude lo arma con los dos "
            "balances y el estado de resultados.",
            "Segundo, un puente desde la utilidad hasta la variación del efectivo.",
            "Tercero, cuánta caja absorbió cada componente del capital de trabajo.",
            "Cuarto, el flujo de caja libre.",
            "Y quinto, la respuesta directa, en cinco líneas.",
        ],
    ),
    dict(
        tipo="svg", titulo="De la utilidad a la caja: el puente de 2025", alto=True,
        svg=graficos.puente_caja(),
        pasos=[
            "Este es el puente de Alerce. Partimos de la utilidad: 485 millones.",
            "Sumamos la depreciación, 170 millones, porque es un gasto que no significó salida de plata.",
            "Y ahora el capital de trabajo. Las cuentas por cobrar aumentaron 960 millones: son ventas ya "
            "facturadas que todavía no se cobran. Esa plata la tienen los clientes.",
            "Los inventarios aumentaron 535 millones: plata que está en la bodega.",
            "Los proveedores financiaron una parte, 151 millones.",
            "Resultado: el flujo de la operación fue negativo en 685 millones. Una empresa que ganó 485 millones "
            "quemó 685 en su operación. El año anterior, la operación había generado 510.",
            "Sigamos. Invirtió 320 millones en activo fijo y pagó 200 millones en dividendos.",
            "¿Cómo lo cubrió? Con 650 millones de deuda bancaria adicional, y con su propia caja, que bajó en "
            "555 millones.",
        ],
    ),
    dict(
        tipo="cifras", titulo="Flujo de caja libre: lo que realmente queda disponible", cols=3,
        sobre=dict(p=0, t="**Flujo de caja libre** = flujo de la operación − inversión · lo disponible para pagar deuda y repartir dividendos"),
        tiles=[
            dict(p=1, eti="Flujo de la operación", val=m(F25["operacional"]), sub="millones de pesos", tono="neg"),
            dict(p=1, eti="Inversión en activo fijo", val=m(F25["capex"]), sub="millones de pesos"),
            dict(p=2, eti="Flujo de caja libre", val=m(F25["flujo_libre"]), sub="millones de pesos", tono="neg"),
        ],
        pie=dict(p=3, t="No había nada que repartir. Y aun así se repartieron 200 millones."),
        pasos=[
            "Un concepto que conviene tener claro: el flujo de caja libre. Es lo que queda del flujo de la "
            "operación después de invertir. Es la plata realmente disponible para pagar deuda y para repartir "
            "dividendos.",
            "En Alerce: una operación que consumió 685 millones, y 320 millones de inversión.",
            "Flujo de caja libre: menos 1.005 millones.",
            "No había nada que repartir. Y aun así se repartieron 200 millones, financiados con deuda.",
        ],
    ),
    dict(
        tipo="respuesta", titulo="¿Por qué la utilidad no se transformó en caja?",
        bloques=[
            dict(p=0, h="La respuesta, en 5 líneas", html=(
                "<ol>"
                '<li data-p="0">La utilidad quedó atrapada en <b>capital de trabajo</b>: cuentas por cobrar e inventarios crecieron 1.495 millones, y los proveedores financiaron solo 151.</li>'
                '<li data-p="1">La operación <b>consumió 685 millones</b>; en 2024 había generado 510.</li>'
                '<li data-p="2">El flujo de caja libre fue de <b class="neg">−1.005 millones</b>.</li>'
                '<li data-p="3">Se cubrió con deuda de corto plazo (+750) y con la caja (−555).</li>'
                '<li data-p="4">Aun así, se repartieron <b>200 millones en dividendos</b>.</li>'
                "</ol>")),
        ],
        notas=[
            dict(p=2, gr="−1.005", tono="neg", t="Flujo de caja libre", d="Operación −685 e inversión −320"),
            dict(p=5, t="La línea que incomoda", d="Es la que un buen analista pone sobre la mesa", clase="rojo"),
        ],
        pasos=[
            "Y esta es la respuesta en cinco líneas. La utilidad quedó atrapada en capital de trabajo: las cuentas "
            "por cobrar y los inventarios crecieron 1.495 millones, y los proveedores financiaron solo 151.",
            "La operación consumió 685 millones, cuando el año anterior había generado 510.",
            "El flujo de caja libre, después de inversiones, fue negativo en 1.005 millones.",
            "Eso se cubrió con deuda de corto plazo y con la caja que había.",
            "Y aun así, la empresa repartió 200 millones en dividendos.",
            "Esa última línea es la que más incomoda, y es exactamente el tipo de frase que un buen analista tiene "
            "que poner sobre la mesa.",
        ],
    ),
    dict(
        tipo="cifras", titulo="Cuánta caja cuesta crecer", cols=3,
        tiles=[
            dict(p=0, eti="Ventas adicionales", val="+" + m(CC["var_ventas"]), sub="millones de pesos"),
            dict(p=1, eti="Capital de trabajo extra", val="+" + m(CC["var_kt"]), sub="millones, neto de proveedores", tono="neg"),
            dict(p=2, eti="Por cada 100 de venta nueva", val=m(CC["por_cada_100"]) + " <small>pesos</small>", sub="de caja", tono="neg"),
        ],
        pie=dict(p=3, t="Con un margen neto de 4,1 %, esa caja equivale a unas 18 veces la utilidad anual de la venta nueva."),
        sub_pie=dict(p=4, t="Crecer no es gratis: con más plazo y más inventario, puede dejarte sin caja"),
        pasos=[
            "Con estos números se puede calcular algo que pocas empresas miden: cuánta caja cuesta crecer. En 2025, "
            "Alerce vendió 1.800 millones más que el año anterior.",
            "Y para lograrlo, tuvo que inmovilizar 1.344 millones adicionales en capital de trabajo, ya descontado "
            "lo que financiaron los proveedores.",
            "Es decir, cada 100 pesos de venta nueva le exigieron 75 pesos de caja.",
            "Esa misma venta, con un margen neto de 4,1 %, deja unos 4 pesos de utilidad al año. O sea: la caja que "
            "exige el crecimiento equivale a unas 18 veces la utilidad anual que genera.",
            "Este es el concepto que quiero que te lleves del caso: crecer no es gratis. Y cuando el crecimiento "
            "viene con más plazo y más inventario, puede ser la forma más rápida de quedarse sin caja.",
        ],
    ),
    dict(
        tipo="tabla", titulo="La siguiente pregunta: ¿de qué clientes vino el crecimiento?",
        previo=dict(p=0, t="Cruza ventas, margen y cuentas por cobrar por tipo de cliente. ¿Dónde se concentran el "
                           "mayor plazo de cobro y el menor margen?"),
        columnas=["Tipo de cliente", "Ventas", "% del total", "Margen bruto", "Por cobrar", "Días de cobro"],
        filas=[
            dict(c=["Cadenas nuevas (2 clientes)", m(CAD["ventas"]), pc(CAD["pct_ventas"], 0), (pc(CAD["margen_pct"]), "neg"),
                    m(CAD["cxc"]), (m(CAD["dias_cobro"]), "neg")], p="1,2", aparece=1),
            dict(c=["Clientes tradicionales", m(TRA["ventas"]), pc(TRA["pct_ventas"], 0), pc(TRA["margen_pct"]),
                    m(TRA["cxc"]), m(TRA["dias_cobro"])], p=3, aparece=1),
            dict(c=["Total", m(R25["ventas"]), "100 %", pc(X25["margen_bruto"]), m(B25["cxc"]), m(X25["dias_cobro"])],
                 cls="total", aparece=1),
        ],
        pie=dict(p=4, t="No es un error venderles: hay que saber cuánto cuesta, y decidir con ese número a la vista."),
        pasos=[
            "Con Claude, cada respuesta abre la siguiente pregunta. Le subí el detalle de ventas y cuentas por "
            "cobrar por cliente, y le pedí que cruzara margen y plazo de cobro por tipo de cliente.",
            "El resultado es muy claro. Las dos cadenas nuevas explican todo el crecimiento del año: 1.800 "
            "millones, el 15 % de las ventas. Pero dejan un margen de 19,4 %, contra 26 % del resto de los clientes.",
            "Y pagan a 100 días, contra 67. O sea: todo el crecimiento vino de los clientes menos rentables y más "
            "lentos para pagar.",
            "Y un dato más: incluso los clientes tradicionales se están demorando más que antes, de 50 a 67 días. "
            "Hay un problema de cobranza que va más allá de las cadenas.",
            "Todo esto no significa que venderles a las cadenas sea un error. Significa que hay que saber cuánto "
            "cuesta, y decidir con ese número a la vista.",
        ],
    ),
    dict(
        tipo="tabla", titulo="Otra repregunta: la antigüedad de las cuentas por cobrar", compacta=True, flex=2.0, alto=True,
        previo=dict(p=0, t="Analiza la antigüedad de las cuentas por cobrar. ¿Qué parte del aumento es crecimiento y "
                           "qué parte es atraso?"),
        columnas=["Millones de pesos", "2024", "%", "2025", "%"],
        clases_col=["", "", "pct", "", "pct"],
        filas=[
            dict(c=["Por vencer", m(A24["tramos"]["Por vencer"]), pc(A24["pct_al_dia"]), m(A25["tramos"]["Por vencer"]),
                    pc(A25["pct_al_dia"])], p=1, aparece=1),
            *[dict(c=[k, m(A24["tramos"][k]), pc(A24["tramos"][k] / A24["total"]), m(A25["tramos"][k]),
                      pc(A25["tramos"][k] / A25["total"])], p=("2,3" if k in ("61 a 90 días", "Más de 90 días") else 2), aparece=1)
              for k in ("1 a 30 días", "31 a 60 días", "61 a 90 días", "Más de 90 días")],
            dict(c=["Total cuentas por cobrar", m(A24["total"]), "100 %", m(A25["total"]), "100 %"], cls="total", aparece=1),
        ],
        notas=[
            dict(p=1, gr="73 % → 55 %", t="Cartera al día"),
            dict(p=2, gr="×2,8", tono="neg", t="Cartera vencida", d="De 370 a 1.050 millones"),
            dict(p=4, gr="680 de 960", tono="neg", t="millones del aumento son atraso", clase="rojo"),
        ],
        pasos=[
            "Otra repregunta que siempre vale la pena cuando las cuentas por cobrar crecen: pedir el análisis de "
            "antigüedad. Le subí el detalle por tramo de vencimiento, y le pregunté qué parte del aumento es "
            "crecimiento del negocio, y qué parte es atraso.",
            "En 2024, el 73 % de la cartera estaba al día. En 2025, solo el 55 %.",
            "Lo vencido pasó de 370 a 1.050 millones: casi se triplicó.",
            "Y el tramo más delicado, lo vencido a más de 60 días, pasó de 40 a 240 millones.",
            "La conclusión de Claude: de los 960 millones que aumentaron las cuentas por cobrar, 680 corresponden a "
            "cartera vencida. No es solo que la empresa venda más. Es que le están pagando más tarde.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Tres preguntas para el capital de trabajo",
        items=[
            dict(p=0, icono="personas", t="Clientes: ¿a quién le doy plazo, y por qué?",
                 d="Plazo real de pago por cliente, contra el plazo pactado"),
            dict(p=1, icono="caja", t="Inventarios: ¿qué no está rotando?", d="Días de inventario por producto o por familia"),
            dict(p=2, icono="camion", t="Proveedores: ¿uso el plazo que me dan?", d="Días de pago reales, contra las condiciones negociadas"),
        ],
        pasos=[
            "Cuando el capital de trabajo es el problema, hay tres preguntas que le puedes hacer a Claude, una por "
            "cada componente. Para los clientes: ¿a quién le estoy dando plazo, y por qué? Pídele el plazo real de "
            "pago por cliente, comparado con el plazo pactado.",
            "Para los inventarios: ¿qué no está rotando? Pídele los días de inventario por producto, o por familia.",
            "Y para los proveedores: ¿estoy usando el plazo que me dan? Alerce paga a 48 días. Si sus condiciones "
            "fueran a 60, estaría financiando a sus proveedores sin necesidad.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Paso 5 en resumen", estilo="check",
        items=[
            dict(p=0, t="La utilidad no es caja", d="Y la caja es la que paga los sueldos"),
            dict(p=1, t="Pide el puente", d="De la utilidad, a la operación, a la variación de la caja"),
            dict(p=2, t="Repregunta por cliente y por antigüedad", d="Ahí aparece quién se está llevando la caja"),
        ],
        pasos=[
            "El paso 5, en resumen. La utilidad no es caja, y la caja es la que paga los sueldos.",
            "Pide siempre el puente, desde la utilidad hasta la variación de la caja.",
            "Y repregunta por cliente y por antigüedad, porque ahí aparece quién se está llevando la caja.",
        ],
    ),
]
