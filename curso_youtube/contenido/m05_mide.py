"""Paso 4: ratios financieros."""

from .. import caso, graficos
from ._util import m, pc, ratio, semaforo

X24, X25 = caso.ratios("2024"), caso.ratios("2025")
V = caso.valor_de_un_dia()


def _veces(clave, dec=2):
    return (m(X24[clave], dec), "v0"), (m(X25[clave], dec), "v")


def _pct(clave, dec=1):
    return (pc(X24[clave], dec), "v0"), (pc(X25[clave], dec), "v")


def _fila(nombre, formula, valores, color, p=None):
    return dict(c=[ratio(nombre, formula), valores[0], valores[1], semaforo(color)], p=p)


COLS = ["Ratio", "2024", "2025", ""]

LAMINAS = [
    dict(
        tipo="seccion", paso=4, etiqueta="Paso 4", nombre="Mide con ratios",
        bajada="Liquidez, endeudamiento, rentabilidad y eficiencia",
        pasos=[
            "Paso 4: medir con ratios. Los ratios convierten los estados financieros en indicadores comparables: "
            "con el año anterior, con tu presupuesto y con tu industria. Los vamos a ordenar en cuatro familias.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Cuatro familias, cuatro preguntas", columnas=2, une=True,
        items=[
            dict(p=0, icono="peso", t="Liquidez", d="¿Puedo pagar lo que debo en el corto plazo?"),
            dict(p=1, icono="banco", t="Endeudamiento", d="¿Cuánto debo, y puedo sostenerlo?"),
            dict(p=2, icono="sube", t="Rentabilidad", d="¿Cuánto gano por cada peso vendido e invertido?"),
            dict(p=3, icono="reloj", t="Eficiencia", d="¿Cuánto tardo en convertir las ventas en caja?"),
        ],
        pie=dict(p=4, t="Un ratio aislado no dice nada: importan la tendencia y la comparación."),
        pasos=[
            "Liquidez: ¿puedo pagar lo que debo en el corto plazo?",
            "Endeudamiento: ¿cuánto debo, y puedo sostenerlo?",
            "Rentabilidad: ¿cuánto gano por cada peso que vendo, y por cada peso invertido?",
            "Y eficiencia: ¿cuánto tardo en convertir mis ventas en caja?",
            "Un consejo antes de seguir: un ratio aislado no significa nada. Lo que importa es la tendencia y la "
            "comparación. Una liquidez de 1,5 puede ser excelente en un supermercado, y preocupante en una "
            "constructora.",
        ],
    ),
    dict(
        tipo="puntos", titulo="¿Contra qué comparo un ratio?", columnas=2,
        items=[
            dict(p=0, icono="reloj", t="Contra tu propia historia", d="La tendencia de los últimos años o de 12 meses"),
            dict(p=1, icono="diana", t="Contra tu presupuesto", d="Lo que te propusiste lograr"),
            dict(p=2, icono="personas", t="Contra tu industria", d="Con cuidado: tamaños y modelos distintos"),
            dict(p=3, icono="banco", t="Contra tus compromisos", d="Los indicadores que exigen tus créditos"),
        ],
        pasos=[
            "Una pregunta que siempre aparece: ¿contra qué comparo un ratio? Tienes cuatro referencias. Tu propia "
            "historia: la tendencia de los últimos años, o de los últimos 12 meses. Es la más útil, y la que siempre "
            "tienes.",
            "Tu presupuesto: lo que te propusiste lograr.",
            "Tu industria, con cuidado, porque las empresas tienen tamaños y modelos de negocio distintos.",
            "Y tus compromisos. Si tus créditos te exigen cumplir ciertos indicadores, pídele a Claude que los "
            "calcule todos los meses, y que te diga cuánta holgura te queda.",
        ],
    ),
    dict(
        tipo="prompt", titulo="Prompt 5: ratios con fórmula, interpretación y semáforo", numero=5, nombre="Ratios financieros",
        segs=[
            ("Calcula los ratios financieros de ambos períodos, en 4\n"
             "grupos: liquidez, endeudamiento, rentabilidad y eficiencia.\n\n", 0),
            ("Para cada ratio entrega:\n"
             "- La fórmula y las cifras que reemplazaste.\n"
             "- El valor de cada período y su variación.\n"
             "- Una interpretación en una línea, en lenguaje simple.\n"
             "- Un semáforo: verde, amarillo o rojo, con el criterio.\n\n", 1),
            ("Incluye el ciclo de conversión de efectivo y el análisis\n"
             "DuPont.\n", 2),
            ("Al final, dime cuáles son los 3 ratios que más me deberían\n"
             "preocupar y por qué.", 3),
        ],
        notas=[
            dict(p=1, t="La fórmula a la vista", d="Para revisar cualquier cálculo"),
            dict(p=2, t="Lo que nadie hace a mano", d="Ciclo de efectivo y DuPont"),
            dict(p=3, t="Los 3 que preocupan", d="Otra vez: priorizar"),
            dict(p=4, t="Ojo con las definiciones", d="Si usas otra fórmula, se la indicas"),
        ],
        pasos=[
            "En el prompt le pido los cuatro grupos.",
            "Y para cada ratio, cuatro cosas. La fórmula con las cifras reemplazadas, para poder revisarla. El "
            "valor de cada período con su variación. Una interpretación en una línea, en lenguaje simple. Y un "
            "semáforo, explicando el criterio.",
            "Agrego dos análisis que casi nunca se hacen a mano, porque toman tiempo: el ciclo de conversión de "
            "efectivo y el análisis DuPont.",
            "Y cierro igual que antes: los tres ratios que más me deberían preocupar.",
            "Un detalle importante: hay ratios que tienen más de una fórmula. Por ejemplo, la rentabilidad sobre el "
            "patrimonio se puede calcular con el patrimonio final o con el promedio. Por eso pido siempre la "
            "fórmula. Si tu empresa usa una definición distinta, se la indicas y recalcula.",
        ],
    ),
    dict(
        tipo="html", titulo="¿Cómo define el semáforo?",
        html=('<div class="formato">'
              '<div class="fm" data-p="0"><div class="st"><span class="semaforo v gr"></span></div><div><b>Verde</b>'
              '<span>El indicador mejora, o se mantiene en un nivel sano</span></div></div>'
              '<div class="fm" data-p="1"><div class="st"><span class="semaforo a gr"></span></div><div><b>Amarillo</b>'
              '<span>Empeora, pero no compromete los pagos</span></div></div>'
              '<div class="fm" data-p="2"><div class="st"><span class="semaforo r gr"></span></div><div><b>Rojo</b>'
              '<span>Empeora, y compromete los pagos de los próximos meses</span></div></div></div>'),
        notas=[dict(p=3, t="El criterio puede ser el tuyo", d="Escribe tus umbrales en las instrucciones del proyecto")],
        pasos=[
            "Una aclaración sobre el semáforo. Como le pedí el criterio, Claude lo explica: verde, si el indicador "
            "mejora o se mantiene en un nivel sano.",
            "Amarillo, si empeora, pero no compromete los pagos.",
            "Y rojo, si empeora y además compromete los pagos de los próximos meses.",
            "Ese criterio es suyo, y es razonable. Pero el tuyo puede ser otro. Si tu empresa tiene umbrales "
            "propios, por ejemplo una liquidez mínima, escríbelos en las instrucciones del proyecto, y el semáforo "
            "va a usar tus reglas.",
        ],
    ),
    dict(
        tipo="tabla", titulo="Liquidez: se ve sana… hasta que miras qué hay dentro", ratios=True, flex=1.7,
        columnas=COLS,
        filas=[
            _fila("Liquidez corriente", "Activos corrientes ÷ pasivos corrientes", _veces("liquidez_corriente"), "a", p="0,1"),
            _fila("Prueba ácida", "(Activos corrientes − inventarios) ÷ pasivos corrientes", _veces("prueba_acida"), "a", p=2),
            _fila("Razón de caja", "Efectivo ÷ pasivos corrientes", _veces("razon_caja"), "r", p=3),
        ],
        notas=[
            dict(p=1, t="Parece sana", d="1,65 pesos de activo corriente por cada peso que debe"),
            dict(p=3, gr="3 de cada 100", tono="neg", t="pesos que debe pagar, en la cuenta", clase="rojo"),
            dict(p=4, t="La lección", d="La liquidez corriente esconde el problema cuando es puro crédito e inventario"),
        ],
        pasos=[
            "Partamos por la liquidez. La liquidez corriente, activos corrientes sobre pasivos corrientes, bajó de "
            "2 a 1,65.",
            "Si miras solo ese número, dirías que la empresa está sana: tiene 1,65 pesos de activos de corto plazo "
            "por cada peso que debe. Pero ojo con lo que hay dentro de esos activos.",
            "La prueba ácida, que saca los inventarios, cayó de 1,3 a 0,99.",
            "Y la razón de caja, que considera solo el efectivo, cayó de 0,39 a 0,03. Es decir, por cada 100 pesos "
            "que la empresa debe pagar en los próximos meses, tiene 3 en la cuenta.",
            "Esta es una de las lecciones del caso: la liquidez corriente puede verse bien y esconder un problema "
            "serio, cuando está llena de cuentas por cobrar e inventario.",
        ],
    ),
    dict(
        tipo="tabla", titulo="Endeudamiento: no solo debe más, debe a menos plazo", ratios=True, flex=1.7, alto=True,
        columnas=COLS,
        filas=[
            _fila("Endeudamiento", "Pasivos ÷ patrimonio", _veces("endeudamiento"), "a", p=0),
            _fila("Deuda financiera neta sobre EBITDA", "(Deuda bancaria − efectivo) ÷ EBITDA", _veces("dfn_ebitda"), "a", p=1),
            _fila("Cobertura de intereses", "Resultado operacional ÷ gastos financieros", _veces("cobertura_intereses", 1), "a", p=2),
            _fila("Deuda de corto plazo", "Deuda de corto plazo ÷ deuda bancaria total", _pct("deuda_cp_pct", 0), "r", p=3),
        ],
        notas=[
            dict(p=1, gr="×2,5", tono="amb", t="Deuda neta sobre EBITDA", d="En un solo año"),
            dict(p=3, gr="59 %", tono="neg", t="de la deuda vence en menos de un año",
                 d="Depende de que el banco renueve la línea", clase="rojo"),
        ],
        pasos=[
            "Endeudamiento. La relación entre pasivos y patrimonio subió de 1 a 1,19: no es alarmante.",
            "La deuda financiera neta sobre EBITDA pasó de 0,76 a 1,87 veces. Tampoco es un nivel crítico, pero se "
            "multiplicó por dos y medio en un año.",
            "La cobertura de intereses bajó de 6,8 a 4,8 veces.",
            "Y el dato que más me importa: la deuda de corto plazo pasó de ser el 31 % de la deuda bancaria a ser "
            "el 59 %. La empresa no solo debe más: debe a menos plazo. Depende de que el banco le renueve la línea.",
        ],
    ),
    dict(
        tipo="tabla", titulo="Rentabilidad: los tres márgenes bajan", ratios=True, flex=1.7, alto=True,
        columnas=COLS,
        filas=[
            _fila("Margen bruto", "Margen bruto ÷ ventas", _pct("margen_bruto"), "a", p=0),
            _fila("Margen operacional", "Resultado operacional ÷ ventas", _pct("margen_operacional"), "a", p=0),
            _fila("Margen neto", "Utilidad neta ÷ ventas", _pct("margen_neto"), "a", p=0),
            _fila("Rentabilidad del patrimonio (ROE)", "Utilidad neta ÷ patrimonio", _pct("roe"), "a", p=1),
        ],
        notas=[
            dict(p=0, t="Bruto, operacional y neto", d="Los tres bajan a la vez"),
            dict(p=1, gr="17,4 %", t="Rentabilidad del patrimonio", d="Sigue siendo buena… ¿pero de qué calidad?"),
        ],
        pasos=[
            "Rentabilidad. Los tres márgenes bajaron: el bruto de 26 a 25 %, el operacional de 7,5 a 7,1, y el neto "
            "de 4,7 a 4,1.",
            "La rentabilidad sobre el patrimonio bajó de 18,7 a 17,4 %. Sigue siendo una buena rentabilidad. Pero "
            "el análisis DuPont muestra algo que el número solo no dice.",
        ],
    ),
    dict(
        tipo="html", titulo="DuPont: de dónde viene la rentabilidad",
        html=graficos.dupont_html(),
        pasos=[
            "DuPont descompone esa rentabilidad en tres factores: el margen neto, la rotación de activos y el "
            "apalancamiento. En 2024, la rentabilidad venía de un margen de 4,7 %, una rotación de 2 veces y un "
            "apalancamiento de 2.",
            "En 2025, el margen bajó, la rotación bajó, y lo único que subió fue el apalancamiento, a 2,19.",
            "En simple: el negocio rinde menos, y la rentabilidad se sostiene porque la empresa usa más deuda. Eso "
            "es una rentabilidad de peor calidad.",
        ],
    ),
    dict(
        tipo="svg", titulo="Eficiencia: el ciclo de conversión de efectivo", alto=True,
        svg=graficos.ciclo_caja(),
        pasos=[
            "Y llegamos a la familia que explica el caso: la eficiencia. Los días de inventario, es decir, cuánto "
            "tiempo pasa la mercadería en bodega, subieron de 55 a 68.",
            "Los días de cobro, cuánto tarda la empresa en cobrar una venta, pasaron de 50 a 72.",
            "Y los días de pago a proveedores bajaron de 50 a 48.",
            "Con esos tres números se calcula el ciclo de conversión de efectivo: días de inventario, más días de "
            "cobro, menos días de pago. Es el tiempo que pasa desde que la empresa paga la mercadería hasta que "
            "recupera la plata. En 2024 eran 55 días. En 2025 son 92.",
            "Son 37 días más que la empresa tiene que financiar. Y como vende más de 30 millones diarios, cada día "
            "adicional es mucha plata. Aquí está, en un solo indicador, la respuesta a por qué la utilidad no se "
            "transformó en caja.",
        ],
    ),
    dict(
        tipo="cifras", titulo="Traduce los días a pesos: el valor de un día", cols=3,
        tiles=[
            dict(p=1, eti="1 día de cobro", val=m(V["cobro"]) + " <small>millones</small>", sub="de caja"),
            dict(p=2, eti="1 día de inventario", val=m(V["inventario"]) + " <small>millones</small>", sub="de caja"),
            dict(p=3, eti="1 día de pago a proveedores", val=m(V["pago"]) + " <small>millones</small>", sub="de financiamiento"),
        ],
        pie=dict(p=4, t="«Cada día que nos demoramos en cobrar nos cuesta 32 millones de caja»"),
        pasos=[
            "Hay una forma de hacer que estos días le hablen a cualquier gerente: traducirlos a plata. Le pedí a "
            "Claude que calculara cuánto vale un día.",
            "En Alerce, un día de cobro equivale a 32 millones de pesos de caja.",
            "Un día de inventario, a 24 millones.",
            "Y cada día de plazo con proveedores, a otros 24 millones de financiamiento.",
            "Con ese dato, la conversación con el equipo comercial cambia. Ya no es «tenemos que cobrar más rápido». "
            "Es «cada día que nos demoramos en cobrar nos cuesta 32 millones de caja».",
        ],
    ),
    dict(
        tipo="cifras", titulo="Los 3 ratios en rojo", cols=3,
        tiles=[
            dict(p=0, eti="Razón de caja", val="0,03", sub="Era 0,39 en 2024", tono="neg"),
            dict(p=1, eti="Ciclo de efectivo", val="92 <small>días</small>", sub="37 días más que en 2024", tono="neg"),
            dict(p=2, eti="Deuda de corto plazo", val="59 %", sub="de la deuda bancaria · era 31 %", tono="neg"),
        ],
        pie=dict(p=3, t="Ninguno aparece en un informe de gestión típico."),
        sub_pie=dict(p=4, t="Antes de presentar, revisa dos o tres fórmulas contra tu propio Excel"),
        pasos=[
            "Los tres ratios que Claude marca en rojo: la razón de caja, en 0,03.",
            "+ El ciclo de conversión de efectivo, que aumentó 37 días.",
            "+ Y el peso de la deuda de corto plazo, 59 % del total.",
            "Ninguno de los tres aparece en un informe de gestión típico, que suele quedarse en ventas, margen y "
            "utilidad. Por eso este paso vale la pena.",
            "Un recordatorio: antes de usar estos números en una presentación, revisa al menos dos o tres fórmulas "
            "contra tu propio Excel. Te toma un minuto, y te da la tranquilidad de que el resto está bien calculado.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Paso 4 en resumen", estilo="check",
        items=[
            dict(p=0, t="Cuatro familias de ratios", d="Liquidez, endeudamiento, rentabilidad y eficiencia"),
            dict(p=1, t="Pide fórmula, interpretación y semáforo", d="Y revisa dos o tres cálculos por tu cuenta"),
            dict(p=2, t="Traduce los días a pesos", d="Es el idioma que entiende toda la empresa"),
        ],
        pasos=[
            "El paso 4, en resumen. Cuatro familias de ratios: liquidez, endeudamiento, rentabilidad y eficiencia.",
            "Pide siempre la fórmula, la interpretación y el semáforo.",
            "Y traduce los días a pesos, porque es el idioma que entiende toda la empresa.",
        ],
    ),
]
