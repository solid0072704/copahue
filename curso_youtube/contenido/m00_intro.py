"""Introducción: gancho, promesa, método y caso."""

LAMINAS = [
    dict(
        tipo="gancho",
        docs=[("Balance", "balance"), ("Estado de resultados", "resultados"), ("Flujo de efectivo", "flujo")],
        pregunta="¿Qué haces con <em>tus números</em>?",
        remate=["Tener los números", "entenderlos", "decidir"],
        p_pregunta=1, p_remate=2,
        pasos=[
            "Cada mes cierras los estados financieros y tienes el balance, el estado de resultados y el flujo de "
            "caja sobre la mesa.",
            "+ La pregunta es qué haces con ellos.",
            "+ Porque tener los números no es lo mismo que entenderlos, y entenderlos no es lo mismo que decidir.",
        ],
    ),
    dict(
        tipo="titulo_curso", une=True,
        sobre="Curso completo · 1 hora",
        titulo_html="Análisis de estados financieros con <em>Claude</em>",
        sub="De los números a las decisiones, paso a paso",
        quien="Rodrigo Barceló",
        pasos=[
            "Hola, soy Rodrigo Barceló. En esta clase de una hora vas a aprender a analizar estados "
            "financieros con Claude, paso a paso: desde preparar la información hasta entregar un informe con "
            "conclusiones y un plan de acción.",
        ],
    ),
    dict(
        tipo="chips", une=True,
        titulo_html="Lo único que <em>necesitas</em>",
        chips=[
            dict(p=0, icono="doc", t="Tus estados financieros", d="Balance, resultados y flujo de efectivo"),
            dict(p=0, icono="mensaje", t="Una cuenta de Claude", d="Sin programar y sin ser experto en IA"),
            dict(p=0, icono="duda", t="Las preguntas correctas", d="Te las llevas de esta clase"),
        ],
        pasos=[
            "No necesitas saber programar ni ser experto en inteligencia artificial. Solo necesitas tus estados "
            "financieros, una cuenta de Claude y las preguntas correctas. Y esas preguntas son justamente lo que "
            "te vas a llevar de aquí.",
        ],
    ),
    dict(
        tipo="puntos", titulo="¿Para quién es este curso?", columnas=2,
        items=[
            dict(p=0, icono="maletin", t="Gerentes de finanzas y controllers", d="Que preparan el cierre y el informe de gestión"),
            dict(p=1, icono="persona", t="Dueños y gerentes generales", d="Que quieren entender sus números sin depender de nadie"),
            dict(p=2, icono="libro", t="Contadores y asesores", d="Que quieren entregar más análisis a sus clientes"),
            dict(p=3, icono="lupa", t="Analistas", d="Que quieren dedicar menos tiempo a armar tablas"),
        ],
        pasos=[
            "Este curso es para ti si preparas el cierre y el informe de gestión, como gerente de finanzas o "
            "controller.",
            "+ Si eres dueño o gerente general, y quieres entender tus números sin depender de nadie.",
            "+ Si eres contador o asesor, y quieres entregar más análisis a tus clientes.",
            "+ O si eres analista, y quieres dedicar menos tiempo a armar tablas, y más a entender qué está pasando.",
        ],
    ),
    dict(
        tipo="metodo", titulo="Un método de 8 pasos: de los datos a la decisión", p0=1,
        pasos=[
            "Vamos a trabajar con un método de 8 pasos.",
            "+ Primero preparas la información y le das contexto a Claude.",
            "+ Después verificas que los números cuadren.",
            "+ Luego entiendes la estructura y la tendencia,",
            "+ mides con ratios,",
            "+ sigues la caja,",
            "+ comparas contra el presupuesto,",
            "+ buscas señales de alerta",
            "+ y, al final, conviertes todo eso en decisiones.",
            "En cada paso te voy a mostrar el prompt exacto que uso, por qué funciona y qué responde Claude.",
        ],
    ),
    dict(
        tipo="cifras", titulo="El caso: una empresa en su «mejor año»", cols=3,
        tiles=[
            dict(p=0, eti="Ventas 2025", val="+18 %", sub="11.800 millones de pesos", tono="pos"),
            dict(p=0, eti="Utilidad neta", val="485 <small>millones</small>", sub="La más alta de su historia", tono="pos"),
            dict(p=1, eti="Caja en diciembre", val="65 <small>millones</small>", sub="Eran 620 a comienzos de año", tono="neg"),
        ],
        pie=dict(p=2, t="¿Cómo se puede vender más, ganar más y quedarse sin caja?"),
        sub_pie=dict(p=3, icono="descarga", t="Ebook y archivo del caso: enlaces en la descripción del video"),
        pasos=[
            "Y lo vamos a hacer con un caso práctico: Distribuidora Alerce, una empresa ficticia que en 2025 vendió "
            "18 % más que el año anterior y tuvo la mayor utilidad de su historia.",
            "+ Aun así, en diciembre casi no pudo pagar los sueldos.",
            "A lo largo de la clase vamos a descubrir, con Claude, qué pasó y qué habría que hacer.",
            "En la descripción encuentras el enlace a nuestro ebook y al archivo del caso, para que repitas cada "
            "paso con los mismos números. Y un consejo: mira el curso con tus propios estados financieros a mano. "
            "Pausa el video en cada prompt, cópialo y pruébalo. Así, al terminar esta hora, no solo vas a conocer "
            "un método: vas a tener tu primer análisis hecho.",
        ],
    ),
]
