"""Paso 7: señales de alerta."""

from .. import caso
from ._util import m

Q = caso.calidad_utilidad()

LAMINAS = [
    dict(
        tipo="seccion", paso=7, etiqueta="Paso 7", nombre="Cuestiona: busca señales de alerta",
        bajada="Ahora le pedimos a Claude que desconfíe",
        pasos=[
            "Paso 7: buscar señales de alerta. Hasta aquí le pedimos a Claude que calcule y que explique. Ahora le "
            "vamos a pedir que desconfíe. Es el paso donde más valor agrega tener una segunda mirada.",
        ],
    ),
    dict(
        tipo="prompt", titulo="Prompt 8: el abogado del diablo", numero=8, nombre="Señales de alerta", alto=True, une=True,
        segs=[
            ("Actúa como un analista de riesgo de crédito escéptico, que\n"
             "debe decidir si le presta dinero a esta empresa.\n\n", 0),
            ("1. Lista las señales de alerta que ves en estos estados\n"
             "   financieros. Para cada una: la evidencia en cifras, qué\n"
             "   podría significar y qué documento pedirías.\n", 1),
            ("2. Evalúa la calidad de la utilidad: ¿está respaldada por\n"
             "   caja?\n", 2),
            ("3. Dime qué supuestos optimistas podría estar haciendo la\n"
             "   administración.\n"
             "4. Clasifica cada alerta: crítica, importante o a observar.\n\n", 3),
            ("Sé directo. No suavices las conclusiones.", 4),
        ],
        notas=[
            dict(p=0, t="Cambia el rol", d="De tu analista a un analista de crédito escéptico"),
            dict(p=1, t="Evidencia y documento", d="Cada alerta, con cifras y con qué pedir"),
            dict(p=2, t="Calidad de la utilidad", d="¿Está respaldada por caja?"),
            dict(p=4, t="Sin diplomacia", d="Pídele que sea directo"),
        ],
        pasos=[
            "Mira cómo cambia el rol. Ya no es mi analista: es un analista de riesgo de crédito, escéptico, que "
            "tiene que decidir si le presta plata a la empresa. Ese cambio de rol modifica por completo la respuesta.",
            "Le pido las señales de alerta con su evidencia en cifras, qué podrían significar y qué documento "
            "pediría.",
            "Que evalúe la calidad de la utilidad, es decir, si está respaldada por caja.",
            "Que identifique los supuestos optimistas de la administración, y que clasifique cada alerta.",
            "Y termino con una instrucción que uso mucho: sé directo, no suavices las conclusiones. Los modelos "
            "tienden a ser diplomáticos. En este paso, no quieres diplomacia.",
        ],
    ),
    dict(
        tipo="alertas", titulo="Las señales de alerta de Alerce", alto=True,
        items=[
            dict(p=0, nivel="cr", t="Utilidad sin caja", d="Utilidad de 485 millones, con un flujo operacional de −685"),
            dict(p=1, nivel="cr", t="Dependencia de la deuda de corto plazo", d="1.150 millones que vencen en un año, con 65 en caja"),
            dict(p=2, nivel="im", t="Cartera vencida y sin provisión", d="45 % vencido (era 27 %) · a más de 90 días: de 10 a 90 millones"),
            dict(p=3, nivel="im", t="Concentración de clientes", d="2 clientes: 20 % de las ventas y 28 % de las cuentas por cobrar"),
            dict(p=4, nivel="im", t="Dividendos con flujo negativo", d="200 millones repartidos mientras la operación consumía caja"),
            dict(p=5, nivel="ob", t="Inventarios", d="+48 % con ventas +18 %: ¿hay productos de baja rotación?"),
        ],
        pasos=[
            "Estas son las alertas de Alerce. Dos críticas. La primera, utilidad sin caja: 485 millones de utilidad "
            "con un flujo operacional negativo de 685.",
            "La segunda, dependencia de la deuda de corto plazo: 1.150 millones que vencen dentro de un año, con "
            "65 millones en caja. Si el banco no renueva la línea, la empresa tiene un problema inmediato.",
            "Luego, tres importantes. La cartera vencida: el 45 % de las cuentas por cobrar está vencido, contra "
            "27 % el año anterior. Y lo vencido a más de 90 días pasó de 10 a 90 millones, sin ninguna provisión. Si "
            "parte de eso no se cobra, la utilidad del año está sobrestimada.",
            "La concentración: dos clientes explican el 20 % de las ventas y el 28 % de las cuentas por cobrar.",
            "Y los dividendos: se repartieron 200 millones en un año en que la operación consumió caja.",
            "Por último, una alerta a observar: los inventarios crecieron 48 %, con ventas que crecieron 18. Claude "
            "pide el detalle por producto, para ver si hay mercadería de baja rotación.",
        ],
    ),
    dict(
        tipo="cifras", titulo="Calidad de la utilidad: ¿se convierte en caja?", cols=2,
        sobre=dict(p=0, t="**Flujo de la operación ÷ utilidad neta** · cuando es cercano o superior a 1, la utilidad se está convirtiendo en caja"),
        tiles=[
            dict(p=1, eti="2024", val=m(Q["2024"], 2), sub="Por cada peso de utilidad, 1,09 de caja", tono="pos"),
            dict(p=2, eti="2025", val=m(Q["2025"], 2), sub="Por cada peso de utilidad, salió 1,41 de caja", tono="neg"),
        ],
        pie=dict(p=3, t="Es el indicador que un banco mira primero."),
        pasos=[
            "Para medir la calidad de la utilidad hay un indicador muy simple: el flujo de la operación dividido "
            "por la utilidad neta. Cuando es cercano o superior a 1, la utilidad se está convirtiendo en caja.",
            "En 2024, Alerce tenía 1,09: por cada peso de utilidad, generaba 1,09 pesos de caja.",
            "En 2025, el indicador es negativo: menos 1,41. Por cada peso de utilidad que mostró, salió 1,41 de la "
            "caja.",
            "Es el indicador que cualquier banco va a mirar primero. Y es muy fácil de pedir todos los meses.",
        ],
    ),
    dict(
        tipo="tabla", titulo="Para cada alerta, el documento que hay que pedir", texto=True, flex=1,
        columnas=["Señal de alerta", "Documento a pedir"],
        filas=[
            dict(c=["Utilidad sin caja", "Flujo de caja mensual y proyección a 13 semanas"], p=0, aparece=0),
            dict(c=["Deuda de corto plazo", "Contratos de crédito: tasas, vencimientos y garantías"], p=1, aparece=1),
            dict(c=["Cartera vencida", "Detalle por cliente y registro de la gestión de cobranza"], p=2, aparece=2),
            dict(c=["Concentración de clientes", "Contratos y condiciones comerciales con las cadenas"], p=3, aparece=3),
            dict(c=["Inventarios", "Rotación por producto y mermas"], p=4, aparece=4),
        ],
        pie=dict(p=5, t="Las alertas dejan de ser preocupaciones y se convierten en una lista de tareas."),
        pasos=[
            "Y esta es la otra mitad de la respuesta: qué documento pedir para cada alerta. Para la utilidad sin "
            "caja, el flujo de caja mensual y una proyección a 13 semanas.",
            "Para la deuda de corto plazo, los contratos de crédito, con sus tasas, vencimientos y garantías.",
            "Para la cartera vencida, el detalle por cliente y el registro de la gestión de cobranza.",
            "Para la concentración, los contratos y las condiciones comerciales con las cadenas.",
            "Y para los inventarios, la rotación por producto y las mermas.",
            "Con esto, las alertas dejan de ser una lista de preocupaciones, y se convierten en una lista de tareas.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Los supuestos optimistas que detecta",
        items=[
            dict(p=0, icono="personas", t="Que las cadenas van a pagar a tiempo", d="Hoy pagan a 101 días, y la cartera vencida crece"),
            dict(p=1, icono="banco", t="Que el banco va a renovar la línea", d="1.150 millones dependen de esa decisión"),
            dict(p=2, icono="caja", t="Que el inventario se va a vender sin descuentos", d="Creció 48 %, con ventas que crecieron 18 %"),
        ],
        pasos=[
            "También le pedimos los supuestos optimistas de la administración. Identifica tres. Que las cadenas van "
            "a pagar a tiempo, cuando hoy pagan a 101 días y la cartera vencida crece.",
            "Que el banco va a renovar la línea de crédito, cuando 1.150 millones dependen de esa decisión.",
            "Y que el inventario adicional se va a vender sin descuentos. Ninguno de estos supuestos está escrito "
            "en los estados financieros, pero los tres están implícitos en ellos. Ponerlos por escrito es la mitad "
            "del trabajo de gestionar el riesgo.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Señales que conviene revisar siempre", columnas=2, compacto=True,
        items=[
            dict(p=0, icono="alerta", t="Cuentas por cobrar que crecen más rápido que las ventas"),
            dict(p=1, icono="alerta", t="Inventarios que crecen más rápido que el costo de ventas"),
            dict(p=2, icono="alerta", t="Utilidad que sube mientras el flujo operacional baja"),
            dict(p=3, icono="alerta", t="Gastos que se activan en vez de ir a resultado"),
            dict(p=4, icono="alerta", t="Deuda de corto plazo que financia activos de largo plazo"),
            dict(p=5, icono="alerta", t="Cambios de criterio contable de un año a otro"),
            dict(p=6, icono="alerta", t="Operaciones relevantes con empresas relacionadas"),
        ],
        pie=dict(p=7, t="Deja esta lista en las instrucciones de tu proyecto: Claude la revisará todos los meses."),
        pasos=[
            "Más allá de este caso, hay señales que conviene revisar siempre. Cuentas por cobrar que crecen más "
            "rápido que las ventas.",
            "+ Inventarios que crecen más rápido que el costo de ventas.",
            "+ Utilidad que sube mientras el flujo operacional baja.",
            "+ Gastos que se activan en vez de ir a resultado.",
            "+ Deuda de corto plazo financiando activos de largo plazo.",
            "+ Cambios de criterio contable de un año a otro.",
            "+ Y operaciones relevantes con empresas relacionadas.",
            "Puedes dejar esta lista en las instrucciones de tu proyecto, para que Claude la revise todos los meses.",
        ],
    ),
    dict(
        tipo="dos", titulo="Cómo usar este paso",
        paneles=[
            dict(p=0, clase="oro", icono="duda", titulo="Una alerta no es una acusación", icono_item="flecha", items=[
                dict(t="Es una pregunta", d="Que se responde con un documento"),
                dict(t="Por eso pedimos el documento", d="Así sabes exactamente qué revisar")]),
            dict(p=1, clase="azul", icono="personas", titulo="Úsalo para prepararte", icono_item="flecha", items=[
                dict(t="Antes de presentar al directorio", d="O antes de pedir un crédito"),
                dict(t="Llegas con las preguntas difíciles", d="Y con las respuestas preparadas")]),
        ],
        pie=dict(p=2, t="Un ensayo general con el director más exigente de la mesa."),
        pasos=[
            "Dos comentarios sobre este paso. El primero: una señal de alerta no es una acusación. Es una pregunta "
            "que hay que responder con un documento. Por eso le pedí que indicara qué documento solicitaría.",
            "El segundo: este prompt es muy útil para prepararte. Si vas a presentar al directorio, o vas a pedir "
            "un crédito, úsalo antes. Vas a llegar a la reunión sabiendo cuáles son las preguntas difíciles, y con "
            "las respuestas preparadas.",
            "Es como tener un ensayo general con el director más exigente de la mesa.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Paso 7 en resumen", estilo="check",
        items=[
            dict(p=0, t="Cambia el rol: de analista a escéptico", d="Y pídele que sea directo"),
            dict(p=1, t="Cada alerta, con su evidencia y su documento", d="Para convertirla en una tarea"),
            dict(p=2, t="Úsalo antes del directorio o del banco", d="Llegas con las respuestas preparadas"),
        ],
        pasos=[
            "El paso 7, en resumen. Cambia el rol: de analista a escéptico.",
            "Pide cada alerta con su evidencia y con el documento que la respalda.",
            "Y úsalo antes de ir al directorio o al banco.",
        ],
    ),
]
