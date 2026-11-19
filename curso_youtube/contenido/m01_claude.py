"""Antes de empezar: qué puede y qué no puede hacer Claude con tus números."""

from ._util import bloques, versus

LAMINAS = [
    dict(
        tipo="seccion", etiqueta="Antes de empezar", nombre="Qué puede y qué no puede hacer Claude",
        bajada="Conoce la herramienta antes de escribir el primer prompt",
        pasos=[
            "Antes de escribir el primer prompt, necesitas tener claro con qué herramienta estás trabajando. "
            "Qué puede hacer Claude con tus estados financieros, y qué no.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Claude: un asistente al que le escribes como a un colega", une=True,
        items=[
            dict(p=0, icono="mensaje", t="Conversas en lenguaje natural",
                 d="Es un asistente de inteligencia artificial desarrollado por Anthropic"),
            dict(p=1, icono="doc", t="Le adjuntas tus archivos", d="Un PDF, una planilla Excel o la foto de un informe"),
            dict(p=2, icono="rayo", t="Lee, calcula, compara, explica y redacta", d="A partir de lo que le entregas"),
        ],
        pasos=[
            "Claude es un asistente de inteligencia artificial desarrollado por la empresa Anthropic. Conversas con "
            "él en lenguaje natural, como si le escribieras a un colega.",
            "Le puedes adjuntar archivos: un PDF con los estados financieros, una planilla Excel, incluso una foto "
            "de un informe.",
            "Y a partir de ahí lee, calcula, compara, explica y redacta.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Lo que Claude hace bien con tus estados financieros", estilo="check", compacto=True,
        items=[
            dict(p=0, t="Lee y ordena", d="Convierte un PDF de 40 páginas en una tabla limpia"),
            dict(p=1, t="Calcula", d="Variaciones, porcentajes y ratios, sin armar las fórmulas"),
            dict(p=2, t="Explica", d="Propone causas y te dice qué dato necesita para confirmarlas"),
            dict(p=3, t="Cuestiona", d="Busca inconsistencias y hace de abogado del diablo"),
            dict(p=4, t="Redacta", d="El resumen para el directorio, el correo al banco o el informe mensual"),
        ],
        pasos=[
            "Hay cinco cosas que hace muy bien. Primero, lee y ordena. En segundos convierte un PDF de 40 páginas "
            "en una tabla limpia.",
            "Segundo, calcula. Variaciones, porcentajes, ratios: todo lo que harías en Excel, sin armar las fórmulas.",
            "Tercero, explica. No solo te dice que el margen cayó: te propone por qué, y te dice qué dato "
            "necesitaría para confirmarlo.",
            "Cuarto, cuestiona. Le puedes pedir que busque inconsistencias, o que haga de abogado del diablo con "
            "tus propias conclusiones.",
            "Y quinto, redacta. Un resumen para el directorio, un correo para el banco o el informe mensual de "
            "gestión.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Lo que Claude no hace", estilo="cruz",
        items=[
            dict(p=0, t="No conoce tu negocio", d="Lo que no le dices, lo va a suponer"),
            dict(p=1, t="Puede equivocarse", d="Leer mal una cifra o mezclar dos columnas"),
            dict(p=2, t="No reemplaza tu criterio", d="La decisión y la responsabilidad siguen siendo tuyas"),
            dict(p=3, t="No audita", d="Si los datos de origen están mal, el análisis estará mal"),
        ],
        pasos=[
            "Ahora, los límites. No conoce tu negocio. No sabe que en marzo cambiaste de proveedor, ni que tu "
            "principal cliente paga a 90 días. Si no se lo dices, lo va a suponer.",
            "Puede equivocarse. Puede leer mal una cifra de un PDF escaneado, o mezclar dos columnas. Por eso el "
            "segundo paso del método es verificar.",
            "No reemplaza tu criterio. Claude te entrega análisis; la decisión, y la responsabilidad, siguen siendo "
            "tuyas.",
            "Y no audita. Trabaja con la información que le das. Si los datos de origen están mal, el análisis va a "
            "estar mal, aunque esté muy bien redactado.",
        ],
    ),
    dict(
        tipo="puntos", titulo="El modelo mental: un analista muy rápido… en su primer día",
        items=[
            dict(p=1, icono="carpeta", t="Dale contexto", d="Tu industria, tus clientes y tu historia"),
            dict(p=2, icono="diana", t="Pide con precisión", d="Qué necesitas y en qué formato"),
            dict(p=3, icono="lupa", t="Revisa su trabajo", d="Antes de presentarlo a nadie"),
        ],
        lateral=[dict(p=0, clase="tinta", eti="Así es Claude",
                      frase="Sabe mucho de finanzas, trabaja en segundos y no se cansa.",
                      ps=["Pero no conoce tu industria, tus clientes ni tu historia."], ancho=620)],
        pasos=[
            "El modelo mental que más me sirve es este: piensa en Claude como un analista muy rápido, que acaba de "
            "llegar a tu empresa. Sabe mucho de finanzas, trabaja en segundos y no se cansa. Pero es su primer día: "
            "no conoce tu industria, tus clientes ni tu historia.",
            "¿Qué harías con un analista así? Le darías contexto,",
            "+ le explicarías con precisión qué necesitas,",
            "+ y revisarías su trabajo antes de presentarlo. Exactamente eso vamos a hacer.",
        ],
    ),
    dict(
        tipo="html", titulo="Las 5 partes de un buen prompt financiero",
        html='<div class="pila">' + bloques([
            dict(p=1, n="01", t="Contexto", d="Qué empresa es, su industria y su tamaño"),
            dict(p=2, n="02", t="Datos", d="Qué le entregas y de qué período"),
            dict(p=3, n="03", t="Tarea", d="Qué quieres exactamente que haga"),
            dict(p=4, n="04", t="Formato", d="Cómo quieres la respuesta"),
            dict(p=5, n="05", t="Verificación", d="Que muestre sus cálculos y sus supuestos"),
        ]).replace('class="bloques"', 'class="bloques" style="flex:none"')
        + versus("<code>Analiza este balance</code>", "Un prompt con las 5 partes: la diferencia es enorme", 6) + "</div>",
        pasos=[
            "Y esto nos lleva a la regla más importante del curso. Un buen prompt financiero tiene cinco partes.",
            "Uno, contexto: qué empresa es, de qué industria, de qué tamaño.",
            "Dos, datos: qué le estás entregando y de qué período.",
            "Tres, tarea: qué quieres exactamente que haga.",
            "Cuatro, formato: cómo quieres la respuesta, por ejemplo una tabla y tres conclusiones.",
            "Y cinco, verificación: pídele que muestre sus cálculos y que te diga qué supuso.",
            "Compara: «analiza este balance», frente a un prompt con estas cinco partes. La diferencia en la "
            "respuesta es enorme, y la vas a ver en cada paso.",
        ],
    ),
    dict(
        tipo="dos", titulo="El mismo balance, dos preguntas",
        paneles=[
            dict(p=0, clase="rojo", icono="x", titulo="«Analiza este balance»", icono_item="mensaje", items=[
                dict(t="«Los activos suman 6.105 y los pasivos 3.320…»", d="«Se recomienda monitorear la liquidez y el endeudamiento»"),
                dict(t="Correcta, pero genérica", d="Describe lo que ya podías ver tú mismo", icono="flecha")]),
            dict(p=1, clase="verde", icono="check", titulo="Con contexto, tarea y formato", icono_item="mensaje", items=[
                dict(t="«La caja cayó 89 % mientras las ventas crecían 18 %…»", d="«El crecimiento se financió con cuentas por cobrar y deuda de corto plazo»"),
                dict(t="Prioriza y explica", d="Y te dice por dónde empezar", icono="flecha")]),
        ],
        pie=dict(p=2, t="El mismo archivo y la misma herramienta. Lo único que cambió fue la pregunta."),
        pasos=[
            "Te muestro la diferencia con el caso que vamos a usar. Si subes el balance y escribes solamente "
            "«analiza este balance», la respuesta suele ser correcta, pero genérica: te describe los totales y te "
            "recomienda monitorear la liquidez y el endeudamiento. No hay nada que no pudieras ver tú mismo.",
            "En cambio, cuando le das contexto, una tarea precisa y un formato, la respuesta cambia de nivel. Te "
            "dice que la caja cayó 89 % mientras las ventas crecían 18, que el crecimiento se financió con cuentas "
            "por cobrar y con deuda de corto plazo, y por dónde empezar.",
            "Es el mismo archivo y la misma herramienta. Lo único que cambió fue la pregunta.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Qué necesitas para seguir el curso",
        items=[
            dict(p=0, icono="persona", t="Una cuenta en claude.ai", d="Puedes empezar con el plan gratuito"),
            dict(p=1, icono="ajustes", t="Análisis de archivos y ejecución de código",
                 d="Revisa en la configuración que estén activados, si tu plan los incluye"),
            dict(p=2, icono="porciento", t="Así Claude calcula, no adivina",
                 d="Escribe y ejecuta un pequeño programa para hacer las cuentas"),
            dict(p=3, icono="ojo", t="Y puedes ver cómo lo hizo", d="El detalle del cálculo queda a la vista"),
        ],
        pasos=[
            "Para seguir el curso necesitas una cuenta en claude.ai. Puedes empezar con el plan gratuito, y ver "
            "después si necesitas uno de pago.",
            "En la configuración, revisa que estén activadas las funciones de análisis de archivos y ejecución de "
            "código, si tu plan las incluye.",
            "Es importante: con ellas, Claude no calcula de memoria. Escribe y ejecuta un pequeño programa para "
            "hacer las cuentas, igual que tú usarías una calculadora.",
            "Y puedes abrir ese detalle para ver qué hizo, paso a paso. Esa transparencia es lo que te permite "
            "revisar su trabajo, en vez de tener que creerle.",
        ],
    ),
]
