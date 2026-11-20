"""Tu rutina mensual con Claude y los errores más comunes."""

from ._util import bloques

LAMINAS = [
    dict(
        tipo="seccion", etiqueta="Para llevar", nombre="Tu rutina mensual con Claude",
        bajada="Cómo convertir el método en un hábito de cierre",
        pasos=[
            "¿Cómo llevas todo esto a tu día a día? Te propongo una rutina mensual de cierre con Claude, en cinco "
            "momentos.",
        ],
    ),
    dict(
        tipo="html", titulo="El cierre mensual, en cinco momentos", une=True,
        html=bloques([
            dict(p=0, n="01", t="Exporta", d="Los estados del mes, desde tu sistema contable"),
            dict(p=1, n="02", t="Sube", d="A tu proyecto, donde ya está el contexto"),
            dict(p=2, n="03", t="Verifica", d="Corre el prompt de cuadraturas"),
            dict(p=3, n="04", t="Analiza", d="Los mismos prompts, en el mismo orden"),
            dict(p=4, n="05", t="Revisa y comparte", d="Corrige, valida y recién entonces envía"),
        ]),
        pie=dict(p=5, t="Menos tiempo armando tablas, más tiempo pensando."),
        pasos=[
            "Uno: exportas los estados financieros del mes desde tu sistema contable.",
            "Dos: los subes a tu proyecto, donde ya están el contexto y las instrucciones.",
            "Tres: corres la verificación.",
            "Cuatro: corres los análisis, siempre los mismos y en el mismo orden, para que los meses sean "
            "comparables.",
            "Y cinco: revisas, corriges y recién entonces compartes.",
            "Las primeras veces te va a tomar más tiempo. Cuando tengas el proyecto armado y los prompts afinados, "
            "vas a dedicar ese tiempo a pensar, y no a armar tablas.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Qué dejar en las instrucciones de tu proyecto", estilo="check", columnas=2, compacto=True,
        items=[
            dict(p=0, t="El contexto de la empresa", d="Giro, tamaño y hechos relevantes"),
            dict(p=1, t="La unidad y el formato de las cifras", d="Millones, miles o pesos"),
            dict(p=2, t="Las definiciones de tus indicadores", d="Tal como los calcula la empresa"),
            dict(p=3, t="Tus reglas de trabajo", d="Calcular con código y separar hechos de hipótesis"),
            dict(p=4, t="La lista de señales de alerta", d="La del paso 7"),
            dict(p=5, t="El formato del informe mensual", d="Para que cada mes sea comparable"),
        ],
        pasos=[
            "Para que la rutina funcione, deja bien armadas las instrucciones de tu proyecto. Yo incluyo seis "
            "cosas. El contexto de la empresa.",
            "+ La unidad y el formato de las cifras.",
            "+ Las definiciones de los indicadores, tal como los calcula la empresa.",
            "+ Las reglas de trabajo: calcular con código, mostrar la fórmula, y separar los hechos de las hipótesis.",
            "+ La lista de señales de alerta que vimos en el paso 7.",
            "+ Y el formato del informe mensual. Lo armas una vez, lo vas afinando, y cada mes partes desde ahí.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Cinco repreguntas que siempre sirven", compacto=True,
        items=[
            dict(p=0, t="¿Por qué?", d="Pide hipótesis, y el dato para confirmarlas"),
            dict(p=1, t="¿Cómo lo calculaste?", d="La fórmula y las cifras, a la vista"),
            dict(p=2, t="¿Qué supusiste?", d="Los supuestos escondidos son el mayor riesgo"),
            dict(p=3, t="¿Qué dato te falta?", d="Te dice exactamente qué ir a buscar"),
            dict(p=4, t="¿Qué pasaría si…?", d="Para abrir escenarios y medir sensibilidad"),
        ],
        pasos=[
            "Si tuviera que resumir el curso en una habilidad, sería repreguntar. Estas son las cinco repreguntas "
            "que uso todos los días. La primera: ¿por qué? Pide hipótesis, y el dato para confirmarlas.",
            "La segunda: ¿cómo lo calculaste? Que muestre la fórmula y las cifras.",
            "La tercera: ¿qué supusiste? Los supuestos escondidos son el mayor riesgo de cualquier análisis.",
            "La cuarta: ¿qué dato te falta? Te dice exactamente qué ir a buscar.",
            "Y la quinta: ¿qué pasaría si cambia un supuesto? Sirve para abrir escenarios y medir sensibilidad. Con "
            "estas cinco preguntas, una respuesta regular se convierte en un buen análisis.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Los 6 errores más comunes", estilo="cruz", columnas=2, compacto=True,
        items=[
            dict(p=0, t="Pedir sin contexto", d="«Analiza este balance» entrega un análisis genérico"),
            dict(p=1, t="Creerle sin verificar", d="Revisa las cifras de origen y algunas fórmulas"),
            dict(p=2, t="Pedir todo en un solo prompt", d="Avanza por pasos y revisa cada respuesta"),
            dict(p=3, t="Quedarse con la primera respuesta", d="Repregunta: por qué, cómo, qué pasaría si"),
            dict(p=4, t="Subir datos sin revisar la confidencialidad", d="Política interna y privacidad de la cuenta"),
            dict(p=5, t="Delegar la conclusión", d="El análisis puede ser de Claude; la decisión es tuya"),
        ],
        pasos=[
            "Y antes de cerrar, los seis errores que más veo. Uno: pedir sin contexto. Si solo escribes «analiza "
            "este balance», recibes un análisis genérico, que podría ser de cualquier empresa.",
            "Dos: creerle sin verificar. Revisa siempre las cifras de origen y algunas fórmulas.",
            "Tres: pedir todo en un solo prompt. Es mucho mejor avanzar por pasos, y revisar cada respuesta antes "
            "de seguir.",
            "Cuatro: quedarse con la primera respuesta. Repregunta: por qué, cómo lo calculaste, qué pasaría si.",
            "Cinco: subir información sin revisar antes la confidencialidad.",
            "Y seis, el más importante: delegar la conclusión. El análisis puede hacerlo Claude. La conclusión, y "
            "la decisión, tienen que ser tuyas.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Antes de enviar: dos minutos de control de calidad", estilo="check",
        items=[
            dict(p=0, t="Las cifras clave coinciden con tu sistema contable"),
            dict(p=1, t="Revisaste al menos tres cálculos por tu cuenta"),
            dict(p=2, t="Cada conclusión tiene su evidencia al lado"),
            dict(p=3, t="Las decisiones, los responsables y los plazos son tuyos"),
        ],
        lateral=[dict(p=4, clase="tinta", frase="Al final, quien firma el informe eres tú.", ancho=480)],
        pasos=[
            "Y antes de enviar cualquier informe, un control de calidad de dos minutos. Uno: las cifras clave "
            "coinciden con tu sistema contable.",
            "Dos: revisaste al menos tres cálculos por tu cuenta.",
            "Tres: cada conclusión tiene su evidencia al lado.",
            "Y cuatro: las decisiones, los responsables y los plazos son tuyos, no del borrador.",
            "Si esos cuatro puntos se cumplen, puedes firmar el informe con tranquilidad. Porque al final, quien "
            "firma eres tú.",
        ],
    ),
    dict(
        tipo="puntos", titulo="El mismo método, para mucho más que el cierre", columnas=2,
        items=[
            dict(p=0, icono="personas", t="Evaluar a un cliente antes de darle crédito", d="Con sus estados financieros y el prompt de riesgo"),
            dict(p=1, icono="banco", t="Preparar la reunión con el banco", d="Con los ratios calculados y las respuestas listas"),
            dict(p=2, icono="lupa", t="Mirar una empresa que quieres comprar", d="Una primera lectura, antes de la revisión formal"),
            dict(p=3, icono="mensaje", t="Explicarle los números a tu equipo", d="En simple, y con foco en lo que cada área puede mover"),
        ],
        pasos=[
            "El mismo método te sirve para mucho más que el cierre mensual. Para evaluar a un cliente antes de "
            "darle crédito: le pides sus estados financieros, y usas el prompt del analista de riesgo.",
            "Para preparar una reunión con el banco, y llegar con los ratios calculados y las respuestas listas.",
            "Para una primera mirada a una empresa que quieres comprar, antes de la revisión formal.",
            "Y para explicarle los números a tu equipo, en lenguaje simple y con foco en lo que cada área puede "
            "mover.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Preguntas que me hacen siempre (1 de 2)",
        items=[
            dict(p=0, icono="duda", t="¿Puedo confiar en los cálculos?",
                 d="En la aritmética, sí, cuando calcula con código. Revisa qué datos leyó y qué fórmula eligió."),
            dict(p=1, icono="duda", t="¿Y si me responde distinto cada vez?",
                 d="La redacción cambia; los números no deberían. Si cambian, pide el cálculo."),
        ],
        pasos=[
            "Antes de terminar, cuatro preguntas que me hacen siempre. ¿Puedo confiar en los cálculos? Cuando Claude "
            "calcula con código, la aritmética es confiable. El riesgo está en otra parte: en los datos que leyó, y "
            "en la fórmula que eligió. Por eso verificamos la lectura, y por eso pedimos siempre la fórmula.",
            "¿Y si me responde distinto cada vez? La redacción cambia, y es normal. Los números no deberían "
            "cambiar, si los datos y las instrucciones son los mismos. Si cambian, pídele que muestre el cálculo: "
            "casi siempre hay una definición distinta detrás.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Preguntas que me hacen siempre (2 de 2)",
        items=[
            dict(p=0, icono="duda", t="¿Sirve para una empresa pequeña?",
                 d="Sí: basta con dos balances y un estado de resultados."),
            dict(p=1, icono="duda", t="¿Reemplaza a mi contador o a mi analista?",
                 d="No. Les quita el trabajo mecánico y les deja más tiempo para el negocio."),
        ],
        pasos=[
            "¿Sirve para una empresa pequeña? Sí. Mientras tengas dos balances y un estado de resultados, el método "
            "funciona igual. De hecho, es donde más ayuda, porque normalmente no hay un equipo de análisis.",
            "¿Y reemplaza a mi contador o a mi analista? No. Les quita el trabajo mecánico, y les deja más tiempo "
            "para lo que realmente vale: entender el negocio y recomendar qué hacer.",
        ],
    ),
    dict(
        tipo="html", titulo="Tu plan para los próximos 30 días",
        html=bloques([
            dict(p=0, n="SEMANA 1", t="Arma tu proyecto", d="Contexto, reglas y los archivos de tu último cierre"),
            dict(p=1, n="SEMANA 2", t="Verifica y entiende", d="Cuadraturas, análisis vertical y horizontal"),
            dict(p=2, n="SEMANA 3", t="Mide y sigue la caja", d="Ratios, ciclo de efectivo y puente de caja"),
            dict(p=3, n="SEMANA 4", t="Compara, cuestiona y decide", d="Desviaciones, alertas y tu primer informe"),
        ]).replace('class="bloques"', 'class="bloques" style="grid-template-columns:repeat(4,1fr)"'),
        pie=dict(p=4, t="En un mes, el método completo funcionando con tus propios números."),
        pasos=[
            "Y para que no se quede en buenas intenciones, un plan para tus próximos 30 días. Semana 1: arma tu "
            "proyecto, con el contexto, las reglas y los archivos de tu último cierre.",
            "Semana 2: corre la verificación, y el análisis vertical y horizontal.",
            "Semana 3: los ratios, el ciclo de efectivo y el puente de caja.",
            "Y semana 4: las desviaciones, las señales de alerta y tu primer informe de una página.",
            "En un mes vas a tener el método completo funcionando con tus propios números.",
        ],
    ),
]
