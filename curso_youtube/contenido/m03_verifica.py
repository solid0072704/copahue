"""Paso 2: verificar que todo cuadre y pedir una primera lectura."""

from ..laminas import ico

OK = ico("check")
AV = ico("alerta")

LAMINAS = [
    dict(
        tipo="seccion", paso=2, etiqueta="Paso 2", nombre="Verifica que todo cuadre",
        bajada="Dos minutos que te ahorran un error frente al directorio",
        pasos=[
            "Paso 2: verificar. Antes de analizar cualquier cosa, hay que confirmar que Claude leyó bien los "
            "archivos, y que los estados financieros son coherentes entre sí. Son dos minutos que te pueden ahorrar "
            "un error grave frente al directorio.",
        ],
    ),
    dict(
        tipo="dos", titulo="Dos tipos de error que hay que atrapar a tiempo", une=True,
        paneles=[
            dict(p=0, clase="oro", icono="ojo", titulo="Errores de lectura", icono_item="x", items=[
                dict(t="Toma mal una cifra"), dict(t="Se salta una fila"), dict(t="Confunde miles con millones")]),
            dict(p=1, clase="azul", icono="doc", titulo="Errores de origen", icono_item="x", items=[
                dict(t="Un balance que no cuadra"), dict(t="Un flujo que no llega al saldo de caja"),
                dict(t="Subtotales que no suman")]),
        ],
        pie=dict(p=2, t="Si no lo detectas ahora, todo lo demás se construye sobre un dato malo."),
        pasos=[
            "Hay dos tipos de errores. Los de lectura: Claude toma mal una cifra, se salta una fila o confunde miles "
            "con millones.",
            "Y los de origen: los estados que te entregaron ya venían con un problema, como un balance que no "
            "cuadra, o un flujo que no llega al saldo de caja.",
            "En los dos casos, si no lo detectas ahora, todo lo que viene después se construye sobre un dato malo.",
        ],
    ),
    dict(
        tipo="prompt", titulo="Prompt 2: verificación", numero=2, nombre="Lectura y cuadraturas", alto=True,
        segs=[
            ("Antes de analizar, verifica la información que te entregué.\n\n", None),
            ("1. Muéstrame en una tabla las cifras que leíste de cada\n"
             "   estado, por período y con su unidad.\n", 1),
            ("2. Comprueba estas cuadraturas y dime si se cumplen:\n"
             "   - Activos = Pasivos + Patrimonio.\n"
             "   - Utilidad del año = variación de utilidades acumuladas\n"
             "     + dividendos.\n"
             "   - Efectivo inicial + flujo neto = efectivo final.\n"
             "   - Cada subtotal = suma de sus partidas.\n", 2),
            ("3. Lista los datos que faltan o que te parecen\n"
             "   inconsistentes.\n", 3),
            ("4. No sigas hasta que yo confirme.", 4),
        ],
        notas=[
            dict(p=1, t="La tabla de lo que leyó", d="La comparas con tu archivo de un vistazo"),
            dict(p=2, t="Cuadraturas básicas", d="Balance, patrimonio, caja y subtotales"),
            dict(p=3, t="Lo que falta o no calza", d="Las primeras pistas aparecen aquí"),
            dict(p=4, t="Una pausa antes de seguir", d="Tú confirmas; recién entonces analiza"),
        ],
        pasos=[
            "Este es el prompt de verificación. Le pido cuatro cosas.",
            "Primero, que me muestre en una tabla las cifras que leyó, con su unidad. Así comparo de un vistazo "
            "contra mi archivo.",
            "Segundo, que compruebe las cuadraturas básicas. Que los activos sean iguales a los pasivos más el "
            "patrimonio. Que la utilidad del año explique el cambio en las utilidades acumuladas, considerando los "
            "dividendos. Que el efectivo inicial más el flujo del año llegue al efectivo final del balance. Y que "
            "los subtotales sumen.",
            "Tercero, que liste lo que falta, o lo que no le hace sentido.",
            "Y cuarto, otra vez: que no siga hasta que yo confirme.",
        ],
    ),
    dict(
        tipo="respuesta", titulo="La verificación de Alerce: todo cuadra… y una pista", alto=True,
        bloques=[
            dict(p=0, h="Cuadraturas", html=(
                '<ul class="chk">'
                f'<li data-p="0">{OK}<span><b>Activos = Pasivos + Patrimonio.</b> 2024: 5.000 = 5.000 · 2025: 6.105 = 6.105</span></li>'
                f'<li data-p="1">{OK}<span><b>Utilidad 485</b> = aumento de utilidades acumuladas 285 + dividendos 200</span></li>'
                f'<li data-p="2">{OK}<span><b>Efectivo:</b> 620 − 555 de flujo neto = 65, igual que el balance</span></li>'
                f'<li data-p="2">{OK}<span>Los subtotales suman</span></li></ul>')),
            dict(p=3, h="Observaciones", html=(
                '<ul class="chk">'
                f'<li class="av">{AV}<span>No hay <b>provisión de incobrables</b>, y lo vencido a más de 90 días subió de 10 a 90 millones</span></li>'
                f'<li class="av">{AV}<span>Falta el detalle de la <b>deuda de corto plazo</b>: tasa y vencimientos</span></li></ul>')),
        ],
        notas=[
            dict(p=2, t="Todo en orden", d="Los tres estados son coherentes entre sí", clase="verde"),
            dict(p=3, t="La primera pista", d="Cartera vencida y sin provisión", clase="rojo"),
            dict(p=4, t="Dos temas para conversar", d="Y todavía no analizamos nada"),
        ],
        pasos=[
            "Y esto es lo que responde con los datos de Alerce. El balance cuadra en los dos años: 6.105 millones de "
            "activos contra 6.105 de pasivos y patrimonio.",
            "La utilidad de 485 millones calza con el aumento de 285 en utilidades acumuladas, más los 200 de "
            "dividendos.",
            "Y el flujo calza con la caja: 620 de saldo inicial, menos 555 de flujo neto, da los 65 del balance. "
            "Todo en orden.",
            "Pero mira las observaciones, porque aquí aparece la primera pista. Claude nota que no existe provisión "
            "de incobrables, aunque las cuentas vencidas a más de 90 días pasaron de 10 a 90 millones. Y pide el "
            "detalle de la deuda de corto plazo.",
            "Todavía no analizamos nada, y ya tenemos dos temas para conversar.",
        ],
    ),
    dict(
        tipo="puntos", titulo="¿Y si algo no cuadra? Un ejemplo",
        items=[
            dict(p=1, t="Pide el detalle", d="«Muéstrame qué partidas sumaste en cada lado»"),
            dict(p=2, t="Compara con tu archivo", d="En inventarios leyó 1.560; el archivo dice 1.650"),
            dict(p=3, t="Corrige y vuelve a verificar", d="«El inventario es 1.650. Repite las cuadraturas»"),
        ],
        lateral=[
            dict(p=0, clase="oro", eti="Lectura de un PDF escaneado", gr="90", tono="neg",
                 ps=["de diferencia: activos por 6.015, contra 6.105 de pasivos y patrimonio."], ancho=560),
            dict(p=4, clase="tinta", eti="Dato de contador",
                 ps=["Una diferencia divisible por 9 suele ser un error de dígitos invertidos."], ancho=560),
        ],
        pasos=[
            "¿Y qué pasa cuando algo no cuadra? Te doy un ejemplo. Supón que, al leer un PDF escaneado, Claude te "
            "dice que los activos suman 6.015, y que los pasivos más el patrimonio suman 6.105. Hay una diferencia "
            "de 90.",
            "Primer paso: pide el detalle. Muéstrame qué partidas sumaste en cada lado.",
            "Segundo: compara con tu archivo. Y ahí aparece: en inventarios leyó 1.560, y el archivo dice 1.650. Dos "
            "dígitos invertidos.",
            "Tercero: corrige y vuelve a verificar. Le indicas el valor correcto, y le pides que repita las "
            "cuadraturas.",
            "Un dato de contador: cuando la diferencia es divisible por 9, casi siempre es un error de dígitos "
            "invertidos. Lo importante es que el error apareció aquí, en el paso 2, y no en la reunión de directorio.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Tres chequeos de 30 segundos, para cualquier respuesta",
        items=[
            dict(p=0, t="Una cifra contra tu archivo", d="Elige una al azar y compárala"),
            dict(p=1, t="Un total que conoces de memoria", d="Las ventas del año o la caja del cierre"),
            dict(p=2, t="Un cálculo hecho por ti", d="Un porcentaje o un ratio, con la calculadora"),
        ],
        lateral=[dict(p=3, clase="oro", frase="El mismo control que le harías al trabajo de cualquier persona de tu equipo.",
                      ancho=560)],
        pasos=[
            "Más allá de las cuadraturas, te recomiendo tres chequeos de 30 segundos, para cualquier respuesta. Uno: "
            "toma una cifra al azar y compárala contra tu archivo.",
            "Dos: busca un total que conozcas de memoria, como las ventas del año o la caja del cierre.",
            "Y tres: repite tú un cálculo, un porcentaje o un ratio, con la calculadora.",
            "Si los tres chequeos calzan, puedes avanzar con confianza. No es desconfianza hacia la herramienta. Es "
            "el mismo control que le harías al trabajo de cualquier persona de tu equipo.",
        ],
    ),
    dict(
        tipo="prompt", titulo="Prompt 3: la primera lectura", numero=3, nombre="Resumen ejecutivo inicial",
        segs=[
            ("Con la información verificada, dame una primera lectura\n"
             "ejecutiva en un máximo de 10 líneas:\n\n", 1),
            ("- Las 3 cifras que mejor resumen el año.\n"
             "- Lo que mejoró y lo que empeoró frente al año anterior.\n"
             "- La pregunta más importante que debería hacerme como\n"
             "  gerente.\n\n", 2),
            ("Escribe para un directorio que no es experto en\n"
             "contabilidad. Sin jerga y sin repetir la tabla.", 3),
        ],
        notas=[
            dict(p=1, t="Un límite de extensión", d="Primero el bosque, después los árboles"),
            dict(p=2, t="Tres cifras y una pregunta", d="Lo obliga a priorizar"),
            dict(p=3, t="Para quién escribe", d="La audiencia define el tono"),
        ],
        pasos=[
            "Con los datos verificados, pido una primera lectura.",
            "Máximo 10 líneas, porque quiero ver el bosque antes que los árboles.",
            "Tres cifras que resuman el año, qué mejoró, qué empeoró, y la pregunta más importante que debería "
            "hacerme.",
            "Y le indico para quién escribe: un directorio que no es experto en contabilidad. Ese detalle cambia "
            "por completo el tono de la respuesta.",
        ],
    ),
    dict(
        tipo="respuesta", titulo="La primera lectura: de tres archivos a una pregunta",
        bloques=[
            dict(p=0, h="Tres cifras", html='<p>Ventas <b class="pos">+18 %</b> · Utilidad neta <b class="pos">+3,9 %</b> · '
                                            'Efectivo <b class="neg">−89,5 %</b></p>'),
            dict(p=1, h="Mejoró", html="<p>Las ventas y el resultado operacional, que creció 12 %.</p>"),
            dict(p=2, h="Empeoró", html="<p>El margen bruto bajó de 26 % a 25 %. La deuda bancaria de corto plazo casi se "
                                        "triplicó: de 400 a 1.150 millones.</p>"),
            dict(p=3, h="La pregunta", html="<p><b>¿En qué se transformó la utilidad, si no se transformó en caja?</b></p>"),
        ],
        notas=[
            dict(p=4, t="De 3 archivos a 1 pregunta de negocio", d="El resto del curso es responderla"),
        ],
        pasos=[
            "Esta es la respuesta. Las tres cifras: ventas 18 % arriba, utilidad casi 4 % arriba, y el efectivo "
            "89 % abajo.",
            "Lo que mejoró: las ventas y el resultado operacional.",
            "Lo que empeoró: el margen bruto cayó un punto, de 26 a 25 %, y la deuda bancaria de corto plazo casi se "
            "triplicó, de 400 a 1.150 millones.",
            "Y la pregunta que propone: ¿en qué se transformó la utilidad, si no se transformó en caja?",
            "Fíjate en lo que acaba de pasar. En pocos minutos pasamos de tres archivos a una pregunta de negocio "
            "bien formulada. El resto del curso es responderla.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Paso 2 en resumen", estilo="check",
        items=[
            dict(p=0, t="Verifica antes de analizar", d="La lectura de los archivos y la coherencia entre estados"),
            dict(p=1, t="Pide la tabla de lo que leyó y las cuadraturas", d="Balance, patrimonio, caja y subtotales"),
            dict(p=2, t="Lee las observaciones", d="Ahí aparecen las primeras pistas"),
        ],
        pasos=[
            "El paso 2, en resumen. Verifica antes de analizar.",
            "Pide la tabla de lo que leyó, y las cuadraturas.",
            "Y lee con atención las observaciones, porque ahí aparecen las primeras pistas.",
        ],
    ),
]
