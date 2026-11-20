"""Paso 8: escenarios, plan de acción e informe."""

from .. import caso, graficos
from ._util import m

E = caso.escenario()
ES = caso.escenarios()
B25 = caso.balance("2025")

LAMINAS = [
    dict(
        tipo="seccion", paso=8, etiqueta="Paso 8", nombre="Decide: del diagnóstico a la acción",
        bajada="Qué vamos a hacer, quién, cuándo y cuánto vale",
        pasos=[
            "Paso 8: decidir. Todo lo anterior no sirve de nada si se queda en un diagnóstico. El último paso es "
            "convertir el análisis en decisiones: qué vamos a hacer, quién, cuándo, y cuánto vale.",
        ],
    ),
    dict(
        tipo="prompt", titulo="Prompt 9: ponerles número a las alternativas", numero=9, nombre="Palancas y escenarios", une=True,
        alto=True,
        segs=[
            ("Con todo lo analizado, cuantifica el impacto de estas\n"
             "palancas:\n\n", 0),
            ("a) Reducir los días de cobro a [60].\n"
             "b) Reducir los días de inventario a [60].\n"
             "c) Recuperar [1] punto de margen bruto.\n"
             "d) Volver el gasto en fletes al % de ventas del año\n"
             "   anterior.\n\n", 1),
            ("Para cada una: impacto en caja, impacto en el resultado\n"
             "anual y supuestos.\n", 2),
            ("Después arma tres escenarios (conservador, base y\n"
             "optimista) y muéstrame cómo quedarían la deuda de corto\n"
             "plazo y el ciclo de conversión de efectivo.", 3),
        ],
        notas=[
            dict(p=1, t="4 palancas del diagnóstico", d="Cobranza, inventario, margen y fletes"),
            dict(p=2, t="Caja, resultado y supuestos", d="Para cada palanca"),
            dict(p=3, t="Tres escenarios", d="Conservador, base y optimista"),
            dict(p=4, t="Tú pones las metas", d="Claude hace la aritmética"),
        ],
        pasos=[
            "Para decidir hay que ponerles número a las alternativas.",
            "En este prompt le entrego cuatro palancas, que salen directamente del diagnóstico. Reducir los días de "
            "cobro a 60. Reducir los días de inventario a 60. Recuperar un punto de margen bruto. Y volver los "
            "fletes al porcentaje de ventas del año anterior.",
            "Para cada una le pido el impacto en caja, el impacto en el resultado anual y los supuestos.",
            "Y después, tres escenarios, para ver cómo quedarían la deuda de corto plazo y el ciclo de conversión "
            "de efectivo.",
            "Las metas las pongo yo, porque yo conozco el negocio y sé qué es alcanzable. Claude hace la aritmética "
            "y me muestra las consecuencias.",
        ],
    ),
    dict(
        tipo="cifras", titulo="Cuánto vale cada palanca", cols=4,
        tiles=[
            dict(p=0, eti="Cobranza a 60 días", val=m(E["libera_cobranza"]), sub="millones de caja liberada", tono="pos"),
            dict(p=1, eti="Inventario a 60 días", val=m(E["libera_inventario"]), sub="millones de caja liberada", tono="pos"),
            dict(p=3, eti="1 punto de margen", val=m(E["punto_margen"]), sub="millones de resultado al año"),
            dict(p=3, eti="Fletes a 4,3 % de las ventas", val=m(E["ahorro_fletes"]), sub="millones de resultado al año"),
        ],
        pie=dict(p=2, t=f"{m(E['caja_liberada'])} millones de caja: la deuda de corto plazo baja de {m(B25['deuda_cp'])} "
                        f"a {m(E['deuda_cp_despues'])}, y el ciclo, de 92 a 72 días"),
        sub_pie=dict(p=4, icono="bandera", t="La prioridad: cobrar mejor y comprar mejor, antes que vender más"),
        pasos=[
            "Estos son los resultados. Bajar los días de cobro de 72 a 60 libera 390 millones de caja.",
            "Bajar los días de inventario de 68 a 60 libera otros 195.",
            "En conjunto, 585 millones. Con eso la deuda de corto plazo baja de 1.150 a 565 millones, a menos de la "
            "mitad, y la empresa se ahorra unos 63 millones al año en intereses. El ciclo de conversión de efectivo "
            "baja de 92 a 72 días.",
            "Las otras dos palancas van al resultado. Recuperar un punto de margen son 118 millones al año. Y "
            "volver los fletes a 4,3 % de las ventas, 53 millones más.",
            "Fíjate en el orden de magnitud. Las palancas de caja valen 585 millones; las de resultado, 171. La "
            "prioridad de esta empresa no es vender más: es cobrar mejor y comprar mejor.",
        ],
    ),
    dict(
        tipo="tabla", titulo="Tres escenarios, para mostrar un rango", compacta=True, flex=1, alto=False,
        columnas=["", "Hoy", "Conservador", "Base", "Optimista"],
        filas=[
            dict(c=["Días de cobro", "72", str(ES["Conservador"]["dias_cobro"]), str(ES["Base"]["dias_cobro"]), str(ES["Optimista"]["dias_cobro"])], aparece=0),
            dict(c=["Días de inventario", "68", str(ES["Conservador"]["dias_inventario"]), str(ES["Base"]["dias_inventario"]), str(ES["Optimista"]["dias_inventario"])], aparece=0),
            dict(c=["Caja liberada (millones)", "—", (m(ES["Conservador"]["caja_liberada"]), "pos"), (m(ES["Base"]["caja_liberada"]), "pos"),
                    (m(ES["Optimista"]["caja_liberada"]), "pos")], cls="total", p="0,1,2", aparece=0),
            dict(c=["Deuda de corto plazo (millones)", m(B25["deuda_cp"]), m(ES["Conservador"]["deuda_cp"]), m(ES["Base"]["deuda_cp"]),
                    m(ES["Optimista"]["deuda_cp"])], aparece=0),
            dict(c=["Ciclo de conversión de efectivo (días)", "92", str(ES["Conservador"]["ciclo"]), str(ES["Base"]["ciclo"]),
                    str(ES["Optimista"]["ciclo"])], aparece=0),
        ],
        pie=dict(p=3, t="Lo valioso no es acertar el número: es mostrar el rango y dejar por escrito los supuestos."),
        pasos=[
            "Y estos son los tres escenarios. En el conservador, la cobranza baja solo a 66 días y el inventario a "
            "64: se liberan cerca de 300 millones, y la deuda de corto plazo queda cerca de 860.",
            "En el escenario base, con 60 y 60 días, se liberan 585 millones y la deuda baja a 565.",
            "Y en el optimista, con 55 días en ambos, se liberan casi 870 millones, y la deuda de corto plazo baja a "
            "menos de 300.",
            "Lo valioso de este ejercicio no es acertar el número. Es mostrarle al directorio el rango: qué pasa si "
            "nos va regular, qué pasa si cumplimos, y qué pasa si nos va muy bien. Y dejar por escrito los supuestos "
            "de cada escenario, para revisarlos en tres meses.",
        ],
    ),
    dict(
        tipo="svg", titulo="Una herramienta más: la proyección de caja a 13 semanas",
        svg=graficos.proyeccion_caja(),
        notas=[
            dict(p=0, t="Tú entregas", d="Saldo inicial, cobranzas esperadas y pagos comprometidos"),
            dict(p=1, gr="3", tono="neg", t="millones en la semana 4", d="Después de pagar los sueldos"),
            dict(p=2, gr="Semana 8", tono="neg", t="La caja es negativa", clase="rojo"),
            dict(p=3, gr="163", tono="pos", t="millones al cierre, con el plan", clase="verde"),
        ],
        pasos=[
            "Con el diagnóstico hecho, hay una herramienta más que te recomiendo pedirle: una proyección de caja a "
            "13 semanas. Le entregas el saldo inicial, las cobranzas esperadas y los pagos comprometidos: "
            "proveedores, sueldos, impuestos y cuotas de créditos.",
            "Esta es la proyección de Alerce si nada cambia. La caja parte en 65 millones. En la semana 4, después "
            "de pagar los sueldos, queda en 3 millones.",
            "Y en la semana 8 es negativa. Sin más crédito, la empresa no llega a fin de trimestre.",
            "Con el plan de cobranza e inventario en marcha, la caja se mantiene positiva, y termina el trimestre "
            "en torno a 160 millones.",
            "Sigue siendo una holgura pequeña para una empresa que paga más de 80 millones en sueldos cada mes. Por "
            "eso esta planilla no es un lujo: es la herramienta de gestión más importante del próximo trimestre.",
        ],
    ),
    dict(
        tipo="tabla", titulo="El plan de acción: cuatro columnas, siempre", texto=True, flex=1,
        columnas=["Acción", "Responsable", "Plazo", "Indicador"],
        filas=[
            dict(c=["Cobranza a 60 días", "Gerencia comercial", "Junio", "Días de cobro"], aparece=1),
            dict(c=["Inventario a 60 días", "Operaciones", "Junio", "Días de inventario"], aparece=1),
            dict(c=["Revisar las condiciones con las cadenas", "Gerencia general", "Marzo", "Margen por cliente"], aparece=2),
            dict(c=["Proyección de caja semanal", "Finanzas", "Desde enero", "Saldo mínimo de caja"], aparece=2),
            dict(c=["Línea de crédito a un plazo mayor", "Finanzas", "Marzo", "Deuda de corto plazo"], aparece=2),
        ],
        pie=dict(p=3, t="Un plan sin responsable y sin fecha es una lista de buenos deseos."),
        pasos=[
            "Con los números claros, le pido el plan de acción en una tabla: acción, responsable, plazo e indicador.",
            "Cobranza a 60 días, a cargo de la gerencia comercial, medida en días de cobro. Inventario a 60 días, a "
            "cargo de operaciones.",
            "Revisar las condiciones con las cadenas, medido en margen por cliente. La proyección de caja semanal. "
            "Y renegociar la línea de crédito a un plazo mayor.",
            "Un plan sin responsable y sin fecha es una lista de buenos deseos. Y un plan sin indicador no se puede "
            "seguir. Pídele siempre las cuatro columnas, y después ajusta tú los nombres y las fechas, porque eso "
            "Claude no lo puede saber.",
        ],
    ),
    dict(
        tipo="prompt", titulo="Prompt 10: el informe para el directorio", numero=10, nombre="Informe ejecutivo de una página",
        alto=True,
        segs=[
            ("Redacta un informe ejecutivo de una página para el\n"
             "directorio.\n\n", 0),
            ("Estructura:\n"
             "1. Mensaje principal, en una frase.\n"
             "2. Qué pasó: 3 cifras clave.\n"
             "3. Por qué pasó: las 3 causas, con su evidencia.\n"
             "4. Qué proponemos: acciones con responsable, plazo e\n"
             "   indicador de seguimiento.\n"
             "5. Qué necesitamos del directorio: decisiones concretas.\n\n", 1),
            ("Tono directo y sin jerga. Cada cifra debe salir de los\n"
             "análisis anteriores. Si un dato es una estimación, dilo.", 2),
        ],
        notas=[
            dict(p=0, t="Una página", d="Lo que un directorio alcanza a leer"),
            dict(p=1, t="5 partes", d="Mensaje, qué pasó, por qué, qué proponemos y qué pedimos"),
            dict(p=2, t="Sin números nuevos", d="Y separando los datos de las estimaciones"),
        ],
        pasos=[
            "Y el último prompt: el informe. Una página, para el directorio.",
            "Con cinco partes. El mensaje principal, en una frase. Qué pasó, con tres cifras. Por qué pasó, con las "
            "tres causas y su evidencia. Qué proponemos, con responsable, plazo e indicador de seguimiento. Y qué "
            "necesitamos del directorio: las decisiones concretas que le estamos pidiendo.",
            "Y dos reglas de cierre. Que cada cifra salga de los análisis anteriores, para que no aparezcan números "
            "nuevos. Y que distinga los datos de las estimaciones.",
        ],
    ),
    dict(
        tipo="memo", titulo="El informe: una página, cinco partes", alto=True,
        encabezado="Distribuidora Alerce · Informe al directorio", fecha="Cierre 2025 · borrador",
        secciones=[
            dict(p=0, h="Mensaje principal", html='<p class="mensaje">Crecimos 18 % en ventas, pero financiamos ese '
                                                  'crecimiento con nuestra caja y con deuda de corto plazo. La prioridad es '
                                                  'recuperar liquidez.</p>'),
            dict(p=1, h="Qué pasó", html='<div class="tres"><div><b>+18 %</b>Ventas</div><div><b>485</b>Utilidad neta (+3,9 %)</div>'
                                         '<div><b>620 → 65</b>Caja</div></div>'),
            dict(p=1, h="Por qué", html="<ul><li>El ciclo de conversión de efectivo pasó de 55 a 92 días</li>"
                                        "<li>El margen bruto bajó 1 punto, por los descuentos a las cadenas</li>"
                                        "<li>La deuda de corto plazo casi se triplicó; los gastos financieros subieron 59 %</li></ul>"),
            dict(p=2, h="Qué proponemos", html="<ul><li>Cobranza a 60 días · Gerencia comercial · junio</li>"
                                               "<li>Inventario a 60 días · Operaciones · junio</li>"
                                               "<li>Revisar las condiciones con las cadenas · Gerencia general · marzo</li>"
                                               "<li>Proyección de caja semanal · Finanzas · desde enero</li></ul>"),
            dict(p=3, h="Qué pedimos al directorio", html="<ul><li>Suspender los dividendos hasta normalizar la caja</li>"
                                                          "<li>Aprobar la renegociación de la línea de crédito a un plazo mayor</li></ul>"),
        ],
        pasos=[
            "Este es el informe. El mensaje principal: Alerce creció 18 % en ventas, pero financió ese crecimiento "
            "con su caja y con deuda de corto plazo; la prioridad es recuperar liquidez.",
            "Después, las tres cifras y las tres causas.",
            "Las acciones, cada una con responsable y plazo.",
            "Y al final, lo que se le pide al directorio: suspender los dividendos hasta normalizar la caja, y "
            "aprobar la renegociación de la línea de crédito a un plazo mayor.",
            "Ahora, una advertencia importante. Este borrador sale muy rápido, pero no lo envíes tal cual. Los "
            "responsables, los plazos y las decisiones que pides son tuyos. Claude te da la estructura y la "
            "redacción; el contenido lo validas tú.",
            "Ese es el equilibrio correcto: la máquina hace el trabajo pesado, y la persona pone el criterio.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Tres consejos para presentar el informe",
        items=[
            dict(p=0, t="Empieza por el mensaje, no por la tabla", d="La primera frase debe sostenerse sola"),
            dict(p=1, t="Lleva respondidas las tres preguntas más difíciles", d="Las encuentras con el prompt del paso 7"),
            dict(p=2, t="Termina con lo que pides", d="Una decisión, un responsable y una fecha"),
        ],
        lateral=[dict(p=3, clase="oro", frase="Un informe que no pide nada no cambia nada.", ancho=500)],
        pasos=[
            "Tres consejos para presentar el informe. Uno: empieza por el mensaje, no por la tabla. Si la primera "
            "frase no se sostiene sola, el informe todavía no está listo.",
            "Dos: lleva respondidas las tres preguntas más difíciles. Ya sabes cómo encontrarlas: con el prompt del "
            "paso 7.",
            "Y tres: termina con lo que pides. Una decisión, un responsable y una fecha.",
            "Un informe que no pide nada no cambia nada.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Paso 8 en resumen", estilo="check",
        items=[
            dict(p=0, t="Ponle número a cada palanca", d="Impacto en caja y en resultado"),
            dict(p=1, t="Muestra un rango", d="Tres escenarios, con sus supuestos por escrito"),
            dict(p=2, t="Plan con cuatro columnas, informe de una página", d="Y el contenido lo validas tú"),
        ],
        pasos=[
            "El paso 8, en resumen. Ponle número a cada palanca.",
            "Muestra un rango, con tres escenarios.",
            "Y termina con un plan de cuatro columnas, y un informe de una página que validas tú.",
        ],
    ),
]
