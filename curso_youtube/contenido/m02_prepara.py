"""Paso 1: preparar los datos y el contexto."""

from .. import caso
from ._util import formato, m, proyecto

R24, R25, RP = caso.eerr("2024"), caso.eerr("2025"), caso.eerr("ppto")

LAMINAS = [
    dict(
        tipo="seccion", paso=1, etiqueta="Paso 1", nombre="Prepara los datos y el contexto",
        bajada="La calidad del análisis depende de lo que le entregas",
        pasos=[
            "Paso 1: preparar la información. La calidad del análisis depende, antes que nada, de lo que le "
            "entregas. Este paso es el menos vistoso, y el que más diferencia hace.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Qué reunir antes de empezar", columnas=2, une=True,
        items=[
            dict(p=0, icono="tabla", t="Balance", d="Activos, pasivos y patrimonio"),
            dict(p=1, icono="barras", t="Estado de resultados", d="Ventas, costos, gastos y utilidad"),
            dict(p=2, icono="peso", t="Flujo de efectivo", d="Operación, inversión y financiamiento"),
            dict(p=3, icono="diana", t="Presupuesto del período", d="Para comparar contra lo planificado"),
            dict(p=4, icono="lapiz", t="Notas y aclaraciones", d="Lo que solo tú sabes del año"),
            dict(p=5, icono="calendario", t="Dos períodos o más", d="Lo que importa es cómo cambió"),
        ],
        pasos=[
            "Reúne cinco cosas. El balance.",
            "+ El estado de resultados.",
            "+ El estado de flujo de efectivo.",
            "+ El presupuesto del período.",
            "+ Y las notas o aclaraciones que solo tú conoces.",
            "Idealmente de dos períodos o más, porque un número aislado no dice mucho: lo que importa es cómo "
            "cambió.",
            "Si no tienes flujo de efectivo, no te preocupes: con dos balances y un estado de resultados, Claude "
            "puede construirlo, y lo vamos a ver en el paso 5.",
        ],
    ),
    dict(
        tipo="html", titulo="El formato importa: Excel antes que una foto",
        html=formato([
            dict(p=0, estrellas=3, t="Planilla Excel o archivo CSV", d="Una fila por cuenta y una columna por período"),
            dict(p=1, estrellas=2, t="PDF generado por tu sistema contable", d="Texto real, que se lee sin errores"),
            dict(p=2, estrellas=1, t="PDF escaneado o foto", d="Claude interpreta la imagen y puede confundir un 3 con un 8"),
        ]),
        notas=[
            dict(p=3, t="Indica la unidad", d="Pesos, miles o millones"),
            dict(p=4, t="Marca los subtotales", d="Que se distingan de las partidas"),
            dict(p=5, t="Usa los mismos nombres", d="De cuenta, en todos los períodos"),
        ],
        pasos=[
            "Sobre el formato. Lo mejor es una planilla Excel o un archivo CSV, con una fila por cuenta y una "
            "columna por período.",
            "Lo segundo mejor es un PDF generado por tu sistema contable.",
            "Y lo que más problemas da es un PDF escaneado o una foto, porque ahí Claude tiene que interpretar la "
            "imagen, y puede confundir un 3 con un 8.",
            "Tres cuidados más. Indica la unidad: si son pesos, miles o millones.",
            "+ Deja los subtotales claramente identificados.",
            "+ Y usa los mismos nombres de cuenta en todos los períodos.",
        ],
    ),
    dict(
        tipo="tabla", titulo="Así se ve un archivo bien preparado", compacta=True, flex=1.9,
        previo=dict(p=0, pill="ARCHIVO", t="Distribuidora Alerce · Estado de resultados · Cifras en millones de pesos"),
        columnas=["Cuenta", "2024", "2025", "Presupuesto 2025"],
        filas=[
            dict(c=["Ventas", m(R24["ventas"]), m(R25["ventas"]), m(RP["ventas"])], aparece=0),
            dict(c=["Costo de ventas", m(-R24["costo"]), m(-R25["costo"]), m(-RP["costo"])], aparece=0),
            dict(c=["Margen bruto", m(R24["margen_bruto"]), m(R25["margen_bruto"]), m(RP["margen_bruto"])], cls="total", p=2, aparece=0),
            dict(c=["Remuneraciones", m(-R24["remuneraciones"]), m(-R25["remuneraciones"]), m(-RP["remuneraciones"])], aparece=0),
            dict(c=["Distribución y fletes", m(-R24["distribucion"]), m(-R25["distribucion"]), m(-RP["distribucion"])], aparece=0),
            dict(c=["Arriendos y otros gastos", m(-R24["otros_gav"]), m(-R25["otros_gav"]), m(-RP["otros_gav"])], aparece=0),
            dict(c=["Depreciación", m(-R24["depreciacion"]), m(-R25["depreciacion"]), m(-RP["depreciacion"])], aparece=0),
            dict(c=["Resultado operacional", m(R24["ebit"]), m(R25["ebit"]), m(RP["ebit"])], cls="total", p=2, aparece=0),
        ],
        notas=[
            dict(p=0, t="Una fila por cuenta", d="Con el mismo nombre en todos los períodos"),
            dict(p=1, t="Una columna por período", d="Y el presupuesto al lado"),
            dict(p=2, t="Subtotales destacados", d="Y la unidad escrita arriba"),
        ],
        pasos=[
            "Así se ve un archivo bien preparado. Una fila por cuenta, con el mismo nombre en todos los períodos.",
            "Una columna por período, y el presupuesto al lado.",
            "Los subtotales, destacados. Y la unidad, escrita arriba: cifras en millones de pesos.",
            "No necesitas nada más sofisticado. De hecho, este es el archivo del caso, que puedes descargar desde "
            "la descripción.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Antes de subir datos reales: confidencialidad",
        items=[
            dict(p=0, icono="escudo", t="Revisa la política de tu empresa",
                 d="¿Se puede usar inteligencia artificial con información financiera?"),
            dict(p=1, icono="candado", t="Revisa la privacidad de tu cuenta",
                 d="Planes para empresas: por defecto no se entrena con tus datos. Planes personales: lo decides tú "
                   "en la configuración."),
            dict(p=2, icono="ojo", t="Si tienes dudas, anonimiza", d="«Empresa A», «Cliente B»: el análisis funciona igual"),
        ],
        pasos=[
            "Antes de subir información real, detente un minuto en la confidencialidad. Primero, revisa la política "
            "de tu empresa: si puedes o no usar herramientas de inteligencia artificial con información financiera.",
            "Segundo, revisa la configuración de privacidad de tu cuenta. En los planes para empresas, Anthropic "
            "indica que por defecto no usa tus datos para entrenar sus modelos. En los planes personales, tú decides "
            "en la configuración si tus conversaciones pueden usarse para ese fin. Las condiciones cambian, así que "
            "revísalas tú mismo.",
            "Y tercero, si tienes dudas, anonimiza: cambia el nombre de la empresa, y reemplaza los clientes por "
            "Cliente A y Cliente B. El análisis funciona exactamente igual.",
        ],
    ),
    dict(
        tipo="cifras", titulo="El caso: Distribuidora Alerce", cols=3,
        sobre=dict(p=0, t="**Mayorista de alimentos y abarrotes** para almacenes, restaurantes y hoteles · empresa ficticia"),
        tiles=[
            dict(p=1, eti="Ventas 2025", val="11.800", sub="+18 % sobre 2024 · millones de pesos", tono="pos"),
            dict(p=2, eti="Utilidad neta", val="485", sub="La más alta de su historia", tono="pos"),
            dict(p=3, eti="Caja al cierre", val="65", sub="Eran 620 a comienzos de año", tono="neg"),
        ],
        pie=dict(p=4, t="¿Cómo es posible vender más, ganar más y quedarse sin caja?"),
        pasos=[
            "Presentemos entonces a la empresa del caso. Distribuidora Alerce es una mayorista de alimentos y "
            "abarrotes, que abastece a almacenes, restaurantes y hoteles. Es ficticia, pero se parece a muchas "
            "empresas reales.",
            "En 2025 vendió 11.800 millones de pesos, 18 % más que el año anterior.",
            "La utilidad fue de 485 millones, la más alta de su historia. El dueño estaba contento.",
            "Hasta que en diciembre miró la cuenta corriente: de 620 millones en caja a comienzos de año, quedaban 65.",
            "¿Cómo es posible vender más, ganar más y quedarse sin caja? Esa es la pregunta que le vamos a hacer a "
            "Claude.",
        ],
    ),
    dict(
        tipo="prompt", titulo="Prompt 1: el contexto, antes de cualquier análisis", numero=1, nombre="Contexto y reglas de trabajo",
        alto=True,
        segs=[
            ("Actúa como un analista financiero sénior. Voy a pedirte que\n"
             "analices los estados financieros de una empresa.\n\n", 1),
            ("CONTEXTO\n"
             "- Empresa: [nombre o «Empresa A»], [industria], [país].\n"
             "- Tamaño: ventas anuales de [monto]. Cifras en [unidad].\n"
             "- Períodos: [2024 y 2025], más el presupuesto [2025].\n", 2),
            ("- Hechos relevantes del año: [qué cambió].\n"
             "- Lo que más me preocupa: [pregunta de negocio].\n\n", 3),
            ("REGLAS\n"
             "1. Usa solo los datos que te entrego. Si falta algo, pregunta.\n"
             "2. Calcula con código y muestra la fórmula.\n"
             "3. Separa los hechos de tus hipótesis.\n", 4),
            ("4. No concluyas todavía: primero confirma qué entendiste.", 5),
        ],
        notas=[
            dict(p=1, t="Un rol", d="Fija el nivel y el tono"),
            dict(p=2, t="El contexto", d="Empresa, tamaño, unidad y períodos"),
            dict(p=3, t="Lo que no está en los números", d="Hechos del año y tu preocupación"),
            dict(p=4, t="Reglas de trabajo", d="Datos, cálculos, hechos e hipótesis"),
            dict(p=5, t="Confirmar antes de concluir", d="Detecta malentendidos a tiempo"),
        ],
        pasos=[
            "Ahora sí, el primer prompt. No es un análisis: es el contexto. Fíjate en la estructura.",
            "Parte con un rol: actúa como un analista financiero sénior.",
            "Después viene el contexto: qué empresa es, de qué industria, de qué tamaño, en qué unidad están las "
            "cifras y qué períodos le entregas.",
            "Incluye los hechos relevantes del año, lo que no aparece en los números. En nuestro caso: se "
            "incorporaron dos cadenas como clientes y se abrió una línea de crédito. Y agrega lo que más te "
            "preocupa, porque eso orienta todo el análisis.",
            "Al final van cuatro reglas. Que use solo tus datos, y pregunte si falta algo. Que calcule con código y "
            "muestre la fórmula. Que separe los hechos de las hipótesis.",
            "Y la que más me gusta: que no concluya todavía, que primero te confirme qué entendió. Con esa regla "
            "detectas cualquier malentendido antes de que contamine el análisis.",
        ],
    ),
    dict(
        tipo="puntos", titulo="¿Qué hechos del año conviene contarle?", columnas=2, compacto=True,
        items=[
            dict(p=0, icono="personas", t="Clientes", d="Quién entró y quién se fue"),
            dict(p=1, icono="porciento", t="Precios y descuentos", d="Qué cambió y desde cuándo"),
            dict(p=2, icono="camion", t="Proveedores y costos", d="Cambios de proveedor o alzas relevantes"),
            dict(p=3, icono="banco", t="Inversiones y créditos", d="Lo que compraste y cómo lo financiaste"),
            dict(p=4, icono="libro", t="Criterios contables", d="Si algo se registra distinto que antes"),
            dict(p=5, icono="alerta", t="Eventos extraordinarios", d="Un siniestro, un juicio o la venta de un activo"),
        ],
        pasos=[
            "¿Qué hechos del año conviene contarle? Los que explican cambios, y no aparecen en los números. "
            "Clientes: quién entró y quién se fue.",
            "+ Cambios de precios y de descuentos.",
            "+ Cambios de proveedores o de costos.",
            "+ Inversiones y créditos nuevos.",
            "+ Cambios de criterio contable.",
            "+ Y eventos extraordinarios, como un siniestro, un juicio o la venta de un activo. Con cinco o seis "
            "líneas basta.",
        ],
    ),
    dict(
        tipo="prompt", titulo="El prompt 1, con los datos de Alerce", numero=1, nombre="Contexto · ejemplo completo",
        alto=True,
        segs=[
            ("Actúa como un analista financiero sénior. Voy a pedirte que\n"
             "analices los estados financieros de una empresa.\n\n", None),
            ("CONTEXTO\n"
             "- Empresa: Distribuidora Alerce, mayorista de alimentos y\n"
             "  abarrotes, Chile.\n"
             "- Tamaño: ventas anuales de 11.800. Cifras en millones de\n"
             "  pesos.\n"
             "- Períodos: 2024 y 2025, más el presupuesto 2025.\n", 0),
            ("- Hechos relevantes del año: incorporamos dos cadenas como\n"
             "  clientes, con descuento y pago a 90 días o más. Ampliamos\n"
             "  el surtido. Abrimos una línea de crédito.\n", 1),
            ("- Lo que más me preocupa: vendimos más y ganamos más, pero\n"
             "  la caja bajó de 620 a 65.", 2),
        ],
        notas=[
            dict(p=0, t="Empresa, unidad y períodos", d="Tres líneas que evitan los errores más comunes"),
            dict(p=1, t="Los hechos del año", d="Un minuto tuyo le ahorra a Claude suponer"),
            dict(p=2, t="La preocupación, en simple", d="Orienta todo el análisis"),
        ],
        pasos=[
            "Así queda el prompt con los datos de Alerce. La empresa, su giro y el país. Las ventas y la unidad: "
            "millones de pesos. Y los períodos.",
            "Fíjate en los hechos relevantes: incorporamos dos cadenas como clientes, con descuento y con pago a "
            "90 días o más; ampliamos el surtido; abrimos una línea de crédito. Son tres líneas que a ti te toman un "
            "minuto, y que le ahorran a Claude suponer.",
            "Y la preocupación, dicha en simple: vendimos más, ganamos más, pero la caja bajó de 620 a 65.",
        ],
    ),
    dict(
        tipo="respuesta", titulo="La respuesta: Claude confirma antes de analizar",
        bloques=[
            dict(p=0, h="Lo que entendí", html="<p>Alerce es una distribuidora mayorista. Analizaré 2024, 2025 y el "
                                               "presupuesto 2025, en millones de pesos. La pregunta central: por qué la caja "
                                               "bajó de 620 a 65 si las ventas y la utilidad crecieron.</p>"),
            dict(p=1, h="Antes de empezar, dos preguntas", html="<ol><li>¿El presupuesto está en la misma unidad que los "
                                                                "estados financieros?</li><li data-p=\"2\">¿Tienes el detalle de "
                                                                "ventas y cuentas por cobrar por cliente? Serviría para medir el "
                                                                "efecto de las dos cadenas.</li></ol>"),
        ],
        notas=[
            dict(p=0, t="Confirma lo que entendió", d="Empresa, períodos, unidad y pregunta central"),
            dict(p=1, t="Pregunta por la unidad", d="Un detalle que causa errores muy frecuentes"),
            dict(p=3, t="Ya piensa en el negocio", d="Pide justo el dato que vamos a necesitar"),
        ],
        pasos=[
            "Y esta es la respuesta. Como le pedimos que no concluyera, Claude parte confirmando lo que entendió: "
            "qué empresa es, qué períodos va a analizar, en qué unidad, y cuál es la pregunta central.",
            "Después hace dos preguntas. Una: si el presupuesto está en la misma unidad que los estados "
            "financieros. Parece un detalle, pero es un error muy frecuente.",
            "Y dos: si tenemos el detalle de ventas y cuentas por cobrar por cliente, para medir el efecto de las "
            "dos cadenas.",
            "Esa segunda pregunta es justamente el análisis que vamos a hacer más adelante. Cuando el contexto es "
            "bueno, Claude empieza a pensar como alguien que conoce el negocio.",
        ],
    ),
    dict(
        tipo="html", titulo="Para no repetir el contexto cada mes: los Proyectos",
        html=proyecto("Proyecto: Distribuidora Alerce", [
            dict(p=1, h="Instrucciones", t="El prompt de contexto y tus reglas de trabajo"),
            dict(p=2, h="Archivos de referencia", items=[("doc", "Plan de cuentas"), ("doc", "Definiciones de los indicadores"),
                                                         ("doc", "Informes de meses anteriores")]),
            dict(p=3, h="Conversaciones", t="Cierre de enero · Cierre de febrero · Cierre de marzo…"),
        ]),
        notas=[
            dict(p=0, t="Un proyecto por empresa", d="Un espacio con instrucciones permanentes y archivos de referencia"),
            dict(p=3, t="Cada conversación parte con contexto", d="Ya sabe quién es la empresa y cómo te gusta trabajar"),
        ],
        pasos=[
            "Un consejo para no repetir este contexto cada mes: usa los Proyectos de Claude. Un proyecto es un "
            "espacio de trabajo donde dejas instrucciones permanentes y archivos de referencia. Yo creo un proyecto "
            "por empresa.",
            "En las instrucciones pego el prompt de contexto.",
            "Y como archivos de referencia subo el plan de cuentas, las definiciones de los indicadores y los "
            "informes de meses anteriores.",
            "Así, cada conversación nueva dentro del proyecto parte sabiendo quién es la empresa y cómo te gusta "
            "trabajar.",
        ],
    ),
    dict(
        tipo="puntos", titulo="Paso 1 en resumen", estilo="check",
        items=[
            dict(p=0, t="Reúne los estados, el presupuesto y tus notas", d="De dos períodos o más"),
            dict(p=1, t="Cuida el formato y la confidencialidad", d="Excel antes que una foto; anonimiza si tienes dudas"),
            dict(p=2, t="Parte por el contexto", d="Y pídele que confirme qué entendió antes de analizar"),
        ],
        pasos=[
            "El paso 1, en resumen. Reúne los estados financieros, el presupuesto y tus notas, de dos períodos o más.",
            "Cuida el formato y la confidencialidad.",
            "Y parte siempre por el contexto, pidiéndole a Claude que confirme qué entendió antes de analizar.",
        ],
    ),
]
