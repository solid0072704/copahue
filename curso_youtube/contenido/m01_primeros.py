"""Primeros pasos: dónde se usa Claude, cómo se instala y qué planes existen.

Lo que se afirma aquí se verificó en las páginas oficiales de Anthropic el 20 de noviembre de 2026:
claude.com/download (aplicaciones y sincronización entre equipos), claude.com/pricing (planes, precios en
dólares sin impuestos, límites y uso de los datos para entrenar) y el centro de ayuda (inicio de sesión con
correo o con Google, sin contraseña; aplicación para Android). Los precios y los nombres de los planes cambian:
por eso la narración los presenta como referencia y remite a la página oficial.
"""

DESCARGA = "claude.com/download"
PRECIOS = "claude.com/pricing"

LAMINAS = [
    dict(
        tipo="seccion", etiqueta="Primeros pasos", nombre="Cómo descargar Claude y qué plan elegir",
        bajada="Dónde se usa, cómo se instala y qué planes existen",
        pasos=[
            "Empecemos por lo primero: tener Claude instalado y saber qué plan te conviene. Si ya lo usas todos "
            "los días, puedes saltar al capítulo siguiente.",
        ],
    ),
    dict(
        tipo="accesos", titulo="Tres formas de usar Claude, con una sola cuenta", une=True,
        accesos=[
            dict(p=0, dibujo="navegador", k="En el navegador", t="claude.ai", d="Sin instalar nada: entras y empiezas"),
            dict(p=1, dibujo="escritorio", k="En tu computador", t="Aplicación de escritorio",
                 d="Para Mac y Windows · Linux, en beta"),
            dict(p=2, dibujo="telefono", k="En tu teléfono", t="Aplicación móvil", d="Para iPhone y Android"),
        ],
        pie=dict(p=3, t="La misma cuenta en las tres: tus conversaciones se sincronizan."),
        pasos=[
            "Claude se usa de tres formas. La más simple es en el navegador: entras a claude.ai y empiezas, sin "
            "instalar nada.",
            "La segunda es la aplicación de escritorio, para Mac y para Windows. Es la más cómoda si vas a "
            "trabajar con Claude todos los días.",
            "Y la tercera es la aplicación para el teléfono, en iPhone y en Android. Sirve para revisar un "
            "análisis o hacer una consulta cuando no estás en tu escritorio.",
            "En las tres inicias sesión con la misma cuenta, y tus conversaciones se sincronizan.",
        ],
    ),
    dict(
        tipo="instalar", titulo="Instalarlo toma un par de minutos",
        url=DESCARGA, rotulo="Sitio oficial", encabezado="Descarga Claude",
        botones=["Para Windows", "Para Mac", "Linux · beta"],
        descarga=dict(p=1, t="Instalador de Claude"),
        cuenta=dict(p=2, t="Crea tu cuenta", opciones=["Continuar con tu correo", "Continuar con Google"],
                    d="Sin contraseña: recibes un enlace para entrar"),
        notas=[
            dict(p=0, gr="01", t="Entra al sitio oficial", d=DESCARGA),
            dict(p=1, gr="02", t="Descarga e instala", d="El instalador para Mac o para Windows"),
            dict(p=2, gr="03", t="Crea tu cuenta", d="Con tu correo o con Google"),
            dict(p=3, gr="04", t="En tu teléfono", d="App Store o Google Play: «Claude by Anthropic»"),
        ],
        aviso=dict(p=4, t="Descarga solo desde el sitio oficial o la tienda de tu teléfono"),
        pasos=[
            "Instalarlo toma un par de minutos. Entra a la página oficial de descargas: la dirección está en "
            "pantalla, y también en la descripción del video.",
            "+ Descarga el instalador para tu sistema, Mac o Windows, y ábrelo.",
            "Después creas tu cuenta, con tu correo o con tu cuenta de Google. Con el correo no hay contraseña: "
            "Claude te envía un enlace para entrar.",
            "En el teléfono, busca la aplicación en la tienda, y fíjate en que el desarrollador sea Anthropic.",
            "Un cuidado: descarga siempre desde el sitio oficial o desde la tienda de tu teléfono. Así evitas las "
            "imitaciones.",
        ],
    ),
    dict(
        tipo="planes", titulo="Planes para personas: de gratis a uso intensivo",
        planes=[
            dict(p=0, nombre="Free", lema="Para probar Claude", precio="US$ 0", unidad="gratis",
                 items=["Web, escritorio y móvil", "Crea archivos y ejecuta código",
                        "Límite de uso cada 5 horas y cada semana"]),
            dict(p=1, nombre="Pro", lema="Para el trabajo de todos los días", precio="US$ 20", unidad="al mes",
                 alt="US$ 17 con pago anual",
                 items=["Más uso que Free", "Más modelos de Claude", "Claude en Chrome y en Microsoft 365"]),
            dict(p=2, nombre="Max", lema="Para trabajar todo el día", precio="US$ 100", unidad="al mes",
                 items=["5 veces el uso de Pro", "Prioridad en horas de alta demanda",
                        "Acceso anticipado a funciones nuevas"]),
            dict(p=3, nombre="Ultra", lema="Para las tareas más largas", precio="US$ 200", unidad="al mes",
                 items=["10 veces el uso semanal de Pro", "Sin el límite de 5 horas, con cuenta verificada"]),
        ],
        nota=dict(p=4, icono="reloj",
                  t=f"Precios de referencia en dólares, sin impuestos (noviembre de 2026) · confírmalos en {PRECIOS}"),
        pasos=[
            "Ahora, los planes. Para personas hay cuatro. El primero es Free, el plan gratuito: sirve para probar "
            "Claude, y tiene límites de uso cada 5 horas y cada semana.",
            "El segundo es Pro, que cuesta alrededor de 20 dólares al mes. Te da más uso, más modelos y más "
            "funciones: es el plan pensado para el trabajo de todos los días.",
            "Para un uso intensivo están los otros dos. Max cuesta 100 dólares al mes, y multiplica por 5 el uso "
            "de Pro.",
            "+ Y Ultra cuesta 200, con 10 veces el uso semanal de Pro.",
            "Son precios de referencia, en dólares y sin impuestos, al momento de grabar este curso. Cambian con "
            "el tiempo, así que confírmalos en la página oficial.",
        ],
    ),
    dict(
        tipo="dos", titulo="Planes para equipos y empresas",
        paneles=[
            dict(p=0, clase="oro", icono="personas", titulo="Team · de 2 a 150 personas", icono_item="check", items=[
                dict(t="Desde US$ 20 por persona al mes", d="Con pago anual; US$ 25 con pago mensual"),
                dict(t="Asientos estándar y premium", d="El premium tiene 5 veces más uso"),
                dict(t="Administración centralizada", d="Una sola factura, inicio de sesión único y control de conectores"),
            ]),
            dict(p=1, clase="azul", icono="banco", titulo="Enterprise · desde 20 personas", icono_item="check", items=[
                dict(t="US$ 20 por asiento al mes, más el uso", d="Pago anual; el uso se cobra a tarifa de API"),
                dict(t="Todo lo del plan Team", d="Y límites de gasto por usuario y por organización"),
                dict(t="Seguridad y cumplimiento", d="Roles, registros de auditoría y retención de datos a medida"),
            ]),
        ],
        pie=dict(p=2, t="En ambos, por defecto, tu contenido no se usa para entrenar los modelos."),
        pasos=[
            "Para equipos y empresas hay dos planes más. Team es para equipos de 2 a 150 personas. Parte en unos "
            "20 dólares por persona al mes, y suma facturación y administración centralizadas.",
            "Enterprise es para organizaciones más grandes, desde 20 personas. Agrega controles de seguridad y de "
            "cumplimiento: roles, registros de auditoría y retención de datos a medida.",
            "Y hay una diferencia que en finanzas importa. En estos dos planes, por defecto, tu contenido no se "
            "usa para entrenar los modelos. En los planes personales lo decides tú, en la configuración de "
            "privacidad.",
        ],
    ),
    dict(
        tipo="tabla", titulo="¿Qué plan te conviene?", texto=True, flex=2.3,
        columnas=["Si…", "Plan", "Por qué"], clases_col=["", "fuerte", ""],
        filas=[
            dict(c=["Vas a seguir este curso", "Free", "Alcanza para empezar; tiene límites de uso"], p=0),
            dict(c=["Analizas tus números todos los meses", "Pro", "Más uso para trabajar con tus archivos"], p=1),
            dict(c=["Lo usas todo el día", "Max o Ultra", "5 y 10 veces el uso de Pro"], p=2),
            dict(c=["Trabajas en equipo con datos de la empresa", "Team o Enterprise",
                    "Administración central y privacidad por defecto"], p=3),
        ],
        notas=[
            dict(p=4, t="También hay un plan Education", d="Para universidades: estudiantes, docentes y personal"),
        ],
        pie=dict(p=5, t="Empieza gratis y cambia de plan solo cuando el límite de uso te frene."),
        pasos=[
            "¿Cuál te conviene? Para seguir este curso, el plan gratuito alcanza para empezar, aunque es probable "
            "que en algún momento llegues al límite de uso.",
            "Si vas a analizar tus estados financieros todos los meses, lo razonable es Pro.",
            "+ Si lo usas todo el día, Max o Ultra.",
            "+ Y si trabajas en equipo con información de la empresa, Team o Enterprise: por la administración "
            "central y por la privacidad.",
            "Para universidades existe, además, un plan educativo.",
            "Mi recomendación: empieza gratis, y cambia de plan solo cuando el límite de uso te frene.",
        ],
    ),
]
