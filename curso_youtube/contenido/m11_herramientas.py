"""Para ir más lejos: artefactos, conectores, skills y plugins de Claude en la gestión financiera.

Lo que se afirma aquí se verificó en la documentación de Anthropic (noviembre de 2026). Los nombres y la
disponibilidad por plan cambian: por eso la narración no nombra planes y remite a la configuración de la cuenta.
"""

from .. import caso
from ._util import artefacto, conectores, m, puntos_con_icono

R24, R25 = caso.ratios("2024"), caso.ratios("2025")
BASE = caso.escenarios()["Base"]

LAMINAS = [
    dict(
        tipo="seccion", etiqueta="Para ir más lejos", nombre="Artefactos, conectores y plugins",
        bajada="Tres herramientas de Claude para la gestión financiera",
        pasos=[
            "Todo lo que hicimos hasta aquí funciona en una conversación normal con Claude, sin instalar nada. Pero "
            "Claude tiene tres herramientas que te permiten ir más lejos en la gestión financiera: los artefactos, "
            "los conectores y los plugins. Te las presento brevemente, para que sepas qué son y cuándo conviene "
            "usarlas.",
        ],
    ),
    dict(
        tipo="html", titulo="Artefactos: del análisis a algo que puedes mostrar",
        html=puntos_con_icono([
            dict(p=0, icono="capas", t="Una ventana al lado de la conversación",
                 d="Un documento, una presentación, un tablero o una herramienta interactiva"),
            dict(p=1, icono="barras", t="Un tablero de ratios", d="Con semáforos y la comparación entre años"),
            dict(p=2, icono="ajustes", t="Un simulador de escenarios", d="Mueves los días y ves cuánta caja se libera"),
            dict(p=3, icono="personas", t="Se ajusta conversando y se comparte con un enlace",
                 d="Antes, revisa qué datos incluye y quién lo va a ver"),
        ]) + artefacto(
            "Tablero · Distribuidora Alerce",
            [("Liquidez corriente", m(R24["liquidez_corriente"], 2), m(R25["liquidez_corriente"], 2), "a"),
             ("Razón de caja", m(R24["razon_caja"], 2), m(R25["razon_caja"], 2), "r"),
             ("Días de cobro", m(round(R24["dias_cobro"])), m(round(R25["dias_cobro"])), "r"),
             ("Ciclo de caja (días)", m(round(R24["ciclo_caja"])), m(round(R25["ciclo_caja"])), "r")],
            dict(p=2, h="Simulador de escenarios",
                 controles=[("Días de cobro", str(BASE["dias_cobro"]), 0.45), ("Días de inventario", str(BASE["dias_inventario"]), 0.5)],
                 resultado=("Caja liberada", f"{m(BASE['caja_liberada'])} millones")),
            dict(p=3, t="Compartir · cualquiera con el enlace"),
        ),
        pasos=[
            "Empecemos por los artefactos. Un artefacto es lo que Claude crea para ti en una ventana aparte, al lado "
            "de la conversación: un documento, una presentación, un tablero o una pequeña herramienta interactiva.",
            "¿Para qué sirve en finanzas? Pídele, por ejemplo, que convierta los ratios del paso 4 en un tablero, "
            "con semáforos y la comparación entre años.",
            "O que arme los escenarios del paso 8 como un simulador: mueves los días de cobro y de inventario, y "
            "ves cuánta caja se libera.",
            "Lo sigues ajustando conversando, y lo puedes compartir con un enlace. Un cuidado: antes de compartirlo, "
            "revisa qué datos incluye y quién va a poder verlo.",
        ],
    ),
    dict(
        tipo="html", titulo="Conectores: Claude lee los datos donde ya están",
        previo=dict(p=1, pill="EJEMPLO", t="Busca en mi carpeta «Cierre 2025» la planilla de diciembre y repite el análisis de ratios."),
        html=conectores(
            [dict(p=0, icono="carpeta", t="Tu carpeta en la nube", d="Google Drive · Microsoft 365"),
             dict(p=0, icono="mensaje", t="Tu correo y tu calendario", d="Gmail · Outlook"),
             dict(p=0, icono="tabla", t="Tus planillas y documentos", d="Hojas de cálculo, PDF y archivos de Office")],
            ("Claude", "busca y lee los archivos donde ya están"),
        ),
        notas=[
            dict(p=2, t="Respeta tus permisos", d="Solo puede ver lo que tú ya puedes ver"),
            dict(p=2, t="Activa solo los que necesites", d="Con la misma política de confidencialidad del paso 1"),
            dict(p=2, t="En los planes para equipos", d="Primero los habilita un administrador"),
        ],
        pasos=[
            "La segunda herramienta son los conectores. Un conector le permite a Claude trabajar con las "
            "herramientas que ya usas: tu carpeta en la nube, tu correo o tus planillas.",
            "En la práctica, en vez de descargar la planilla del cierre y subirla cada mes, le pides que la busque "
            "y la lea directamente desde tu carpeta.",
            "Tres cuidados. Claude respeta tus permisos: solo puede ver lo que tú ya puedes ver. Activa solo los "
            "conectores que realmente necesites, con la misma política de confidencialidad que vimos en el paso 1. "
            "Y en los planes para equipos, primero tiene que habilitarlos un administrador.",
        ],
    ),
    dict(
        tipo="dos", titulo="Skills y plugins: flujos de trabajo ya armados",
        paneles=[
            dict(clase="azul", icono="capas", titulo="Skills (habilidades)", icono_item="check", items=[
                dict(p=0, t="Conocimiento y pasos de trabajo", d="Claude los carga cuando la tarea lo requiere"),
                dict(p=0, t="Algunas ya vienen incluidas", d="Planillas Excel, documentos, presentaciones y PDF"),
                dict(p=2, t="Y puedes crear las tuyas", d="Con tus definiciones y tu formato de informe"),
            ]),
            dict(clase="oro", icono="caja", titulo="Plugin de finanzas, de Anthropic", icono_item="check", p=1, items=[
                dict(p=1, t="Reúne skills, conectores y comandos", d="Se instala de una sola vez"),
                dict(p=1, t="Asientos contables y conciliaciones"),
                dict(p=1, t="Estados financieros y análisis de desviaciones"),
                dict(p=1, t="Cierre mensual", d="Lo que entregue también se revisa"),
            ]),
        ],
        pie=dict(p=2, t="Los 10 prompts del curso son un buen punto de partida para tu propia skill."),
        pasos=[
            "Y la tercera: las skills y los plugins. Una skill, o habilidad, es un paquete de conocimiento y de "
            "pasos de trabajo, que Claude carga cuando la tarea lo requiere. Claude ya trae algunas: por ejemplo, "
            "para crear planillas Excel, documentos y presentaciones.",
            "Un plugin reúne varias skills, conectores y comandos para un tipo de trabajo, y se instala de una sola "
            "vez. Anthropic publica un plugin de finanzas, con flujos ya armados para asientos contables, "
            "conciliaciones, estados financieros, análisis de desviaciones y el cierre mensual. Y como siempre: lo "
            "que entregue se revisa antes de usarlo.",
            "También puedes crear tus propias skills. Los diez prompts de este curso, con tus definiciones y tu "
            "formato de informe, son un muy buen punto de partida.",
        ],
    ),
    dict(
        tipo="tabla", titulo="¿Cuándo conviene usar cada una?", texto=True, flex=2.3,
        columnas=["Herramienta", "Úsala cuando…", "Por ejemplo"],
        filas=[
            dict(c=["Artefacto", "Necesitas mostrar y compartir el resultado", "El tablero de ratios para el directorio"], p=0),
            dict(c=["Conector", "Los datos cambian todos los meses", "La planilla del cierre, leída desde tu carpeta"], p=1),
            dict(c=["Skill o plugin", "Repites siempre el mismo flujo de trabajo", "El cierre mensual, paso a paso"], p=2),
        ],
        notas=[
            dict(p=3, t="Revisa qué tienes disponible", d="Estas funciones dependen de tu plan, y sus nombres cambian con el tiempo"),
        ],
        pie=dict(p=4, t="Ninguna reemplaza el método: primero pregunta bien y verifica; después, automatiza."),
        pasos=[
            "¿Cuándo usar cada una? El artefacto, cuando necesitas mostrar y compartir el resultado.",
            "+ El conector, cuando los datos cambian todos los meses y no quieres subirlos a mano.",
            "+ Y la skill o el plugin, cuando repites siempre el mismo flujo de trabajo.",
            "Una advertencia: estas funciones dependen de tu plan, y sus nombres cambian con el tiempo. Revisa en la "
            "configuración de tu cuenta cuáles tienes disponibles.",
            "Y lo más importante: ninguna de estas herramientas reemplaza el método. Primero aprende a preguntar "
            "bien y a verificar. Después, automatiza.",
        ],
    ),
]
