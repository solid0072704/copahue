"""Las evaluaciones del programa: el diagnóstico inicial y el banco de preguntas de cada módulo.

Tipos de pregunta:
    unica     una alternativa correcta (`r`: su posición en `o`)
    multiple  varias correctas (`r`: lista de posiciones); solo cuenta si se marcan todas y ninguna de más
    numero    respuesta numérica (`r`), con tolerancia `tol` y, si se quiere, `unidad`
    orden     `o` está en el orden correcto; la plataforma lo muestra desordenado

Cada pregunta lleva su explicación (`e`), que se muestra al corregir, y la lección donde se enseña (`l`).
El banco de un módulo tiene más preguntas de las que se muestran en una prueba (`programa.EVALUACION`), para que
repetirla no sea repetir las mismas. Los bancos se escriben junto con el módulo: una pregunta solo entra cuando su
lección existe.
"""

from __future__ import annotations

from . import programa


def P(p: str, o: list[str], r, e: str, l: str = "", tipo: str = "unica", **extra) -> dict:  # noqa: E741
    return dict(tipo=tipo, p=p, o=o, r=r, e=e, l=l, **extra)


# El diagnóstico no se aprueba ni se reprueba: ubica al alumno. `tema` dice a qué módulo apunta cada pregunta.
DIAGNOSTICO = [
    dict(P("Le pides a Claude el margen bruto de tu empresa sin darle ningún dato. ¿Qué es lo más probable?",
           ["Lo busca en una base de datos financiera", "Te pide los datos o responde con una cifra que no puede respaldar",
            "Lo calcula a partir de tus conversaciones de otros meses", "Se niega a responder por seguridad"], 1,
           "Claude no consulta una base de datos de tu empresa: trabaja con lo que está en la conversación. Sin datos, o los pide o estima sin respaldo."),
         tema="m01"),
    dict(P("Subes un libro mayor de 12.000 filas y pides el total por cuenta. ¿Qué conviene exigir?",
           ["Que responda de memoria, para que sea rápido", "Que haga el cálculo con código y muestre el total de control",
            "Que resuma las primeras 100 filas", "Que redondee a millones"], 1,
           "Con la herramienta de análisis Claude ejecuta código sobre el archivo completo: el resultado se puede cuadrar contra un total de control."),
         tema="m04"),
    dict(P("¿Cuál de estos pedidos va a dar el mejor análisis?",
           ["«Analiza este balance»", "«Eres analista financiero. Con el balance adjunto de una distribuidora mayorista, calcula liquidez y endeudamiento "
            "de 2024 y 2025, muestra la fórmula y dime en cinco líneas qué cambió y por qué importa»",
            "«Dime si la empresa está bien»", "«Haz un resumen ejecutivo»"], 1,
           "Contexto, tarea concreta, formato y criterio: con eso la respuesta se puede usar y revisar."),
         tema="m02"),
    dict(P("¿Para qué sirve un proyecto en Claude?",
           ["Para pagar menos", "Para dejar instrucciones y documentos de referencia que valen en todas sus conversaciones",
            "Para que varias personas escriban al mismo tiempo en un Excel", "Para guardar las contraseñas del ERP"], 1,
           "Un proyecto guarda instrucciones permanentes y conocimiento (plan de cuentas, políticas, formato de informes), para no repetir el contexto."),
         tema="m03"),
    dict(P("Necesitas que el gerente comercial mueva los días de cobro y vea cuánta caja se libera. ¿Qué le pides a Claude?",
           ["Un correo con la explicación", "Un artefacto: un simulador interactivo que puedes compartir con un enlace",
            "Una tabla en texto", "Un conector"], 1,
           "Un artefacto es lo que Claude crea en una ventana aparte: un tablero o una herramienta interactiva que se comparte."),
         tema="m05"),
    dict(P("¿Qué es un conector?",
           ["Un cable para proyectar la pantalla", "Una vía para que Claude lea o actualice datos en las aplicaciones que ya usas, con tus permisos",
            "Un modelo más potente", "Una plantilla de prompt"], 1,
           "Los conectores usan el protocolo MCP para que Claude trabaje con tus carpetas, tu correo o tu CRM, respetando lo que tú puedes ver."),
         tema="m07"),
    dict(P("El área comercial registra sus oportunidades en un CRM. ¿Qué gana finanzas al conectarlo con Claude?",
           ["Nada: el CRM es solo para vendedores", "Anticipación: ver la venta probable y proyectar cobros antes de que exista la factura",
            "Reemplazar la contabilidad", "Eliminar la cobranza"], 1,
           "El pipeline muestra ventas probables con semanas de anticipación: es la base de una proyección de ingresos y de caja."),
         tema="m08"),
    dict(P("Tienes un tablero en Power BI con un total que no cuadra. ¿Cómo puede ayudar Claude?",
           ["No puede: Power BI y Claude son incompatibles", "Explicando y corrigiendo la medida DAX, y proponiendo cómo validar el resultado",
            "Borrando el tablero", "Cambiando el tipo de gráfico"], 1,
           "Claude escribe, explica y depura medidas DAX y consultas de Power Query; con los servidores MCP de Power BI puede además consultar el modelo."),
         tema="m09"),
    dict(P("Repites cada mes el mismo informe, con el mismo formato y las mismas definiciones. ¿Qué herramienta lo resuelve mejor?",
           ["Escribir el prompt de nuevo cada vez", "Una skill con el procedimiento, las definiciones y la plantilla",
            "Un modelo más caro", "Un artefacto distinto cada mes"], 1,
           "Una skill empaqueta conocimiento y pasos de trabajo; Claude la carga cuando la tarea lo requiere."),
         tema="m06"),
    dict(P("Claude te entrega un análisis impecable de la cartera vencida. ¿Qué haces antes de enviarlo al directorio?",
           ["Lo envías: está bien escrito", "Cuadras los totales con el auxiliar y revisas los supuestos",
            "Le pides que lo traduzca", "Le cambias el título"], 1,
           "Lo que entrega Claude se verifica: totales de control, fuentes y supuestos. La responsabilidad profesional no se delega."),
         tema="m15"),
]

# Lo que recomienda el diagnóstico, según las respuestas correctas (de 0 a 10).
RECOMENDACIONES = [
    (0, "Parte por el módulo 1. El bloque «Cómo funciona Claude» es la base de todo lo demás."),
    (5, "Ya tienes la base. Repasa los módulos 1 a 3 rindiendo sus pruebas y concéntrate desde el módulo 4."),
    (8, "Conoces bien las herramientas. Rinde las pruebas de los módulos 1 a 7 y dedica tu tiempo a CRM, Power BI y a las aplicaciones."),
]

BANCOS: dict[str, list[dict]] = {
    "m01": [
        P("¿Qué describe mejor a un modelo de lenguaje como Claude?",
          ["Una base de datos que busca la respuesta exacta", "Un sistema que genera la continuación más probable de un texto, a partir de lo que aprendió y de lo que le das",
           "Una planilla con fórmulas muy complejas", "Un buscador de internet"], 1,
          "Genera texto prediciendo qué viene a continuación. Por eso escribe y razona bien, pero no «consulta» un registro de hechos.", "m01-l01"),
        P("Claude responde con seguridad una cifra del sector que no le entregaste. ¿Cómo la tratas?",
          ["Como un dato verificado", "Como una hipótesis: pides la fuente o la confirmas por tu cuenta",
           "Como un error seguro", "Como una cifra oficial"], 1,
          "El tono seguro no es evidencia. Un dato que no está en la conversación ni trae fuente se confirma antes de usarlo.", "m01-l01"),
        P("¿Qué es la ventana de contexto?",
          ["El tamaño de la pantalla", "La cantidad de texto que Claude puede tener presente en una conversación",
           "El horario en que Claude está disponible", "El número de archivos que guarda un proyecto"], 1,
          "Todo lo que Claude considera al responder (tus mensajes, archivos y sus propias respuestas) tiene que caber en esa ventana.", "m01-l02"),
        P("En una conversación muy larga, Claude deja de respetar una definición que le diste al comienzo. ¿Qué conviene hacer?",
          ["Insistir en mayúsculas", "Abrir una conversación nueva con el contexto esencial, o dejar la definición en las instrucciones del proyecto",
           "Cambiar de computador", "Subir el archivo otra vez, tres veces"], 1,
          "Las conversaciones largas presionan la ventana de contexto. Lo permanente va en las instrucciones del proyecto; lo demás, en una conversación nueva.", "m01-l02"),
        P("Ordena, de menor a mayor, lo que consume de tu límite de uso:",
          ["Una pregunta corta sin archivos", "Una pregunta sobre un PDF de 20 páginas", "Un análisis con razonamiento extendido sobre un libro mayor completo"],
          None, "Mientras más texto entra y más trabajo se pide, más se consume: los archivos grandes y el razonamiento extendido pesan.", "m01-l02", tipo="orden"),
        P("Tienes que clasificar 300 glosas de gastos en ocho categorías. ¿Qué modelo conviene?",
          ["El más potente, siempre", "Uno rápido: la tarea es simple y repetitiva", "Ninguno: Claude no clasifica", "El que tenga más años"], 1,
          "Para tareas simples y repetitivas, un modelo rápido rinde igual y gasta menos de tu límite. El modelo más capaz se reserva para el análisis difícil.", "m01-l03"),
        P("¿Cuándo vale la pena activar el razonamiento extendido?",
          ["Para saludar", "En problemas de varios pasos: una cuadratura que no cierra, un escenario con muchos supuestos",
           "Nunca: solo hace más lenta la respuesta", "Solo para traducir"], 1,
          "El razonamiento extendido le da a Claude espacio para pensar paso a paso. Ayuda en problemas complejos; en tareas simples solo agrega tiempo.", "m01-l03"),
        P("¿En cuáles de estas situaciones es más probable que Claude invente? (marca todas)",
          ["Le preguntas por el tipo de cambio de una fecha exacta sin darle la fuente", "Le pides resumir un contrato que está adjunto",
           "Le pides el número de una norma que «debe existir»", "Le pides sumar una columna con la herramienta de análisis"], [0, 2],
          "Inventa cuando se le pide un hecho preciso que no está en la conversación. Con el documento adjunto o con código, el riesgo baja mucho.", "m01-l04", tipo="multiple"),
        P("¿Qué instrucción reduce más el riesgo de cifras inventadas?",
          ["«Sé creativo»", "«Usa solo los archivos adjuntos; si un dato no está, dilo y no lo estimes»",
           "«Responde rápido»", "«No uses tablas»"], 1,
          "Limitar la respuesta a los datos entregados y permitir un «no está» es la forma más directa de evitar que rellene vacíos.", "m01-l04"),
        P("Pides el total de una columna de 4.000 filas. ¿Cómo sabes si Claude lo calculó o lo estimó?",
          ["Por la cantidad de decimales", "Porque usó la herramienta de análisis: se ve el código y el resultado",
           "Porque responde en negrita", "No hay forma de saberlo"], 1,
          "Cuando calcula, ejecuta código y lo puedes ver. Si no hay ejecución, la cifra es una estimación del modelo y no un cálculo.", "m01-l05"),
        P("Las ventas pasaron de 10.000 a 11.800 millones. ¿Cuánto crecieron?",
          [], 18, "(11.800 ÷ 10.000 − 1) × 100 = 18 %. Es el tipo de cálculo que conviene pedir «con la fórmula a la vista».", "m01-l05",
          tipo="numero", tol=0.1, unidad="%"),
        P("Antes de usar una cifra calculada por Claude en un informe, ¿qué control mínimo aplicas?",
          ["Ninguno, si usó código", "Un total de control: que la suma coincida con el saldo del sistema o del archivo",
           "Pedirle que confirme que está seguro", "Repetir la pregunta en mayúsculas"], 1,
          "El código también puede partir de un rango equivocado. Un total de control independiente detecta ese error en segundos.", "m01-l05"),
        P("En los planes para equipos y empresas, ¿qué ocurre con tus conversaciones respecto del entrenamiento de modelos?",
          ["Se usan siempre", "Por defecto no se usan para entrenar", "Se publican", "Se borran a la hora"], 1,
          "En los planes Team y Enterprise los datos no se usan para entrenar por defecto. En los planes personales, la opción se controla en la configuración de privacidad.", "m01-l06"),
        P("Vas a analizar las remuneraciones de tu empresa con una cuenta personal. ¿Qué haces primero?",
          ["Subir la planilla completa", "Revisar la política de la empresa y quitar nombres y RUT: con cargos y montos basta para el análisis",
           "Compartir la conversación con todo el equipo", "Nada especial"], 1,
          "Datos personales y sensibles: primero la política de la empresa, y después minimizar. Casi ningún análisis necesita identificar a las personas.", "m01-l06"),
        P("¿Cuál es el modelo mental correcto para trabajar con Claude en finanzas?",
          ["Un oráculo que no se equivoca", "Un analista muy rápido y capaz, que solo sabe lo que le das y cuyo trabajo revisas",
           "Un buscador", "Una calculadora"], 1,
          "Rápido, capaz y dependiente del contexto que recibe. Tú defines el encargo y respondes por el resultado.", "m01-l01"),
    ],
}


def validar() -> None:
    """Revisa la forma de cada pregunta y que apunte a una lección que existe."""
    ids = {x["id"] for x in programa.lecciones()}
    claves = {m["clave"] for m in programa.modulos()}
    for nombre, banco in [("diagnostico", DIAGNOSTICO)] + list(BANCOS.items()):
        assert nombre == "diagnostico" or nombre in claves, nombre
        for q in banco:
            donde = f'{nombre}: {q["p"][:50]}'
            assert q["e"], donde
            assert not q["l"] or q["l"] in ids, donde
            if q["tipo"] == "unica":
                assert len(q["o"]) >= 3 and 0 <= q["r"] < len(q["o"]), donde
            elif q["tipo"] == "multiple":
                assert len(q["r"]) >= 2 and all(0 <= i < len(q["o"]) for i in q["r"]) and len(q["r"]) < len(q["o"]), donde
            elif q["tipo"] == "numero":
                assert isinstance(q["r"], (int, float)) and q.get("tol", 0) >= 0, donde
            elif q["tipo"] == "orden":
                assert len(q["o"]) >= 3, donde
            else:
                raise AssertionError(f"tipo desconocido · {donde}")
            if nombre == "diagnostico":
                assert q["tema"] in claves, donde
    for clave, banco in BANCOS.items():
        assert len(banco) >= programa.EVALUACION["banco_minimo"], f"{clave}: faltan preguntas en el banco"


def para_la_web() -> dict:
    """Lo que necesita la plataforma: el diagnóstico, los bancos y las reglas."""
    validar()
    return dict(reglas=programa.EVALUACION, diagnostico=DIAGNOSTICO, recomendaciones=RECOMENDACIONES, bancos=BANCOS)
