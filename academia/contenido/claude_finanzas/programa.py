"""El programa del curso de pago «Claude para Finanzas y Contabilidad».

Es la única fuente del temario: de aquí salen los módulos y las lecciones que la academia muestra (se cargan con
`manage.py importar_curso`), las horas de cada módulo y el presupuesto de narración. Cada lección dura lo que dice
`min` (minutos de narración); al producirla, ese número se reemplaza por la duración real.

    python -m academia.contenido.claude_finanzas.programa     # módulos, lecciones, horas y créditos de voz por etapa

Las funciones de Claude cambian con frecuencia. Lo que el temario nombra se verificó el 22 de noviembre de 2026
(directorio de conectores de Claude, documentación de Microsoft sobre los servidores MCP de Power BI, notas de versión
de Claude); cada módulo se vuelve a verificar al producirlo.
"""

from __future__ import annotations

NOMBRE = "Claude para Finanzas y Contabilidad"
BAJADA = "Programa profesional: de entender cómo funciona Claude a conectarlo con tu CRM, tus planillas y Power BI"
AUTOR = dict(nombre="Rodrigo Barceló", titulo="Contador auditor · Magíster en gestión financiera")
CARACTERES_POR_MINUTO = 17.1 * 60      # ritmo medido de la voz del autor (curso_youtube.voz)
CREDITOS_POR_CARACTER = 1.0            # ElevenLabs, modelo eleven_v4: un crédito por carácter enviado

# Lo que el alumno va a poder hacer al terminar (la portada lo muestra tal cual).
LOGROS = [
    "Explicar cómo funciona Claude y por qué se equivoca, para saber cuándo confiar y cuándo verificar",
    "Convertir planillas, cartolas y contratos en análisis revisables, con los cálculos hechos con código",
    "Construir tableros, simuladores y calculadoras que tu equipo usa sin instalar nada",
    "Conectar Claude a tu CRM para proyectar ventas y cobros antes de que exista la factura",
    "Escribir y depurar medidas de Power BI, y convertir un tablero en un comentario de gestión",
    "Dejar el cierre, el presupuesto y el informe al directorio como procedimientos que se repiten, con sus controles",
]

# Las etapas ordenan la producción y el lanzamiento: la 1 es lo que se pidió primero (cómo funciona Claude, sus
# herramientas y las conexiones con CRM y Power BI); la 2 y la 3 se agregan al programa a medida que se producen.
ETAPAS = {
    1: "Fundamentos, herramientas y conexiones",
    2: "Aplicaciones en contabilidad y finanzas",
    3: "Automatización, gobierno e implementación",
}


def L(t: str, d: str, minutos: int, tipo: str = "clase", **extra) -> dict:
    """Una lección. `tipo`: clase (se explica), demo (se muestra en Claude) o laboratorio (lo hace el alumno)."""
    return dict(t=t, d=d, min=minutos, tipo=tipo, **extra)


BLOQUES = [
    dict(
        clave="b1", etapa=1, t="Cómo funciona Claude",
        d="Lo que hay que entender antes de confiarle un número: qué hace bien, por qué se equivoca y cómo pedirle las cosas.",
        modulos=[
            dict(
                clave="m01", t="Cómo piensa Claude, y por qué a veces se equivoca",
                d="El modelo mental correcto: un analista muy rápido, que escribe y razona bien, pero que no sabe lo que no le das.",
                logras=["Explicar con tus palabras qué es un modelo de lenguaje y qué no es",
                        "Elegir el modelo y el modo de razonamiento según la tarea",
                        "Reconocer cuándo Claude calcula y cuándo estima, y obligarlo a calcular",
                        "Decidir qué datos de tu empresa puedes subir y con qué plan"],
                lecciones=[
                    L("Un modelo de lenguaje, explicado para financieros", "Predice texto, no consulta una base de datos: qué significa eso para tus números.", 8),
                    L("Tokens, ventana de contexto y límites de uso", "Por qué «olvida» un archivo largo y cómo repartir el trabajo.", 7),
                    L("Los modelos de Claude y cuándo usar cada uno", "Rápido, equilibrado o profundo; con o sin razonamiento extendido.", 7),
                    L("Por qué inventa y cómo evitarlo", "Datos en la conversación, citas a la fuente y permiso para decir «no lo sé».", 8),
                    L("Cuándo calcula y cuándo adivina", "La herramienta de análisis ejecuta código: pide siempre el cálculo y la fórmula.", 8, "demo"),
                    L("Planes, privacidad y datos de la empresa", "Qué pasa con lo que subes, qué cambia en los planes de equipo y qué conviene anonimizar.", 8),
                ],
                practica="Clasifica doce tareas de tu área: cuáles puede hacer Claude solo, cuáles con verificación y cuáles no conviene delegar.",
            ),
            dict(
                clave="m02", t="Prompts profesionales para finanzas",
                d="Del pedido vago al encargo que un analista entendería a la primera, con su verificación incluida.",
                logras=["Escribir encargos con contexto, tarea, formato y criterio de calidad",
                        "Repreguntar hasta llegar a la causa, no solo a la descripción",
                        "Exigir cuadraturas y trazabilidad en cada respuesta",
                        "Armar tu biblioteca de prompts con variables"],
                lecciones=[
                    L("Anatomía de un encargo financiero", "Rol, contexto del negocio, tarea, formato y criterio para darlo por bueno.", 8),
                    L("El contexto que cambia la respuesta", "Giro, tamaño, moneda, criterios contables y para quién es el análisis.", 7),
                    L("Pedir el formato: tablas, memos y anexos", "Cómo obtener algo que se pueda pegar en el informe sin rehacerlo.", 7, "demo"),
                    L("Repreguntar: del primer borrador al análisis fino", "Las cinco repreguntas que separan una descripción de una explicación.", 8, "demo"),
                    L("Verificación dentro del prompt", "Cuadraturas, totales de control y «muéstrame de dónde sale cada cifra».", 8),
                    L("Tu biblioteca de prompts", "Plantillas con variables para el cierre, la cobranza y el informe mensual.", 6),
                ],
                practica="Reescribe cinco pedidos reales de tu trabajo y compara las dos respuestas con una pauta.",
            ),
            dict(
                clave="m03", t="Tu espacio de trabajo: proyectos, memoria y estilos",
                d="Para no explicar tu empresa en cada conversación: un proyecto que ya conoce tu plan de cuentas y tu forma de informar.",
                logras=["Crear un proyecto con instrucciones y conocimiento de tu empresa",
                        "Elegir qué documentos dejar como conocimiento y cuáles no",
                        "Usar la memoria y la búsqueda en conversaciones anteriores",
                        "Compartir un proyecto con tu equipo sin perder el control"],
                lecciones=[
                    L("Proyectos: el analista que ya conoce tu empresa", "Instrucciones permanentes y documentos de referencia.", 8, "demo"),
                    L("Qué dejar en el conocimiento del proyecto", "Plan de cuentas, políticas contables, presupuesto, glosario y formato de informes.", 7),
                    L("Instrucciones que se cumplen", "Cómo escribirlas para que no se diluyan: reglas, definiciones y ejemplos.", 7),
                    L("Memoria y conversaciones anteriores", "Qué recuerda Claude, cómo revisarlo y cómo borrarlo.", 6),
                    L("Estilos: que escriba como tu área", "El tono del informe al directorio no es el del correo de cobranza.", 5),
                    L("Proyectos compartidos", "Permisos, versiones del conocimiento y buenas prácticas de equipo.", 6),
                ],
                practica="Arma el proyecto «Finanzas» de tu empresa (o el de Distribuidora Alerce) y pruébalo con tres preguntas de control.",
            ),
        ],
    ),
    dict(
        clave="b2", etapa=1, t="Herramientas de Claude para el financiero",
        d="Las funciones que convierten una conversación en trabajo terminado: archivos, artefactos y skills.",
        modulos=[
            dict(
                clave="m04", t="Excel, CSV y PDF: del archivo al análisis",
                d="Cómo lee Claude tus archivos, cómo analiza miles de movimientos con código y cómo devuelve planillas con fórmulas.",
                logras=["Preparar un archivo para que Claude lo lea sin errores",
                        "Analizar un libro mayor completo con la herramienta de análisis",
                        "Extraer tablas de cartolas, facturas y contratos en PDF",
                        "Obtener Excel con fórmulas vivas, documentos y presentaciones"],
                lecciones=[
                    L("Cómo lee Claude una planilla", "Hojas, encabezados, celdas combinadas y fórmulas: lo que ve y lo que se pierde.", 7),
                    L("Analizar un libro mayor con código", "Agrupar, cruzar y buscar valores atípicos en miles de movimientos.", 9, "demo"),
                    L("PDF y documentos escaneados", "Cartolas bancarias, facturas y contratos: de la imagen a la tabla, con control de totales.", 8, "demo"),
                    L("Crear archivos: Excel, Word y PowerPoint", "Planillas con fórmulas que se recalculan, y cómo revisarlas.", 8, "demo"),
                    L("Claude dentro de Excel", "El complemento que trabaja en la planilla: explica modelos, corrige fórmulas y llena plantillas.", 8, "demo"),
                    L("Límites y trucos", "Tamaño de archivos, precisión, y cuándo dividir el trabajo en partes.", 6),
                ],
                practica="Laboratorio con el libro mayor de Distribuidora Alerce (12.000 movimientos): tres preguntas con respuesta numérica.",
            ),
            dict(
                clave="m05", t="Artefactos: tableros, simuladores y herramientas",
                d="Cuando el resultado tiene que verse y usarse: un tablero con semáforos, un simulador de caja o una calculadora que compartes con un enlace.",
                logras=["Pedir un tablero de indicadores y ajustarlo conversando",
                        "Construir un simulador de escenarios de capital de trabajo",
                        "Crear calculadoras financieras para tu equipo",
                        "Compartir un artefacto controlando quién lo ve y qué datos lleva"],
                lecciones=[
                    L("Qué es un artefacto y para qué sirve en finanzas", "Documento, tablero, simulador o herramienta: cuál elegir.", 6),
                    L("Un tablero de indicadores con semáforos", "De la tabla de ratios a un tablero que el gerente entiende solo.", 9, "demo"),
                    L("Un simulador de capital de trabajo", "Mueves días de cobro e inventario y ves la caja que se libera.", 9, "demo"),
                    L("Calculadoras: punto de equilibrio, créditos y VAN", "Herramientas pequeñas que evitan diez correos.", 8, "demo"),
                    L("Artefactos con datos al día y con Claude dentro", "Cuando el artefacto lee tus fuentes o responde preguntas por sí mismo.", 7),
                    L("Compartir y publicar con cuidado", "Enlaces, permisos y la revisión de datos antes de enviar.", 6),
                ],
                practica="Publica el tablero de indicadores de Alerce y compártelo con una persona de tu equipo.",
            ),
            dict(
                clave="m06", t="Skills y plugins: enséñale cómo se trabaja en tu empresa",
                d="Un procedimiento bien escrito, guardado una vez y usado siempre: el informe mensual, la conciliación, el cierre.",
                logras=["Explicar qué es una skill y cuándo la carga Claude",
                        "Crear una skill con tu formato de informe de gestión",
                        "Usar el plugin de finanzas: asientos, conciliaciones, cierre y desviaciones",
                        "Mantener y compartir las skills del equipo"],
                lecciones=[
                    L("Qué es una skill", "Conocimiento y pasos de trabajo que Claude carga solo cuando la tarea lo pide.", 6),
                    L("Las skills que ya vienen", "Planillas, documentos, presentaciones y PDF: lo que resuelven.", 6, "demo"),
                    L("Tu primera skill: el informe de gestión", "De tus diez prompts a un procedimiento reutilizable.", 9, "demo"),
                    L("Skills con plantillas y archivos", "La conciliación bancaria con tu formato y tus reglas.", 8, "demo"),
                    L("El plugin de finanzas", "Flujos ya armados para asientos, conciliaciones, estados financieros y cierre.", 8, "demo"),
                    L("Mantener y compartir skills", "Versiones, responsables y revisión antes de publicarlas al equipo.", 5),
                ],
                practica="Escribe la skill de un procedimiento que repites cada mes y pruébala con datos de dos meses distintos.",
            ),
        ],
    ),
    dict(
        clave="b3", etapa=1, t="Conectar Claude con tus sistemas",
        d="Que Claude lea los datos donde ya están: tus carpetas, tu CRM y tus modelos de Power BI.",
        modulos=[
            dict(
                clave="m07", t="Conectores y MCP",
                d="El estándar que permite a Claude trabajar con tus aplicaciones, explicado sin tecnicismos y con los cuidados que exige.",
                logras=["Explicar qué es MCP y qué diferencia a un conector de subir un archivo",
                        "Activar conectores y decidir sus permisos",
                        "Usar los conectores de carpetas y correo en el cierre mensual",
                        "Evaluar qué hacer cuando tu ERP no tiene conector"],
                lecciones=[
                    L("Qué es MCP, sin tecnicismos", "Un enchufe común entre Claude y tus sistemas: qué viaja y qué no.", 7),
                    L("El directorio de conectores", "Qué hay para finanzas, quién publica cada conector y cómo evaluarlo.", 7, "demo"),
                    L("Permisos y seguridad", "Claude ve lo que tú puedes ver: lectura, escritura y aprobaciones.", 8),
                    L("Carpetas, correo y calendario en el cierre", "Buscar la planilla del mes, leerla y dejar el borrador del informe.", 8, "demo"),
                    L("Cuando tu sistema no tiene conector", "Exportaciones programadas, conectores de escritorio y conectores a medida.", 7),
                    L("Buenas prácticas de gobierno", "Qué conectores habilitar, a quién y cómo dejar registro.", 6),
                ],
                practica="Diseña el mapa de datos de tu cierre: de qué sistema sale cada cifra y por qué vía podría leerla Claude.",
            ),
            dict(
                clave="m08", t="CRM + Claude: del pipeline a la caja",
                d="La venta existe en el CRM semanas antes de llegar a la contabilidad. Conectarlo le da al financiero lo que más le falta: anticipación.",
                logras=["Conectar un CRM a Claude y entender qué datos entrega",
                        "Convertir el pipeline comercial en una proyección de ventas defendible",
                        "Llevar esa proyección al flujo de caja con fechas de cobro realistas",
                        "Priorizar la cobranza cruzando cartera vencida e historial del cliente",
                        "Medir rentabilidad por cliente y segmento"],
                lecciones=[
                    L("Por qué el CRM le importa al financiero", "Oportunidades, etapas y probabilidades: la venta antes de la factura.", 7),
                    L("Conectar HubSpot a Claude, paso a paso", "El conector oficial, sus permisos y qué preguntas responde. Alternativas para otros CRM.", 9, "demo"),
                    L("Del pipeline a la proyección de ventas", "Pipeline ponderado, tasas de cierre reales y duración del ciclo de venta.", 9, "demo"),
                    L("De la proyección de ventas al flujo de caja", "Plazos de cobro por cliente, estacionalidad y escenarios.", 9, "demo"),
                    L("Cobranza con criterio", "Cartera vencida más historial comercial: a quién llamar primero y qué decirle.", 8, "demo"),
                    L("Rentabilidad por cliente y segmento", "Cruzar ventas del CRM con márgenes de la contabilidad.", 8, "demo"),
                    L("Calidad del dato y datos personales", "Lo que no está en el CRM no existe: controles y cuidados legales.", 6),
                ],
                practica="Laboratorio con el CRM de Distribuidora Alerce (datos ficticios): proyecta la venta y la cobranza del próximo trimestre.",
            ),
            dict(
                clave="m09", t="Power BI + Claude",
                d="Power BI muestra; Claude explica, escribe las medidas y responde las preguntas que el tablero no previó.",
                logras=["Decidir qué resolver en Power BI y qué con Claude",
                        "Escribir y depurar medidas DAX y consultas de Power Query con ayuda de Claude",
                        "Diseñar un modelo de datos financiero en estrella",
                        "Consultar un modelo semántico en lenguaje natural con los servidores MCP de Power BI",
                        "Convertir un tablero en un comentario de gestión"],
                lecciones=[
                    L("Power BI y Claude: para qué sirve cada uno", "Tablero gobernado frente a análisis conversacional, y por qué se complementan.", 7),
                    L("DAX con Claude", "Medidas, inteligencia de tiempo y cómo depurar un total que no cuadra.", 9, "demo"),
                    L("Power Query: limpiar datos contables", "Del mayor exportado a tablas listas para el modelo.", 8, "demo"),
                    L("El modelo: hechos y dimensiones financieras", "Estrella, calendario, plan de cuentas y centros de costo.", 8),
                    L("Los servidores MCP de Power BI", "Preguntarle al modelo semántico en lenguaje natural: qué permite la versión preliminar de Microsoft y qué exige.", 9, "demo"),
                    L("Del tablero al comentario de gestión", "Claude lee las cifras del mes y escribe la explicación para el directorio.", 8, "demo"),
                    L("Documentar y auditar un modelo heredado", "Entender medidas ajenas, encontrar errores y dejar documentación.", 7, "demo"),
                ],
                practica="Laboratorio con el modelo de Alerce: escribe tres medidas, valídalas contra el mayor y redacta el comentario del mes.",
            ),
        ],
    ),
    dict(
        clave="b4", etapa=2, t="Contabilidad y finanzas con Claude",
        d="El método aplicado a las tareas de todos los meses: cierre, presupuesto, tesorería e informes.",
        modulos=[
            dict(
                clave="m10", t="Contabilidad y cierre mensual",
                d="Conciliar, analizar cuentas y preparar asientos con Claude como asistente, y con el contador firmando.",
                logras=["Organizar el cierre con un calendario y controles",
                        "Conciliar bancos y auxiliares con apoyo de Claude",
                        "Preparar y revisar asientos con su respaldo",
                        "Hacer consultas técnicas y verificarlas contra la norma"],
                lecciones=[
                    L("El mapa del cierre con Claude", "Qué tareas acelera, cuáles revisa y cuáles siguen siendo tuyas.", 7),
                    L("Conciliación bancaria asistida", "Cartola contra mayor: partidas en tránsito, diferencias y propuesta de ajustes.", 9, "demo"),
                    L("Análisis de cuentas y auxiliares", "Antigüedad, partidas abiertas y saldos que no deberían existir.", 8, "demo"),
                    L("Asientos: propuesta, respaldo y revisión", "Devengos, provisiones y reclasificaciones con su explicación.", 9, "demo"),
                    L("Cortes y provisiones de fin de mes", "Lo que falta por facturar y lo que falta por recibir.", 7),
                    L("Consultas técnicas: NIIF e impuestos", "Cómo preguntar, cómo pedir la fuente y por qué siempre se confirma.", 8),
                    L("Preparar la auditoría", "Muestras, respaldos y respuestas a los requerimientos.", 7),
                ],
                practica="Concilia el banco de Alerce de diciembre y propone los asientos de ajuste.",
            ),
            dict(
                clave="m11", t="Planificación y control de gestión",
                d="Presupuesto, proyecciones y desviaciones: menos tiempo armando planillas y más entendiendo qué pasó.",
                logras=["Construir un presupuesto basado en supuestos explícitos",
                        "Mantener una proyección móvil",
                        "Explicar desviaciones por precio, volumen y mezcla",
                        "Evaluar proyectos y escenarios"],
                lecciones=[
                    L("Presupuesto por supuestos", "Volúmenes, precios, dotación y estacionalidad a la vista.", 8, "demo"),
                    L("Proyección móvil", "Actualizar el año con lo real y lo que viene.", 7, "demo"),
                    L("Desviaciones: precio, volumen y mezcla", "La descomposición que el directorio siempre pregunta.", 9, "demo"),
                    L("Escenarios y sensibilidades", "Qué pasa si baja el margen, sube el dólar o se atrasa la cobranza.", 8, "demo"),
                    L("Modelo de tres estados", "Resultados, balance y flujo que cuadran entre sí.", 9, "demo"),
                    L("Evaluación de proyectos", "VAN, TIR y período de recuperación, con sus supuestos.", 8, "demo"),
                    L("Rentabilidad por línea y por producto", "Costos directos, asignaciones y márgenes de contribución.", 8, "demo"),
                ],
                practica="Explica la desviación de margen de Alerce contra su presupuesto y propón tres acciones con su efecto.",
            ),
            dict(
                clave="m12", t="Tesorería, capital de trabajo y financiamiento",
                d="La caja, semana a semana: proyectarla, liberarla y financiarla al menor costo.",
                logras=["Armar un flujo de caja de 13 semanas",
                        "Gestionar crédito a clientes, inventarios y proveedores con datos",
                        "Comparar alternativas de financiamiento por costo efectivo",
                        "Controlar covenants y vencimientos"],
                lecciones=[
                    L("Flujo de caja de 13 semanas", "Cobros, pagos y saldo mínimo, semana a semana.", 9, "demo"),
                    L("Crédito y cobranza", "Límites por cliente, plazos y provisión de incobrables.", 8, "demo"),
                    L("Inventarios y proveedores", "Rotación, quiebres y condiciones de pago.", 7, "demo"),
                    L("Deuda: tablas, costo efectivo y covenants", "Leer un contrato de crédito y seguir sus compromisos.", 8, "demo"),
                    L("Comparar ofertas de financiamiento", "Línea, factoring, leasing y crédito: en la misma unidad.", 8, "demo"),
                    L("Tipo de cambio e inflación", "Dólar y UF en la proyección y en los contratos.", 6),
                ],
                practica="Proyecta la caja de Alerce para el primer trimestre y decide cómo cubrir el punto más bajo.",
            ),
            dict(
                clave="m13", t="Informes y comunicación financiera",
                d="El análisis no sirve si no se entiende: informes, presentaciones y respuestas para quienes deciden.",
                logras=["Escribir un informe de gestión de una página",
                        "Preparar la presentación al directorio",
                        "Anticipar las preguntas difíciles"],
                lecciones=[
                    L("El informe de gestión mensual", "Una página, cinco partes y ninguna tabla repetida.", 8, "demo"),
                    L("La presentación al directorio", "De los hallazgos a diez láminas con una idea cada una.", 8, "demo"),
                    L("Comentar resultados", "Escribir como director de finanzas: causa, efecto y decisión.", 7),
                    L("Memos y minutas", "Decisiones, responsables y plazos.", 6, "demo"),
                    L("Preparar las preguntas difíciles", "Claude como directorio exigente.", 7, "demo"),
                ],
                practica="Escribe el informe de diciembre de Alerce y la minuta del comité.",
            ),
        ],
    ),
    dict(
        clave="b5", etapa=3, t="Automatización, gobierno e implementación",
        d="De usar Claude a tenerlo trabajando: flujos que se repiten solos, con controles y con el equipo a bordo.",
        modulos=[
            dict(
                clave="m14", t="De conversar a delegar: tareas y agentes",
                d="Cuando el trabajo deja de ser una pregunta y pasa a ser un encargo de varias horas, o uno que se repite cada lunes.",
                logras=["Distinguir una conversación de una tarea delegada",
                        "Encargar trabajos largos sobre tus carpetas",
                        "Programar tareas recurrentes",
                        "Definir los puntos de control humanos"],
                lecciones=[
                    L("De chatear a delegar", "Qué es un agente y qué cambia cuando Claude trabaja solo.", 7),
                    L("Cowork: Claude trabajando en tus carpetas", "Encargos largos con archivos reales y un resultado para revisar.", 9, "demo"),
                    L("Tareas programadas", "El informe de cobranza que llega solo cada lunes.", 8, "demo"),
                    L("Claude en el navegador", "Portales bancarios y sistemas sin conector: usos y límites.", 7, "demo"),
                    L("Un cierre de punta a punta", "Encadenar lectura, conciliación, asientos e informe.", 9, "demo"),
                    L("Supervisión", "Qué aprobar siempre a mano y cómo dejar evidencia.", 6),
                ],
                practica="Diseña y programa un flujo semanal de tu área, con su punto de control.",
            ),
            dict(
                clave="m15", t="Riesgos, control interno y ética",
                d="Usar IA en finanzas sin perder el control: confidencialidad, errores, evidencia y responsabilidad profesional.",
                logras=["Identificar los riesgos de usar IA en procesos financieros",
                        "Escribir la política de uso del área",
                        "Dejar trazabilidad para auditoría"],
                lecciones=[
                    L("Confidencialidad y datos personales", "Qué se puede subir, con qué plan y con qué autorización.", 7),
                    L("Errores típicos y controles compensatorios", "Cifras inventadas, fórmulas mal copiadas y supuestos no dichos.", 8),
                    L("La política de uso del área de finanzas", "Una página que el equipo puede cumplir.", 7),
                    L("Trazabilidad para auditoría", "Qué guardar de cada análisis hecho con Claude.", 6),
                    L("Lo que no se delega", "El juicio profesional y la firma siguen siendo tuyos.", 6),
                ],
                practica="Escribe la política de uso de IA de tu área a partir de la plantilla del curso.",
            ),
            dict(
                clave="m16", t="Implementación: 90 días en tu área",
                d="Del curso a la práctica: qué casos partir, cómo medir el resultado y cómo sumar al equipo.",
                logras=["Priorizar casos de uso por impacto y dificultad",
                        "Medir horas, plazos y calidad antes y después",
                        "Presentar tu plan de 90 días"],
                lecciones=[
                    L("Diagnóstico y priorización", "Una matriz de impacto y dificultad para las tareas del área.", 7),
                    L("Medir el retorno", "Horas liberadas, días de cierre y errores evitados.", 7),
                    L("Capacitar al equipo", "De la persona que sabe a un equipo que lo usa.", 6),
                    L("Tu plan de 30, 60 y 90 días", "Hitos, responsables y riesgos.", 7),
                    L("Proyecto final", "Aplica el programa a un proceso real de tu empresa.", 8, "laboratorio"),
                ],
                practica="Proyecto final: un proceso real de tu área, antes y después, con su plan de 90 días.",
            ),
        ],
    ),
]

# Cómo se evalúa. Cada módulo cierra con una prueba; el banco tiene más preguntas de las que se muestran, para que un
# segundo intento no repita la misma prueba.
EVALUACION = dict(preguntas_por_modulo=10, banco_minimo=15, aprobacion=70, examen_final=40, aprobacion_final=75)


def modulos() -> list[dict]:
    """Los módulos en orden, cada uno con su bloque, su etapa, su número y sus lecciones identificadas."""
    out = []
    for b in BLOQUES:
        for m in b["modulos"]:
            n = len(out) + 1
            assert m["clave"] == f"m{n:02d}", f'{m["clave"]}: los módulos van numerados en orden'
            lecciones = [dict(x, id=f'{m["clave"]}-l{i:02d}', n=i) for i, x in enumerate(m["lecciones"], 1)]
            out.append(dict(m, n=n, bloque=b["clave"], etapa=b["etapa"], lecciones=lecciones, min=sum(x["min"] for x in lecciones)))
    return out


def lecciones() -> list[dict]:
    return [dict(x, modulo=m["clave"]) for m in modulos() for x in m["lecciones"]]


def totales(etapa: int | None = None) -> dict:
    """Módulos, lecciones, minutos de narración y créditos de voz (del programa completo o de una etapa)."""
    ms = [m for m in modulos() if etapa is None or m["etapa"] == etapa]
    minutos = sum(m["min"] for m in ms)
    caracteres = round(minutos * CARACTERES_POR_MINUTO)
    return dict(modulos=len(ms), lecciones=sum(len(m["lecciones"]) for m in ms), minutos=minutos, horas=round(minutos / 60, 1),
                caracteres=caracteres, creditos=round(caracteres * CREDITOS_POR_CARACTER))


if __name__ == "__main__":
    for b in BLOQUES:
        print(f'\n{b["t"].upper()}  (etapa {b["etapa"]})')
        for m in modulos():
            if m["bloque"] == b["clave"]:
                print(f'  {m["n"]:>2}. {m["t"]}  ·  {len(m["lecciones"])} lecciones  ·  {m["min"]} min')
    print()
    miles = lambda n: f"{n:,}".replace(",", ".")  # noqa: E731
    for e, nombre in list(ETAPAS.items()) + [(None, "Programa completo")]:
        t = totales(e)
        print(f'{"Etapa " + str(e) + " · " if e else ""}{nombre}: {t["modulos"]} módulos, {t["lecciones"]} lecciones, '
              f'{str(t["horas"]).replace(".", ",")} h, {miles(t["creditos"])} créditos de voz')
