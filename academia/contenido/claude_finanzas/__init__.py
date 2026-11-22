"""El curso «Claude para Finanzas y Contabilidad»: su ficha, su programa y sus evaluaciones.

    python academia/manage.py importar_curso academia.contenido.claude_finanzas

Un curso nuevo es otro paquete como este (con `CURSO`, `programa` y `evaluaciones`), o se crea a mano en el panel de
administración. El precio y la publicación no van aquí: se definen en el panel.
"""

from . import evaluaciones, programa

CURSO = dict(
    slug="claude-para-finanzas-y-contabilidad",
    titulo=programa.NOMBRE,
    bajada=programa.BAJADA,
    descripcion=("Aprende cómo funciona Claude y úsalo en serio: con tus planillas, tu CRM y Power BI, en el cierre, el presupuesto y los "
                 "informes. Con evaluaciones, casos prácticos y certificado."),
    categoria="Finanzas con IA",
    logros=programa.LOGROS,
    muestra_titulo="Artefactos, conectores y plugins",
    instructor=dict(
        nombre=programa.AUTOR["nombre"], titulo=programa.AUTOR["titulo"],
        bio=("Gerente de administración y finanzas y docente en transformación digital financiera e inteligencia artificial. "
             "Autor del curso gratuito «Análisis de estados financieros con Claude».")),
    preguntas=[
        ("¿Necesito saber programar?",
         "No. Cuando hace falta código, lo escribe y lo ejecuta Claude; tú aprendes a pedirlo y a revisar el resultado."),
        ("¿Qué plan de Claude necesito?",
         "Puedes empezar con el plan gratuito. Para trabajar con archivos grandes, proyectos y conectores conviene un plan de pago. "
         "Cada lección indica qué plan requiere lo que muestra."),
        ("Mi empresa usa otro CRM, o no usa Power BI. ¿Me sirve?",
         "Sí. Los módulos usan HubSpot y Power BI como ejemplo y muestran la alternativa con archivos exportados, que funciona con cualquier sistema."),
        ("¿Los datos del curso son reales?",
         "No. Distribuidora Alerce es una empresa ficticia creada para el curso, con estados financieros, libro mayor y CRM que cuadran entre sí."),
        ("¿Este programa es de Anthropic?",
         "No. Claude es un producto de Anthropic; este programa es independiente y no tiene afiliación con Anthropic."),
    ],
)

__all__ = ["CURSO", "programa", "evaluaciones"]
