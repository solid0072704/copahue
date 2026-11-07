# Agente · Alertas diarias de empleos CFO en inmobiliarias y constructoras (Sudamérica)

Agente que corre en la nube de Claude Code como **tarea programada**. Una vez al día busca en Internet
ofertas de CFO / Director de Finanzas / Gerente de Administración y Finanzas en inmobiliarias,
desarrolladoras y constructoras de Sudamérica y envía un resumen a `r.diaz.barcelo@gmail.com`
desde la propia cuenta de Gmail (conector Gmail de claude.ai).

No necesita código ni servidor: no hay nada que instalar ni ejecutar desde este repositorio.
Este archivo documenta cómo está configurado para poder ajustarlo o reproducirlo.

| Parámetro | Valor |
|---|---|
| Nombre de la tarea | `Alertas CFO inmobiliarias y constructoras Sudamérica` |
| Horario | todos los días a las **7:41, hora de Chile** (`CRON_TZ=America/Santiago 41 7 * * *`) |
| Sesión | se ejecuta dentro de la sesión de Claude Code donde se creó (https://claude.ai/code/session_01D34wMugEoqmrqALvxbuFae), que tiene conectado Gmail. **No archivar ni borrar esa sesión**, o la tarea deja de poder enviar correos |
| Conectores | Gmail (buscar resúmenes anteriores y enviar el correo) |
| Destinatario | r.diaz.barcelo@gmail.com |
| Asunto | `[CFO Sudamérica] N ofertas · <fecha>` |

## Qué hace en cada corrida

1. Hace al menos 10 búsquedas web (español, portugués e inglés) combinando cargo + rubro + país,
   incluidas búsquedas acotadas a portales (LinkedIn, Computrabajo, Bumeran, Laborum, Trabajando,
   Indeed, elempleo, Multitrabajos, Michael Page, Page Executive, Robert Half, Glassdoor, Catho, Jooble).
2. Filtra por cargo (CFO, Director/a de Finanzas, Gerente de Finanzas, Gerente de Administración y
   Finanzas, Diretor Financeiro) y por rubro (inmobiliarias, desarrolladoras, constructoras,
   incorporadoras). Países: Chile (prioridad), Argentina, Perú, Colombia, Uruguay, Paraguay, Bolivia,
   Ecuador y Brasil.
3. Revisa en Gmail los resúmenes de los últimos 7 días y marca como **NUEVA** cada oferta que no
   aparecía antes.
4. Envía un correo HTML con una sección por país, enlace a cada aviso y una sección "Sin resultados".
   Si no encuentra nada, igual avisa.

Nota: desde una sesión de Claude Code no se pueden adjuntar conectores a una tarea que abra una sesión nueva
cada día (la organización no lo permite), por eso la tarea reutiliza esta sesión. Si prefieres que corra en una
sesión limpia cada día, créala desde la interfaz de tareas programadas de claude.ai, marca el conector Gmail y
pega el prompt de más abajo.

Limitación conocida: la red del entorno de Claude Code no permite abrir las páginas de los avisos,
así que el agente trabaja con los títulos, extractos y enlaces que devuelve la búsqueda web. Por eso
cada correo pide confirmar la vigencia en el enlace.

## Cómo pausarlo, cambiar la hora o el criterio

En claude.ai → Code → tareas programadas aparece la tarea con el nombre de arriba. Desde ahí se puede
pausar, borrar, cambiar el horario o editar el prompt (por ejemplo, agregar países, cargos o portales).
También se puede pedir en una sesión de Claude Code: "pausa la tarea Alertas CFO..." o "cámbiala a las 8:30".

## Prompt completo de la tarea

```text
Eres el agente diario de búsqueda de empleo de Rodrigo (r.diaz.barcelo@gmail.com). Tu misión de hoy: encontrar ofertas de trabajo vigentes de CFO en empresas inmobiliarias, desarrolladoras y constructoras de Sudamérica y enviarle un resumen por correo. Trabaja de forma autónoma, sin hacer preguntas: nadie está mirando la sesión.

ALCANCE
- Cargos: CFO, Chief Financial Officer, Director/a Financiero/a, Director/a de Finanzas, Gerente de Finanzas, Gerente de Administración y Finanzas, Gerente Financiero/a, Vicepresidente de Finanzas; en portugués: CFO, Diretor(a) Financeiro(a). Incluye subgerencias o jefaturas de finanzas solo si el rubro es claramente inmobiliario o de construcción, marcándolas como "(nivel subgerencia)" o "(nivel jefatura)".
- Empresas: inmobiliarias, desarrolladoras o promotoras inmobiliarias, constructoras, incorporadoras y construtoras (Brasil), gestoras o fondos de activos inmobiliarios, holdings de ingeniería y construcción. Descarta proveedores de materiales, bancos y otros rubros.
- Países: Chile (prioridad), Argentina, Perú, Colombia, Uruguay, Paraguay, Bolivia, Ecuador, Brasil. Excluye México, Centroamérica, el Caribe y España.

CÓMO BUSCAR
- Usa la herramienta WebSearch (si no aparece cargada, cárgala con ToolSearch "select:WebSearch"). Haz al menos 10 búsquedas distintas en modo "extended", combinando cargo + rubro + país o ciudad, en español, portugués e inglés. Incluye búsquedas acotadas a portales con site: (linkedin.com/jobs, computrabajo, bumeran, laborum.cl, trabajando.cl, getonbrd.com, cl.indeed.com, ar.indeed.com, pe.indeed.com, co.indeed.com, elempleo.com, multitrabajos.com, michaelpage.cl, michaelpage.com.ar, michaelpage.com.pe, michaelpage.com.co, michaelpage.com.br, pageexecutive.com, roberthalf.com, hays, glassdoor, catho.com.br, vagas.com.br, jooble).
- La red de esta sesión bloquea la apertura directa de las páginas de los avisos (WebFetch y curl fallan con 403). No insistas: trabaja con los títulos, enlaces y extractos que devuelve WebSearch y no inventes datos que no estén en los resultados.
- Prioriza avisos publicados en los últimos 30 días. Descarta los que digan expresamente que están cerrados o vencidos; si la fecha de cierre ya pasó, indícalo como "posiblemente vencida". Si la fecha no es visible, dilo.
- Antes de armar el correo, busca en Gmail (search_threads con query: subject:"[CFO Sudamérica]" newer_than:7d) los resúmenes de días anteriores y marca con la etiqueta "NUEVA" las ofertas que no aparecían en ellos. Si la búsqueda falla, continúa sin esa marca.

CORREO
- Envíalo con la herramienta de Gmail send_message a r.diaz.barcelo@gmail.com, en HTML (htmlBody) y con versión en texto plano (body).
- Asunto: "[CFO Sudamérica] N ofertas · <fecha de hoy, por ejemplo 8 nov 2026>" donde N es la cantidad de ofertas listadas.
- Contenido en español: una introducción breve (fecha, cuántas ofertas, cuántas nuevas); después una sección por país con bandera y nombre, Chile primero; cada oferta en una línea: cargo en negrita, empresa o headhunter, ciudad, fecha de publicación si se conoce, una frase con lo relevante (requisitos, sueldo si aparece), el enlace al aviso y la etiqueta NUEVA cuando corresponda. Si el enlace lleva a una página de resultados y no al aviso directo, indícalo como "(listado)". Cierra con una sección "Sin resultados hoy" con los países sin ofertas y una nota de que conviene confirmar la vigencia en cada enlace.
- Si no encuentras ninguna oferta, envía igual un correo corto diciendo que hoy no hubo novedades y qué búsquedas hiciste.
- Envía un solo correo. No escribas a nadie más, no crees borradores ni modifiques otros correos.

Termina con una sola línea que diga cuántas ofertas enviaste y cuántas eran nuevas.
```
