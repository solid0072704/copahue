# Agente diario · Ofertas freelance finanzas + IA

Tarea programada en Claude Code (claude.ai/code → Scheduled tasks) que cada día:

1. Hace 6 a 10 búsquedas web de ofertas vigentes para **consultores freelance de finanzas con conocimientos en IA**
   (consultoría de proyecto, CFO fraccional, contratos por horas, entrenamiento/evaluación de modelos de IA).
2. Selecciona las 6 a 12 más relevantes (recientes, remotas, abiertas a Argentina/LATAM/España, mejor tarifa).
3. Envía un correo HTML en español a `r.diaz.barcelo@gmail.com` con: ofertas concretas y enlaces,
   contexto de mercado y portales donde seguir buscando.

| Parámetro | Valor |
|---|---|
| Nombre de la tarea | `Ofertas freelance finanzas IA` |
| Horario | `CRON_TZ=America/Argentina/Buenos_Aires 33 8 * * *` (arranca 8:33 para que el correo llegue ≈ 9:00) |
| Sesión | Nueva por cada ejecución |
| Conectores | Gmail (envío) |
| Notificación | Push al terminar |

## Prompt de la tarea

El prompt completo que recibe cada ejecución está en [`prompt.md`](prompt.md). Para cambiar el destinatario,
los criterios de búsqueda o el formato del correo, edita la tarea desde claude.ai/code (Scheduled tasks) y
actualiza este archivo para mantenerlos sincronizados.

## Cómo pausar o ajustar

- Pausar / reanudar / cambiar la hora: desde la lista de tareas programadas en claude.ai/code.
- Ver qué hizo una ejecución: cada firing crea una sesión con su transcripción.
