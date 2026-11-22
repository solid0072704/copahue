# Academia · «Claude para Finanzas y Contabilidad»

El programa de pago que sigue al curso gratuito de YouTube: más profundo, por módulos, con evaluaciones y con su
propia plataforma web. Igual que en `curso_youtube/`, todo sale de una sola fuente.

| Qué | Dónde |
|---|---|
| Temario: bloques, módulos, lecciones, prácticas y etapas | `programa.py` |
| Diagnóstico y bancos de preguntas por módulo | `evaluaciones.py` |
| La plataforma (portada, mi curso, lección, pruebas, certificado) | `web/app.js` y `web/app.css` |
| El generador del sitio | `sitio.py` |

```bash
python -m academia.programa                           # módulos, lecciones, horas y créditos de voz por etapa
python -m academia.sitio <salida> --curso curso.mp4   # el sitio, con la lección de muestra cortada del curso gratuito
```

`<salida>/index.html` es el sitio completo y se puede publicar en cualquier alojamiento estático;
`vista_previa.html` es la misma página sin envoltorio, para la vista previa privada.

## El programa

16 módulos en tres etapas, 97 lecciones y unas 12 horas de clases:

1. **Fundamentos, herramientas y conexiones** (9 módulos, 7 h): cómo funciona Claude, prompts, proyectos, archivos,
   artefactos, skills y plugins, conectores y MCP, CRM + Claude, Power BI + Claude.
2. **Aplicaciones en contabilidad y finanzas** (4 módulos): cierre mensual, control de gestión, tesorería, informes.
3. **Automatización, gobierno e implementación** (3 módulos): tareas y agentes, riesgos y control interno, plan de 90 días.

Cada lección dura entre 5 y 10 minutos (`min`). Con el ritmo medido de la voz (17,1 caracteres por segundo) y un
crédito de ElevenLabs por carácter, `programa.totales()` da el presupuesto de narración: unos 428.000 créditos para
la etapa 1 y 745.000 para el programa completo.

Lo que el temario nombra de Claude (conector de HubSpot, servidores MCP de Power BI, Claude en Excel, Cowork) se
verificó en la fecha que indica `programa.py`. Estas funciones cambian: cada módulo se vuelve a verificar al producirlo.

## Evaluaciones

- **Diagnóstico**: 10 preguntas, sin nota; recomienda por dónde partir y qué módulos reforzar.
- **Prueba por módulo**: 10 preguntas tomadas de un banco de al menos 15, con las alternativas barajadas; se aprueba
  con 70 % y cada intento trae preguntas distintas. Al entregar se ven la respuesta correcta, la explicación y la
  lección donde se enseña.
- **Tipos de pregunta**: una correcta, varias correctas, respuesta numérica con tolerancia y ordenar pasos.
- **Examen final**: 40 preguntas de todo el programa, con 75 %. Con las pruebas aprobadas, da derecho al certificado.

Un banco se escribe junto con su módulo: `evaluaciones.validar()` exige que cada pregunta apunte a una lección que
existe. Hoy están el diagnóstico y el banco del módulo 1.

## Lo que falta

- **Lecciones.** Se producen con el mismo motor del curso gratuito (láminas animadas y la voz del autor), una por
  video. La plataforma muestra «En producción» mientras una lección no tiene video.
- **Cuentas, cobro y entrega protegida.** El sitio es estático y el avance se guarda en el navegador (`Almacen`, en
  `web/app.js`). Vender el programa exige un servicio de cuentas, un medio de pago y que los videos solo se
  entreguen a quien pagó; son decisiones del autor y todavía no están tomadas.
