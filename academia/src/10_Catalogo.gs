/**
 * TPDV Academy · Catálogo inicial del programa.
 *
 * Estructura del documento «PILARES Y CONTENIDOS» (carpeta TPDV ACADEMY): 4 semanas, pilares
 * Ventas, Marketing y Legal, aprobación 80%. Cada lección apunta al archivo real de Drive.
 *
 * instalar() copia este catálogo a las hojas «Módulos», «Lecciones» y «Preguntas» SOLO si están vacías.
 * Después, el programa se administra desde la planilla (no hace falta tocar este archivo).
 * Las preguntas se redactaron a partir del material de cada pilar (la columna «Fuente» indica el archivo).
 */

function catalogo_() {
  var m = function (id, orden, semana, pilar, titulo, descripcion, objetivos, tipo) {
    return {
      'ID': id, 'Orden': orden, 'Semana': semana, 'Pilar': pilar, 'Título': titulo, 'Descripción': descripcion,
      'Objetivos': objetivos.join('\n'), 'Tipo': tipo || 'Contenido', 'Visible': 'SI'
    };
  };
  var l = function (id, modulo, orden, seccion, titulo, tipo, idArchivo, descripcion, opciones) {
    opciones = opciones || {};
    return {
      'ID': id, 'ID módulo': modulo, 'Orden': orden, 'Sección': seccion, 'Título': titulo, 'Tipo': tipo,
      'ID archivo': idArchivo || '', 'URL': opciones.url || '', 'Descripción': descripcion,
      'Duración (min)': opciones.min || '', 'Obligatoria': opciones.opcional ? 'NO' : 'SI', 'Visible': 'SI',
      'Origen': 'Catálogo inicial'
    };
  };
  var p = function (modulo, n, pregunta, opciones, correcta, explicacion, fuente) {
    return {
      'ID módulo': modulo, 'N°': n, 'Pregunta': pregunta, 'A': opciones[0], 'B': opciones[1], 'C': opciones[2],
      'D': opciones[3], 'Correcta': correcta, 'Explicación': explicacion, 'Fuente': fuente
    };
  };
  var x = function (id, titulo, tipo, carpeta, nota, modulo) {
    return {
      'ID archivo': id, 'Título': titulo, 'Tipo': tipo, 'Carpeta': carpeta, 'URL': 'https://drive.google.com/open?id=' + id,
      'Detectado': new Date(), 'Asignar a módulo (ID)': modulo || '', 'Incluir (SI/NO)': 'NO', 'Nota': nota
    };
  };

  var modulos = [
    m('INDUCCION', 1, 'Semana 1', 'Inducción', 'Te damos la bienvenida a Tu Proyecto de Vida',
      'Conoce la empresa, su propuesta de valor y cómo funciona tu programa de inducción de 4 semanas.',
      ['Explicar qué hace Tu Proyecto de Vida y qué la diferencia', 'Conocer la visión, la misión y los valores de la empresa',
        'Entender cómo avanzarás en la academia y cómo se te evaluará']),
    m('VENTAS-1', 2, 'Semana 1', 'Ventas', 'Ventas I: fundamentos, proyectos y HubSpot básico',
      'El rol del asesor, tus metas comerciales, el proceso de venta, HubSpot básico y cómo presentar un proyecto.',
      ['Explicar el proceso de venta completo', 'Proyectar tus metas a partir de contactos, visitas y conversión',
        'Leer y usar fichas técnicas de forma confiable', 'Registrar clientes y actividades en HubSpot',
        'Preparar y conducir una visita a terreno']),
    m('VENTAS-2', 3, 'Semana 2', 'Ventas', 'Ventas II: seguimiento, guiones y cierre',
      'HubSpot avanzado, guiones de llamada, plantillas de WhatsApp y correo, manejo de objeciones y técnicas de cierre.',
      ['Gestionar negocios y etapas del embudo en HubSpot', 'Hablar con fluidez en el primer contacto, el seguimiento y el cierre',
        'Aplicar la cadencia de contacto con leads que no responden', 'Responder objeciones con honestidad y seguridad',
        'Elegir la técnica de cierre adecuada para cada cliente']),
    m('MARKETING-1', 4, 'Semana 2', 'Marketing', 'Marketing I: leads, canales y portafolio',
      'Del marketing tradicional al digital, la clasificación de leads, los canales de contacto, el portafolio por zona y la marca.',
      ['Distinguir un IQL, un MQL y un SQL', 'Elegir el canal adecuado (WhatsApp Business o correo) según el cliente',
        'Ubicar cada proyecto en su zona y reconocer sus colores corporativos', 'Usar correctamente el logo y el material de marca']),
    m('MARKETING-2', 5, 'Semana 3', 'Marketing', 'Marketing II: contenido para ventas y CRM',
      'Contenido desde terreno, uso correcto del material gráfico, solicitudes a marketing, grilla de contenidos y velocidad de contacto.',
      ['Generar contenido real desde terreno para mantenerte en el «top of mind»', 'Usar las creatividades de marketing sin modificarlas',
        'Pedir materiales con el Formulario de Solicitud de Creatividades', 'Contactar rápido a cada lead que entra al CRM']),
    m('LEGAL-1', 6, 'Semana 3', 'Legal y Escrituración', 'Legal I: fundamentos legales de la parcelación',
      'Derecho de propiedad, DL 3.516, subdivisión y rol propio, construcción en terreno agrícola, venta regularizada y carpeta legal.',
      ['Explicar cómo se acredita la propiedad de una parcela', 'Aplicar las reglas del DL 3.516 sobre subdivisión de predios rústicos',
        'Diferenciar una parcela con rol propio de la compra de «derechos»', 'Reconocer los documentos de la carpeta legal de un proyecto']),
    m('LEGAL-2', 7, 'Semana 4', 'Legal y Escrituración', 'Legal II: documentos de venta y preguntas frecuentes',
      'Promesa y compraventa, reserva y plazos, crédito hipotecario, compra con instrucciones y las preguntas legales más frecuentes.',
      ['Diferenciar la promesa de compraventa de la escritura definitiva', 'Explicar los plazos desde la reserva hasta la inscripción',
        'Orientar una compra con crédito hipotecario', 'Responder las preguntas frecuentes con la información vigente del área']),
    m('FINAL', 8, 'Semana 4', 'Evaluación final', 'Evaluación final: simulación comercial completa',
      'Role play del proceso de venta, desde el primer contacto hasta el cierre, con retroalimentación de tu tutora o supervisor.',
      ['Explicar cualquier proyecto TPDV sin apoyo', 'Llevar un lead por las etapas del embudo en HubSpot',
        'Manejar objeciones y resolver preguntas frecuentes', 'Guiar a un cliente desde la consulta inicial hasta la reserva'],
      'Práctica')
  ];

  var lecciones = [
    // Semana 1 · Bienvenida
    l('IND-01', 'INDUCCION', 1, '', 'Cómo funciona TPDV Academy', 'Texto', '', 'Tu ruta de 4 semanas, cómo avanzas y cómo se te evalúa.', { min: 5 }),
    l('IND-02', 'INDUCCION', 2, '', 'Quiénes somos', 'Texto', '', 'Historia, visión, misión, valores y clientes de Tu Proyecto de Vida.', { min: 6 }),
    l('IND-03', 'INDUCCION', 3, '', 'Nuestra propuesta de valor y dónde estamos', 'Texto', '', 'Lo que nos diferencia, nuestras zonas y oficinas.', { min: 6 }),

    // Semana 1 · Ventas I
    l('V1-01', 'VENTAS-1', 1, 'Bienvenida al pilar', 'TPDV Academy · Ventas (video)', 'Video', '13l2t4fGjzQUzMnuWktLnXhEuvLnzVE76',
      'Video principal del pilar de Ventas.'),
    l('V1-02', 'VENTAS-1', 2, 'Bienvenida al pilar', 'Qué es vender en TPDV', 'Documento', '1O4OZ-XHRwwQom1C-UBPQH9ND5FZZ-sVuGUG0fohDQ9I',
      'Qué significa vender en Tu Proyecto de Vida y las claves del proceso.'),
    l('V1-03', 'VENTAS-1', 3, 'Tu rol y tus metas', 'Rol y objetivos de un asesor', 'PDF', '1r77TlfDTEHNYCNam6vO3SnsFWcHpOUHf',
      'Los roles del asesor, la conversión y cómo se proyectan tus ingresos.'),
    l('V1-04', 'VENTAS-1', 4, 'Tu rol y tus metas', 'Pilar de Ventas: guía detallada', 'Documento', '1KFvt6BgZGly9oJWTDUyJwtj_V9A2x9ay0AGD7kobfK8',
      '«Vender no es presionar, es guiar con conocimiento y confianza»: objetivos comerciales, proceso de venta, fichas técnicas, CRM y guiones.'),
    l('V1-05', 'VENTAS-1', 5, 'Tu rol y tus metas', 'Presentación del pilar de Ventas', 'PowerPoint', '109mpUQN0qSHH3UQL-EfOEMMjgsTYcvsi',
      'Presentación oficial del pilar: metas, proceso de venta en 6 etapas y herramientas.'),
    l('V1-06', 'VENTAS-1', 6, 'Tu rol y tus metas', 'Proyecto 100: tu lista de contactos', 'PDF', '1Ia8_0T-4D0vvjK5EF_K6FNKLIUXlKroK',
      'Ejercicio: identifica 100 contactos de tu entorno para comenzar a prospectar.'),
    l('V1-07', 'VENTAS-1', 7, 'Proceso de venta', 'Manual de procesos del asesor', 'PDF', '10JHGNuwlXW6yTUosUlx6GBcr_5P6zwZh',
      'Los pasos del asesor, desde que llega el lead hasta la postventa.'),
    l('V1-08', 'VENTAS-1', 8, 'Proceso de venta', 'Proceso de ventas y plazos', 'PDF', '1RprffDnGSTyNOHh0GK1JGbWw9TjxTcLo',
      'Plazos de cada etapa: cotización, reserva, promesa y escritura.'),
    l('V1-09', 'VENTAS-1', 9, 'HubSpot básico', 'HubSpot 1 · Visualización del panel', 'Video', '10XevcIDsv2orHEvBf7wCkY_X9fsaYDsj',
      'Recorrido por el panel de HubSpot.'),
    l('V1-10', 'VENTAS-1', 10, 'HubSpot básico', 'HubSpot 2 · Cómo crear un contacto', 'Video', '1yBhxGHm2XjDOrK4wWokzG8KhTo5Dcnzf',
      'Registro de un cliente nuevo en el CRM.'),
    l('V1-11', 'VENTAS-1', 11, 'Proyectos y fichas', 'Proyectos zona centro y sur', 'Excel', '1K3Ep50HqVYko_vnR3KYLwKZyNxARdmeT',
      'Ficha comercial de cada proyecto: región, topografía, luz, agua, rango de precios y recorrido 360.'),
    l('V1-12', 'VENTAS-1', 12, 'Proyectos y fichas', 'Mensajes rápidos por proyecto', 'PDF', '1cldyn8eXuPtwojhtPiWv7YmMdMsI6iR4',
      'Textos breves para presentar cada proyecto por WhatsApp.'),
    l('V1-13', 'VENTAS-1', 13, 'Presentar un proyecto', 'Manual de visitas y cierre de ventas de parcelas', 'PDF', '1Kp7wW_C17x-Zvifo0MRmenqxpJqlZ4L5',
      'Cómo preparar la visita, qué mostrar en terreno y cómo cerrar.'),

    // Semana 2 · Ventas II
    l('V2-01', 'VENTAS-2', 1, 'HubSpot avanzado', 'Flujo de seguimiento en HubSpot', 'PDF', '1EaUqY05FY-VMtbdZT2cGSiGDiqNdQYcD',
      'Las etapas del embudo, preguntas clave y qué hacer en cada una.'),
    l('V2-02', 'VENTAS-2', 2, 'HubSpot avanzado', 'HubSpot 3 · Negocios y etapas del embudo', 'Video', '1i4lblqmJVzfDy7l7NQajB7Vf828VTFmp',
      'Cómo crear un negocio y moverlo por las etapas.'),
    l('V2-03', 'VENTAS-2', 3, 'HubSpot avanzado', 'HubSpot 4 · Cómo crear una plantilla', 'Video', '15ou6lejYGdkplSNDPM5X47B2w6PWPjSw',
      'Plantillas de correo para ahorrar tiempo en el seguimiento.'),
    l('V2-04', 'VENTAS-2', 4, 'HubSpot avanzado', 'HubSpot 5 · Cómo crear un fragmento', 'Video', '1Yif3n8t1RESmNpFK2j8wI5wI6EdrmCCt',
      'Fragmentos de texto reutilizables en HubSpot.'),
    l('V2-05', 'VENTAS-2', 5, 'Guiones de llamada', 'Guion telefónico por perfil de cliente', 'PDF', '1DRUOOTfuJcKTCGt5RTCHGnigueZocAz0',
      'Primer contacto y preguntas de sondeo para una venta consultiva.'),
    l('V2-06', 'VENTAS-2', 6, 'Guiones de llamada', 'Primera llamada · zona sur', 'PDF', '1aKYJa7ssztrRBmbU-JcGmb1ko0eTqMCn',
      'Guion detallado de la primera llamada.'),
    l('V2-07', 'VENTAS-2', 7, 'Guiones de llamada', 'Segunda y tercera llamada · zona sur', 'PDF', '1ke9akVtm_HfN_PTaLRuwdrZBFDEyUqSm',
      'Propuesta de valor, reserva y cierre por teléfono.'),
    l('V2-08', 'VENTAS-2', 8, 'Guiones de llamada', 'Resumen: primera, segunda y tercera llamada', 'PDF', '1fAw4OIkQEwi-bDqSHYUfOdz3Ef24Wj4Y',
      'Versión resumida de los tres guiones.', { opcional: true }),
    l('V2-09', 'VENTAS-2', 9, 'Guiones de llamada', 'Ejemplo de pitch de llamada', 'Documento', '1i22ef8IfAFTYVtE1s8w778xGmTBlWSQLra9ormafy8o',
      'Un pitch real y breve usado por el equipo.', { opcional: true }),
    l('V2-10', 'VENTAS-2', 10, 'Guiones de llamada', 'Leads que no responden: etapas y mensajes', 'PDF', '1tNwyJc6_gDYTP1DJA40ytK1DaFKggnFN',
      'Cadencia de contacto de 7 días y textos para cada intento.'),
    l('V2-11', 'VENTAS-2', 11, 'Guiones de llamada', 'Leads que sí responden: la propuesta de valor', 'PDF', '1nU_q7906GWpLSg4H0uIyfATmUqjeXY_C',
      'Correo de propuesta de valor para la zona sur.', { opcional: true }),
    l('V2-12', 'VENTAS-2', 12, 'Plantillas', 'Plantillas de WhatsApp', 'PDF', '1z58l14W1c6giPVeAWdyS9BYZY6IBDqc-',
      'Mensajes para cada etapa del proceso.'),
    l('V2-13', 'VENTAS-2', 13, 'Plantillas', 'Formato de correos', 'PDF', '1Hm35-INpQltHEIURr4dYNxIkBnGS-n_p',
      'Plantillas de correo comercial.'),
    l('V2-14', 'VENTAS-2', 14, 'Objeciones y cierre', 'Manejo de objeciones', 'Word', '1A_6_DuA0wof5VkU5qP-cUTwux3KTn_3_',
      'Las objeciones y dudas más comunes, con respuestas sugeridas.'),
    l('V2-15', 'VENTAS-2', 15, 'Objeciones y cierre', 'Preguntas difíciles de clientes y cómo abordarlas', 'PDF', '1YLz4wbAwzgQoJMl9orkzyadzauEJSKTR',
      'Preguntas frecuentes y difíciles, incluidas contribuciones y DL 3.516.'),
    l('V2-16', 'VENTAS-2', 16, 'Objeciones y cierre', 'Técnicas de cierre', 'PDF', '1BOXH6Oi8juQTUfxnXNa9zJPaV37bAH1U',
      'Cinco tipos de cierre: escasez, comparación, consultivo, emocional y con alternativa.'),
    l('V2-17', 'VENTAS-2', 17, 'Práctica', 'Simulación de llamadas: tu plan de práctica', 'Texto', '',
      'Escucha activa, role play y llamadas supervisadas.', { min: 4 }),

    // Semana 2 · Marketing I
    l('M1-01', 'MARKETING-1', 1, '', 'TPDV Academy · Marketing (video)', 'Video', '17xHm4U4yH807INBRgctSqvR__xx3moQY',
      'Video principal del pilar de Marketing.'),
    l('M1-02', 'MARKETING-1', 2, '', 'Pilar I: fundamentos del marketing aplicado', 'Documento', '100JEbybm2MVA1miyABrK9bS0K25fd1U7a1OlxhJBVEM',
      'Lee las secciones 1 a 4: del marketing tradicional al digital, leads (IQL, MQL y SQL), canales de contacto, Buk y portafolio por zona.'),
    l('M1-03', 'MARKETING-1', 3, '', 'Manual de Marca TPDV', 'PDF', '1lTjPKCV0x2LGu4JoVbbJj_AdlvyLEZKO',
      'Logotipo y sus variantes, usos correctos e incorrectos, paleta, tipografías, fotografía e íconos.'),
    l('M1-04', 'MARKETING-1', 4, '', 'Logos oficiales 2026', 'Carpeta', '1X22sXtdWtcQK3DdadRGIv3i29hl2VLLW',
      'Versiones horizontal, vertical e isotipo para descargar.', { opcional: true }),

    // Semana 3 · Marketing II
    l('M2-01', 'MARKETING-2', 1, '', 'Pilar II: contenido y gestión para ventas', 'Documento', '100JEbybm2MVA1miyABrK9bS0K25fd1U7a1OlxhJBVEM',
      'Lee las secciones 5 y 6: contenido desde terreno, uso correcto del material gráfico, solicitudes a marketing, grilla de contenidos y HubSpot.'),
    l('M2-02', 'MARKETING-2', 2, '', 'Checklist: contenido que vende', 'Texto', '',
      'Resumen práctico para publicar desde terreno y trabajar con marketing.', { min: 4 }),
    l('M2-03', 'MARKETING-2', 3, '', 'Ejemplo: Grilla de contenidos 2026', 'Planilla', '1M8DwwJLl8W6vkTdZ0WtOFtFVN8thKUbNXtG0xFaa6ug',
      'Así planifica marketing el calendario editorial: pilar de contenido, proyecto, plataforma y formato.', { opcional: true }),

    // Semana 3 · Legal I
    l('L1-01', 'LEGAL-1', 1, '', 'TPDV Academy · Escrituración (video)', 'Video', '14Sb6QYJ5NYvLdyZEBDAAYeRmvhazgzny',
      'Video principal del pilar Legal y Escrituración.'),
    l('L1-02', 'LEGAL-1', 2, '', 'Presentación del pilar Legal y Escrituración', 'PowerPoint', '1dGo2IefiBg2bdI3xM5KWGQfJFPtpgPyJ',
      'Revisa los módulos 1 y 2: derecho de propiedad, subdivisión (DL 3.516), construcción en terreno agrícola y venta regularizada.'),
    l('L1-03', 'LEGAL-1', 3, '', 'Inducción del área de Escrituración', 'PDF', '1l38Ys8K5_KDWVd15GsHvOWq1yLv5znkA',
      'Marco normativo completo y la carpeta legal de un proyecto (láminas 1 a 12 y 24 a 26).'),
    l('L1-04', 'LEGAL-1', 4, '', 'Glosario legal TPDV', 'PDF', '15vaLGmH0p1G0GxKQmhujK8NVeBqzSvJK',
      'Los documentos legales de una compra, explicados en simple.'),
    l('L1-05', 'LEGAL-1', 5, '', 'Checklist de documentos legales', 'PDF', '1r0GnJTIMfnHTyFGZ1KnE8zh_fSoryL1q',
      'Los documentos que respaldan la compra de una parcela.'),
    l('L1-06', 'LEGAL-1', 6, '', 'Guion de la inducción de Escrituración', 'Word', '1bzSlO1UpUL-yC_zf7dXMhBgLJ9hBUniT',
      'Texto narrado del pilar, con ejemplos y consejos para el cierre. Útil para repasar.', { opcional: true }),

    // Semana 4 · Legal II
    l('L2-01', 'LEGAL-2', 1, '', 'Documentos de venta: promesa y compraventa', 'PowerPoint', '1dGo2IefiBg2bdI3xM5KWGQfJFPtpgPyJ',
      'Revisa el módulo 3: rol del área de escrituración, promesa y compraventa, cláusulas a revisar, crédito hipotecario y compra con instrucciones.'),
    l('L2-02', 'LEGAL-2', 2, '', 'Reserva, crédito hipotecario y plazos', 'PDF', '1l38Ys8K5_KDWVd15GsHvOWq1yLv5znkA',
      'Qué se necesita para pedir promesa o compraventa, el crédito hipotecario y los plazos (láminas 13 a 29).'),
    l('L2-03', 'LEGAL-2', 3, '', 'Preguntas frecuentes de clientes', 'Documento', '1pBt806NPAcvkM7Hh0vJkh_DavV_UWgyy9p0mFjt-bsA',
      'Respuestas vigentes del área de Escrituración a las preguntas legales más frecuentes.'),

    // Semana 4 · Evaluación final
    l('F-01', 'FINAL', 1, '', 'Cómo es la evaluación final', 'Texto', '', 'Qué incluye la simulación y cómo se califica.', { min: 4 }),
    l('F-02', 'FINAL', 2, '', 'Prepárate: cuestionario de alternativas y casos', 'Word', '1UDzAMLlh10NY5m7zX7QqOLjF1PHtxrVD',
      'Diez preguntas de alternativa y diez casos de clientes para practicar antes de tu evaluación.')
  ];

  var fuenteManual = 'Manual de Marca TPDV (1lTjPKCV0x2LGu4JoVbbJj_AdlvyLEZKO)';
  var fuenteMkt = '202507 - Desarrollo Pilares Marketing (100JEbybm2MVA1miyABrK9bS0K25fd1U7a1OlxhJBVEM)';
  var preguntas = [
    p('INDUCCION', 1, '¿Qué comercializa Tu Proyecto de Vida?',
      ['Departamentos en la ciudad', 'Parcelas de agrado de 5.000 m² con rol propio', 'Derechos sobre campos sin subdividir', 'Casas prefabricadas'],
      'B', 'TPDV vende parcelas de 5.000 m², cada una con rol propio individual y aprobación del SAG.', 'Lección «Quiénes somos»'),
    p('INDUCCION', 2, '¿Cuál de estos es uno de los valores de TPDV según el Manual de Marca?',
      ['Rapidez', 'Exclusividad', 'Confianza', 'Austeridad'],
      'C', 'Los valores son Innovación, Confianza, Excelencia, Comunidad, Compromiso y Sostenibilidad.', fuenteManual),
    p('INDUCCION', 3, '¿Cuál es la visión de Tu Proyecto de Vida?',
      ['Ser la inmobiliaria con más departamentos vendidos de Santiago', 'Liderar el mercado de parcelas agrícolas, promoviendo una vida rural sustentable y cercana a la naturaleza',
        'Ofrecer los precios más bajos del mercado', 'Construir una casa en cada parcela vendida'],
      'B', 'Es la visión declarada en el Manual de Marca.', fuenteManual),
    p('INDUCCION', 4, '¿Hasta cuándo acompaña TPDV a sus clientes?',
      ['Hasta que firman la reserva', 'Hasta que firman la promesa', 'Durante todo el proceso de compra, hasta la entrega oficial del terreno', 'Solo durante la visita al campo'],
      'C', '«Su trabajo no termina en la firma: acompañan, asesoran y gestionan todo el proceso de compra hasta la entrega oficial del terreno.»', fuenteManual),
    p('INDUCCION', 5, '¿Qué NO vende Tu Proyecto de Vida?',
      ['Parcelas con rol propio individual', 'Parcelas con subdivisión aprobada por el SAG', 'Derechos o cesiones informales sobre un campo', 'Parcelas inscritas en el Conservador de Bienes Raíces'],
      'C', 'No se comercializan derechos ni cesiones informales: cada parcela tiene rol propio e inscripción individual.', 'Lección «Nuestra propuesta de valor»'),
    p('INDUCCION', 6, '¿Cuáles son los tres pilares de TPDV Academy?',
      ['Ventas, Marketing y Legal', 'Finanzas, Operaciones y Ventas', 'Construcción, Paisajismo y Legal', 'Marketing, Diseño y Tecnología'],
      'A', 'La academia forma en los pilares de la gestión comercial: Ventas, Marketing y Legal.', 'PILARES Y CONTENIDOS (1A8oe09l_AEDsqAq06ubx_R8Ao8LpzyF8)'),

    p('VENTAS-1', 1, '¿Qué regla de oro aplica TPDV al uso de HubSpot?',
      ['Registrar solo a quienes ya reservaron', 'Lo que no está en el CRM, no existe', 'Registrar las llamadas una vez por semana', 'El CRM lo actualiza el equipo legal'],
      'B', 'Toda interacción con el cliente debe quedar registrada en el CRM.', 'Pilar de Ventas: guía detallada (1KFvt6BgZGly9oJWTDUyJwtj_V9A2x9ay0AGD7kobfK8)'),
    p('VENTAS-1', 2, 'En el proceso de venta de seis etapas, ¿qué viene justo después de la calificación?',
      ['Seguimiento', 'Cierre', 'Presentación del proyecto', 'Manejo de objeciones'],
      'C', 'El orden es captación, calificación, presentación del proyecto, objeciones, seguimiento y cierre.', 'Pilar de Ventas: guía detallada (1KFvt6BgZGly9oJWTDUyJwtj_V9A2x9ay0AGD7kobfK8)'),
    p('VENTAS-1', 3, 'Visitaste a 10 personas y cerraste 2 ventas. ¿Cuál es tu porcentaje de conversión?',
      ['2%', '10%', '20%', '50%'],
      'C', 'Ventas ÷ visitas × 100 = 2 ÷ 10 × 100 = 20%.', 'Pilar de Ventas: guía detallada (1KFvt6BgZGly9oJWTDUyJwtj_V9A2x9ay0AGD7kobfK8)'),
    p('VENTAS-1', 4, 'Si necesitas 5 visitas por venta y tu meta es cerrar 2 ventas en el mes, ¿cuántas visitas debes generar como mínimo?',
      ['5', '7', '10', '20'],
      'C', '5 visitas por venta × 2 ventas = 10 visitas en el mes.', 'Presentación del pilar de Ventas (109mpUQN0qSHH3UQL-EfOEMMjgsTYcvsi)'),
    p('VENTAS-1', 5, 'Una ficha indica 5.000 m² totales y 4.368 m² útiles. ¿A qué se debe la diferencia?',
      ['El SAG reserva una parte del terreno', 'A la servidumbre del camino de tránsito de la parcelación', 'La pendiente impide usar el resto', 'Se descuenta el bosque nativo'],
      'B', 'La diferencia corresponde a la servidumbre del camino interior.', 'Pilar de Ventas: guía detallada (1KFvt6BgZGly9oJWTDUyJwtj_V9A2x9ay0AGD7kobfK8)'),
    p('VENTAS-1', 6, 'Al coordinar una visita a terreno, ¿cuántos lotes conviene preseleccionar?',
      ['Solo 1', '2 o 3 lotes estratégicos', 'Al menos 5', 'Todos los disponibles'],
      'B', 'El manual recomienda llevar 2 o 3 lotes estratégicos preseleccionados.', 'Manual de procesos del asesor (10JHGNuwlXW6yTUosUlx6GBcr_5P6zwZh)'),

    p('VENTAS-2', 1, 'Un lead no responde después de 5 a 7 intentos. ¿Qué corresponde hacer en HubSpot?',
      ['Eliminarlo', 'Llamarlo todos los días hasta que conteste', 'Moverlo a «Perdido» o «Contactar más adelante», dejando una tarea', 'Traspasarlo sin registrar nada'],
      'C', 'Se pausa el lead con una tarea para retomarlo más adelante sin saturarlo.', 'Flujo de seguimiento en HubSpot (1EaUqY05FY-VMtbdZT2cGSiGDiqNdQYcD)'),
    p('VENTAS-2', 2, 'Con un lead que no responde, ¿qué se hace el día 1?',
      ['Un llamado y un correo', '3 intentos de llamada en 3 horarios distintos', 'Solo un WhatsApp con el video', 'Darlo de baja'],
      'B', 'La cadencia parte con 3 intentos en horarios distintos durante las primeras 24 horas.', 'Leads que no responden: etapas y mensajes (1tNwyJc6_gDYTP1DJA40ytK1DaFKggnFN)'),
    p('VENTAS-2', 3, 'Según el guion de la segunda llamada, ¿por cuánto tiempo la reserva bloquea el lote para visitar y concretar la compra o cambiar de lote?',
      ['24 horas', '10 días', '30 días', '90 días'],
      'B', 'La reserva mantiene el lote por 10 días.', 'Segunda y tercera llamada · zona sur (1ke9akVtm_HfN_PTaLRuwdrZBFDEyUqSm)'),
    p('VENTAS-2', 4, 'Un cliente dice «el sector es muy seco» y en ese proyecto no es recomendable perforar un pozo. ¿Qué indica la guía?',
      ['Asegurar que igual podrá tener pozo', 'Cambiar de tema', 'Ser honesto y mostrar alternativas reales, como camión aljibe o estanques acumuladores', 'Ofrecer un descuento'],
      'C', 'La guía pide claridad y honestidad, mostrando alternativas reales.', 'Manejo de objeciones (1A_6_DuA0wof5VkU5qP-cUTwux3KTn_3_)'),
    p('VENTAS-2', 5, 'En el sondeo, el cliente cuenta que busca un lugar para desconectarse. ¿Qué conviene destacar?',
      ['La cercanía al pueblo', 'La privacidad y la naturaleza del entorno', 'Que es el precio más bajo', 'El financiamiento'],
      'B', 'El discurso se adapta a lo que el cliente valora.', 'Guion telefónico por perfil de cliente (1DRUOOTfuJcKTCGt5RTCHGnigueZocAz0)'),
    p('VENTAS-2', 6, '«¿Te gustaría visitarlo este fin de semana o prefieres coordinar durante la semana?» ¿Qué técnica de cierre es?',
      ['Por escasez', 'Por comparación', 'Emocional', 'Con alternativa'],
      'D', 'Ofrece dos opciones de cuándo, en vez de preguntar si quiere visitar.', 'Técnicas de cierre (1BOXH6Oi8juQTUfxnXNa9zJPaV37bAH1U)'),

    p('MARKETING-1', 1, '¿Qué caracteriza a un SQL (Sales Qualified Lead)?',
      ['Dejó sus datos tras ver una publicidad en Instagram', 'Está listo para comprar: pidió cotización, reserva o disponibilidad de lotes', 'Solo interactuó con contenido orgánico', 'Es un contacto interno de Buk'],
      'B', 'Los SQL están listos para la compra y requieren seguimiento inmediato.', fuenteMkt),
    p('MARKETING-1', 2, 'Una persona busca en Google «Parcelas en la Región de Los Lagos», encuentra TPDV y completa el formulario web. ¿Cómo se clasifica?',
      ['IQL', 'MQL', 'SQL', 'No es un lead'],
      'B', 'Es el ejemplo de MQL del documento: interés específico y una intención de compra más clara.', fuenteMkt),
    p('MARKETING-1', 3, 'Alguien vio una publicidad en su feed de Instagram, le llamó la atención y completó el formulario. ¿Qué tipo de lead es?',
      ['IQL', 'MQL', 'SQL', 'Cliente'],
      'A', 'Un IQL dejó su información para conocer más, sin una intención clara de compra aún.', fuenteMkt),
    p('MARKETING-1', 4, '¿Qué canal es el más adecuado para enviar cotizaciones y respaldar acuerdos con un cliente?',
      ['Estados de WhatsApp', 'Instagram', 'Correo electrónico (Gmail)', 'Muro de Buk'],
      'C', 'El correo es el canal formal y sirve como evidencia ante situaciones complejas.', fuenteMkt),
    p('MARKETING-1', 5, '¿Para qué usa TPDV el muro de noticias de Buk?',
      ['Enviar fichas técnicas a los leads', 'Anunciar logros, concursos, campañas internas y metas comerciales', 'Publicar avisos pagados', 'Registrar leads nuevos'],
      'B', 'Buk es un canal interno de cultura y motivación.', fuenteMkt),
    p('MARKETING-1', 6, '¿Qué color corporativo identifica a los proyectos de la zona Sur?',
      ['#FFB829 (amarillo)', '#85B500 (verde)', '#007264 (verde azulado)', '#005A82 (azul)'],
      'C', 'Norte #FFB829, Centro #85B500, Sur #007264 y Patagonia #005A82.', fuenteMkt),
    p('MARKETING-1', 7, '¿A qué zona pertenece la Región de Aysén?',
      ['Norte', 'Centro', 'Sur', 'Patagonia'],
      'D', 'Aysén corresponde a la zona Patagonia.', fuenteMkt),

    p('MARKETING-2', 1, 'Marketing te envía una creatividad para una campaña especial. ¿Qué corresponde hacer?',
      ['Cambiar el logo por el del proyecto', 'Usarla sin modificar ni cubrir los logos o textos originales', 'Agregarle un texto propio encima', 'Recortarla para que se vea más grande'],
      'B', 'Las piezas ya están optimizadas para comunicar de forma clara y efectiva.', fuenteMkt),
    p('MARKETING-2', 2, 'Gracias a Zapier, cuando recibes un lead en HubSpot, ¿hace cuánto ingresó?',
      ['Menos de 1 minuto', 'Una hora', 'Un día', 'Una semana'],
      'A', 'Zapier conecta los formularios con el CRM casi en tiempo real: cuanto más rápido contactas, mayor la probabilidad de venta.', fuenteMkt),
    p('MARKETING-2', 3, '¿Qué busca publicar fotos o videos del campo en tus estados de WhatsApp Business?',
      ['Reemplazar el CRM', 'Reactivar el recuerdo («top of mind») de leads ya contactados', 'Cumplir un requisito legal', 'Bajar el precio de la parcela'],
      'B', 'Es una técnica de «top of mind» que puede gatillar una decisión de compra.', fuenteMkt),
    p('MARKETING-2', 4, 'Para pedir a marketing materiales específicos para un proyecto, ¿qué se usa?',
      ['La Grilla Orgánica de Contenidos', 'El Formulario de Solicitud de Creatividades', 'El muro de Buk', 'Un correo al cliente'],
      'B', 'El formulario existe para solicitar materiales para proyectos o campañas.', fuenteMkt),
    p('MARKETING-2', 5, 'Según el Manual de Marca, ¿qué tipografías usa TPDV?',
      ['Arial y Times New Roman', 'Montserrat para títulos y Lato para textos', 'Roboto para todo', 'Lato para títulos y Montserrat para textos'],
      'B', 'Montserrat es la tipografía principal (títulos y destacados) y Lato la secundaria (textos extensos).', fuenteManual),

    p('LEGAL-1', 1, 'Un cliente pregunta qué lo convierte legalmente en dueño de su parcela. ¿Qué le respondes?',
      ['La firma de la ficha de reserva', 'El pago total del precio', 'La inscripción a su nombre en el Conservador de Bienes Raíces', 'La promesa firmada ante notario'],
      'C', 'La inscripción en el CBR acredita el dominio: es como la «cédula de identidad» del terreno.', 'Guion de la inducción de Escrituración (1bzSlO1UpUL-yC_zf7dXMhBgLJ9hBUniT)'),
    p('LEGAL-1', 2, '¿Qué garantiza el derecho de propiedad en Chile?',
      ['El Código Civil', 'La Ordenanza General de Urbanismo y Construcciones', 'La Constitución Política, artículo 19 N° 24', 'El DL 3.516'],
      'C', 'La Constitución reconoce la facultad de usar, gozar y disponer de los bienes.', 'Inducción del área de Escrituración (1l38Ys8K5_KDWVd15GsHvOWq1yLv5znkA)'),
    p('LEGAL-1', 3, '¿Cuál es la superficie mínima de un lote al subdividir un predio rústico?',
      ['1.000 m²', '2.500 m²', '5.000 m²', '10.000 m²'],
      'C', 'El DL 3.516 fija un mínimo de media hectárea (5.000 m²).', 'Cuestionario oficial del pilar Legal'),
    p('LEGAL-1', 4, '¿Qué regula el Decreto Ley 3.516?',
      ['La construcción de viviendas', 'La subdivisión de predios rústicos', 'Los créditos hipotecarios', 'Las propiedades urbanas'],
      'B', 'Es la norma sobre división de predios rústicos.', 'Cuestionario oficial del pilar Legal'),
    p('LEGAL-1', 5, 'En una parcela de 5.000 m², ¿cuánta superficie se puede destinar a la vivienda según el material del área?',
      ['250 m²', '500 m² (hasta el 10% del terreno)', '1.000 m²', 'No se puede construir vivienda'],
      'B', 'Se permite hasta el 10% del terreno para la vivienda del propietario o de sus trabajadores, con permiso de la DOM.', 'Inducción del área de Escrituración (1l38Ys8K5_KDWVd15GsHvOWq1yLv5znkA)'),
    p('LEGAL-1', 6, 'Otra empresa le ofrece al cliente «derechos» sobre un campo sin subdividir. ¿Cuál es el riesgo principal?',
      ['Paga el doble de contribuciones', 'No obtiene dominio exclusivo sobre un lote determinado', 'Solo puede destinarlo a turismo rural', 'Debe construir dentro de un año'],
      'B', 'Comprar derechos no garantiza propiedad exclusiva porque no hay inscripción individual del lote.', 'Inducción del área de Escrituración (1l38Ys8K5_KDWVd15GsHvOWq1yLv5znkA)'),

    p('LEGAL-2', 1, '¿Qué documento es un contrato preliminar en el que las partes se obligan a celebrar la compraventa en una fecha futura?',
      ['La escritura pública', 'La promesa de compraventa', 'La ficha de reserva', 'El certificado de hipotecas y gravámenes'],
      'B', 'La promesa es el paso previo a la escritura definitiva de compraventa.', 'Inducción del área de Escrituración (1l38Ys8K5_KDWVd15GsHvOWq1yLv5znkA)'),
    p('LEGAL-2', 2, 'Un cliente comprará con crédito hipotecario. ¿Qué pie mínimo debe pagar?',
      ['10%', '20%', '30%', '50%'],
      'C', 'Con crédito hipotecario el pie es del 30%; el 50% corresponde a la promesa con saldo en cuotas.', 'Material del área de Escrituración'),
    p('LEGAL-2', 3, 'En un crédito hipotecario, ¿sobre qué valor financia el banco?',
      ['El precio de venta pactado', 'El valor que arroja la tasación', 'El avalúo fiscal del SII', 'El monto de la reserva'],
      'B', 'El banco financia sobre el valor de la tasación, no sobre el precio de venta.', 'Presentación del pilar Legal y Escrituración (1dGo2IefiBg2bdI3xM5KWGQfJFPtpgPyJ)'),
    p('LEGAL-2', 4, 'Una vez completada la reserva, ¿de cuánto plazo dispone el cliente para desistir con devolución de su dinero?',
      ['48 horas', '5 días hábiles', '10 días corridos', '15 días hábiles'],
      'C', 'Las 48 horas son para completar la reserva y los 15 días hábiles para firmar la promesa o la compraventa.', 'Inducción del área de Escrituración (1l38Ys8K5_KDWVd15GsHvOWq1yLv5znkA)'),
    p('LEGAL-2', 5, '¿Qué es una «compra con instrucciones»?',
      ['Pagar el 50% y el saldo en cuotas', 'Que el vale vista quede en custodia en la notaría hasta que la parcela se inscriba a nombre del cliente en el CBR',
        'Que el banco redacte la escritura', 'Firmar mediante un poder apostillado'],
      'B', 'Una vez inscrita la parcela se libera el vale vista; la comisión se paga después de liberada la instrucción.', 'Presentación del pilar Legal y Escrituración (1dGo2IefiBg2bdI3xM5KWGQfJFPtpgPyJ)'),
    p('LEGAL-2', 6, 'Una clienta casada en sociedad conyugal quiere comprar sin la firma de su cónyuge. ¿Qué norma lo permite?',
      ['El artículo 55 de la LGUC', 'El artículo 150 del Código Civil, acreditando ingresos propios', 'Un certificado CONADI', 'Un usufructo a su favor'],
      'B', 'El artículo 150 regula el patrimonio reservado; se acredita con liquidaciones de sueldo o carpeta tributaria.', 'Preguntas frecuentes de clientes (1pBt806NPAcvkM7Hh0vJkh_DavV_UWgyy9p0mFjt-bsA)')
  ];

  var notas = 'Notas de Gemini: actas de producción de la academia (no es material para alumnos).';
  var excluidos = [
    x('1HCjmbRztJQtF00AWrgYEWj9uLMCB6zDRBqqrvOS8z7w', 'TPDV ACADEMY · 2026/01/27 · Notas de Gemini', 'Documento', 'TPDV ACADEMY', notas),
    x('1-EearoZZ_CGn4Ee0xjYzK-VCphbP_tpfIOnosOOAq8Q', 'Avances y continuidad TPDV Academy · 2026/03/10 · Notas de Gemini', 'Documento', 'TPDV ACADEMY', notas),
    x('1kGc_wY2WaapjIEif-H1iKJmripFc24YuSvJxSzPixhM', 'ACADEMIA TPDV STEP 2 · 2025/07/18 · Notas de Gemini', 'Documento', 'TPDV ACADEMY', notas),
    x('1gAQGDI4MN12LS0xw6CxzDqfc7hL376kcFn8X_M0J-tw', 'Seguimiento Guiones Pilar Legal · 2025/07/21 · Notas de Gemini', 'Documento', 'TPDV ACADEMY', notas),
    x('1NXCYuEd2ICfxtsRmLefBdZbscr8mz_yUcNj7Dt74X-8', 'GUIONES ACADEMIA Y PROGRAMACION BAJADAS · 2025/08/13 · Notas de Gemini', 'Documento', 'TPDV ACADEMY', notas),
    x('1pp_73Q_pkjX4D6P-YdHY8bQRGTsHtFZnp_7WbT5Euwg', 'Revisión Pilar Ventas · 2026/03/17 · Notas de Gemini', 'Documento', 'TPDV ACADEMY / Ventas', notas),
    x('1v21MCzN7c2xyifhnSPUGdphx5N1f1LtFn4UOLZXb5HI', 'Seguimientos Guiones Pilar Ventas · 2025/07/21 · Notas de Gemini', 'Documento', 'TPDV ACADEMY / Ventas', notas),
    x('12OEeSEvBcMTXoLsLsbJEWH9fHdxnecofcobpryzzohk', 'REVISION MATERIAL VENTAS · 2026/04/09 · Notas de Gemini', 'Documento', 'TPDV ACADEMY / Ventas', notas),
    x('16JMBYTNRAPeVi2N1yGq3q8ip6lvfUcRPifXOhvGigMc', 'Bajada de guión MKT-VENTAS · 2025/09/09 · Notas de Gemini', 'Documento', 'TPDV ACADEMY / Ventas', notas),
    x('15dqccNMrmI3fbVCj967lFH9ISD0ywn1gJeHosBwmgS0', 'Entrega Contenidos Pilar Ventas · 2025/07/07 · Notas de Gemini', 'Documento', 'TPDV ACADEMY / Ventas', notas),
    x('15yGvBbAAheB7GwHg-Vn57qjUrZqVIanw6hK72rtdu7E', 'Bajada de guión MKT-VENTAS · 2025/08/26 · Notas de Gemini', 'Documento', 'TPDV ACADEMY / Ventas', notas),
    x('1eMc77OzSlc2C5yHSxPLAmnfo11djzDCQ5tHVqZm-K88', 'Bajada de guión MKT-Escrituración · 2025/09/10 · Notas de Gemini', 'Documento', 'TPDV ACADEMY / Escrituración', notas),
    x('1mFf1R-Mq0xWFF6YYiDamvQiaMS81T9BEg3O00_UI1_M', 'SEGUIMIENTO PILAR ESCRITURACION · 2026/04/06 · Notas de Gemini', 'Documento', 'TPDV ACADEMY / Escrituración', notas),
    x('1oNFzD4ldxdKfSaV5Dr9Jz9WOf8qDjJHaRkhjxeh-A-I', 'Revisión Pilar Escrituración · 2026/03/17 · Notas de Gemini', 'Documento', 'TPDV ACADEMY / Escrituración', notas),
    x('1DqLRimmdjS-g0Ea9A9oQ3lZ5VtfRr8JhBCqspZcSl0Y', 'Bajada de guión MKT-Escrituración · 2025/08/27 · Notas de Gemini', 'Documento', 'TPDV ACADEMY / Escrituración', notas),
    x('18qXAXm_sY5OY6-_0wcgA5kHr1VycoBt28bvD1Arunyw', 'Academia TPDV · Pilar MKT · 2025/06/23 · Notas de Gemini', 'Documento', 'TPDV ACADEMY / Marketing', notas),

    x('1A8oe09l_AEDsqAq06ubx_R8Ao8LpzyF8', 'PILARES Y CONTENIDOS', 'PDF', 'TPDV ACADEMY', 'Lineamientos para formadores; resumido en la lección «Cómo funciona TPDV Academy».'),
    x('1gV3r3GV8FJvoO_pmSFoGuwHOA1RhJwka_ECtdF4emTc', 'Guión Pilares de ventas', 'Documento', 'Ventas / Introducción Pilar I', 'Guion de producción del video (incluye notas de filmación).'),
    x('1nLsyXG6R-03tb8Hw2g_9KhbKZqMBSOdz3SJG4eFPzMw', 'Guión hablado Pilar de ventas', 'Documento', 'Ventas / Introducción Pilar I', 'Guion de producción; el plan de role play está resumido en la lección V2-17.'),
    x('139xstRSc0OtVT0nZ6MaYks5g6Besdfu99xmgcHqyQSA', 'Guión para videos con pausas', 'Documento', 'Ventas / Introducción Pilar I', 'Guion del video de Ventas. Puede servir como transcripción; revisar tramos de comisión antes de publicarlo.'),
    x('1g9HunNAcnxma4TrmNLmtUquQYnXNfUj8', '202509 - Ejemplo Informe Semanal I.S.A -TPDV Academy.xlsx', 'Excel', 'Ventas', 'Contiene datos personales de prospectos reales: anonimizar antes de usarlo como ejemplo.'),
    x('192yTgAl8AdmB1OOIVDkT8-wPLpnFh-o5', 'Presentación de Inducción para Nueva Asesora Comercial.docx', 'Word', 'Ventas', 'Borrador de inducción anterior (reemplazado por este programa).'),
    x('1KgfRZjg3U4gU5MgKsNm3BMDF-Z0hkzaH', 'GUION-Presentación de Inducción para Nueva Asesora Comercial.docx', 'Word', 'Ventas', 'Borrador de inducción anterior (reemplazado por este programa).'),
    x('1BCMT_Ht0GqHL5-chkUMZ5CjDRADZWd16VLPT94-P784', '2025 - Ventas - TPDV Academy (Google Vids)', 'Archivo', 'Ventas', 'Sin acceso para la cuenta que revisó la carpeta; se usa la versión .pptx (V1-05).'),
    x('1ysJ0dPad1sS985-cj_6CikXPnIAmmIfZ', 'Preguntas Difíciles de Clientes y Cómo Abordarlas.pdf (07-07-2025)', 'PDF', 'Ventas / Pilar II', 'Versión superada; se usa la del 11-07-2025 (V2-15).'),
    x('1xxKoxAkrnNXFdPHmwVCVUNdCpd0rfO1M', 'TPDV - Llamadas Que NO Responden - Las Etapas.pdf', 'PDF', 'Ventas / Llamada telefonica', 'Superado por «Etapas + Correspondencia» (V2-10).'),
    x('1tdg3rq9qTcECDrrQhTHsmjPPJOAZTQ_o', 'puntos de valor zona centro.xlsx', 'Excel', 'Ventas / Pilar II',
      'Revisar antes de incluir: proyecciones de plusvalía sin respaldo y precios de la competencia. Si se aprueba, escribe SI en «Incluir».', 'VENTAS-1'),
    x('1-vmtt-C-y6Hm2q-rWIjxdj8PSYWWI67A', 'Catálogo de precios todo Chile', 'Excel', 'INFORMACION CAPACITACION / Ventas', 'Catálogo vivo de precios y stock: compartir solo como enlace interno, no como lección.'),
    x('15apwVU50slg0ndoOgE03msGSqMgskeIX', 'Capacitación vendedores TPDV.pdf', 'PDF', 'INFORMACION CAPACITACION / Ventas', 'Contiene certificados reales con datos personales: no publicar.'),
    x('1D6dAWmo15U7oYIjv_yEdpVJPGjvW2YwN', '🎤 Guion Hablado Inducción – Área de Escrituración (1).docx', 'Word', 'Escrituración', 'Versión anterior; se usa el guion actualizado (L1-06).'),
    x('1RZSvkJrafwRexfvc29objmZIpeeM1Dh3', 'Preguntas y Respuestas desarrollo escrituracion.xlsx', 'Excel', 'Escrituración', 'Pauta con respuestas esperadas para el evaluador: no mostrar a los alumnos.'),
    x('1PLKHyog8kpZGk5MaO6GSz1CnwwFYiucW', 'Cuestionario para evaluador.docx', 'Word', 'Escrituración', 'Clave de respuestas del evaluador: no mostrar a los alumnos.'),
    x('1nPLGreRIxjVMqtlMtwNiG02mTmrMAxYR', '2025 - Marketing - TPDV Academy.key', 'Archivo', 'Marketing', 'Keynote de 1,4 GB: Drive no lo previsualiza. Exportarlo a PDF y agregar ese PDF a MARKETING-1.'),
    x('1AmARyuFXCclOBaBZqaAOQfIChf9jp5bO', '202507 - Desarrollo Pilares Marketing.pdf', 'PDF', 'Marketing', 'Mismo contenido que el Google Doc usado en M1-02 y M2-01.'),
    x('1doBOlgA0yZjlTGT04-_qnZTVNXaXePsiPyHxY_59vZ4', '202507 - Desarrollo Pilares Marketing (segunda versión)', 'Documento', 'Marketing', 'Sin acceso para la cuenta que revisó la carpeta: confirmar si reemplaza a la versión usada.'),
    x('1jFh40srYkfNZDX9enqNn8w5e8E2Pa_6i', 'Carpeta legal de ejemplo: Villa Alegre', 'Carpeta', 'INFORMACION CAPACITACION / Area Legal',
      'Candidata a ejemplo de carpeta legal para LEGAL-1: verificar que no contenga datos personales antes de incluirla.', 'LEGAL-1')
  ];

  return { modulos: modulos, lecciones: lecciones, preguntas: preguntas, excluidos: excluidos };
}

/**
 * Lecturas propias de la academia (lecciones de tipo «Texto»). HTML confiable, escrito para la plataforma.
 * {{PUNTAJE_APROBACION}} se reemplaza por el valor de la hoja «Config».
 */
function contenidoInterno_() {
  return {
    'IND-01':
      '<h2>El objetivo</h2>' +
      '<p>TPDV Academy forma a los nuevos asesores comerciales en los tres pilares de la gestión comercial: <strong>Ventas, Marketing y Legal</strong>. ' +
      'Es una inducción integral para que inicies tu gestión comercial efectiva a partir del segundo mes.</p>' +
      '<h2>Tu ruta de 4 semanas</h2>' +
      '<table><thead><tr><th>Semana</th><th>Pilares</th><th>Qué lograrás</th><th>Te acompaña</th></tr></thead><tbody>' +
      '<tr><td>1</td><td>Bienvenida + Ventas I</td><td>Introducción, herramientas básicas y recorrido por los proyectos</td><td>Mentora o supervisora de Ventas</td></tr>' +
      '<tr><td>2</td><td>Ventas II + Marketing I</td><td>Dominio del CRM, técnicas de venta y presentación del portafolio</td><td>Supervisora de Ventas y Marketing</td></tr>' +
      '<tr><td>3</td><td>Marketing II + Legal I</td><td>Creación de contenido y marco legal básico</td><td>Marketing y Escrituración</td></tr>' +
      '<tr><td>4</td><td>Legal II + simulaciones</td><td>Documentos, casos frecuentes y role playing</td><td>Escrituración y equipo comercial</td></tr>' +
      '</tbody></table>' +
      '<h2>Cómo avanzas</h2><ul>' +
      '<li>Cada módulo tiene lecciones: videos, presentaciones, documentos y lecturas. Márcalas como completadas a medida que las revisas.</li>' +
      '<li>Al terminar las lecciones se habilita la <strong>evaluación del módulo</strong>. Se aprueba con <strong>{{PUNTAJE_APROBACION}}%</strong> o más; si no lo logras, repasa y vuelve a rendirla.</li>' +
      '<li>Los módulos se abren en orden: el siguiente se desbloquea al completar el anterior.</li>' +
      '<li>La <strong>evaluación final</strong> es una simulación completa del proceso de venta (role play) que realiza tu tutora o supervisor.</li>' +
      '<li>Al completar todo, descargas tu <strong>certificado</strong>.</li></ul>' +
      '<div class="destacado"><strong>Meta de tu primera semana.</strong> Explicar cualquier proyecto TPDV, manejar HubSpot correctamente, realizar un primer contacto comercial, ' +
      'resolver preguntas frecuentes básicas y guiar a un cliente desde la consulta inicial hasta una eventual reserva.</div>' +
      '<p>Además del contenido en línea, tu tutora te acompañará en visitas a los campos, en la escucha de llamadas y en ejercicios de role play. ' +
      'Aprovecha cada visita a terreno: es la mejor forma de conocer lo que vendes.</p>',

    'IND-02':
      '<p>Tu Proyecto de Vida (TPDV) es una empresa chilena dedicada a la venta de parcelas, que ha construido un compromiso real con cada cliente. ' +
      'Nuestro trabajo no termina en la firma: acompañamos, asesoramos y gestionamos todo el proceso de compra hasta la entrega oficial del terreno, ' +
      'asegurando una experiencia clara, confiable y personalizada.</p>' +
      '<div class="grilla-datos">' +
      '<div class="dato"><strong>2019</strong>año de fundación</div>' +
      '<div class="dato"><strong>+20</strong>proyectos activos</div>' +
      '<div class="dato"><strong>+3.500</strong>clientes con su proyecto de vida cumplido</div>' +
      '<div class="dato"><strong>5.000 m²</strong>por parcela, con rol propio</div></div>' +
      '<h2>Visión</h2><p>«Liderar el mercado de parcelas agrícolas, promoviendo una vida rural sustentable y cercana a la naturaleza.»</p>' +
      '<h2>Misión</h2><p>«Impulsar y promover la vida rural, brindando un servicio personalizado, confiable y comprometido con nuestros valores.»</p>' +
      '<h2>Nuestros valores</h2><p><strong>Innovación · Confianza · Excelencia · Comunidad · Compromiso · Sostenibilidad</strong></p>' +
      '<h2>A quién ayudamos</h2><ul>' +
      '<li><strong>Inversionista patrimonial:</strong> busca resguardar su capital en un activo real, la tierra, con una mirada de mediano y largo plazo.</li>' +
      '<li><strong>Familias que buscan una segunda vivienda:</strong> quieren un lugar para el fin de semana o las vacaciones en un entorno natural, cerca de lagos, ríos o la costa.</li>' +
      '<li><strong>Migración ciudad-campo o personas próximas a jubilar:</strong> buscan instalar su vivienda en el campo o en el sur, con conectividad y tranquilidad.</li></ul>' +
      '<h2>Cómo hablamos</h2><p>Con un tono cercano, inspirador, empático, transparente y profesional. Usamos conceptos claros y verdaderos: ' +
      '«Tu Proyecto de Vida», «Raíces», «certeza jurídica», «rol propio», «aprobado por el SAG», «entornos naturales».</p>',

    'IND-03':
      '<h2>Lo que nos diferencia</h2><ol>' +
      '<li><strong>Certeza y seguridad jurídica.</strong> Cada parcela tiene rol propio individual, subdivisión aprobada por el Servicio Agrícola y Ganadero (SAG) ' +
      'e inscripción en el Conservador de Bienes Raíces, conforme al Decreto Ley 3.516. No vendemos derechos ni cesiones informales.</li>' +
      '<li><strong>Entornos naturales.</strong> Proyectos que respetan la topografía, el bosque nativo, los esteros y ríos, con vistas a volcanes y lagos.</li>' +
      '<li><strong>Habilitación y conectividad.</strong> Caminos interiores estabilizados, accesos despejados, estacado de cada lote y conexión con rutas principales y centros urbanos.</li>' +
      '<li><strong>Accesibilidad.</strong> Valores de entrada competitivos, promociones de lanzamiento y reserva digital.</li></ol>' +
      '<h2>Dónde estamos</h2><p>En toda nuestra comunicación los proyectos se ordenan por zona, cada una con su color corporativo:</p>' +
      '<div class="zonas">' +
      '<div class="zona" style="background:#FFB829;color:#1A1A1A">Norte<small>Región de Coquimbo</small></div>' +
      '<div class="zona" style="background:#85B500;color:#1A1A1A">Centro<small>Valparaíso y Maule</small></div>' +
      '<div class="zona" style="background:#007264">Sur<small>Los Ríos y Los Lagos</small></div>' +
      '<div class="zona" style="background:#005A82">Patagonia<small>Región de Aysén</small></div></div>' +
      '<table><thead><tr><th>Zona</th><th>Proyectos</th></tr></thead><tbody>' +
      '<tr><td>Centro</td><td>Viñedos de Santo Domingo, Raíces de Santo Domingo, Raíces de Curicó, Raíces de Putú, Raíces de San Javier, Raíces de Villa Alegre</td></tr>' +
      '<tr><td>Sur</td><td>Raíces de La Unión, Raíces de Osorno, Raíces del Río Rahue, Raíces de Mulpulmo, Raíces del Río Chifín, Raíces de Purranque, Raíces de Puerto Octay, ' +
      'Raíces Los Coihues, Raíces Los Ulmos, Raíces de Rupanco, Raíces de Miramonte, Raíces de Concordia, Santuario de Hornopirén, Fundo Las Ortigas</td></tr>' +
      '<tr><td>Patagonia</td><td>Raíces del Río Cóndor, Raíces del Río Paloma</td></tr></tbody></table>' +
      '<p class="chico">La cartera cambia con cada lanzamiento y entrega: confirma siempre con las fichas y el catálogo vigente que te entregue tu supervisora.</p>' +
      '<h2>Nuestras oficinas</h2><table><tbody>' +
      '<tr><td>Región Metropolitana</td><td>San Patricio 4099, of. 201, Vitacura · Málaga 85 / Badajoz 130, Las Condes</td></tr>' +
      '<tr><td>Los Lagos</td><td>Francisco Bilbao 1129, of. 309 y 410, Osorno · Santa Rosa 575, of. S11, Puerto Varas · Camino Punta Larga km 1, Frutillar</td></tr>' +
      '<tr><td>O\'Higgins</td><td>Quechereguas 786, San Fernando</td></tr></tbody></table>',

    'V2-17':
      '<p>Hablar con seguridad se logra practicando. En esta etapa vas a ensayar las tres conversaciones clave: el primer contacto, el seguimiento y el cierre.</p><ol>' +
      '<li><strong>Escucha activa.</strong> Escucha llamadas reales de tu tutora y de asesores de tu zona. Fíjate en cómo se presentan, qué preguntan y cómo cierran cada conversación.</li>' +
      '<li><strong>Role play.</strong> Realiza 5 simulaciones diarias durante 3 días con tu tutora, que hará de cliente.</li>' +
      '<li><strong>Llamadas supervisadas.</strong> Luego realiza 10 llamadas reales con supervisión y conversa la retroalimentación después de cada una.</li>' +
      '<li><strong>Registra todo en HubSpot.</strong> Cada llamada, WhatsApp o reunión: lo que no está en el CRM, no existe.</li></ol>' +
      '<div class="destacado">Usa como guía los guiones de este módulo: primer contacto, segunda y tercera llamada, y la cadencia para leads que no responden.</div>' +
      '<h2>Antes de cada llamada</h2><ul>' +
      '<li>Revisa el historial del lead en HubSpot: mientras más sepas, más amena será la conversación.</li>' +
      '<li>Ten a mano la ficha del proyecto y el mensaje rápido para enviarlo apenas cortes.</li>' +
      '<li>Define el próximo paso que buscas: enviar la ficha, agendar una visita o explicar la reserva.</li></ul>',

    'M2-02':
      '<h2>Desde terreno</h2><ul>' +
      '<li>Cuando visites un campo, graba videos o toma fotos del lugar, los paisajes, los accesos o las visitas destacadas.</li>' +
      '<li>Publícalos en tus estados de WhatsApp Business o en Facebook e Instagram: reactivan a los leads que ya contactaste («top of mind»).</li>' +
      '<li>Busca la luz natural y los tonos cálidos (la «golden hour» es ideal), horizontes rectos y nada de filtros intensos: la imagen debe verse real.</li></ul>' +
      '<div class="destacado">Ejemplo: «Hoy recorrimos Raíces del Río Rahue. ¡Miren esta vista al río al atardecer!»</div>' +
      '<h2>Con el material de marketing</h2><ul>' +
      '<li>Usa las creatividades de campaña tal como llegan: <strong>no modifiques ni cubras logos o textos</strong>.</li>' +
      '<li>El logo nunca se rota, deforma, estira ni cambia de color, y no va sobre fondos sin contraste.</li>' +
      '<li>¿Necesitas una pieza para un proyecto? Pídela con el <strong>Formulario de Solicitud de Creatividades</strong>.</li></ul>' +
      '<h2>Con cada lead nuevo</h2><ul>' +
      '<li>Gracias a Zapier, el lead llega a HubSpot menos de 1 minuto después de completar el formulario.</li>' +
      '<li>Todo lead entra con prioridad media: contáctalo rápido. Cuanto antes llames, mayor es la probabilidad de concretar una visita o una venta.</li></ul>',

    'F-01':
      '<p>La evaluación final comprueba que puedes llevar una venta de principio a fin. La realiza tu tutora o supervisor y queda registrada en la academia.</p>' +
      '<h2>Qué incluye</h2><ul>' +
      '<li><strong>Simulación completa del proceso de venta (role play):</strong> desde el primer contacto hasta el cierre.</li>' +
      '<li><strong>Práctica en HubSpot:</strong> creas tu contacto y tu negocio en el embudo y lo llevas por las distintas etapas, enviando fragmentos y plantillas.</li>' +
      '<li><strong>Casos de clientes:</strong> respondes situaciones frecuentes, como un cliente desconfiado, la compra de derechos, el crédito hipotecario, los plazos o la sociedad conyugal.</li>' +
      '<li><strong>Presentación comercial estructurada</strong> de un proyecto.</li></ul>' +
      '<h2>Cómo se califica</h2>' +
      '<p>Cada caso o criterio se evalúa como <strong>Logrado</strong> (3 puntos), <strong>Por lograr</strong> (2) o <strong>Debe trabajar</strong> (1). ' +
      'Se aprueba con <strong>{{PUNTAJE_APROBACION}}%</strong> o más del puntaje total y recibirás retroalimentación individual con tus fortalezas y puntos de mejora.</p>' +
      '<div class="destacado">Prepárate con el cuestionario de esta sección y repasa las preguntas frecuentes de clientes del módulo Legal II.</div>'
  };
}
