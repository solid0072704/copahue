# Guion · Análisis de estados financieros con Claude

108 láminas · 96 audios · 57.936 caracteres de narración · ≈ 59 minutos

Cada párrafo es un cambio en pantalla. Las cifras se escriben como se ven; la voz las recibe en palabras.
Para cambiar el texto, edita los archivos de `contenido/` y vuelve a generar este documento.


## Introducción

### m00_s01 · Gancho

Cada mes cierras los estados financieros y tienes el balance, el estado de resultados y el flujo de caja sobre la mesa.

↳ La pregunta es qué haces con ellos.

↳ Porque tener los números no es lo mismo que entenderlos, y entenderlos no es lo mismo que decidir.

### m00_s02 · Título del curso

Hola, soy Rodrigo, de Hyperion Advisory. En esta clase de una hora vas a aprender a analizar estados financieros con Claude, paso a paso: desde preparar la información hasta entregar un informe con conclusiones y un plan de acción.

### m00_s03 · Lo único que necesitas

No necesitas saber programar ni ser experto en inteligencia artificial. Solo necesitas tus estados financieros, una cuenta de Claude y las preguntas correctas. Y esas preguntas son justamente lo que te vas a llevar de aquí.

### m00_s04 · ¿Para quién es este curso?

Este curso es para ti si preparas el cierre y el informe de gestión, como gerente de finanzas o controller.

↳ Si eres dueño o gerente general, y quieres entender tus números sin depender de nadie.

↳ Si eres contador o asesor, y quieres entregar más análisis a tus clientes.

↳ O si eres analista, y quieres dedicar menos tiempo a armar tablas, y más a entender qué está pasando.

### m00_s05 · Un método de 8 pasos: de los datos a la decisión

Vamos a trabajar con un método de 8 pasos.

↳ Primero preparas la información y le das contexto a Claude.

↳ Después verificas que los números cuadren.

↳ Luego entiendes la estructura y la tendencia,

↳ mides con ratios,

↳ sigues la caja,

↳ comparas contra el presupuesto,

↳ buscas señales de alerta

↳ y, al final, conviertes todo eso en decisiones.

En cada paso te voy a mostrar el prompt exacto que uso, por qué funciona y qué responde Claude.

### m00_s06 · El caso: una empresa en su «mejor año»

Y lo vamos a hacer con un caso práctico: Distribuidora Alerce, una empresa ficticia que en 2025 vendió 18 % más que el año anterior y tuvo la mayor utilidad de su historia.

↳ Aun así, en diciembre casi no pudo pagar los sueldos.

A lo largo de la clase vamos a descubrir, con Claude, qué pasó y qué habría que hacer.

En la descripción encuentras el enlace a nuestro ebook y al archivo del caso, para que repitas cada paso con los mismos números. Y un consejo: mira el curso con tus propios estados financieros a mano. Pausa el video en cada prompt, cópialo y pruébalo. Así, al terminar esta hora, no solo vas a conocer un método: vas a tener tu primer análisis hecho.


## Qué puede y qué no puede hacer Claude

### m01_s01 · Qué puede y qué no puede hacer Claude

Antes de escribir el primer prompt, necesitas tener claro con qué herramienta estás trabajando. Qué puede hacer Claude con tus estados financieros, y qué no.

### m01_s02 · Claude: un asistente al que le escribes como a un colega

Claude es un asistente de inteligencia artificial desarrollado por la empresa Anthropic. Conversas con él en lenguaje natural, como si le escribieras a un colega.

Le puedes adjuntar archivos: un PDF con los estados financieros, una planilla Excel, incluso una foto de un informe.

Y a partir de ahí lee, calcula, compara, explica y redacta.

### m01_s03 · Lo que Claude hace bien con tus estados financieros

Hay cinco cosas que hace muy bien. Primero, lee y ordena. Convierte un PDF de muchas páginas en una tabla limpia, en muy poco tiempo.

Segundo, calcula. Variaciones, porcentajes, ratios: todo lo que harías en Excel, sin armar las fórmulas.

Tercero, explica. No solo te dice que el margen cayó: te propone por qué, y te dice qué dato necesitaría para confirmarlo.

Cuarto, cuestiona. Le puedes pedir que busque inconsistencias, o que haga de abogado del diablo con tus propias conclusiones.

Y quinto, redacta. Un resumen para el directorio, un correo para el banco o el informe mensual de gestión.

### m01_s04 · Lo que Claude no hace

Ahora, los límites. No conoce tu negocio. No sabe que en marzo cambiaste de proveedor, ni que tu principal cliente paga a 90 días. Si no se lo dices, lo va a suponer.

Puede equivocarse. Puede leer mal una cifra de un PDF escaneado, o mezclar dos columnas. Por eso el segundo paso del método es verificar.

No reemplaza tu criterio. Claude te entrega análisis; la decisión, y la responsabilidad, siguen siendo tuyas.

Y no audita. Trabaja con la información que le das. Si los datos de origen están mal, el análisis va a estar mal, aunque esté muy bien redactado.

### m01_s05 · El modelo mental: un analista muy rápido… en su primer día

El modelo mental que más me sirve es este: piensa en Claude como un analista muy rápido, que acaba de llegar a tu empresa. Sabe mucho de finanzas, trabaja muy rápido y no se cansa. Pero es su primer día: no conoce tu industria, tus clientes ni tu historia.

¿Qué harías con un analista así? Le darías contexto,

↳ le explicarías con precisión qué necesitas,

↳ y revisarías su trabajo antes de presentarlo. Exactamente eso vamos a hacer.

### m01_s06 · Las 5 partes de un buen prompt financiero

Y esto nos lleva a la regla más importante del curso. Un buen prompt financiero tiene cinco partes.

Uno, contexto: qué empresa es, de qué industria, de qué tamaño.

Dos, datos: qué le estás entregando y de qué período.

Tres, tarea: qué quieres exactamente que haga.

Cuatro, formato: cómo quieres la respuesta, por ejemplo una tabla y tres conclusiones.

Y cinco, verificación: pídele que muestre sus cálculos y que te diga qué supuso.

Compara un prompt de tres palabras, «analiza este balance», con uno que tiene estas cinco partes. La diferencia en la respuesta es enorme, y la vas a ver en cada paso.

### m01_s07 · El mismo balance, dos preguntas

Te muestro la diferencia con el caso que vamos a usar. Si subes el balance y escribes solamente «analiza este balance», la respuesta suele ser correcta, pero genérica: te describe los totales y te recomienda monitorear la liquidez y el endeudamiento. No hay nada que no pudieras ver tú mismo.

En cambio, cuando le das contexto, una tarea precisa y un formato, la respuesta cambia de nivel. Te dice que la caja cayó casi 90 % mientras las ventas crecían 18, que el crecimiento quedó atrapado en cuentas por cobrar e inventario, que se financió con deuda de corto plazo, y por dónde empezar.

Es el mismo archivo y la misma herramienta. Lo único que cambió fue la pregunta.

### m01_s08 · Qué necesitas para seguir el curso

Para seguir el curso necesitas una cuenta en claude.ai. Puedes empezar con el plan gratuito, que tiene límites de uso, y ver después si necesitas uno de pago.

En la configuración, revisa que esté activada la función de ejecución de código y creación de archivos. El nombre puede cambiar con el tiempo, así que búscala en la sección de capacidades.

Es importante: con ella, Claude puede escribir y ejecutar un pequeño programa para hacer las cuentas, en vez de calcular de memoria. Por eso, en cada prompt le vamos a pedir que calcule con código.

Y puedes abrir ese detalle para ver qué hizo, paso a paso. Esa transparencia es lo que te permite revisar su trabajo, en vez de tener que creerle.


## Paso 1: preparar los datos y el contexto

### m02_s01 · Prepara los datos y el contexto

Paso 1: preparar la información. La calidad del análisis depende, antes que nada, de lo que le entregas. Este paso es el menos vistoso, y el que más diferencia hace.

### m02_s02 · Qué reunir antes de empezar

Reúne cinco cosas. El balance.

↳ El estado de resultados.

↳ El estado de flujo de efectivo.

↳ El presupuesto del período.

↳ Y las notas o aclaraciones que solo tú conoces.

Idealmente de dos períodos o más, porque un número aislado no dice mucho: lo que importa es cómo cambió.

Si no tienes flujo de efectivo, no te preocupes: con dos balances y un estado de resultados, Claude puede construirlo, y lo vamos a ver en el paso 5.

### m02_s03 · El formato importa: Excel antes que una foto

Sobre el formato. Lo mejor es una planilla Excel o un archivo CSV, con una fila por cuenta y una columna por período.

Lo segundo mejor es un PDF generado por tu sistema contable.

Y lo que más problemas da es un PDF escaneado o una foto, porque ahí Claude tiene que interpretar la imagen, y puede confundir un 3 con un 8.

Tres cuidados más. Indica la unidad: si son pesos, miles o millones.

↳ Deja los subtotales claramente identificados.

↳ Y usa los mismos nombres de cuenta en todos los períodos.

### m02_s04 · Así se ve un archivo bien preparado

Así se ve un archivo bien preparado. Una fila por cuenta, con el mismo nombre en todos los períodos.

Una columna por período, y el presupuesto al lado.

Los subtotales, destacados. Y la unidad, escrita arriba: cifras en millones de pesos.

No necesitas nada más sofisticado. De hecho, este es el archivo del caso, que puedes descargar desde la descripción.

### m02_s05 · Antes de subir datos reales: confidencialidad

Antes de subir información real, detente un minuto en la confidencialidad. Primero, revisa la política de tu empresa: si puedes o no usar herramientas de inteligencia artificial con información financiera.

Segundo, revisa la configuración de privacidad de tu cuenta. En los planes para empresas, Anthropic indica que por defecto no usa tus datos para entrenar sus modelos. En los planes personales, tú decides en la configuración si tus conversaciones pueden usarse para ese fin. Las condiciones cambian, así que revísalas tú mismo.

Y tercero, si tienes dudas, anonimiza: cambia el nombre de la empresa, y reemplaza los clientes por Cliente A y Cliente B. El análisis funciona exactamente igual.

### m02_s06 · El caso: Distribuidora Alerce

Presentemos entonces a la empresa del caso. Distribuidora Alerce es una mayorista de alimentos y abarrotes, que abastece a almacenes, restaurantes y hoteles. Es ficticia, pero se parece a muchas empresas reales.

En 2025 vendió 11.800 millones de pesos, 18 % más que el año anterior.

La utilidad fue de 485 millones, la más alta de su historia. El dueño estaba contento.

Hasta que en diciembre miró la cuenta corriente: de 620 millones en caja a comienzos de año, quedaban 65.

¿Cómo es posible vender más, ganar más y quedarse sin caja? Esa es la pregunta que le vamos a hacer a Claude.

### m02_s07 · Prompt 1: el contexto, antes de cualquier análisis

Ahora sí, el primer prompt. No es un análisis: es el contexto. Fíjate en la estructura.

Parte con un rol: actúa como un analista financiero sénior.

Después viene el contexto: qué empresa es, de qué industria, de qué tamaño, en qué unidad están las cifras y qué períodos le entregas.

Incluye los hechos relevantes del año, lo que no aparece en los números. En nuestro caso: se incorporaron dos cadenas como clientes y se abrió una línea de crédito. Y agrega lo que más te preocupa, porque eso orienta todo el análisis.

Al final van cuatro reglas. Que use solo tus datos, y pregunte si falta algo. Que calcule con código y muestre la fórmula. Que separe los hechos de las hipótesis.

Y la que más me gusta: que no concluya todavía, que primero te confirme qué entendió. Con esa regla detectas cualquier malentendido antes de que contamine el análisis.

### m02_s08 · ¿Qué hechos del año conviene contarle?

¿Qué hechos del año conviene contarle? Los que explican cambios, y no aparecen en los números. Clientes: quién entró y quién se fue.

↳ Cambios de precios y de descuentos.

↳ Cambios de proveedores o de costos.

↳ Inversiones y créditos nuevos.

↳ Cambios de criterio contable.

↳ Y eventos extraordinarios, como un siniestro, un juicio o la venta de un activo. Con cinco o seis líneas basta.

### m02_s09 · El prompt 1, con los datos de Alerce

Así queda el prompt con los datos de Alerce. La empresa, su giro y el país. Las ventas y la unidad: millones de pesos. Y los períodos.

Fíjate en los hechos relevantes: incorporamos dos cadenas como clientes, con descuento y con pago a 90 días o más; ampliamos el surtido; abrimos una línea de crédito. Son tres líneas que a ti te toman un minuto, y que le ahorran a Claude suponer.

Y la preocupación, dicha en simple: vendimos más, ganamos más, pero la caja bajó de 620 a 65.

### m02_s10 · La respuesta: Claude confirma antes de analizar

Y esta es la respuesta. Como le pedimos que no concluyera, Claude parte confirmando lo que entendió: qué empresa es, qué períodos va a analizar, en qué unidad, y cuál es la pregunta central.

Después hace dos preguntas. Una: si el presupuesto está en la misma unidad que los estados financieros. Parece un detalle, pero es un error muy frecuente.

Y dos: si tenemos el detalle de ventas y cuentas por cobrar por cliente, para medir el efecto de las dos cadenas.

Esa segunda pregunta es justamente el análisis que vamos a hacer más adelante. Cuando el contexto es bueno, Claude empieza a pensar como alguien que conoce el negocio.

### m02_s11 · Para no repetir el contexto cada mes: los Proyectos

Un consejo para no repetir este contexto cada mes: usa los Proyectos de Claude. Un proyecto es un espacio de trabajo donde dejas instrucciones permanentes y archivos de referencia. Yo creo un proyecto por empresa.

En las instrucciones pego el prompt de contexto.

Y como archivos de referencia subo el plan de cuentas, las definiciones de los indicadores y los informes de meses anteriores.

Así, cada conversación nueva dentro del proyecto parte sabiendo quién es la empresa y cómo te gusta trabajar.

### m02_s12 · Paso 1 en resumen

El paso 1, en resumen. Reúne los estados financieros, el presupuesto y tus notas, de dos períodos o más.

Cuida el formato y la confidencialidad.

Y parte siempre por el contexto, pidiéndole a Claude que confirme qué entendió antes de analizar.


## Paso 2: verificar que todo cuadre

### m03_s01 · Verifica que todo cuadre

Paso 2: verificar. Antes de analizar cualquier cosa, hay que confirmar que Claude leyó bien los archivos, y que los estados financieros son coherentes entre sí. Son dos minutos que te pueden ahorrar un error grave frente al directorio.

### m03_s02 · Dos tipos de error que hay que atrapar a tiempo

Hay dos tipos de errores. Los de lectura: Claude toma mal una cifra, se salta una fila o confunde miles con millones.

Y los de origen: los estados que te entregaron ya venían con un problema, como un balance que no cuadra, o un flujo que no llega al saldo de caja.

En los dos casos, si no lo detectas ahora, todo lo que viene después se construye sobre un dato malo.

### m03_s03 · Prompt 2: verificación

Este es el prompt de verificación. Le pido cuatro cosas.

Primero, que me muestre en una tabla las cifras que leyó, con su unidad. Así comparo de un vistazo contra mi archivo.

Segundo, que compruebe las cuadraturas básicas. Que los activos sean iguales a los pasivos más el patrimonio. Que la utilidad del año explique el cambio en las utilidades acumuladas, considerando los dividendos. Que el efectivo inicial más el flujo del año llegue al efectivo final del balance. Y que los subtotales sumen.

Tercero, que liste lo que falta, o lo que no le hace sentido.

Y cuarto, otra vez: que no siga hasta que yo confirme.

### m03_s04 · La verificación de Alerce: todo cuadra… y una pista

Y esto es lo que responde con los datos de Alerce. El balance cuadra en los dos años: 6.105 millones de activos contra 6.105 de pasivos y patrimonio.

La utilidad de 485 millones calza con el aumento de 285 en utilidades acumuladas, más los 200 de dividendos.

Y el flujo calza con la caja: 620 de saldo inicial, menos 555 de flujo neto, da los 65 del balance. Todo en orden.

Pero mira las observaciones, porque aquí aparece la primera pista. Claude nota que no existe provisión de incobrables, aunque las cuentas vencidas a más de 90 días pasaron de 10 a 90 millones. Y pide el detalle de la deuda de corto plazo.

Todavía no analizamos nada, y ya tenemos dos temas para conversar.

### m03_s05 · ¿Y si algo no cuadra? Un ejemplo

¿Y qué pasa cuando algo no cuadra? Te doy un ejemplo. Supón que, al leer un PDF escaneado, Claude te dice que los activos suman 6.015, y que los pasivos más el patrimonio suman 6.105. Hay una diferencia de 90.

Primer paso: pide el detalle. Pídele que te muestre qué partidas sumó en cada lado.

Segundo: compara con tu archivo. Y ahí aparece: en inventarios leyó 1.560, y el archivo dice 1.650. Dos dígitos invertidos.

Tercero: corrige y vuelve a verificar. Le indicas el valor correcto, y le pides que repita las cuadraturas.

Un dato de contador: cuando la diferencia es divisible por 9, suele ser un error de dígitos invertidos. Lo importante es que el error apareció aquí, en el paso 2, y no en la reunión de directorio.

### m03_s06 · Tres chequeos de 30 segundos, para cualquier respuesta

Más allá de las cuadraturas, te recomiendo tres chequeos de 30 segundos, para cualquier respuesta. Uno: toma una cifra al azar y compárala contra tu archivo.

Dos: busca un total que conozcas de memoria, como las ventas del año o la caja del cierre.

Y tres: repite tú un cálculo, un porcentaje o un ratio, con la calculadora.

Si los tres chequeos calzan, puedes avanzar con confianza. No es desconfianza hacia la herramienta. Es el mismo control que le harías al trabajo de cualquier persona de tu equipo.

### m03_s07 · Prompt 3: la primera lectura

Con los datos verificados, pido una primera lectura.

Máximo 10 líneas, porque quiero ver el bosque antes que los árboles.

Tres cifras que resuman el año, qué mejoró, qué empeoró, y la pregunta más importante que debería hacerme.

Y le indico para quién escribe: un directorio que no es experto en contabilidad. Ese detalle cambia por completo el tono de la respuesta.

### m03_s08 · La primera lectura: de tres estados a una pregunta

Esta es la respuesta. Las tres cifras: ventas 18 % arriba, utilidad casi 4 % arriba, y el efectivo casi 90 % abajo.

Lo que mejoró: las ventas y el resultado operacional.

Lo que empeoró: el margen bruto cayó un punto, de 26 a 25 %, y la deuda bancaria de corto plazo casi se triplicó, de 400 a 1.150 millones.

Y la pregunta que propone: ¿en qué se transformó la utilidad, si no se transformó en caja?

Fíjate en lo que acaba de pasar. En pocos minutos pasamos de tres estados financieros a una pregunta de negocio bien formulada. El resto del curso es responderla.

### m03_s09 · Paso 2 en resumen

El paso 2, en resumen. Verifica antes de analizar.

Pide la tabla de lo que leyó, y las cuadraturas.

Y lee con atención las observaciones, porque ahí aparecen las primeras pistas.


## Paso 3: análisis vertical y horizontal

### m04_s01 · Entiende la estructura y la tendencia

Paso 3: entender la estructura y la tendencia. Son las dos lecturas clásicas del análisis financiero: el análisis vertical y el análisis horizontal. Con Claude, las tienes en un solo prompt.

### m04_s02 · Dos lecturas que se complementan

Un repaso rápido. El análisis vertical mira la estructura. En el estado de resultados, expresa cada línea como porcentaje de las ventas: de cada 100 pesos que vendo, cuánto se va en costo, cuánto en gastos y cuánto queda. En el balance, expresa cada cuenta como porcentaje del total de activos.

El análisis horizontal mira la tendencia: cuánto cambió cada cuenta de un período a otro, en monto y en porcentaje.

Por separado dicen poco. Juntos responden una pregunta muy útil: qué creció más rápido que las ventas. Porque lo que crece más rápido que las ventas se está comiendo el margen o la caja.

### m04_s03 · Prompt 4: análisis vertical y horizontal

El prompt pide las dos lecturas, para el estado de resultados y para el balance.

Pero lo valioso está en los puntos 3 y 4. En el 3 le pido que marque las partidas que crecieron más que las ventas, y que explique qué implica cada una.

Y en el 4, que termine con los tres hallazgos más relevantes, ordenados por impacto en pesos. Esa instrucción evita el problema típico de estos análisis: una lista de 20 observaciones donde todo parece igual de importante.

### m04_s04 · Resultados: vende más, pero gana menos por cada peso

Veamos el estado de resultados de Alerce. Las ventas crecieron 18 %.

Pero el costo de ventas creció 19,6 %. Por eso el margen bruto bajó de 26 a 25 % de las ventas. Un punto parece poco. Sobre 11.800 millones de ventas, son 118 millones.

Los gastos de administración y ventas crecieron menos que las ventas, lo que es bueno. Salvo una línea: distribución y fletes, que subió 30 %.

Y más abajo, los gastos financieros crecieron 59 %. Es la partida que más creció de todo el estado.

Resultado: las ventas subieron 18 %, el resultado operacional 12 %, y la utilidad neta apenas 3,9 %. La empresa vende mucho más, pero en cada peso de venta gana menos: el margen neto bajó de 4,7 a 4,1 %.

### m04_s05 · El hábito más rentable: repreguntar

Y aquí viene un hábito que quiero que te lleves: repreguntar. La tabla dice que el margen bajó un punto, pero no dice por qué. Entonces le pregunto: dame las tres hipótesis más probables y, para cada una, el dato exacto que necesitas para confirmarla o descartarla.

Primera hipótesis: la mezcla de clientes. Los clientes nuevos compran con descuento. Dato necesario: el margen por cliente.

Segunda: un aumento de costos que no se traspasó a precios. Dato: la evolución del costo y del precio promedio, por familia de productos.

Tercera: la mezcla de productos. Se está vendiendo más de las líneas de menor margen. Dato: ventas y margen por línea.

Fíjate que Claude no adivina la causa. Te entrega una lista ordenada de qué ir a buscar. En el caso de Alerce, como sabemos por el contexto que entraron dos cadenas con descuento, la primera hipótesis es la candidata. La vamos a confirmar en el paso 5.

### m04_s06 · Balance: aquí está la historia

Ahora el balance, que es donde está la historia. Recuerda: las ventas crecieron 18 %.

Las cuentas por cobrar crecieron 70 %. Pasaron de 1.370 a 2.330 millones, y hoy son el 38 % de todos los activos.

Los inventarios crecieron 48 %.

¿Y cómo se financió ese crecimiento? Los proveedores aumentaron solo 15 %. La diferencia la puso el banco: la deuda de corto plazo pasó de 400 a 1.150 millones, casi el triple.

Y la caja pagó el resto: de ser el 12 % de los activos, hoy es el 1 %.

Mira también la última fila. El patrimonio pasó de financiar el 50 % de los activos a financiar el 46 %. La empresa depende cada vez más de recursos de terceros.

### m04_s07 · Los 3 hallazgos, ordenados por impacto

Estos son los tres hallazgos que entrega Claude, ordenados por impacto. Uno: el capital de trabajo. Entre cuentas por cobrar e inventarios, la empresa inmovilizó casi 1.500 millones de pesos adicionales.

Dos: el financiamiento. Ese crecimiento se pagó con 750 millones de deuda de corto plazo, y por eso los gastos financieros subieron 59 %.

Y tres: el margen. Un punto menos de margen bruto equivale a 118 millones, y los fletes subieron 130 millones más.

Ya tenemos una hipótesis sólida: la empresa creció dando más plazo, cargando más inventario y cediendo margen. Ahora hay que medirlo.

### m04_s08 · Paso 3 en resumen

El paso 3, en resumen. El análisis vertical muestra la estructura, y el horizontal, la tendencia.

Busca siempre lo que crece más rápido que las ventas.

Y pide tres hallazgos ordenados por impacto. Después, repregunta por las causas.


## Paso 4: ratios financieros

### m05_s01 · Mide con ratios

Paso 4: medir con ratios. Los ratios convierten los estados financieros en indicadores comparables: con el año anterior, con tu presupuesto y con tu industria. Los vamos a ordenar en cuatro familias.

### m05_s02 · Cuatro familias, cuatro preguntas

Liquidez: ¿puedo pagar lo que debo en el corto plazo?

Endeudamiento: ¿cuánto debo, y puedo sostenerlo?

Rentabilidad: ¿cuánto gano por cada peso que vendo, y por cada peso invertido?

Y eficiencia: ¿cuánto tardo en convertir mis ventas en caja?

Un consejo antes de seguir: un ratio aislado no significa nada. Lo que importa es la tendencia y la comparación. Una liquidez de 1,5 puede ser excelente en un supermercado, y preocupante en una constructora.

### m05_s03 · ¿Contra qué comparo un ratio?

Una pregunta que siempre aparece: ¿contra qué comparo un ratio? Tienes cuatro referencias. Tu propia historia: la tendencia de los últimos años, o de los últimos 12 meses. Es la más útil, y la que siempre tienes.

Tu presupuesto: lo que te propusiste lograr.

Tu industria, con cuidado, porque las empresas tienen tamaños y modelos de negocio distintos.

Y tus compromisos. Si tus créditos te exigen cumplir ciertos indicadores, pídele a Claude que los calcule todos los meses, y que te diga cuánta holgura te queda.

### m05_s04 · Prompt 5: ratios con fórmula, interpretación y semáforo

En el prompt le pido los cuatro grupos.

Y para cada ratio, cuatro cosas. La fórmula con las cifras reemplazadas, para poder revisarla. El valor de cada período con su variación. Una interpretación en una línea, en lenguaje simple. Y un semáforo, explicando el criterio.

Agrego dos análisis que casi nunca se hacen a mano, porque toman tiempo: el ciclo de conversión de efectivo y el análisis DuPont.

Y cierro igual que antes: los tres ratios que más me deberían preocupar.

Un detalle importante: hay ratios que tienen más de una fórmula. Por ejemplo, la rentabilidad sobre el patrimonio se puede calcular con el patrimonio final o con el promedio. Por eso pido siempre la fórmula. Si tu empresa usa una definición distinta, se la indicas y recalcula.

### m05_s05 · ¿Cómo define el semáforo?

Una aclaración sobre el semáforo. Como le pedí el criterio, Claude lo explica: verde, si el indicador mejora o se mantiene en un nivel sano.

Amarillo, si empeora, pero no compromete los pagos.

Y rojo, si empeora y además compromete los pagos de los próximos meses.

Ese criterio es suyo, y es razonable. Pero el tuyo puede ser otro. Si tu empresa tiene umbrales propios, por ejemplo una liquidez mínima, escríbelos en las instrucciones del proyecto, y el semáforo va a usar tus reglas.

### m05_s06 · Liquidez: se ve sana… hasta que miras qué hay dentro

Partamos por la liquidez. La liquidez corriente, activos corrientes sobre pasivos corrientes, bajó de 2 a 1,65.

Si miras solo ese número, dirías que la empresa está sana: tiene 1,65 pesos de activos de corto plazo por cada peso que debe a corto plazo. Pero ojo con lo que hay dentro de esos activos.

La prueba ácida, que saca los inventarios, cayó de 1,3 a 0,99.

Y la razón de caja, que considera solo el efectivo, cayó de 0,39 a 0,03. Es decir, por cada 100 pesos que la empresa debe pagar en los próximos meses, tiene 3 en la cuenta.

Esta es una de las lecciones del caso: la liquidez corriente puede verse bien y esconder un problema serio, cuando está llena de cuentas por cobrar e inventario.

### m05_s07 · Endeudamiento: no solo debe más, debe a menos plazo

Endeudamiento. La relación entre pasivos y patrimonio subió de 1 a 1,19: no es alarmante.

La deuda financiera neta sobre EBITDA pasó de 0,76 a 1,87 veces. Tampoco es un nivel crítico, pero se multiplicó por dos y medio en un año.

La cobertura de intereses bajó de 6,8 a 4,8 veces.

Y el dato que más me importa: la deuda de corto plazo pasó de ser el 31 % de la deuda bancaria a ser el 59 %. La empresa no solo debe más: debe a menos plazo. Depende de que el banco le renueve la línea.

### m05_s08 · Rentabilidad: los tres márgenes bajan

Rentabilidad. Los tres márgenes bajaron: el bruto de 26 a 25 %, el operacional de 7,5 a 7,1, y el neto de 4,7 a 4,1.

La rentabilidad sobre el patrimonio bajó de 18,7 a 17,4 %. Sigue siendo una buena rentabilidad. Pero el análisis DuPont muestra algo que el número solo no dice.

### m05_s09 · DuPont: de dónde viene la rentabilidad

DuPont descompone esa rentabilidad en tres factores: el margen neto, la rotación de activos y el apalancamiento. En 2024, la rentabilidad venía de un margen cercano a 4,7 %, una rotación de 2 veces y un apalancamiento de 2.

En 2025, el margen bajó, la rotación bajó, y lo único que subió fue el apalancamiento, a 2,19.

En simple: el negocio rinde menos, y la rentabilidad se sostiene porque la empresa usa más deuda. Eso es una rentabilidad de peor calidad.

### m05_s10 · Eficiencia: el ciclo de conversión de efectivo

Y llegamos a la familia que explica el caso: la eficiencia. Los días de inventario, es decir, cuánto tiempo pasa la mercadería en bodega, subieron de 55 a 68.

Los días de cobro, cuánto tarda la empresa en cobrar una venta, pasaron de 50 a 72.

Y los días de pago a proveedores bajaron de 50 a 48.

Con esos tres números se calcula el ciclo de conversión de efectivo: días de inventario, más días de cobro, menos días de pago. Es el tiempo que pasa desde que la empresa paga la mercadería hasta que recupera la plata. En 2024 eran 55 días. En 2025 son 92.

Son 37 días más que la empresa tiene que financiar. Y como vende más de 30 millones diarios, cada día adicional es mucha plata. Aquí está, en un solo indicador, la respuesta a por qué la utilidad no se transformó en caja.

### m05_s11 · Traduce los días a pesos: el valor de un día

Hay una forma de hacer que estos días le hablen a cualquier gerente: traducirlos a plata. Le pedí a Claude que calculara cuánto vale un día.

En Alerce, un día de cobro equivale a 32 millones de pesos de caja.

Un día de inventario, a 24 millones.

Y cada día de plazo con proveedores, a otros 24 millones de financiamiento.

Con ese dato, la conversación con el equipo comercial cambia. Ya no le dices que hay que cobrar más rápido. Le dices que cada día de demora en cobrar cuesta 32 millones de caja.

### m05_s12 · Los 3 ratios en rojo

Los tres ratios que Claude marca en rojo: la razón de caja, en 0,03.

↳ El ciclo de conversión de efectivo, que aumentó 37 días.

↳ Y el peso de la deuda de corto plazo, 59 % del total.

Ninguno de los tres aparece en un informe de gestión típico, que suele quedarse en ventas, margen y utilidad. Por eso este paso vale la pena.

Un recordatorio: antes de usar estos números en una presentación, revisa al menos dos o tres fórmulas contra tu propio Excel. Te toma un minuto, y te da la tranquilidad de que el resto está bien calculado.

### m05_s13 · Paso 4 en resumen

El paso 4, en resumen. Cuatro familias de ratios: liquidez, endeudamiento, rentabilidad y eficiencia.

Pide siempre la fórmula, la interpretación y el semáforo.

Y traduce los días a pesos, porque es el idioma que entiende toda la empresa.


## Paso 5: flujo de efectivo y capital de trabajo

### m06_s01 · Sigue la caja

Paso 5: seguir la caja. Si tuviera que quedarme con un solo paso del método, sería este. Porque una empresa puede sobrevivir varios años con poca utilidad, pero no sobrevive ni un mes sin caja.

### m06_s02 · Por qué la utilidad no es caja

La utilidad y la caja son cosas distintas, por tres razones principales. Una: vendes hoy, pero cobras después. La venta ya está en el estado de resultados; la plata todavía no está en el banco.

Dos: compras mercadería que todavía no vendes. La pagaste, pero no es costo hasta que la vendes.

Y tres: hay pagos que no pasan por el resultado, como las inversiones, las cuotas de los créditos y los dividendos.

El estado de flujo de efectivo ordena todo esto en tres bloques: la operación, la inversión y el financiamiento.

### m06_s03 · Prompt 6: qué pasó con la caja

El prompt de la caja.

Primero le pido que construya el flujo de efectivo por el método indirecto, partiendo de la utilidad. Si no tienes flujo de efectivo, y muchas empresas pequeñas no lo preparan, Claude lo arma con los dos balances y el estado de resultados.

Segundo, un puente desde la utilidad hasta la variación del efectivo.

Tercero, cuánta caja absorbió cada componente del capital de trabajo.

Cuarto, el flujo de caja libre.

Y quinto, la respuesta directa, en cinco líneas.

### m06_s04 · De la utilidad a la caja: el puente de 2025

Este es el puente de Alerce. Partimos de la utilidad: 485 millones.

Sumamos la depreciación, 170 millones, porque es un gasto que no significó salida de plata.

Y ahora el capital de trabajo. Las cuentas por cobrar aumentaron 960 millones: son ventas ya facturadas que todavía no se cobran. Esa plata la tienen los clientes.

Los inventarios aumentaron 535 millones: plata que está en la bodega.

Los proveedores financiaron una parte, 151 millones.

Resultado: el flujo de la operación fue negativo en 685 millones. Una empresa que ganó 485 millones quemó 685 en su operación. El año anterior, la operación había generado 510.

Sigamos. Invirtió 320 millones en activo fijo y pagó 200 millones en dividendos.

¿Cómo lo cubrió? Con 650 millones de deuda bancaria adicional, y con su propia caja, que bajó en 555 millones.

### m06_s05 · Flujo de caja libre: lo que realmente queda disponible

Un concepto que conviene tener claro: el flujo de caja libre. Es lo que queda del flujo de la operación después de invertir. Es la plata realmente disponible para pagar deuda y para repartir dividendos.

En Alerce: una operación que consumió 685 millones, y 320 millones de inversión.

Flujo de caja libre: menos 1.005 millones.

No había nada que repartir. Y aun así se repartieron 200 millones, financiados con deuda.

### m06_s06 · ¿Por qué la utilidad no se transformó en caja?

Y esta es la respuesta en cinco líneas. La utilidad quedó atrapada en capital de trabajo: las cuentas por cobrar y los inventarios crecieron 1.495 millones, y los proveedores financiaron solo 151.

La operación consumió 685 millones, cuando el año anterior había generado 510.

El flujo de caja libre, después de inversiones, fue negativo en 1.005 millones.

Eso se cubrió con deuda de corto plazo y con la caja que había.

Y aun así, la empresa repartió 200 millones en dividendos.

Esa última línea es la que más incomoda, y es exactamente el tipo de frase que un buen analista tiene que poner sobre la mesa.

### m06_s07 · Cuánta caja cuesta crecer

Con estos números se puede calcular algo que pocas empresas miden: cuánta caja cuesta crecer. En 2025, Alerce vendió 1.800 millones más que el año anterior.

Y para lograrlo, tuvo que inmovilizar 1.344 millones adicionales en capital de trabajo, ya descontado lo que financiaron los proveedores.

Es decir, cada 100 pesos de venta nueva le exigieron 75 pesos de caja.

Esa misma venta, con un margen neto de 4,1 %, deja unos 4 pesos de utilidad al año. O sea: la caja que exige el crecimiento equivale a unas 18 veces la utilidad anual que genera.

Este es el concepto que quiero que te lleves del caso: crecer no es gratis. Y cuando el crecimiento viene con más plazo y más inventario, puede ser la forma más rápida de quedarse sin caja.

### m06_s08 · La siguiente pregunta: ¿de qué clientes vino el crecimiento?

Con Claude, cada respuesta abre la siguiente pregunta. Le subí el detalle de ventas y cuentas por cobrar por cliente, y le pedí que cruzara margen y plazo de cobro por tipo de cliente.

El resultado es muy claro. Las dos cadenas nuevas explican todo el crecimiento del año: 1.800 millones, el 15 % de las ventas. Pero dejan un margen de 19,4 %, contra 26 % del resto de los clientes.

Y pagan a 100 días, contra 67. O sea: todo el crecimiento vino de los clientes menos rentables y más lentos para pagar.

Y un dato más: incluso los clientes tradicionales se están demorando más que antes, de 50 a 67 días. Hay un problema de cobranza que va más allá de las cadenas.

Todo esto no significa que venderles a las cadenas sea un error. Significa que hay que saber cuánto cuesta, y decidir con ese número a la vista.

### m06_s09 · Otra repregunta: la antigüedad de las cuentas por cobrar

Otra repregunta que siempre vale la pena cuando las cuentas por cobrar crecen: pedir el análisis de antigüedad. Le subí el detalle por tramo de vencimiento, y le pregunté qué parte del aumento es crecimiento del negocio, y qué parte es atraso.

En 2024, el 73 % de la cartera estaba al día. En 2025, solo el 55 %.

Lo vencido pasó de 370 a 1.050 millones: casi se triplicó.

Y el tramo más delicado, lo vencido a más de 60 días, pasó de 40 a 240 millones.

La conclusión de Claude: de los 960 millones que aumentaron las cuentas por cobrar, 680 corresponden a cartera vencida. No es solo que la empresa venda más. Es que le están pagando más tarde.

### m06_s10 · Tres preguntas para el capital de trabajo

Cuando el capital de trabajo es el problema, hay tres preguntas que le puedes hacer a Claude, una por cada componente. Para los clientes: ¿a quién le estoy dando plazo, y por qué? Pídele el plazo real de pago por cliente, comparado con el plazo pactado.

Para los inventarios: ¿qué no está rotando? Pídele los días de inventario por producto, o por familia.

Y para los proveedores: ¿estoy usando el plazo que me dan? Alerce paga a 48 días. Si sus condiciones fueran a 60, estaría financiando a sus proveedores sin necesidad.

### m06_s11 · Paso 5 en resumen

El paso 5, en resumen. La utilidad no es caja, y la caja es la que paga los sueldos.

Pide siempre el puente, desde la utilidad hasta la variación de la caja.

Y repregunta por cliente y por antigüedad, porque ahí aparece quién se está llevando la caja.


## Paso 6: desviaciones contra el presupuesto

### m07_s01 · Compara contra el presupuesto

Paso 6: comparar contra el presupuesto. Hasta ahora comparamos contra el año anterior. Pero la pregunta de gestión es otra: ¿cumplimos lo que nos propusimos? Y si no, ¿dónde nos desviamos y por qué?

### m07_s02 · Un buen análisis de desviaciones hace tres cosas

Un buen análisis de desviaciones hace tres cosas. Uno: mide. Cuánto se desvió cada línea, en pesos y en porcentaje.

Dos: prioriza. No todas las desviaciones importan; hay que fijar un umbral de materialidad y concentrarse en lo que lo supera.

Y tres: explica. Separa el efecto del volumen del efecto del precio o del margen.

Porque no es lo mismo ganar menos porque vendiste menos, que ganar menos porque vendiste más barato. Las acciones son completamente distintas.

### m07_s03 · Prompt 7: real contra presupuesto

Vamos al prompt.

Primero, la tabla clásica, indicando si cada desviación es favorable o desfavorable. Eso evita confusiones con los signos: gastar más que el presupuesto es desfavorable, vender más es favorable.

Segundo, la materialidad: le digo que analice solo las desviaciones mayores a 5 % y a 20 millones. Ajusta esos umbrales a tu empresa.

Tercero, que descomponga la desviación del margen bruto en efecto volumen y efecto margen.

Cuarto, y esto es clave: para cada desviación, la causa más probable y el dato que necesita para confirmarla. Claude no sabe por qué subieron los fletes. Pero puede decirte qué revisar.

Y quinto, que ordene por impacto en la utilidad.

### m07_s04 · Alerce contra su presupuesto: vendió más y ganó menos

Veamos Alerce contra su presupuesto. Las ventas superaron el presupuesto en 600 millones, un 5,4 %. Hasta ahí, una buena noticia.

Pero el margen bruto prácticamente no se movió: solo 38 millones más. Vendió 600 millones adicionales y casi no ganó nada con ellos.

Los gastos se pasaron en 90 millones, y toda la desviación viene de una línea: distribución y fletes, 19 % sobre el presupuesto.

Los gastos financieros se pasaron en 55 millones, 46 % más de lo previsto.

Y el resultado final: la utilidad quedó 82 millones bajo el presupuesto, un 14,5 % menos. La empresa vendió más de lo que planificó, y ganó menos de lo que planificó.

### m07_s05 · 600 millones más de venta, solo 38 más de margen

¿Cómo se explica que 600 millones más de venta dejen solo 38 de margen? Con la descomposición.

El efecto volumen: si esos 600 millones adicionales se hubieran vendido con el margen presupuestado, de 26 %, habrían aportado 156 millones.

El efecto margen: como toda la venta se hizo con un punto menos de margen, se perdieron 118 millones.

Son 156 menos 118: los 38 que vimos. Dicho de otra forma, tres cuartas partes del beneficio de vender más se fueron en vender más barato.

Este cálculo, que a mano toma un buen rato y en el que es fácil equivocarse, Claude lo entrega en un momento, y con la fórmula a la vista.

### m07_s06 · Las dos fórmulas, por si quieres revisarlas

Las dos fórmulas, por si quieres revisarlas. Efecto volumen: la diferencia entre las ventas reales y las presupuestadas, multiplicada por el margen presupuestado.

Efecto margen: la diferencia entre el margen real y el presupuestado, multiplicada por las ventas reales.

Si sumas los dos efectos, obtienes exactamente la desviación del margen bruto. Esa es tu comprobación.

### m07_s07 · Dónde poner el foco

Y este es el foco que propone, ordenado por impacto. Primero, el margen: 118 millones. Causa probable, los descuentos a las cadenas. Dato para confirmar, el margen por cliente, que ya vimos.

Segundo, los fletes: 90 millones. Causa probable, más despachos y a mayor distancia. Dato para confirmar, el costo por despacho y por ruta.

Tercero, los gastos financieros: 55 millones, por el uso de la línea de crédito para financiar el capital de trabajo.

Fíjate que las tres desviaciones tienen un origen común: la forma en que la empresa creció. Y eso es lo que un buen análisis de desviaciones debería entregarte. No una tabla con 30 líneas en rojo y verde, sino dos o tres focos con nombre y apellido.

### m07_s08 · Tres consejos para este paso

Tres consejos para sacarle más partido a este paso. Uno: compara mes a mes, y también en acumulado. Una desviación pequeña que se repite todos los meses termina siendo una desviación grande a fin de año, y el acumulado la muestra antes.

Dos: pídele una proyección de cierre. Con lo que va del año, y con lo que queda del presupuesto, ¿cómo terminaríamos? Es la pregunta que más le interesa a un directorio a mitad de año.

Y tres: guarda las explicaciones. Las causas que identificaste este mes son el contexto del mes siguiente. Si las guardas como un archivo en tu proyecto, Claude las va a tener presentes en el próximo cierre.

### m07_s09 · Paso 6 en resumen

El paso 6, en resumen. Mide, prioriza y explica.

Separa el efecto volumen del efecto margen.

Y termina con dos o tres focos, no con 30 líneas en rojo y verde.


## Paso 7: señales de alerta

### m08_s01 · Cuestiona: busca señales de alerta

Paso 7: buscar señales de alerta. Hasta aquí le pedimos a Claude que calcule y que explique. Ahora le vamos a pedir que desconfíe. Es el paso donde más valor agrega tener una segunda mirada.

### m08_s02 · Prompt 8: el abogado del diablo

Mira cómo cambia el rol. Ya no es mi analista: es un analista de riesgo de crédito, escéptico, que tiene que decidir si le presta plata a la empresa. Ese cambio de rol modifica por completo la respuesta.

Le pido las señales de alerta con su evidencia en cifras, qué podrían significar y qué documento pediría.

Que evalúe la calidad de la utilidad, es decir, si está respaldada por caja.

Que identifique los supuestos optimistas de la administración, y que clasifique cada alerta.

Y termino con una instrucción que uso mucho: sé directo, no suavices las conclusiones. Los modelos tienden a ser diplomáticos. En este paso, no quieres diplomacia.

### m08_s03 · Las señales de alerta de Alerce

Estas son las alertas de Alerce. Dos críticas. La primera, utilidad sin caja: 485 millones de utilidad con un flujo operacional negativo de 685.

La segunda, dependencia de la deuda de corto plazo: 1.150 millones que vencen dentro de un año, con 65 millones en caja. Si el banco no renueva la línea, la empresa tiene un problema inmediato.

Luego, tres importantes. La cartera vencida: el 45 % de las cuentas por cobrar está vencido, contra 27 % el año anterior. Y lo vencido a más de 90 días pasó de 10 a 90 millones, sin ninguna provisión. Si parte de eso no se cobra, la utilidad del año está sobrestimada.

La concentración: dos clientes explican el 15 % de las ventas y el 21 % de las cuentas por cobrar.

Y los dividendos: se repartieron 200 millones en un año en que la operación consumió caja.

Por último, una alerta a observar: los inventarios crecieron 48 %, con ventas que crecieron 18. Claude pide el detalle por producto, para ver si hay mercadería de baja rotación.

### m08_s04 · Calidad de la utilidad: ¿se convierte en caja?

Para medir la calidad de la utilidad hay un indicador muy simple: el flujo de la operación dividido por la utilidad neta. Cuando es cercano o superior a 1, la utilidad se está convirtiendo en caja.

En 2024, Alerce tenía 1,09: por cada peso de utilidad, generaba 1,09 pesos de caja.

En 2025, el indicador es negativo: menos 1,41. Por cada peso de utilidad que mostró, salió 1,41 de la caja.

Es el indicador que cualquier banco va a mirar primero. Y es muy fácil de pedir todos los meses.

### m08_s05 · Para cada alerta, el documento que hay que pedir

Y esta es la otra mitad de la respuesta: qué documento pedir para cada alerta. Para la utilidad sin caja, el flujo de caja mensual y una proyección a 13 semanas.

Para la deuda de corto plazo, los contratos de crédito, con sus tasas, vencimientos y garantías.

Para la cartera vencida, el detalle por cliente y el registro de la gestión de cobranza.

Para la concentración, los contratos y las condiciones comerciales con las cadenas.

Para los dividendos, la política de dividendos y las actas del directorio.

Y para los inventarios, la rotación por producto y las mermas.

Con esto, las alertas dejan de ser una lista de preocupaciones, y se convierten en una lista de tareas.

### m08_s06 · Los supuestos optimistas que detecta

También le pedimos los supuestos optimistas de la administración. Identifica tres. Que las cadenas van a pagar a tiempo, cuando hoy pagan a 100 días y la cartera vencida crece.

Que el banco va a renovar la línea de crédito, cuando 1.150 millones dependen de esa decisión.

Y que el inventario adicional se va a vender sin descuentos. Ninguno de estos supuestos está escrito en los estados financieros, pero los tres están implícitos en ellos. Ponerlos por escrito es la mitad del trabajo de gestionar el riesgo.

### m08_s07 · Señales que conviene revisar siempre

Más allá de este caso, hay señales que conviene revisar siempre. Cuentas por cobrar que crecen más rápido que las ventas.

↳ Inventarios que crecen más rápido que el costo de ventas.

↳ Utilidad que sube mientras el flujo operacional baja.

↳ Gastos que se activan en vez de ir a resultado.

↳ Deuda de corto plazo financiando activos de largo plazo.

↳ Cambios de criterio contable de un año a otro.

↳ Y operaciones relevantes con empresas relacionadas.

Puedes dejar esta lista en las instrucciones de tu proyecto, para que Claude la revise todos los meses.

### m08_s08 · Cómo usar este paso

Dos comentarios sobre este paso. El primero: una señal de alerta no es una acusación. Es una pregunta que hay que responder con un documento. Por eso le pedí que indicara qué documento solicitaría.

El segundo: este prompt es muy útil para prepararte. Si vas a presentar al directorio, o vas a pedir un crédito, úsalo antes. Vas a llegar a la reunión sabiendo cuáles son las preguntas difíciles, y con las respuestas preparadas.

Es como tener un ensayo general con el director más exigente de la mesa.

### m08_s09 · Paso 7 en resumen

El paso 7, en resumen. Cambia el rol: de analista a escéptico.

Pide cada alerta con su evidencia y con el documento que la respalda.

Y úsalo antes de ir al directorio o al banco.


## Paso 8: escenarios, plan de acción e informe

### m09_s01 · Decide: del diagnóstico a la acción

Paso 8: decidir. Todo lo anterior no sirve de nada si se queda en un diagnóstico. El último paso es convertir el análisis en decisiones: qué vamos a hacer, quién, cuándo, y cuánto vale.

### m09_s02 · Prompt 9: ponerles número a las alternativas

Para decidir hay que ponerles número a las alternativas.

En este prompt le entrego cuatro palancas, que salen directamente del diagnóstico. Reducir los días de cobro a 60. Reducir los días de inventario a 60. Recuperar un punto de margen bruto. Y volver los fletes al porcentaje de ventas del año anterior.

Para cada una le pido el impacto en caja, el impacto en el resultado anual y los supuestos.

Y después, tres escenarios, para ver cómo quedarían la deuda de corto plazo y el ciclo de conversión de efectivo.

Las metas las pongo yo, porque yo conozco el negocio y sé qué es alcanzable. Claude hace la aritmética y me muestra las consecuencias.

### m09_s03 · Cuánto vale cada palanca

Estos son los resultados. Bajar los días de cobro de 72 a 60 libera 390 millones de caja.

Bajar los días de inventario de 68 a 60 libera otros 195.

En conjunto, 585 millones. Con eso la deuda de corto plazo baja de 1.150 a 565 millones, a menos de la mitad, y la empresa se ahorra unos 63 millones al año en intereses. El ciclo de conversión de efectivo baja de 92 a 72 días.

Las otras dos palancas van al resultado. Recuperar un punto de margen son 118 millones al año. Y volver los fletes a 4,3 % de las ventas, 53 millones más.

Fíjate en el orden de magnitud. Las palancas de caja valen 585 millones; las de resultado, 171. La prioridad de esta empresa no es vender más: es cobrar mejor y comprar mejor.

### m09_s04 · Tres escenarios, para mostrar un rango

Y estos son los tres escenarios. En el conservador, la cobranza baja solo a 66 días y el inventario a 64: se liberan cerca de 300 millones, y la deuda de corto plazo queda cerca de 860.

En el escenario base, con 60 y 60 días, se liberan 585 millones y la deuda baja a 565.

Y en el optimista, con 55 días en ambos, se liberan casi 870 millones, y la deuda de corto plazo baja a menos de 300.

Lo valioso de este ejercicio no es acertar el número. Es mostrarle al directorio el rango: qué pasa si nos va regular, qué pasa si cumplimos, y qué pasa si nos va muy bien. Y dejar por escrito los supuestos de cada escenario, para revisarlos en tres meses.

### m09_s05 · Una herramienta más: la proyección de caja a 13 semanas

Con el diagnóstico hecho, hay una herramienta más que te recomiendo pedirle: una proyección de caja a 13 semanas. Le entregas el saldo inicial, las cobranzas esperadas y los pagos comprometidos: proveedores, sueldos, impuestos y cuotas de créditos.

Esta es la proyección de Alerce si nada cambia. La caja parte en 65 millones. En la semana 4, después de pagar los sueldos, queda en 3 millones.

Y en la semana 8 es negativa. Sin más crédito, la empresa no llega a fin de trimestre.

Con el plan de cobranza e inventario en marcha, la caja se mantiene positiva, y termina el trimestre en torno a 160 millones.

Sigue siendo una holgura pequeña para una empresa que paga más de 80 millones en sueldos cada mes. Por eso esta planilla no es un lujo: es la herramienta de gestión más importante del próximo trimestre.

### m09_s06 · El plan de acción: cuatro columnas, siempre

Con los números claros, le pido el plan de acción en una tabla: acción, responsable, plazo e indicador.

Cobranza a 60 días, a cargo de la gerencia comercial, medida en días de cobro. Inventario a 60 días, a cargo de operaciones.

Revisar las condiciones con las cadenas, medido en margen por cliente. La proyección de caja semanal. Y renegociar la línea de crédito a un plazo mayor.

Un plan sin responsable y sin fecha es una lista de buenos deseos. Y un plan sin indicador no se puede seguir. Pídele siempre las cuatro columnas, y después ajusta tú los nombres y las fechas, porque eso Claude no lo puede saber.

### m09_s07 · Prompt 10: el informe para el directorio

Y el último prompt: el informe. Una página, para el directorio.

Con cinco partes. El mensaje principal, en una frase. Qué pasó, con tres cifras. Por qué pasó, con las tres causas y su evidencia. Qué proponemos, con responsable, plazo e indicador de seguimiento. Y qué necesitamos del directorio: las decisiones concretas que le estamos pidiendo.

Y dos reglas de cierre. Que cada cifra salga de los análisis anteriores, para que no aparezcan números nuevos. Y que distinga los datos de las estimaciones.

### m09_s08 · El informe: una página, cinco partes

Este es el informe. El mensaje principal: Alerce creció 18 % en ventas, pero financió ese crecimiento con su caja y con deuda de corto plazo; la prioridad es recuperar liquidez.

Después, las tres cifras y las tres causas.

Las acciones, cada una con responsable y plazo.

Y al final, lo que se le pide al directorio: suspender los dividendos hasta normalizar la caja, y aprobar la renegociación de la línea de crédito a un plazo mayor.

Ahora, una advertencia importante. Este borrador sale muy rápido, pero no lo envíes tal cual. Los responsables, los plazos y las decisiones que pides son tuyos. Claude te da la estructura y la redacción; el contenido lo validas tú.

Ese es el equilibrio correcto: la máquina hace el trabajo pesado, y la persona pone el criterio.

### m09_s09 · Tres consejos para presentar el informe

Tres consejos para presentar el informe. Uno: empieza por el mensaje, no por la tabla. Si la primera frase no se sostiene sola, el informe todavía no está listo.

Dos: lleva respondidas las tres preguntas más difíciles. Ya sabes cómo encontrarlas: con el prompt del paso 7.

Y tres: termina con lo que pides. Una decisión, un responsable y una fecha.

Un informe que no pide nada no cambia nada.

### m09_s10 · Paso 8 en resumen

El paso 8, en resumen. Ponle número a cada palanca.

Muestra un rango, con tres escenarios.

Y termina con un plan de cuatro columnas, y un informe de una página que validas tú.


## Tu rutina mensual y los errores más comunes

### m10_s01 · Tu rutina mensual con Claude

¿Cómo llevas todo esto a tu día a día? Te propongo una rutina mensual de cierre con Claude, en cinco momentos.

### m10_s02 · El cierre mensual, en cinco momentos

Uno: exportas los estados financieros del mes desde tu sistema contable.

Dos: los subes a tu proyecto, donde ya están el contexto y las instrucciones.

Tres: corres la verificación.

Cuatro: corres los análisis, siempre los mismos y en el mismo orden, para que los meses sean comparables.

Y cinco: revisas, corriges y recién entonces compartes.

Las primeras veces te va a tomar más tiempo. Cuando tengas el proyecto armado y los prompts afinados, vas a dedicar ese tiempo a pensar, y no a armar tablas.

### m10_s03 · Qué dejar en las instrucciones de tu proyecto

Para que la rutina funcione, deja bien armadas las instrucciones de tu proyecto. Yo incluyo seis cosas. El contexto de la empresa.

↳ La unidad y el formato de las cifras.

↳ Las definiciones de los indicadores, tal como los calcula la empresa.

↳ Las reglas de trabajo: calcular con código, mostrar la fórmula, y separar los hechos de las hipótesis.

↳ La lista de señales de alerta que vimos en el paso 7.

↳ Y el formato del informe mensual. Lo armas una vez, lo vas afinando, y cada mes partes desde ahí.

### m10_s04 · Cinco repreguntas que siempre sirven

Si tuviera que resumir el curso en una habilidad, sería repreguntar. Estas son las cinco repreguntas que uso todos los días. La primera: ¿por qué? Pide hipótesis, y el dato para confirmarlas.

La segunda: ¿cómo lo calculaste? Que muestre la fórmula y las cifras.

La tercera: ¿qué supusiste? Los supuestos escondidos son el mayor riesgo de cualquier análisis.

La cuarta: ¿qué dato te falta? Te dice exactamente qué ir a buscar.

Y la quinta: ¿qué pasaría si cambia un supuesto? Sirve para abrir escenarios y medir sensibilidad. Con estas cinco preguntas, una respuesta regular se convierte en un buen análisis.

### m10_s05 · Los 6 errores más comunes

Y antes de cerrar, los seis errores que más veo. Uno: pedir sin contexto. Si solo escribes «analiza este balance», recibes un análisis genérico, que podría ser de cualquier empresa.

Dos: creerle sin verificar. Revisa siempre las cifras de origen y algunas fórmulas.

Tres: pedir todo en un solo prompt. Es mucho mejor avanzar por pasos, y revisar cada respuesta antes de seguir.

Cuatro: quedarse con la primera respuesta. Repregunta: por qué, cómo lo calculaste, qué pasaría si.

Cinco: subir información sin revisar antes la confidencialidad.

Y seis, el más importante: delegar la conclusión. El análisis puede hacerlo Claude. La conclusión, y la decisión, tienen que ser tuyas.

### m10_s06 · Antes de enviar: dos minutos de control de calidad

Y antes de enviar cualquier informe, un control de calidad de dos minutos. Uno: las cifras clave coinciden con tu sistema contable.

Dos: revisaste al menos tres cálculos por tu cuenta.

Tres: cada conclusión tiene su evidencia al lado.

Y cuatro: las decisiones, los responsables y los plazos son tuyos, no del borrador.

Si esos cuatro puntos se cumplen, puedes firmar el informe con tranquilidad. Porque al final, quien firma eres tú.

### m10_s07 · El mismo método, para mucho más que el cierre

El mismo método te sirve para mucho más que el cierre mensual. Para evaluar a un cliente antes de darle crédito: le pides sus estados financieros, y usas el prompt del analista de riesgo.

Para preparar una reunión con el banco, y llegar con los ratios calculados y las respuestas listas.

Para una primera mirada a una empresa que quieres comprar, antes de la revisión formal.

Y para explicarle los números a tu equipo, en lenguaje simple y con foco en lo que cada área puede mover.

### m10_s08 · Preguntas que me hacen siempre (1 de 2)

Antes de terminar, cuatro preguntas que me hacen siempre. ¿Puedo confiar en los cálculos? Cuando Claude calcula con código, la aritmética es confiable. El riesgo está en otra parte: en los datos que leyó, y en la fórmula que eligió. Por eso verificamos la lectura, y por eso pedimos siempre la fórmula.

¿Y si me responde distinto cada vez? La redacción cambia, y es normal. Los números no deberían cambiar, si los datos y las instrucciones son los mismos. Si cambian, pídele que muestre el cálculo: casi siempre hay una definición distinta detrás.

### m10_s09 · Preguntas que me hacen siempre (2 de 2)

¿Sirve para una empresa pequeña? Sí. Mientras tengas dos balances y un estado de resultados, el método funciona igual. De hecho, es donde más ayuda, porque normalmente no hay un equipo de análisis.

¿Y reemplaza a mi contador o a mi analista? No. Les quita el trabajo mecánico, y les deja más tiempo para lo que realmente vale: entender el negocio y recomendar qué hacer.

### m10_s10 · Tu plan para los próximos 30 días

Y para que no se quede en buenas intenciones, un plan para tus próximos 30 días. Semana 1: arma tu proyecto, con el contexto, las reglas y los archivos de tu último cierre.

Semana 2: corre la verificación, y el análisis vertical y horizontal.

Semana 3: los ratios, el ciclo de efectivo y el puente de caja.

Y semana 4: las desviaciones, las señales de alerta y tu primer informe de una página.

En un mes vas a tener el método completo funcionando con tus propios números.


## Resumen y próximos pasos

### m11_s01 · Las cuatro lecciones del caso Alerce

Antes de despedirnos, las cuatro lecciones que deja el caso de Alerce. Uno: la utilidad no es caja. La empresa ganó 485 millones, y su operación consumió 685.

Dos: un buen ratio puede esconder un problema. Una liquidez corriente de 1,65, con una razón de caja de 0,03.

Tres: crecer cuesta caja. Son 75 pesos por cada 100 de venta nueva.

Y cuatro: las desviaciones suelen tener un mismo origen. Aquí, la forma en que la empresa creció. Si te llevas estas cuatro ideas, ya vas a leer tus propios estados financieros con otros ojos.

### m11_s02 · El método completo: 8 pasos

Repasemos. Ocho pasos. Preparar la información y el contexto.

↳ Verificar que todo cuadre.

↳ Entender la estructura y la tendencia.

↳ Medir con ratios.

↳ Seguir la caja.

↳ Comparar contra el presupuesto.

↳ Buscar señales de alerta.

↳ Y decidir.

En el caso de Alerce, ese recorrido nos llevó de una empresa que parecía estar en su mejor año, a un diagnóstico claro: creció financiando a sus clientes con su propia caja. Y a un plan concreto para corregirlo.

Ninguno de esos pasos requiere ser experto en inteligencia artificial. Requiere saber qué preguntar.

### m11_s03 · Tres formas de seguir

Si quieres aplicar este método en tu empresa, tienes tres formas de seguir. La primera: descarga el ebook, el archivo del caso y los prompts. El enlace está en la descripción. Repite cada prompt con los mismos números, y después hazlo con los tuyos.

La segunda: si prefieres hacerlo acompañado, en Hyperion Advisory ofrecemos un diagnóstico financiero sin costo, de 30 minutos. Revisamos juntos cómo están hoy tus reportes, y te llevas dos o tres mejoras concretas. También encuentras el enlace en la descripción.

Y la tercera: suscríbete al canal, y cuéntame en los comentarios qué paso te gustaría ver con más profundidad.

Gracias por acompañarme durante esta hora. Nos vemos en el próximo video.
