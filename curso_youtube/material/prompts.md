# Los prompts del curso · Análisis de estados financieros con Claude

Copia cada prompt, reemplaza lo que está entre [corchetes] y úsalo en orden. Funcionan mejor dentro de un Proyecto de Claude que ya tenga el contexto de tu empresa.

## Prompt 1 · Contexto y reglas de trabajo

```
Actúa como un analista financiero sénior. Voy a pedirte que
analices los estados financieros de una empresa.

CONTEXTO
- Empresa: [nombre o «Empresa A»], [industria], [país].
- Tamaño: ventas anuales de [monto]. Cifras en [unidad].
- Períodos: [2024 y 2025], más el presupuesto [2025].
- Hechos relevantes del año: [qué cambió].
- Lo que más me preocupa: [pregunta de negocio].

REGLAS
1. Usa solo los datos que te entrego. Si falta algo, pregunta.
2. Calcula con código y muestra la fórmula.
3. Separa los hechos de tus hipótesis.
4. No concluyas todavía: primero confirma qué entendiste.
```

## Prompt 2 · Lectura y cuadraturas

```
Antes de analizar, verifica la información que te entregué.

1. Muéstrame en una tabla las cifras que leíste de cada
   estado, por período y con su unidad.
2. Comprueba estas cuadraturas y dime si se cumplen:
   - Activos = Pasivos + Patrimonio.
   - Utilidad del año = variación de utilidades acumuladas
     + dividendos.
   - Efectivo inicial + flujo neto = efectivo final.
   - Cada subtotal = suma de sus partidas.
3. Lista los datos que faltan o que te parecen
   inconsistentes.
4. No sigas hasta que yo confirme.
```

## Prompt 3 · Resumen ejecutivo inicial

```
Con la información verificada, dame una primera lectura
ejecutiva en un máximo de 10 líneas:

- Las 3 cifras que mejor resumen el año.
- Lo que mejoró y lo que empeoró frente al año anterior.
- La pregunta más importante que debería hacerme como
  gerente.

Escribe para un directorio que no es experto en
contabilidad. Sin jerga y sin repetir la tabla.
```

## Prompt 4 · Estructura y tendencia

```
Haz un análisis vertical y horizontal de los dos períodos.

1. Estado de resultados: cada línea como % de las ventas,
   y su variación en monto y en %.
2. Balance: cada cuenta como % del total de activos,
   y su variación en monto y en %.
3. Marca las partidas que crecieron MÁS que las ventas
   y explica qué implica cada una.
4. Termina con los 3 hallazgos más relevantes, ordenados
   por impacto en pesos.

Entrega tablas y calcula con código.
```

## Prompt 5 · Ratios financieros

```
Calcula los ratios financieros de ambos períodos, en 4
grupos: liquidez, endeudamiento, rentabilidad y eficiencia.

Para cada ratio entrega:
- La fórmula y las cifras que reemplazaste.
- El valor de cada período y su variación.
- Una interpretación en una línea, en lenguaje simple.
- Un semáforo: verde, amarillo o rojo, con el criterio.

Incluye el ciclo de conversión de efectivo y el análisis
DuPont.
Al final, dime cuáles son los 3 ratios que más me deberían
preocupar y por qué.
```

## Prompt 6 · Flujo de efectivo y capital de trabajo

```
Explícame qué pasó con la caja.

1. Construye el flujo de efectivo del año por el método
   indirecto, partiendo de la utilidad neta. Si te entregué
   el flujo, concílialo con el balance.
2. Muestra un puente: utilidad neta → flujo de la
   operación → variación del efectivo.
3. Cuantifica cuánta caja absorbió cada componente del
   capital de trabajo.
4. Calcula el flujo de caja libre.
5. Responde en 5 líneas: ¿por qué la utilidad no se
   transformó en caja?
```

## Prompt 7 · Análisis de desviaciones

```
Compara el resultado real con el presupuesto del período.

1. Tabla: presupuesto, real, desviación en monto y en %.
   Indica si cada una es favorable o desfavorable.
2. Considera material toda desviación mayor a [5 %] y a
   [20 millones]. Analiza solo esas.
3. Descompón la desviación del margen bruto en efecto
   volumen y efecto margen.
4. Para cada desviación material, propón la causa más
   probable y el dato que necesitas para confirmarla.
5. Ordena de mayor a menor impacto en la utilidad.
```

## Prompt 8 · Señales de alerta

```
Actúa como un analista de riesgo de crédito escéptico, que
debe decidir si le presta dinero a esta empresa.

1. Lista las señales de alerta que ves en estos estados
   financieros. Para cada una: la evidencia en cifras, qué
   podría significar y qué documento pedirías.
2. Evalúa la calidad de la utilidad: ¿está respaldada por
   caja?
3. Dime qué supuestos optimistas podría estar haciendo la
   administración.
4. Clasifica cada alerta: crítica, importante o a observar.

Sé directo. No suavices las conclusiones.
```

## Prompt 9 · Palancas y escenarios

```
Con todo lo analizado, cuantifica el impacto de estas
palancas:

a) Reducir los días de cobro a [60].
b) Reducir los días de inventario a [60].
c) Recuperar [1] punto de margen bruto.
d) Volver el gasto en fletes al % de ventas del año
   anterior.

Para cada una: impacto en caja, impacto en el resultado
anual y supuestos.
Después arma tres escenarios (conservador, base y
optimista) y muéstrame cómo quedarían la deuda de corto
plazo y el ciclo de conversión de efectivo.
```

## Prompt 10 · Informe ejecutivo de una página

```
Redacta un informe ejecutivo de una página para el
directorio.

Estructura:
1. Mensaje principal, en una frase.
2. Qué pasó: 3 cifras clave.
3. Por qué pasó: las 3 causas, con su evidencia.
4. Qué proponemos: acciones con responsable, plazo e
   indicador de seguimiento.
5. Qué necesitamos del directorio: decisiones concretas.

Tono directo y sin jerga. Cada cifra debe salir de los
análisis anteriores. Si un dato es una estimación, dilo.
```

## Repreguntas usadas en el caso

- El margen bruto bajó 1 punto. Dame las 3 hipótesis más probables y, para cada una, el dato exacto que necesitas para confirmarla o descartarla.
- Cruza ventas, margen y cuentas por cobrar por tipo de cliente. ¿Dónde se concentran el mayor plazo de cobro y el menor margen?
- Analiza la antigüedad de las cuentas por cobrar. ¿Qué parte del aumento es crecimiento y qué parte es atraso?

## Cinco repreguntas que siempre sirven

- ¿Por qué? Dame hipótesis y el dato para confirmarlas.
- ¿Cómo lo calculaste? Muéstrame la fórmula y las cifras.
- ¿Qué supusiste?
- ¿Qué dato te falta?
- ¿Qué pasaría si…?
