# Tu Cartera · suscripción, límites de tokens y cobro

Propuesta de modelo de negocio con los números que la sostienen. Los planes, cuotas y precios viven en
`tucartera/planes.py`: cambiarlos ahí actualiza la app, la página **Plan y uso** y los límites.

## 1. Modelo de IA fijo

Tu Cartera usa siempre **`claude-opus-5-5`**, sin desvío automático a otros modelos. Así el costo por token
no cambia según la solicitud y los márgenes se pueden calcular de antemano. Si el modelo rechaza una
solicitud, la app responde con sus plantillas y reglas.

Precio de lista de Anthropic (US$ por millón de tokens, noviembre de 2026):

| Entrada | Salida | Lectura de caché | Escritura de caché |
|---|---|---|---|
| 4,00 | 20,00 | 0,20 | 5,00 |

## 2. Cuánto consume cada acción (estimado)

| Acción | Tokens aprox. | Costo aprox. |
|---|---|---|
| Pedido en la barra ⌘K (busca, lee la cartera y deja propuestas) | 16.000 | US$ 0,067 (≈ $64) |
| Correo pulido por una automatización | 1.400 | US$ 0,014 (≈ $13) |

Con una mezcla normal de pedidos y correos, el costo ronda **US$ 6 por millón de tokens**. Para fijar
precios se usa un supuesto conservador de **US$ 10 por millón** (≈ $9.500 con dólar a $950).

Estas cifras son estimaciones de diseño. Con clientes reales, `python -m tucartera uso` muestra el costo
efectivo por millón de tokens de tu cartera; úsalo para ajustar el primer mes.

Cada llamada tiene topes propios: máximo 6.000 tokens de respuesta y 6 vueltas de herramientas por pedido,
y 3.000 tokens por correo. Ninguna acción puede disparar un gasto grande por sí sola.

## 3. Planes y cuota por usuario

Se cobra **por usuario activo al mes, IVA aparte**. Cada usuario trae su propia cuota mensual de tokens.

| Plan | Precio usuario/mes | Usuarios | Clientes | Tokens por usuario/mes | Equivale a |
|---|---|---|---|---|---|
| Esencial | $19.990 | hasta 2 | hasta 40 | 300.000 | ≈ 18 pedidos ⌘K al mes |
| Profesional | $34.990 | hasta 10 | hasta 200 | 1.000.000 | ≈ 62 pedidos (3 por día hábil) |
| Firma | $54.990 | ilimitados | ilimitados | 2.000.000 | ≈ 125 pedidos (6 por día hábil) |

- **Prueba gratis:** 14 días del plan Profesional, con 200.000 tokens por usuario.
- **Paquete adicional:** 1.000.000 tokens por $14.990 + IVA. Va a una bolsa común del estudio y vence a fin de mes.
- **Cómo se descuenta:** los pedidos en ⌘K usan la cuota de quien pide; los correos pulidos por las
  automatizaciones, la del responsable del cliente. Cuando una persona agota su cuota, usa la bolsa de
  paquetes si existe; si no, sigue trabajando con plantillas y reglas (nunca queda bloqueada).
- **Qué cuenta como token:** todo lo que la IA lee y escribe (entrada, caché y salida). Es simple de explicar
  y el costo real se mide aparte.

## 4. Margen de la IA por plan

Costo de IA de un usuario que gasta el **100 %** de su cuota, frente a lo que paga:

| Plan | Costo IA (US$ 6/M – 10/M) | Precio | Peso del costo IA |
|---|---|---|---|
| Esencial | $1.710 – $2.850 | $19.990 | 9 % – 14 % |
| Profesional | $5.700 – $9.500 | $34.990 | 16 % – 27 % |
| Firma | $11.400 – $19.000 | $54.990 | 21 % – 35 % |
| Paquete 1 M | $5.700 – $9.500 | $14.990 | 38 % – 63 % |

La mayoría de los usuarios no gasta toda su cuota, así que el peso real suele ser menor. A esto se suman el
servidor, el correo saliente y la comisión de la pasarela de pago (normalmente entre 3 % y 4 %).

**Regla para revisar cada mes:** si el costo efectivo pasa de US$ 10 por millón, o si el costo de IA de un plan
supera el 35 % de su precio, sube el precio del plan o baja su cuota en `tucartera/planes.py`.

## 5. Cómo cobrarlo

1. **Cobro recurrente con tarjeta.** En Chile puedes usar Flow (suscripciones), Mercado Pago (suscripciones)
   o Transbank Oneclick. El estudio inscribe su tarjeta una vez y se le cobra cada mes:
   usuarios activos × precio del plan + IVA.
2. **Activación en Tu Cartera.** Hoy, cuando confirmas el pago, activas o renuevas el plan con
   `python -m tucartera plan correo@estudio.cl profesional --meses 1`. El siguiente paso es conectar el aviso
   automático (webhook) de la pasarela elegida para que esto ocurra solo.
3. **Paquetes.** Cobro único; se activan con `python -m tucartera paquete correo@estudio.cl`. Desde la app, el
   botón «Comprar paquete» escribe a `TUCARTERA_VENTAS_EMAIL`.
4. **Documento tributario.** Emite boleta o factura electrónica por cada cobro. La mayoría de los estudios
   pedirá factura para recuperar el IVA.
5. **Plan anual (opcional).** 12 meses al precio de 10, pagados por adelantado: mejora la caja y reduce las
   bajas.
6. **Si no pagan.** Hay 3 días de gracia tras el vencimiento. Después se pausan la IA y las automatizaciones,
   pero el estudio sigue viendo y exportando sus datos. Al pagar, todo vuelve a funcionar.

## 6. Qué ve el cliente

- En **Plan y uso**: su plan, días de prueba, total mensual, el consumo de cada persona con barra de progreso
  y los pedidos restantes aproximados.
- En la barra lateral: aviso de prueba o de suscripción inactiva.
- En ⌘K: cuánto lleva de su cuota, o «modo básico» si se agotó.
