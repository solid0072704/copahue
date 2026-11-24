# Base de conocimiento del agente: plantilla

El agente Breeze responde **solo** con lo que encuentre en sus fuentes. Crea en HubSpot
(**Base de conocimientos** o **Contenido del agente**) un artículo por proyecto con esta estructura
y uno de preguntas frecuentes generales. También puedes agregar como fuente las páginas de cada
proyecto en el sitio web.

Reglas para que funcione bien:

- Una sola versión de cada dato. Si un precio cambia, actualízalo el mismo día (y anota la fecha).
- Frases cortas y concretas. Evita "consulta con tu ejecutivo" dentro del artículo: el agente ya sabe derivar.
- Si un dato no está definido, no lo escribas. Es mejor que el agente diga "lo confirma un ejecutivo" a que repita algo viejo.
- Reemplaza todo lo que está entre `[CORCHETES]`.

---

## Plantilla por proyecto

```markdown
# [Nombre del proyecto] — [comuna], [región]

Última actualización: [AAAA-MM-DD]
Zona: [Centro | Sur]
Estado de venta: [En venta | Preventa | Últimos lotes | Agotado]

## Ubicación y acceso
- Dirección o referencia: [ ... ]
- Distancia: [X] km de [ciudad] ([X] minutos en auto).
- Acceso: [camino pavimentado / ripio], [transporte público sí/no].
- Ubicación en Google Maps: [link]

## Parcelas
- Superficie: desde [X] m² hasta [Y] m².
- Precio: desde $[X] (valor al [fecha]). [Si no quieres que el agente informe precios, escribe: "Los precios los informa el ejecutivo".]
- Formas de pago: [contado / pie + cuotas con la empresa (N cuotas) / crédito hipotecario con bancos].
- Reserva: [monto o "la gestiona el ejecutivo"].

## Servicios y factibilidades
- Luz: [factibilidad / empalme / postación].
- Agua: [derecho de agua / pozo / APR / factibilidad].
- Rol: [rol propio / en trámite].
- Otros: [portón de acceso, caminos interiores, áreas comunes, vista, río, bosque nativo...].

## Visitas
- Días y horarios: [ ... ].
- Punto de encuentro: [ ... ].
- Se agenda con: el ejecutivo del proyecto (el agente toma el interés y lo deriva).

## Material
- Ficha / brochure: [link]
- Video o tour: [link]
- Página web: [link]
```

---

## Preguntas frecuentes generales

```markdown
# Preguntas frecuentes — Tu Proyecto de Vida

Última actualización: [AAAA-MM-DD]

## ¿Quiénes son?
[2-3 frases: años en el mercado, proyectos, regiones.]

## ¿Cómo es el proceso de compra?
1. [Reserva] 2. [Promesa / validación] 3. [Escritura de compraventa en notaría] 4. [Inscripción en el CBR] 5. [Entrega].

## ¿Puedo comprar con crédito hipotecario?
[Sí/No, con qué bancos, requisitos generales.]

## ¿Ofrecen cuotas sin banco?
[Condiciones generales: pie mínimo, número de cuotas, reajuste. Si varía por proyecto, dilo.]

## ¿Qué gastos adicionales hay?
[Notaría, CBR, contribuciones, etc.]

## ¿Se puede construir? ¿Qué permisos se necesitan?
[Respuesta general sobre subdivisión predial, permisos de edificación y DOM.]

## Horario de atención
Lunes a viernes de 9:00 a 19:00 y sábados de 10:00 a 14:00. [Debe coincidir con ruteo.yaml.]

## Ya soy cliente, ¿con quién hablo?
Postventa: [correo / teléfono / el agente lo deriva a Postventa].

## Proveedores y trabajo
Proveedores: [correo]. Postulaciones: [correo o link].
```
