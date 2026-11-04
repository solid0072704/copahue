# Conciliación bancaria predictiva (Google Apps Script)

Aplicativo para Google Sheets que **clasifica automáticamente los movimientos de las cartolas bancarias** usando la historia de la hoja `FLUJO`. Lo que no logra identificar con seguridad lo deja marcado para que el analista lo clasifique a mano.

## Cómo funciona

```
 Cartolas (.csv/.xls/.xlsx)          Historia (hoja FLUJO)
            │                                │
            ▼                                ▼
   Lectura por cuenta ─────────► Motor predictivo (vecinos más cercanos)
   (Banco de Chile consolidado,     • texto del detalle (palabras ponderadas, números enmascarados)
    Santander, genérico)            • misma empresa / cuenta
            │                       • cargo o abono, monto parecido, antigüedad
            │                       • patrón de la empresa para contrapartes nuevas
            │                       • se vota la cuenta contable; el centro de costo
            │                         sigue la forma de asignarlo de los últimos meses
            │                       + REGLAS manuales (prioridad)
            │                       + cruce con PAGOS POR HACER (monto exacto + nombre)
            │                       + traspasos entre cuentas propias ("Traspaso De
            │                         Cuenta:008860515009" → 1101221 Banco de Chile
            │                         Los Coihues 886-05150-09, según el plan de cuentas)
            ▼
   Hoja CONCILIACION
     🟩 IDENTIFICADO  (confianza ≥ 80 %, queda aprobado)
     🟨 POR REVISAR   (55–80 %, sugerencia a confirmar)
     🟥 PENDIENTE     (< 55 %, clasificar a mano)
     ⬜ YA EN FLUJO   (ya registrado, no se duplica)
            │  el analista completa / corrige
            ▼
   "Traspasar aprobados a FLUJO"  ──►  la historia crece y el modelo aprende
```

Para cada movimiento propone: `DETALLE 1`, `C.COSTO`, `GLOSA`, `CUENTA CONTABLE`, `NOMBRE CUENTA`, `CENTRO COSTO`, `RUT`, `RAZON SOCIAL`, y explica de dónde sale la sugerencia (columna `EXPLICACIÓN`, p.ej. *"5 de 6 movimientos parecidos usan esta cuenta; similitud del mejor: 97 %. Ej.: 2026-06-10 'Traspaso A:constructora Jahuel Spa' → Proyecto Ingenieria La Vinilla"*).

### Precisión medida con su historia

Probado con `BANCO_2025.xlsm` en abr–sep 2026. Cada mes se clasificó entrenando **solo con los meses anteriores**, como si fuera una cartola nueva:

| Estado | % de los movimientos | Cuenta contable correcta |
|---|---|---|
| IDENTIFICADO | ~53 % | **~94 %** (cuenta + centro de costo: ~83 %) |
| POR REVISAR | ~24 % | ~65 % (el analista confirma o corrige) |
| PENDIENTE | ~23 % | (clasificación manual) |

Con las cartolas reales del Banco de Chile, simulando el aplicativo completo:

| Cartola | Movimientos | Identificados | Por revisar | Pendientes |
|---|---|---|---|---|
| `Consolidado_cuentas_01-11-2026-04-11-2026.csv` | 51 | **34** | 7 | 10 |
| Consolidado de julio 2026 (como si fuera nueva) | 185 | **116** | 44 | 25 |

Los pendientes son sobre todo contrapartes que nunca habían aparecido (personas nuevas que transfieren, proveedores nuevos) y movimientos genéricos como `Dep.cheq.otros Bancos`, donde el proyecto o la parcela no se deducen del texto. A medida que se traspasan meses a `FLUJO`, y si se agregan **REGLAS** para los casos repetitivos, el porcentaje identificado sube. Puede volver a medirse cuando quiera desde el menú *Evaluar precisión del modelo*.

## Facturas del SII y pagos a proveedores

Junto con las cartolas puede subir el **Registro de Compras del SII** de cada empresa
(`RCV_COMPRA_REGISTRO_<RUT>_<AAAAMM>.csv`, tal como se descarga del SII). El aplicativo:

1. Lee las facturas, notas de crédito y otros documentos, y los acumula en la hoja **FACTURAS COMPRA** sin duplicarlos.
2. Identifica la empresa por el RUT del nombre del archivo. Cada vez que se carga una cartola, el RUT del titular queda asociado a sus cuentas en `CUENTAS BANCARIAS`. Si el RUT todavía no tiene cuentas, la pantalla pregunta desde qué empresa se pagan.
3. Busca, para cada documento, la transferencia que lo paga entre los cargos de `CONCILIACION` y de `FLUJO`:
   - **monto exacto**, una factura con una transferencia;
   - **varias facturas en un solo pago** del mismo proveedor, descontando las notas de crédito;
   - **una factura en varias transferencias** (cuotas o abonos);
   - **nombre del beneficiario**: el banco lo trunca, por ejemplo "Traspaso A:sociedad De Servicios Agropec";
   - **RUT** del proveedor, si aparece en el detalle;
   - **alias aprendidos de la historia**: "Traspaso A:cerda Sofia" paga a DISEÑO GRÁFICO FOLLOWUS, "Pago:eds Frutillar" a ESMAX RED, "Pac Movistar" a Telefónica;
   - **fecha**: de 30 días antes a 180 días después de la factura, prefiriendo el pago más cercano;
   - si no lo encuentra en la empresa, lo busca en **las demás empresas del grupo** y lo deja POR REVISAR.
4. Muestra en FACTURAS COMPRA, para cada factura, la transferencia propuesta: fecha, cuenta, detalle, monto, ubicación (`CONCILIACION fila N` o `FLUJO fila N`) y los otros documentos que se pagaron junto con ella.

| Estado en FACTURAS COMPRA | Significado |
|---|---|
| 🟩 CONCILIADA | Pago encontrado con alta confianza (≥ 85 %). Se aplica solo al movimiento. |
| 🟦 CONFIRMADA | El analista marcó **CONFIRMAR**. |
| ⬜ YA REGISTRADA | El pago ya está en FLUJO y su glosa menciona el folio. |
| 🟨 POR REVISAR | Hay una propuesta: revísela y marque CONFIRMAR. |
| 🟥 SIN PAGO | No se encontró la transferencia (factura por pagar). Si hay transferencias al mismo proveedor por otros montos, se indican. |

Al aplicarse una factura, el movimiento de `CONCILIACION` queda IDENTIFICADO con:
- GLOSA **"Pago FC.69 COMPRAVENTA FRESIA SPA"** (con varias facturas: "Pago FC.304, FC.305 …");
- RUT y RAZÓN SOCIAL del proveedor;
- cuenta 2102001 Proveedores Nacionales (parámetro `cuentaProveedores` en CONFIG).

Si la fila la había editado el analista (MANUAL), solo se completan los campos vacíos. En los pagos que ya están en FLUJO se completan el RUT y la razón social si estaban vacíos.

Si una propuesta está mal, **desmarque CONFIRMAR**: el aplicativo deshace los datos que había puesto en `CONCILIACION`, recuerda ese movimiento como rechazado y, en el siguiente cruce (menú **4**), busca otro.

**Prueba con su historia.** Con los 139 documentos de la hoja `COMPRAS SII` cruzados contra FLUJO, sin usar las glosas y con alias aprendidos solo de meses anteriores:

| Estado | Documentos |
|---|---|
| CONCILIADA | 66 |
| POR REVISAR | 30 |
| SIN PAGO | 42 |

En 24 de las 66 conciliadas se puede verificar el resultado, porque la glosa que puso el analista menciona el folio, y 22 coinciden. Los 2 casos que no coinciden son facturas mensuales del mismo monto (Laudus), donde la propia historia es ambigua sobre qué mes pagó cuál.

En la simulación completa (cartolas de TPDV de enero y febrero de 2026 tratadas como nuevas, más el RCV), 61 movimientos de CONCILIACION quedaron completados con la factura.

## Instalación (una sola vez)

### Archivos a instalar

| Archivo en esta carpeta | Nombre en el editor de Apps Script | Tipo |
|---|---|---|
| `Motor.gs` | `Motor` | Secuencia de comandos |
| `Cartola.gs` | `Cartola` | Secuencia de comandos |
| `Sii.gs` | `Sii` | Secuencia de comandos |
| `Facturas.gs` | `Facturas` | Secuencia de comandos |
| `Codigo.gs` | `Codigo` (reemplaza al `Código` que trae el proyecto) | Secuencia de comandos |
| `Dialogo.html` | `Dialogo` | HTML |
| `appsscript.json` | `appsscript.json` (manifiesto) | — |

La carpeta `test/` no se instala: son pruebas que se ejecutan con Node fuera de Google.

### Paso 1 – Preparar la planilla en Google Sheets

1. En Excel, en la hoja `FLUJO`, convierta en valores las columnas con fórmulas que apuntan a otro libro (`categoria 4`, `Fecha Participación`, `categoria 2`, `categoria 3`): seleccione la columna, copie y use *Pegado especial → Valores*. Google Sheets no puede leer el libro externo `[1]data`.
2. Suba el `.xlsm` a Google Drive, ábralo con Google Sheets y elija *Archivo → Guardar como Hojas de cálculo de Google*.
3. Revise que existan estas hojas, con los nombres exactos:
   - `FLUJO`: obligatoria, es la historia.
   - `BBDD`: plan de cuentas y tabla *Cta. Provisión de Fondos*.
   - `PAGOS POR HACER` y `base flujo caja`: opcionales, se usan si existen.
4. En *Archivo → Configuración*, deje la zona horaria en **(GMT-03:00) Santiago**.

### Paso 2 – Crear el proyecto Apps Script

1. En la planilla: *Extensiones → Apps Script*. Se abre el editor con un archivo `Código.gs`.
2. Cambie el nombre de `Código` a `Codigo` y reemplace todo su contenido por el de `Codigo.gs`.
3. Para cada uno de `Motor`, `Cartola`, `Sii` y `Facturas`:
   - pulse **＋ → Secuencia de comandos**;
   - escriba el nombre sin extensión;
   - pegue el contenido del archivo `.gs` correspondiente.
4. Pulse **＋ → HTML**, llámelo `Dialogo` y pegue el contenido de `Dialogo.html`.
5. Configure el manifiesto:
   - en ⚙️ *Configuración del proyecto*, marque **"Mostrar el archivo de manifiesto appsscript.json en el editor"**;
   - vuelva al editor, abra `appsscript.json` y reemplace su contenido por el de esta carpeta.
6. Guarde con 💾 (Ctrl+S).
7. Vuelva a la planilla y recárguela (F5). Aparece el menú **🏦 Conciliación bancaria**.
8. La primera vez que use el menú, Google pedirá autorización:
   - elija su cuenta;
   - pulse *Configuración avanzada → Ir a (nombre del proyecto)*;
   - pulse *Permitir*.

   El script solo accede a esta planilla.

### Paso 3 – Configuración inicial

1. Menú **🏦 → Configuración inicial (crear hojas)**. Crea `CONCILIACION`, `REGLAS`, `CUENTAS BANCARIAS`, `CONFIG` y `CARTOLA`.
2. La hoja `FACTURAS COMPRA` se crea sola la primera vez que se carga un RCV del SII.
3. Opcional: menú **Evaluar precisión del modelo**, para ver cómo clasifica con su historia antes de usarlo.

### Actualizar a una versión nueva

Reemplace el contenido de cada archivo en el editor, guarde y recargue la planilla. Las hojas y los datos no se tocan. Si la versión trae parámetros nuevos, se agregan solos a `CONFIG`.

> También puede subirse con [clasp](https://github.com/google/clasp): `clasp clone <scriptId>` y después `clasp push` desde esta carpeta, con `test/**` en `.claspignore`.

## Uso mensual

1. **🏦 → 1. Cargar cartolas y facturas SII (archivos)…** y arrastre las cartolas del período y, si quiere cruzar pagos a proveedores, los RCV de compras de cada empresa. La pantalla reconoce sola cada tipo de archivo.
   - El archivo típico es el **CSV consolidado del Banco de Chile** (`Consolidado_cuentas_DD-MM-AAAA-DD-MM-AAAA.csv`): separado por `;`, una línea `Titular (RUT) cta:NNNN` y su fila de encabezados por cada cuenta, y montos con ceros a la izquierda (`0000000000000025473339`). El aplicativo lo separa por cuenta y omite las que no tienen movimientos.
   - Para cada cuenta se sugiere la empresa como aparece en `FLUJO`. La sugerencia sale de la hoja `CUENTAS BANCARIAS`, de la tabla *Cta. Provisión de Fondos* de `BBDD` (p.ej. `886-05140-01 → Octay`) o del nombre del titular. Confírmela la primera vez; queda guardada y el mes siguiente sale con ✔.
   - Si una cuenta queda en "omitir", no se procesa.
   - Alternativa: pegue la cartola en la hoja `CARTOLA` y use **1b**.
2. Revise la hoja **CONCILIACION**. **2. Ver solo pendientes / por revisar** filtra lo que requiere trabajo.
   - Al elegir o escribir una `CUENTA CONTABLE` (hay lista desplegable), el `NOMBRE CUENTA` se completa solo y la fila pasa a **MANUAL** con `APROBAR` marcado.
   - Para los POR REVISAR que estén bien, basta con marcar `APROBAR`.
   - Si un caso se repite todos los meses, use **Crear regla desde la fila seleccionada**.
3. En **FACTURAS COMPRA**, confirme las propuestas POR REVISAR. El menú **4. Cruzar facturas SII con movimientos** vuelve a cruzar cuando quiera, por ejemplo al cargar más cartolas; las facturas SIN PAGO se vuelven a buscar cada vez.
4. **3. Traspasar aprobados a FLUJO**. Antes de traspasar aplica los cruces confirmados. Agrega las filas aprobadas al final de `FLUJO` con todas sus columnas: `MES`, `CONCEPTO`, `DEBE`, `HABER`, `categoria 1`, `Flujo`, `categoria 4`, `mes año`, `categoria 2`, `categoria 3`. Lo no aprobado queda en `CONCILIACION` para después.

## Hojas de configuración

| Hoja | Para qué |
|---|---|
| `CONFIG` | Umbrales de confianza (`umbralIdentificado` 0,80 y `umbralRevisar` 0,55), uso de `PAGOS POR HACER`, peso de otras empresas y cantidad de vecinos. Para facturas: `umbralFactura` (0,85), `umbralFacturaRevisar` (0,50), `diasAntes` (30), `diasDespues` (180), `buscarOtrasEmpresas` (SI) y `cuentaProveedores` (2102001). Si la hoja ya existía, los parámetros nuevos se agregan solos. |
| `REGLAS` | Reglas fijas con prioridad sobre el modelo: empresa (opcional), texto o `/regex/` del detalle, tipo CARGO/ABONO, rango de monto y la clasificación a aplicar. |
| `CUENTAS BANCARIAS` | N° de cuenta → EMPRESA y RUT del titular. Se llena sola al confirmar cada cartola; el RUT dice en qué cuentas buscar el pago de las facturas de cada empresa. |
| `FACTURAS COMPRA` | Documentos del SII cargados y el pago encontrado para cada uno (ver *Facturas del SII*). |

## Archivos

| Archivo | Contenido |
|---|---|
| `Motor.gs` | Motor predictivo, reglas y cruce con pagos (JavaScript puro, probado en Node). |
| `Cartola.gs` | Lectura de cartolas: varias cuentas por archivo, encabezados Banco de Chile/Santander/genéricos, montos chilenos, fechas dd/mm/aaaa y seriales de Excel, reparación de tildes mal codificadas. |
| `Sii.gs` | Lectura del RCV del SII y motor de cruce facturas ↔ pagos (JavaScript puro, probado en Node). |
| `Facturas.gs` | Hoja FACTURAS COMPRA, carga sin duplicados, cruce, confirmación y rechazo, y aplicación a CONCILIACION/FLUJO. |
| `Codigo.gs` | Menú, hojas, clasificación, edición manual, traspaso a FLUJO y evaluación. |
| `Dialogo.html` | Pantalla de carga de archivos. Lee xls/xlsx con SheetJS y csv con `;` `,` o tabulador. |
| `test/backtest.js` | Mide la precisión mes a mes: `node test/backtest.js flujo.json 6` |
| `test/prueba_cartola.js` | Prueba completa con una cartola: `node test/prueba_cartola.js flujo.json cartola.json [AAAA-MM]` |
| `test/prueba_facturas.js` | Mide el cruce de facturas contra la historia: `node test/prueba_facturas.js flujo.json RCV_COMPRA_REGISTRO_<RUT>_<AAAAMM>.csv "EMPRESA - BANCO" [AAAA-MM-DD]` |
| `test/simulacion.js` | Simula el aplicativo completo con una planilla falsa (cargar → editar → traspasar → recargar): `[RCV=rcv.csv] node test/simulacion.js flujo.json cartola.csv bbdd.json [AAAA-MM]` (la cartola también puede ser `.json`) |

Los `.json` de prueba son las hojas exportadas como arreglos de filas. No se incluyen en el repositorio porque contienen datos de la empresa.
