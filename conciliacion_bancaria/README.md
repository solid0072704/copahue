# Conciliación bancaria predictiva (Google Apps Script)

Aplicativo para Google Sheets que **clasifica automáticamente los movimientos de las cartolas bancarias** usando la historia de la hoja `FLUJO`. Lo que no logra identificar con seguridad lo deja marcado para que el analista lo clasifique a mano.

## Cómo funciona

```
 Cartolas (.xls/.xlsx/.csv)          Historia (hoja FLUJO)
            │                                │
            ▼                                ▼
   Lectura por cuenta ─────────► Motor predictivo (vecinos más cercanos)
   (Banco de Chile consolidado,     • texto del detalle (palabras ponderadas, números enmascarados)
    Santander, genérico)            • misma empresa / cuenta
            │                       • cargo o abono, monto parecido, antigüedad
            │                       • patrón de la empresa para contrapartes nuevas
            │                       + REGLAS manuales (prioridad)
            │                       + cruce con PAGOS POR HACER (monto exacto + nombre)
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
| IDENTIFICADO | ~39 % | **~95 %** |
| POR REVISAR | ~26 % | ~75 % (el analista solo confirma) |
| PENDIENTE | ~35 % | (clasificación manual) |

Con la cartola consolidada de julio (31 cuentas, 185 movimientos), simulando que era nueva: **93 identificados, 55 por revisar y 37 pendientes**.

Los pendientes son sobre todo contrapartes que nunca habían aparecido (personas nuevas que transfieren, proveedores nuevos) y movimientos genéricos como `Dep.cheq.otros Bancos`, donde el proyecto o la parcela no se deducen del texto. A medida que se traspasan meses a `FLUJO`, y si se agregan **REGLAS** para los casos repetitivos, el porcentaje identificado sube. Puede volver a medirse cuando quiera desde el menú *Evaluar precisión del modelo*.

## Instalación (una sola vez)

1. **Preparar la planilla en Google Sheets**
   - En Excel, en la hoja `FLUJO`, convierta en valores las columnas con fórmulas que apuntan a otro libro (`categoria 4`, `Fecha Participación`, `categoria 2`, `categoria 3`): seleccione la columna, copie y use *Pegado especial → Valores*. Google Sheets no puede leer el libro externo `[1]data`.
   - Suba el `.xlsm` a Google Drive, ábralo con Google Sheets y elija *Archivo → Guardar como Hojas de cálculo de Google*.
   - Deben existir las hojas `FLUJO` (obligatoria), `BBDD` (plan de cuentas y cuentas de provisión), `PAGOS POR HACER` y `base flujo caja` (opcionales, se usan si están).
2. **Crear el proyecto Apps Script**
   - En la planilla: *Extensiones → Apps Script*.
   - Cree los archivos con el mismo nombre y copie su contenido: `Motor.gs`, `Cartola.gs`, `Codigo.gs` (puede reemplazar el `Código.gs` inicial) y un archivo HTML llamado `Dialogo` (con el contenido de `Dialogo.html`).
   - En *Configuración del proyecto* marque "Mostrar el archivo de manifiesto appsscript.json" y reemplácelo por el de esta carpeta (zona horaria `America/Santiago`, la misma que debe tener la planilla en *Archivo → Configuración*).
   - Guarde y recargue la planilla. Aparece el menú **🏦 Conciliación bancaria**.
   - La primera vez Google pedirá autorización. El script solo accede a esta planilla.
3. Menú **Configuración inicial**. Crea las hojas `CONCILIACION`, `REGLAS`, `CUENTAS BANCARIAS`, `CONFIG` y `CARTOLA`.

> También puede subirse con [clasp](https://github.com/google/clasp): `clasp create --type sheets` o `clasp clone <scriptId>` y después `clasp push` desde esta carpeta. La carpeta `test/` no se sube; si usa clasp, agréguela a `.claspignore`.

## Uso mensual

1. **🏦 → 1. Cargar cartolas (archivos)…** y arrastre las cartolas del período.
   - La cartola consolidada del Banco de Chile (varias cuentas en un archivo) se separa por cuenta automáticamente.
   - Para cada cuenta se sugiere la empresa como aparece en `FLUJO`. La sugerencia sale de la hoja `CUENTAS BANCARIAS`, de la tabla *Cta. Provisión de Fondos* de `BBDD` (p.ej. `886-05140-01 → Octay`) o del nombre del titular. Confírmela la primera vez; queda guardada y el mes siguiente sale con ✔.
   - Si una cuenta queda en "omitir", no se procesa.
   - Alternativa: pegue la cartola en la hoja `CARTOLA` y use **1b**.
2. Revise la hoja **CONCILIACION**. **2. Ver solo pendientes / por revisar** filtra lo que requiere trabajo.
   - Al elegir o escribir una `CUENTA CONTABLE` (hay lista desplegable), el `NOMBRE CUENTA` se completa solo y la fila pasa a **MANUAL** con `APROBAR` marcado.
   - Para los POR REVISAR que estén bien, basta con marcar `APROBAR`.
   - Si un caso se repite todos los meses, use **Crear regla desde la fila seleccionada**.
3. **3. Traspasar aprobados a FLUJO**. Agrega las filas aprobadas al final de `FLUJO` con todas sus columnas: `MES`, `CONCEPTO`, `DEBE`, `HABER`, `categoria 1`, `Flujo`, `categoria 4`, `mes año`, `categoria 2`, `categoria 3`. Lo no aprobado queda en `CONCILIACION` para después.

## Hojas de configuración

| Hoja | Para qué |
|---|---|
| `CONFIG` | Umbrales de confianza (`umbralIdentificado` 0,80 y `umbralRevisar` 0,55), uso de `PAGOS POR HACER`, peso de otras empresas y cantidad de vecinos. |
| `REGLAS` | Reglas fijas con prioridad sobre el modelo: empresa (opcional), texto o `/regex/` del detalle, tipo CARGO/ABONO, rango de monto y la clasificación a aplicar. |
| `CUENTAS BANCARIAS` | N° de cuenta → EMPRESA. Se llena sola al confirmar cada cartola. |

## Archivos

| Archivo | Contenido |
|---|---|
| `Motor.gs` | Motor predictivo, reglas y cruce con pagos (JavaScript puro, probado en Node). |
| `Cartola.gs` | Lectura de cartolas: varias cuentas por archivo, encabezados Banco de Chile/Santander/genéricos, montos chilenos, fechas dd/mm/aaaa y seriales de Excel, reparación de tildes mal codificadas. |
| `Codigo.gs` | Menú, hojas, clasificación, edición manual, traspaso a FLUJO y evaluación. |
| `Dialogo.html` | Pantalla de carga de archivos. Lee xls/xlsx con SheetJS y csv con `;` `,` o tabulador. |
| `test/backtest.js` | Mide la precisión mes a mes: `node test/backtest.js flujo.json 6` |
| `test/prueba_cartola.js` | Prueba completa con una cartola: `node test/prueba_cartola.js flujo.json cartola.json [AAAA-MM]` |
| `test/simulacion.js` | Simula el aplicativo completo con una planilla falsa (cargar → editar → traspasar → recargar): `node test/simulacion.js flujo.json cartola.json bbdd.json [AAAA-MM]` |

Los `.json` de prueba son las hojas exportadas como arreglos de filas. No se incluyen en el repositorio porque contienen datos de la empresa.
