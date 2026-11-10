# TPDV Academy · plataforma de inducción en Google Apps Script

Plataforma de aprendizaje para los nuevos ingresos de **Tu Proyecto de Vida**. Toma el material de la carpeta
compartida **TPDV ACADEMY** de Drive, lo ordena en el programa oficial de 4 semanas y lleva el registro del avance,
las evaluaciones y los certificados de cada persona. **Solo entran cuentas @tuproyectodevida.cl** y usa el kit de
marca oficial.

| Para el colaborador | Para la coordinación |
|---|---|
| Ruta de 4 semanas con módulos que se desbloquean en orden | Panel con el avance de cada colaborador |
| Videos, PDF, presentaciones y documentos de Drive dentro de la plataforma | Registro de la evaluación práctica (role play, rúbrica 3/2/1) |
| Evaluación por módulo (aprobación 80%) con revisión de respuestas | Verificación de permisos de cada archivo de Drive |
| Certificado de finalización para imprimir o guardar en PDF | Detección de archivos nuevos en la carpeta («Por clasificar») |
| Funciona en computador y celular | Todo se administra desde una planilla de Google Sheets |

---

## 1. El programa

Estructura tomada del documento **«PILARES Y CONTENIDOS»** (carpeta TPDV ACADEMY) y del cronograma del asesor comercial.

| Semana | Módulo | Lecciones | Evaluación |
|---|---|---|---|
| 1 | Te damos la bienvenida a Tu Proyecto de Vida | 3 lecturas (cómo funciona la academia, quiénes somos, propuesta de valor y zonas) | 6 preguntas |
| 1 | Ventas I: fundamentos, proyectos y HubSpot básico | 13 (video del pilar, rol y metas, proceso de venta, HubSpot 1–2, fichas, manual de visitas) | 6 preguntas |
| 2 | Ventas II: seguimiento, guiones y cierre | 17 (HubSpot 3–5, guiones de llamada, plantillas WhatsApp/correo, objeciones, cierre, role play) | 6 preguntas |
| 2 | Marketing I: leads, canales y portafolio | 4 (video del pilar, Pilar I de Marketing, Manual de Marca, logos 2026) | 7 preguntas |
| 3 | Marketing II: contenido para ventas y CRM | 3 (Pilar II de Marketing, checklist de contenido, grilla 2026) | 5 preguntas |
| 3 | Legal I: fundamentos legales de la parcelación | 6 (video del pilar, presentación, inducción de Escrituración, glosario, checklist legal, guion) | 6 preguntas |
| 4 | Legal II: documentos de venta y preguntas frecuentes | 3 (promesa y compraventa, crédito y plazos, preguntas frecuentes) | 6 preguntas |
| 4 | Evaluación final: simulación comercial completa | 2 (cómo es la evaluación, cuestionario de práctica) | Práctica, la registra la tutora |

Cada lección apunta al archivo real de Drive (ver `src/10_Catalogo.gs`). Las preguntas se redactaron a partir del
material de cada pilar y la columna «Fuente» de la hoja «Preguntas» indica de qué archivo sale cada una.

---

## 2. Seguridad: solo @tuproyectodevida.cl

1. **Google bloquea la entrada**: la aplicación web se publica con acceso «Cualquier usuario de tuproyectodevida.cl»
   (`appsscript.json → webapp.access = DOMAIN`).
2. **El código vuelve a verificar**: cada página y cada acción comprueba que el correo termine exactamente en
   `@tuproyectodevida.cl` (`src/01_Acceso.gs`). Una cuenta externa ve la página «Acceso restringido» y no recibe
   ningún dato. El dominio está fijo en el código, así que no se puede ampliar editando la planilla.
3. **Respuestas en el servidor**: las respuestas correctas nunca viajan al navegador antes de enviar la evaluación.
4. **Administradores**: el dueño del script y los correos de la clave `ADMINISTRADORES`, siempre del dominio.

> ⚠️ El proyecto de Apps Script debe crearse con una **cuenta @tuproyectodevida.cl** (ideal: una cuenta del área, por
> ejemplo `academia@tuproyectodevida.cl`). Si se crea con una cuenta personal, Google no entrega el correo de los
> visitantes y la plataforma bloquea a todos.

---

## 3. Instalación (15 minutos)

### Opción A: copiar y pegar en el editor

1. Con la cuenta @tuproyectodevida.cl, entra a <https://script.google.com> → **Nuevo proyecto** → nómbralo `TPDV Academy`.
2. En **Configuración del proyecto** (engranaje) activa **«Mostrar el archivo de manifiesto appsscript.json»**.
3. Crea un archivo por cada uno de `src/` con el mismo nombre y pega su contenido:
   - Archivos de **secuencia de comandos** (`+ → Secuencia de comandos`): `00_Config`, `01_Acceso`, `02_Datos`,
     `03_Contenido`, `04_Progreso`, `05_Evaluaciones`, `06_Api`, `07_Instalacion`, `10_Catalogo`.
   - Archivos **HTML** (`+ → HTML`): `Index`, `Estilos`, `App`, `AccesoDenegado`.
   - Reemplaza el contenido de `appsscript.json` y borra el `Código.gs` vacío.

### Opción B: con clasp (línea de comandos)

```bash
npm install -g @google/clasp
```

```bash
clasp login
```

```bash
cd academia && clasp create --type standalone --title "TPDV Academy" --rootDir src
```

```bash
clasp push
```

(`.clasp.json.example` muestra el formato si ya tienes un proyecto creado.)

### Después de copiar el código

4. En el editor elige la función **`instalar`** y presiona **Ejecutar**. Autoriza los permisos (Drive en modo lectura,
   Hojas de cálculo, tu correo y activadores). En el registro de ejecución aparecerá el enlace a la planilla
   **«TPDV Academy · Base de datos»**, ya cargada con el programa.
5. **Implementar → Nueva implementación → Aplicación web**:
   - Ejecutar como: **Yo**
   - Quién tiene acceso: **Cualquier usuario de tuproyectodevida.cl**
   - Copia la URL `/exec`: ese es el enlace de la academia para los nuevos ingresos.
6. Entra a la URL, abre **Panel → Verificar permisos** y comparte con el dominio los archivos que aparezcan con aviso
   («Cualquier persona de tuproyectodevida.cl con el enlace: lector»). Lo más simple es compartir así las carpetas
   TPDV ACADEMY, INFORMACION CAPACITACION y la carpeta de logos 2026.
7. (Opcional) Ejecuta **`activarSincronizacionDiaria`** para que la academia revise la carpeta de Drive cada mañana.

Cada vez que cambies el código, ve a **Implementar → Administrar implementaciones → Editar → Nueva versión** para
publicarlo en la misma URL.

---

## 4. Administración diaria (todo desde la planilla)

| Hoja | Para qué |
|---|---|
| **Config** | Nombre, puntaje de aprobación (`PUNTAJE_APROBACION`), intentos (`INTENTOS_MAXIMOS`, 0 = sin límite), orden obligatorio, administradores, correo de ayuda, colores, tipografías y logos. |
| **Módulos** | Título, descripción, objetivos (uno por línea), semana, pilar, orden y `Visible`. `Tipo = Práctica` agrega la evaluación que registra la tutora. |
| **Lecciones** | Una fila por material: módulo, orden, sección, título, tipo, ID de Drive o URL (YouTube o enlace), duración, obligatoria y visible. |
| **Preguntas** | Selección múltiple: módulo, pregunta, opciones A–D, letra correcta, explicación y fuente. |
| **Por clasificar** | Archivos nuevos que aparecieron en la carpeta. Escribe el ID del módulo, pon `SI` en «Incluir» y presiona **Sincronizar con Drive** en el panel. Los que tienen `NO` ya fueron revisados (actas, borradores, claves del evaluador) y no se publican. |
| **Usuarios · Progreso · Resultados** | Registro automático de cada colaborador: ingresos, lecciones vistas, puntajes, intentos y fecha del certificado. |

**Evaluación práctica (semana 4).** En **Panel → Registrar evaluación práctica** elige al colaborador, ingresa los
puntos de la rúbrica (por ejemplo 26 de 30; la plataforma calcula el %) y la retroalimentación. Con 80% o más se
completa el programa y se habilita el certificado.

**Kit de marca.** Viene configurado con el *Manual de Marca TPDV* (dic-2025): colores principales `#005A82`,
`#007264`, `#85B500` y `#FFB829`; Montserrat para títulos y Lato para textos; logo `horizontal_blanco.png` en la
cabecera azul y `Horizontal_multicolor.png` en el certificado (set de logos 2026). Si el manual cambia, edita las
claves `COLOR_*`, `FUENTE_*` y `LOGO_*` de la hoja «Config»; los valores inválidos se ignoran y se usa el kit oficial.

---

## 5. Revisar antes de abrir la academia

Al leer la carpeta TPDV ACADEMY se detectó lo siguiente. Conviene resolverlo con cada área:

**Acceso y archivos sensibles**
- Las 16 **«Notas de Gemini»** se compartieron solo como accesos directos; los documentos en sí no están compartidos.
  No se usan como lecciones (son actas de producción).
- **Claves del evaluador** en la misma carpeta que verán los alumnos: *Cuestionario para evaluador.docx* y
  *Preguntas y Respuestas desarrollo escrituración.xlsx*. Muévelas a una carpeta restringida.
- *202509 - Ejemplo Informe Semanal I.S.A* contiene datos de prospectos reales, y *Capacitación vendedores TPDV.pdf*
  incluye certificados reales con datos personales. Ninguno de los dos se publicó.

**Contradicciones en el material** (se dejaron fuera de las evaluaciones)
- Ventas: los tramos de comisión, el momento en que se paga la primera comisión, el bono trimestral y algunos
  supuestos de metas (valor promedio de la parcela, contactos necesarios por venta) no coinciden entre documentos.
- Legal: luz y agua («No» frente a «con factibilidad»), cambio de uso de suelo, extranjeros (10 km de frontera frente
  a nacionales de países limítrofes), plazos de notaría y CBR, y el plano de loteo (DOM o SAG).
- La rúbrica de Escrituración aprueba desde 18/30 (60%), mientras que el programa exige 80%. La plataforma usa
  `PUNTAJE_APROBACION` (80) para todo.

**Material pendiente**
- El Keynote de Marketing (1,4 GB) no se puede previsualizar en Drive: exportarlo a PDF y agregarlo al módulo.
- No hay material específico de SPIN Selling, de errores frecuentes en la promesa ni una promesa de ejemplo.
- *Preguntas frecuentes de clientes* todavía tiene notas internas de revisión.
- Las cifras corporativas de la lección «Quiénes somos» (fundación 2019, +20 proyectos, +3.500 clientes) vienen del
  documento de contexto corporativo. Validarlas con Marketing.

---

## 6. Pruebas

El código del servidor se prueba en Node con un simulador de Apps Script (Drive, Sheets, Session y HtmlService):

```bash
node --test academia/tests/*.test.js
```

Las 19 pruebas cubren el bloqueo de cuentas externas, los permisos de administración, la instalación, el recorrido
completo hasta el certificado, la sincronización con Drive, la verificación de permisos, la seguridad de la página y la
consistencia del catálogo.

Para generar capturas de cada pantalla con Chromium (requiere Playwright):

```bash
node academia/tests/vista_previa.js --salida /tmp/capturas
```

`--logo <png>` y `--fuentes <node_modules/@fontsource>` son opcionales y sirven para ver el logo y las tipografías reales.

## Archivos

```
academia/
├── src/                     ← lo que se copia a Apps Script
│   ├── appsscript.json      manifiesto (zona horaria, acceso por dominio, permisos)
│   ├── 00_Config.gs         dominio permitido, carpeta y valores por defecto (marca incluida)
│   ├── 01_Acceso.gs         control de acceso y administradores
│   ├── 02_Datos.gs          base de datos en Google Sheets
│   ├── 03_Contenido.gs      módulos, lecciones, visores de Drive, sincronización, permisos y marca
│   ├── 04_Progreso.gs       avance, desbloqueo y certificado
│   ├── 05_Evaluaciones.gs   cuestionarios y evaluación práctica
│   ├── 06_Api.gs            doGet y funciones que llama la página
│   ├── 07_Instalacion.gs    instalar() y sincronización diaria
│   ├── 10_Catalogo.gs       programa inicial, preguntas y lecturas propias
│   └── Index / Estilos / App / AccesoDenegado .html   interfaz
└── tests/                   simulador de Apps Script, pruebas y vista previa
```
