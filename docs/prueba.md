# Probar Tu Cartera

Guía para usar Tu Cartera como si fuera tu estudio y anotar lo que le falta.

## 1. Entrar

**Opción A · GitHub Codespaces (recomendada, gratis, sin instalar nada)**

1. Abre <https://codespaces.new/solid0072704/copahue/tree/claude/zen-davinci-dyky36> con tu cuenta de GitHub
   y toca **Create codespace**. (También: en el repositorio elige la rama `claude/zen-davinci-dyky36` →
   **Code** → **Codespaces** → **Create codespace on…**).
2. Espera 2 a 3 minutos la primera vez. Tu Cartera se abre sola en una pestaña nueva
   (si no, ve a la pestaña **Ports** y abre el puerto **8000 · Tu Cartera**).
3. Ingresa con **prueba@tucartera.cl** y contraseña **prueba1234**.

Tus cambios se guardan dentro del codespace y siguen ahí cuando lo vuelves a abrir. GitHub detiene el codespace
tras 30 minutos sin uso; ábrelo de nuevo desde <https://github.com/codespaces>. El plan gratuito de GitHub incluye
horas de uso mensuales suficientes para probar.

Para abrirlo en el celular: en la pestaña **Ports**, clic derecho sobre el puerto 8000 → **Port Visibility** →
**Public**, y copia la dirección. Vuélvelo a **Private** al terminar.

**Opción B · Render (dirección fija para usar desde cualquier lugar)**

1. Crea una cuenta en <https://render.com> entrando con GitHub.
2. **New** → **Blueprint** → elige el repositorio `copahue`, rama `claude/zen-davinci-dyky36`.
3. Render pide los valores de `TUCARTERA_PRUEBA_EMAIL` y `TUCARTERA_PRUEBA_CLAVE`: usa tu correo y una
   contraseña tuya. Los demás puedes dejarlos vacíos.
4. Al terminar, Render te da una dirección `https://tu-cartera-xxxx.onrender.com`.

En el plan gratuito el servidor se duerme tras 15 minutos sin uso (la primera visita tarda ~1 minuto en
despertar) y los datos vuelven a cero en cada actualización. Para conservarlos, plan Starter con un disco en `/data`.

**Opción C · En tu computador (localhost)**

1. Instala Python 3.11 o superior desde <https://www.python.org/downloads/>
   (en Windows marca **Add python.exe to PATH** al instalar).
2. Descarga el código:
   <https://github.com/solid0072704/copahue/archive/refs/heads/claude/zen-davinci-dyky36.zip> y descomprímelo.
3. Abre la carpeta y haz doble clic en **Iniciar Tu Cartera.bat** (Windows) o **Iniciar Tu Cartera.command** (Mac;
   la primera vez, clic derecho → Abrir).
4. Se abre <http://localhost:8000>. Ingresa con **prueba@tucartera.cl** / **prueba1234**.

Los datos quedan en la carpeta `data/` y siguen ahí la próxima vez. Para abrirla desde el celular en la misma
red Wi-Fi, inicia con `python iniciar.py --red` y usa la dirección que aparece en la ventana.

## 2. Recorrido sugerido (20 minutos)

Hazlo pensando en tu estudio real. Cada vez que algo no calce, toca **Sugerir una mejora** (barra lateral).

1. **Hoy:** ¿lo que ves es lo que necesitas saber al empezar el día? ¿Falta o sobra algo?
2. **Para revisar:** abre tres propuestas, edita un correo y apruébalo; descarta otra. ¿El tono de los correos
   sirve para tus clientes? ¿Los motivos convencen?
3. **Clientes:** crea un cliente real tuyo (con datos ficticios si prefieres). ¿Faltan campos? ¿Se generaron sus
   declaraciones y documentos?
4. **Ficha de un cliente:** cambia el estado de una declaración, marca documentos, registra un pago.
5. **Vencimientos:** revisa noviembre y diciembre. ¿Están las obligaciones que manejas? ¿Las fechas calzan?
6. **Documentos y Cobranza:** pide lo que falta y prepara recordatorios.
7. **⌘K (o el ícono de búsqueda en el celular):** prueba pedidos como «pide los documentos que faltan» o el
   nombre de un cliente.
8. **Automatizaciones:** cambia una a «Actuar sola» y ejecútala. ¿Te daría confianza dejarla así?
9. **Plan y uso:** ¿los planes y precios se entienden?
10. **Celular:** repite 1, 2 y 4 desde el teléfono.

Al final: **Ajustes → Sugerencias anotadas → Copiar todas**, y pégalas en el chat para trabajarlas.

## 3. Lo que ya sabemos que falta

| Tema | Hoy | Qué faltaría |
|---|---|---|
| Datos reales | Solo datos de ejemplo o cargados a mano | Importar clientes desde Excel y ventas/compras desde el Registro de Compras y Ventas del SII. Sin esto, «Revisión de cifras» y los gráficos no funcionan con clientes reales. |
| Recepción de documentos | El cliente responde por correo y se marca «recibido» a mano | Portal o enlace para que el cliente suba archivos, y marcar recibido automáticamente. |
| Correo saliente | Requiere configurar SMTP | Enviar desde la cuenta Gmail u Outlook del estudio. |
| Cobro de la suscripción | Activación manual con un comando | Pasarela de pago (Flow o Mercado Pago) y boleta o factura automática. |
| Honorarios del estudio | Se registran como datos | Emitir las facturas electrónicas al cliente. |
| Equipo | Todos ven toda la cartera | Roles (solo lectura, cartera propia) y permisos. |
| Edición | Tareas: crear y completar; clientes: editar y pausar | Editar o borrar tareas, eliminar clientes, exportar a Excel o PDF. |
| Avisos | Solo dentro de la app | Resumen diario por correo y notificaciones en el celular. |
| Cuenta | Sin recuperación de contraseña | «Olvidé mi contraseña» por correo. |
| Calendario | F29, Previred, LRE y F22 | Declaraciones juradas de marzo, patentes municipales y otras obligaciones. |
| IA y Google/Microsoft | Probados con respuestas simuladas | Probar en vivo con `ANTHROPIC_API_KEY` y las credenciales OAuth. |
