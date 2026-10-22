# Copahue · herramientas de finanzas

| App | Para qué sirve | Cómo se ejecuta |
|---|---|---|
| `tablero_trading.py` | **Agente de inversiones**: depositas tu dinero, el agente propone qué comprar o vender y **te consulta cada orden** (Telegram o tablero) antes de ejecutarla | `streamlit run tablero_trading.py` |
| `tablero_ebook.py` | **Tablero con agentes** que analizan las descargas del ebook, te avisan por correo, hacen seguimiento y generan reuniones con potenciales clientes | `streamlit run tablero_ebook.py` |
| `app.py` | Generador de reportes Forpay (cuotas por empresa a Excel) | `streamlit run app.py` |
| `grok.py` | Chat con tus documentos PDF (Groq + Gemini) | `streamlit run grok.py` |

---

## 🤖 Agente de inversiones (con tu aprobación)

Tú depositas el dinero. Los agentes arman la cartera según tu perfil de riesgo, la vigilan y te proponen
cada compra o venta. **Ninguna orden se ejecuta sin que toques ✅ Ejecutar** en Telegram o en el tablero.

```mermaid
flowchart LR
    D[💵 Depositas] --> B[(Broker<br/>simulado o Alpaca)]
    B --> E[🧭 Estratega<br/>cartera objetivo, tendencia,<br/>rebalanceo, stop]
    E --> G[🛡️ Guardia de riesgo<br/>límites que nadie puede saltar]
    G --> IA[🧠 Claude opcional<br/>segunda opinión]
    IA --> N[🔔 Notificador]
    N -->|✅ / ❌| T[📱 Tu Telegram o tablero]
    T --> X[⚙️ Ejecutor<br/>revalida precio y riesgo]
    X --> B
```

| Agente | Qué hace |
|---|---|
| **Estratega** | Lleva tu cartera a la de tu perfil (`conservador`, `moderado`, `agresivo` o la tuya en `CARTERA_OBJETIVO`) con ETFs diversificados. **Filtro de tendencia**: si un activo cae bajo su media de 200 días, su peso pasa a bonos cortos (SHY). **Rebalancea** solo si un activo se desvía más del umbral, e invierte el dinero nuevo cuando depositas. **Stop de pérdida** si una posición cae más del X % bajo su costo. |
| **Guardia de riesgo** | Revisa cada orden al proponerla y otra vez al ejecutarla: solo activos permitidos, sin apalancamiento ni ventas en corto, reserva de efectivo, máximo por activo, máximo de órdenes por día y **freno de pérdidas** (pausa compras si la cartera retrocede mucho). |
| **IA (opcional)** | Claude da una segunda opinión (👍 / 🤔 / 👎) y te explica cada orden en simple. No puede crear órdenes ni cambiar montos. |
| **Notificador** | Te consulta por **Telegram con botones ✅ Ejecutar / ❌ Rechazar** (solo acepta respuestas de tu chat), te avisa por correo y responde `/cartera`. |
| **Ejecutor** | Ejecuta solo lo aprobado, primero ventas y luego compras. Si el precio se movió más del límite desde que lo viste, no ejecuta. Si no respondes a tiempo, la propuesta vence. Nunca ejecuta dos veces la misma orden. |

### Puesta en marcha

1. **Pruébalo sin riesgo** (dinero ficticio, precios reales):
   ```bash
   pip install -r requirements.txt
   cp .env.example .env          # deja BROKER=simulado
   streamlit run tablero_trading.py
   ```
   Deposita un monto en la barra lateral y aprueba o rechaza lo que te proponga. Para probar sin internet: botón **🧪 Cargar demo**.
2. **Telegram** (para que te consulte en el celular): crea un bot con [@BotFather](https://t.me/BotFather), escríbele
   cualquier cosa y copia tu `chat.id` desde `https://api.telegram.org/bot<TOKEN>/getUpdates`. Completa
   `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID`. Comandos: `/cartera` y `/plan`.
3. **Broker real**: abre una cuenta en [Alpaca](https://alpaca.markets) (ETFs de EE.UU., acepta fracciones de acción),
   crea claves **paper** y usa `BROKER=alpaca`, `ALPACA_PAPER=true`. Tu dinero se deposita y queda en Alpaca, no aquí.
4. **Dinero real** solo cuando estés conforme con la práctica: `ALPACA_PAPER=false` con claves de la cuenta real y
   `TRADING_REAL=ENTIENDO_EL_RIESGO`. Sin esa frase, el sistema se niega a operar con dinero real.
5. **Que trabaje solo**: `python run_trading.py --loop 5` en un PC o servidor, o el flujo
   `.github/workflows/agentes-trading.yml` (cada 10 min con la bolsa abierta; actívalo con la variable `TRADING_ACTIVO=true`
   y los Secrets que indica el archivo). En Actions usa `BROKER=alpaca`: la memoria se guarda en la caché y la cartera, en el broker.

> ⚠️ Invertir tiene riesgo y puedes perder dinero. Esta herramienta no garantiza rentabilidad ni es asesoría
> financiera; las estrategias son reglas simples y públicas. Empieza en modo práctica y con montos que puedas perder.

Archivos: `agentes_trading/` (`estratega.py`, `guardia.py`, `ejecutor.py`, `notificador.py`, `brokers.py`,
`mercado.py`, `llm.py`, `orquestador.py`), `run_trading.py`, `tablero_trading.py` y `tests/test_trading.py`.

---

## 📘 Tablero de agentes del ebook

Cada vez que alguien descarga el ebook desde tu sitio, FormSubmit te envía a Gmail un correo
**«Nueva descarga del ebook — Hyperion»** con nombre, correo, teléfono, cargo, empresa y motivo de descarga.
Los agentes leen esos correos y hacen el trabajo comercial por ti:

```mermaid
flowchart LR
    F[FormSubmit<br/>notificación] --> G[(Tu Gmail)]
    G --> C[🛰️ Captura<br/>lee y ordena las descargas]
    C --> A[🧠 Analista<br/>score 0-100 y segmento]
    A --> R[🤝 Reuniones<br/>propone horarios e invitaciones]
    A --> S[✉️ Seguimiento<br/>secuencia de correos]
    R --> S
    S -->|borradores o envío| G
    G -->|¿respondió? ¿lo enviaste?| S
    A --> N[🔔 Notificador]
    S --> N
    N -->|alerta + resumen diario| Y[Tu correo]
```

| Agente | Qué hace |
|---|---|
| **Captura** | Lee en Gmail las notificaciones de FormSubmit (o un CSV/Excel), evita duplicados y detecta descargas repetidas. |
| **Analista** | Puntúa cada lead (correo corporativo, cargo, empresa, motivo, país, teléfono, descargas repetidas, si respondió) y lo clasifica en 🔥 Caliente, 🌤️ Tibio o ❄️ Frío. Escribe una lectura comercial y cómo abordarlo. Resume tendencias. |
| **Seguimiento** | Arma una secuencia de 4 correos según el segmento (bienvenida → contenido de valor → invitación → último contacto). Los deja como **borradores en tu Gmail**, los envía solo o espera tu aprobación. Se detiene apenas el lead responde. |
| **Reuniones** | A los calientes y a quienes respondieron les propone 3 horarios hábiles. Genera el evento de Google Calendar con el lead como invitado, el archivo `.ics`, el link de WhatsApp y exporta los contactos a tu CRM (CSV compatible con HubSpot). |
| **Notificador** | Te envía una **alerta inmediata** cuando aparece un lead caliente o alguien te responde (con botones para responder, escribir por WhatsApp o agendar) y un **resumen diario** con las cifras del día. |

### Puesta en marcha (10 minutos)

```bash
pip install -r requirements.txt
cp .env.example .env          # completa GMAIL_USER y GMAIL_APP_PASSWORD
streamlit run tablero_ebook.py
```

1. **Contraseña de aplicación de Gmail**: activa la verificación en 2 pasos y crea una en
   <https://myaccount.google.com/apppasswords>. Pégala en `GMAIL_APP_PASSWORD`. (IMAP viene activo por defecto en Gmail.)
2. Abre el tablero y presiona **▶️ Ejecutar agentes ahora**: leerá las descargas de los últimos 120 días.
3. ¿Quieres verlo antes de conectar Gmail? En la barra lateral, **🧪 Datos de demostración → Cargar demo**
   (leads ficticios con dominios `.example`, que nunca reciben correos).

Personaliza en `.env` tu firma, el nombre del ebook, la oferta de reunión y, si tienes Calendly o
agenda de citas de Google, `LINK_AGENDA` (los correos usarán ese link en vez de proponer horarios).

### Modos de seguimiento

| Modo | Qué pasa con cada correo a un lead |
|---|---|
| `borradores` *(recomendado)* | Queda listo en **Borradores** de tu Gmail. Tú lo revisas y lo envías; el agente detecta que lo enviaste y sigue con el próximo paso. |
| `automatico` | Se envía solo, con un máximo de `SEGUIMIENTO_MAX_DIA` correos al día (primero los más calientes). |
| `manual` | Espera tu aprobación en la pestaña **✉️ Seguimiento** del tablero. |

Puedes cambiar el modo en la barra lateral del tablero. Los leads que descargaron hace más de 3 días y
nunca contactaste reciben un correo de **reactivación** (en vez de la bienvenida); los de más de 60 días quedan
para gestión manual, para no escribir en masa a una base antigua.

### Que trabajen solos las 24 horas

Elige una opción (no las dos con el mismo Gmail, para no duplicar borradores):

- **Tu PC o un servidor**: `python run_agentes.py --loop 15` revisa Gmail cada 15 minutos.
  El tablero (`streamlit run tablero_ebook.py`) lee la misma base de datos.
- **GitHub Actions** (gratis, sin PC encendido): el flujo `.github/workflows/agentes-ebook.yml` corre cada 30 minutos.
  En *Settings → Secrets and variables → Actions* crea los secrets `GMAIL_USER`, `GMAIL_APP_PASSWORD`
  (y opcionalmente `NOTIFY_EMAIL`, `ANTHROPIC_API_KEY`) y la variable `AGENTES_ACTIVOS=true`.
  Recibirás las alertas y el resumen por correo, y los borradores aparecerán en tu Gmail. En este modo
  el tablero es para consultar: usa allí el modo `manual` y gestiona los correos desde Gmail.

Comandos útiles:

```bash
python run_agentes.py              # un ciclo
python run_agentes.py --resumen    # fuerza el resumen diario
python run_agentes.py --sin-notificar
python run_agentes.py --demo       # carga datos ficticios
```

### IA opcional (Claude)

Con `ANTHROPIC_API_KEY`, el analista escribe una lectura más fina de los leads calientes y tibios, y
el agente de seguimiento personaliza cada correo (sin inventar datos y conservando horarios y links).
Usa el modelo `claude-opus-5-5` con esfuerzo bajo y respaldo automático del servidor (`fallbacks: "default"`)
si una solicitud es rechazada. Sin API key todo funciona igual con reglas y plantillas.

### ¿Cómo se calcula el score?

| Factor | Puntos |
|---|---|
| Correo corporativo (no Gmail/Hotmail/Outlook…) | +20 |
| Cargo: decisor (gerente, CFO, socio, fundador, dueño…) / jefatura / especialista / otro | +30 / +20 / +10 / +5 |
| Informó su empresa (independiente: +5) | +10 |
| Motivo: necesidad en su empresa / consultor-asesor / aprendizaje / otro | +25 / +10 / +3 / +5 |
| Dejó teléfono | +5 |
| País en `MERCADO_OBJETIVO` (por prefijo telefónico o dominio) | +10 |
| Descargó más de una vez | +10 |
| Respondió un correo | +25 |
| Descarga en los últimos 3 días | +5 |

🔥 Caliente ≥ `UMBRAL_CALIENTE` (70) · 🌤️ Tibio ≥ `UMBRAL_TIBIO` (40) · ❄️ Frío el resto.
Las reglas están en `agentes_ebook/analista.py` y las plantillas de correo en `agentes_ebook/plantillas.py`.

### Privacidad y seguridad

- Los datos de tus leads se guardan solo en `data/tablero_ebook.db` (ignorado por git) y, si usas Actions, en su caché privada.
- Las credenciales van en `.env` o en los Secrets de Streamlit/GitHub, nunca en el código.
- Los agentes leen Gmail en modo solo lectura (no marcan correos como leídos) y solo escriben borradores o
  envían correos según el modo que elijas.

### Pruebas

```bash
pip install pytest
pytest -q
```

Estructura:

```
agentes_ebook/
  config.py        configuración desde .env / secrets
  db.py            base SQLite: leads, descargas, seguimientos, reuniones, bitácora
  gmail.py         IMAP/SMTP: lee FormSubmit, detecta respuestas, crea borradores, envía
  captura.py       agente de captura (Gmail, CSV, demo)
  analista.py      agente analista (score, segmento, hallazgos)
  plantillas.py    correos, cadencias, links de WhatsApp y mailto
  seguimiento.py   agente de seguimiento
  reuniones.py     agente de reuniones (horarios, Google Calendar, .ics, CRM)
  notificador.py   alertas y resumen diario por correo
  llm.py           IA opcional con Claude
  orquestador.py   ciclo completo de los agentes
tablero_ebook.py   tablero Streamlit
run_agentes.py     ejecución por terminal / cron / GitHub Actions
```
