# Instalación en tu equipo con Visual Studio Code

Funciona en Windows, macOS y Linux. Necesitas Python 3.11 o 3.12, Git y VS Code.

## 1. Programas

| Programa | Dónde | Detalle |
|---|---|---|
| Python 3.12 | https://www.python.org/downloads/ | En Windows marca **"Add python.exe to PATH"** en la primera pantalla del instalador |
| Git | https://git-scm.com/downloads | Instalación por defecto |
| VS Code | https://code.visualstudio.com/ | Luego instala la extensión **Python** (Microsoft); VS Code la sugiere al abrir la carpeta |

## 2. Clonar el repositorio

En VS Code: `Ctrl+Shift+P` → **Git: Clone** → pega `https://github.com/solid0072704/copahue` → elige una carpeta → **Open**.

Cambia a la rama con el agente de trading: en la barra inferior izquierda pulsa el nombre de la rama (`main`) y elige `origin/claude/new-session-bb9omp`. O en la terminal:

```bash
git checkout claude/new-session-bb9omp
```

## 3. Entorno virtual e instalación

Abre la terminal de VS Code (`Ctrl+ñ` o menú Terminal → New Terminal) y ejecuta:

Windows (PowerShell):
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt pytest
```

Si PowerShell se niega a activar el entorno ("running scripts is disabled"), ejecuta una vez
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a intentarlo.

macOS / Linux:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt pytest
```

Luego `Ctrl+Shift+P` → **Python: Select Interpreter** → elige el que termina en `.venv`.
VS Code lo recordará (está en `.vscode/settings.json`).

## 4. Comprobar que todo funciona (sin claves ni dinero)

```bash
pytest -q tests
python run_trader.py --mode backtest --synthetic --candles 900 --seed 7
```

Deben pasar las 88 pruebas (64 del agente de trading y 24 del tablero del ebook) y el backtest imprime métricas (fills, drawdown, Brier). En Windows no uses el comodín `tests/test_trading_*.py`: PowerShell no lo expande y pytest responde "file or directory not found"; `pytest -q tests` funciona en todos los sistemas.
En **Run and Debug** (`Ctrl+Shift+D`) tienes estos mismos comandos como botones.

## 5. Configurar claves

Copia `.env.example` a `.env` (en la terminal: `copy .env.example .env` en Windows,
`cp .env.example .env` en macOS/Linux) y rellena:

- `JEV_API_KEY`: de https://console.typesafe.ai/keys
- `ANTHROPIC_API_KEY`: de https://platform.claude.com
- `EXCHANGE_API_KEY` y `EXCHANGE_API_SECRET`: Binance (sólo lectura basta para empezar)
- `EXCHANGE_MARKET=swap`, `EXCHANGE_SANDBOX=true` mientras pruebas

`.env` está en `.gitignore`: nunca se sube al repositorio.

## 6. Primeras conexiones reales

```bash
python run_trader.py --check-exchange     # mercados, comisiones, saldo, posiciones
python run_trader.py --check-jev          # una llamada real a Jev
python run_trader.py --mode paper         # datos reales de Binance, fills simulados
```

El modo papel se queda corriendo; lo detienes con `Ctrl+C`. La revisión de cada día:
`python run_trader.py --mode review`. Los informes quedan en `docs/trading/reviews/`.

## Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `python` no se reconoce (Windows) | Python no está en el PATH: reinstala marcando "Add to PATH", o usa `py` en lugar de `python` |
| `ModuleNotFoundError: trading_agent` | La terminal no está en la carpeta del proyecto o el entorno no está activado |
| `ccxt` tarda mucho en instalar | Es normal, es un paquete grande |
| `-2015 Invalid API-key, IP, or permissions` | La clave de Binance tiene restricción de IP y tu IP cambió, o faltan permisos |
| `JEV_API_KEY is empty` | El `.env` no está en la raíz del proyecto o no se ha guardado |
