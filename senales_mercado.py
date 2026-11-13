"""Señales de mercado por correo: cuándo conviene entrar, mantener o reducir.

Ejecuta tres skills gratuitas (sin claves de API) instaladas en .claude/skills/:

    market-breadth-analyzer   amplitud del mercado de EE.UU. (0-100)
    uptrend-analyzer          proporción de acciones en tendencia alcista (0-100)
    crypto-regime-analyzer    régimen del mercado cripto (0-100)

Con esos puntajes calcula una señal de tres estados para EE.UU. y otra para
cripto, la compara con la de la ejecución anterior y avisa solo cuando cambia
(más un resumen semanal). No compra ni vende nada: solo informa.

Ejemplos:
    python senales_mercado.py                # revisa y avisa si la señal cambió
    python senales_mercado.py --resumen      # además envía el resumen semanal
    python senales_mercado.py --sin-enviar   # solo muestra el resultado en pantalla
    python senales_mercado.py --sin-cripto   # omite cripto (tarda varios minutos)

El aviso sale por Gmail si existen GMAIL_USER y GMAIL_APP_PASSWORD (los mismos
de los agentes del ebook). Si no, y corre en GitHub Actions, abre un issue en el
repositorio, que GitHub te notifica por correo.
"""

import argparse
import html
import json
import os
import re
import smtplib
import subprocess
import sys
from datetime import date, datetime, timezone
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
SKILLS = RAIZ / ".claude" / "skills"
DIR_DATOS = RAIZ / "data" / "senales"

# Estados de la señal, de mejor a peor.
FAVORABLE, NEUTRAL, DEFENSIVO = "FAVORABLE", "NEUTRAL", "DEFENSIVO"
ORDEN = {DEFENSIVO: 0, NEUTRAL: 1, FAVORABLE: 2}

# Límites de puntaje (inferior de NEUTRAL, inferior de FAVORABLE).
# EE.UU.: 40 y 60 son los cortes de zona que usan las dos skills de amplitud.
# Cripto: 40 y 80 son los cortes RISK_OFF / NEUTRAL / RISK_ON de la skill.
LIMITES_EEUU = (40, 60)
LIMITES_CRIPTO = (40, 80)
# Para cambiar de estado el puntaje debe superar el límite por este margen;
# evita correos de ida y vuelta cuando el puntaje oscila justo en el borde.
MARGEN = 2.0
# Con datos más antiguos que esto no se cambia la señal.
MAX_DIAS_DATOS = 5

MERCADOS = {"eeuu": "Mercado de EE.UU.", "cripto": "Cripto"}

ACCION = {
    FAVORABLE: "Conviene entrar",
    NEUTRAL: "Mantener, sin aumentar",
    DEFENSIVO: "Reducir o salir",
}

QUE_HACER = {
    "eeuu": {
        FAVORABLE: (
            "Condiciones favorables para iniciar o aumentar posiciones (por ejemplo, un ETF "
            "del S&P 500). Entra de forma escalonada, no todo de una vez."
        ),
        NEUTRAL: (
            "Señales mixtas. Mantén lo que tienes; si aún no has entrado, espera o entra con "
            "montos pequeños."
        ),
        DEFENSIVO: (
            "Amplitud débil. No abras posiciones nuevas; considera reducir y dejar una parte "
            "mayor en efectivo."
        ),
    },
    "cripto": {
        FAVORABLE: (
            "Régimen de apetito por riesgo amplio. Condiciones favorables para entrar, con "
            "montos acotados: cripto es mucho más volátil que las acciones."
        ),
        NEUTRAL: "Condiciones mixtas, sin una conclusión clara. Mantén; no aumentes exposición.",
        DEFENSIVO: (
            "Régimen defensivo. No abras posiciones nuevas; considera reducir la exposición."
        ),
    },
}

ZONAS_ES = {
    "Strong": "Fuerte", "Healthy": "Saludable", "Neutral": "Neutral",
    "Weakening": "Debilitándose", "Critical": "Crítica",
    "Strong Bull": "Alcista fuerte", "Bull": "Alcista", "Cautious": "Cautelosa",
    "Bear": "Bajista",
    "RISK_ON": "Apetito por riesgo", "NEUTRAL": "Neutral", "RISK_OFF": "Aversión al riesgo",
}

AVISO_LEGAL = (
    "Estas señales son indicadores estadísticos automáticos, no una recomendación "
    "personalizada ni una garantía de resultados. Pueden equivocarse. La decisión de "
    "comprar o vender, y el monto, son tuyos."
)


# --------------------------------------------------------------------------
# Cálculo de la señal (funciones puras, cubiertas por tests/test_senales.py)
# --------------------------------------------------------------------------
def clasificar(puntaje, limites, zona_previa=None, margen=MARGEN):
    """Convierte un puntaje 0-100 en FAVORABLE / NEUTRAL / DEFENSIVO.

    Si hay una zona previa, solo cambia cuando el puntaje cruza el límite por
    más de `margen` puntos (histéresis).
    """
    bajo, alto = limites

    def zona_simple(p):
        if p >= alto:
            return FAVORABLE
        return NEUTRAL if p >= bajo else DEFENSIVO

    nueva = zona_simple(puntaje)
    if zona_previa not in ORDEN or nueva == zona_previa:
        return nueva
    # Al subir se exige más puntaje; al bajar, menos.
    ajuste = -margen if ORDEN[nueva] > ORDEN[zona_previa] else margen
    confirmada = zona_simple(puntaje + ajuste)
    if ORDEN[nueva] > ORDEN[zona_previa]:
        return confirmada if ORDEN[confirmada] > ORDEN[zona_previa] else zona_previa
    return confirmada if ORDEN[confirmada] < ORDEN[zona_previa] else zona_previa


def senal_eeuu(amplitud, tendencia, previa=None):
    """Combina los dos indicadores de EE.UU. (promedio simple de sus puntajes)."""
    indicadores = [i for i in (amplitud, tendencia) if i and i.get("puntaje") is not None]
    if not indicadores:
        return None
    vigentes = [i for i in indicadores if not i.get("desactualizado")]
    if not vigentes:
        return {"zona": (previa or {}).get("zona"), "puntaje": None, "indicadores": indicadores,
                "nota": "Datos desactualizados: se mantiene la señal anterior."}
    puntaje = round(sum(i["puntaje"] for i in vigentes) / len(vigentes), 1)
    senal = {
        "zona": clasificar(puntaje, LIMITES_EEUU, (previa or {}).get("zona")),
        "puntaje": puntaje,
        "indicadores": indicadores,
    }
    if len(vigentes) < 2:
        senal["nota"] = "Señal calculada con un solo indicador (el otro no tuvo datos vigentes)."
    return senal


def senal_cripto(regimen, previa=None):
    if not regimen or regimen.get("puntaje") is None:
        return None
    senal = {
        "zona": clasificar(regimen["puntaje"], LIMITES_CRIPTO, (previa or {}).get("zona")),
        "puntaje": regimen["puntaje"],
        "indicadores": [regimen],
    }
    if regimen.get("componentes"):
        senal["nota"] = f"Calculada con {regimen['componentes']} componentes de datos."
    return senal


def detectar_cambios(senales, estado):
    """Lista de (mercado, zona_anterior, zona_nueva). La primera vez, anterior es None."""
    cambios = []
    for mercado, senal in senales.items():
        if not senal or not senal.get("zona"):
            continue
        anterior = (estado.get(mercado) or {}).get("zona")
        if anterior != senal["zona"]:
            cambios.append((mercado, anterior, senal["zona"]))
    return cambios


def actualizar_estado(estado, senales, hoy, resumen_enviado):
    nuevo = dict(estado)
    for mercado, senal in senales.items():
        if senal and senal.get("zona"):
            nuevo[mercado] = {
                "zona": senal["zona"],
                "puntaje": senal.get("puntaje"),
                "desde": (estado.get(mercado) or {}).get("desde", hoy)
                if (estado.get(mercado) or {}).get("zona") == senal["zona"] else hoy,
            }
    historial = [h for h in estado.get("historial", []) if h.get("fecha") != hoy]
    historial.append({"fecha": hoy, **{m: (s or {}).get("puntaje") for m, s in senales.items()}})
    nuevo["historial"] = historial[-30:]
    nuevo["actualizado"] = hoy
    if resumen_enviado:
        nuevo["ultimo_resumen"] = hoy
    return nuevo


# --------------------------------------------------------------------------
# Redacción del mensaje
# --------------------------------------------------------------------------
def asunto_mensaje(cambios, senales, es_resumen):
    if cambios:
        partes = []
        for mercado, anterior, nueva in cambios:
            verbo = "inicia en" if anterior is None else "pasa a"
            partes.append(f"{MERCADOS[mercado]} {verbo} {nueva} ({ACCION[nueva].lower()})")
        return "Señal de mercado: " + "; ".join(partes)
    estado = ", ".join(
        f"{MERCADOS[m]} {s['zona']}" for m, s in senales.items() if s and s.get("zona")
    )
    return f"Resumen semanal de señales: {estado or 'sin datos'}"


def _fmt(puntaje):
    return "s/d" if puntaje is None else f"{puntaje:.1f}".replace(".", ",")


def secciones_mensaje(cambios, senales, estado, es_resumen):
    """Devuelve el mensaje como lista de (título, [líneas]) para texto y HTML."""
    secciones = []
    if cambios:
        lineas = []
        for mercado, anterior, nueva in cambios:
            if anterior is None:
                lineas.append(f"{MERCADOS[mercado]}: primera señal, {nueva} ({ACCION[nueva]}).")
            else:
                lineas.append(
                    f"{MERCADOS[mercado]}: cambió de {anterior} a {nueva} ({ACCION[nueva]})."
                )
        secciones.append(("Qué cambió", lineas))
    elif es_resumen:
        secciones.append(("Esta semana", ["No hubo cambios de señal desde el último aviso."]))

    historial = estado.get("historial") or []
    referencia = (historial[-7] if len(historial) >= 7 else historial[0]) if historial else {}
    for mercado, senal in senales.items():
        if not senal:
            secciones.append((MERCADOS[mercado], ["Sin datos en esta ejecución."]))
            continue
        zona = senal.get("zona")
        lineas = [f"Señal: {zona} — {ACCION.get(zona, 'sin señal')} (puntaje {_fmt(senal.get('puntaje'))} de 100)."]
        if zona in ORDEN:
            lineas.append(QUE_HACER[mercado][zona])
        previo = referencia.get(mercado)
        if es_resumen and previo is not None and senal.get("puntaje") is not None:
            lineas.append(
                f"Puntaje el {referencia.get('fecha')}: {_fmt(previo)}; hoy: {_fmt(senal['puntaje'])}."
            )
        for ind in senal.get("indicadores", []):
            detalle = f"{ind['nombre']}: {_fmt(ind['puntaje'])} de 100, zona {ZONAS_ES.get(ind.get('zona'), ind.get('zona') or 's/d')}"
            rango = re.search(r"\d+\s*-\s*\d+%|\d+%", ind.get("exposicion") or "")
            if rango:
                detalle += f", exposición sugerida a acciones {rango.group(0)}"
            if ind.get("fecha_datos"):
                detalle += f" (datos al {ind['fecha_datos']})"
            lineas.append(detalle + ".")
        if senal.get("nota"):
            lineas.append(senal["nota"])
        secciones.append((MERCADOS[mercado], lineas))

    secciones.append(("Cómo leer la señal", [
        "FAVORABLE: conviene entrar. NEUTRAL: mantener, sin aumentar. DEFENSIVO: reducir o salir.",
        "EE.UU. es el promedio de dos indicadores de amplitud del mercado (cortes en 40 y 60).",
        AVISO_LEGAL,
    ]))
    return secciones


def mensaje_texto(secciones):
    bloques = []
    for titulo, lineas in secciones:
        bloques.append(titulo.upper() + "\n" + "\n".join(f"- {linea}" for linea in lineas))
    return "\n\n".join(bloques) + "\n"


def mensaje_html(secciones):
    partes = ['<div style="font-family:Arial,sans-serif;font-size:15px;line-height:1.5;color:#222">']
    for titulo, lineas in secciones:
        partes.append(f'<h3 style="margin:18px 0 6px">{html.escape(titulo)}</h3><ul style="margin:0">')
        partes.extend(f"<li>{html.escape(linea)}</li>" for linea in lineas)
        partes.append("</ul>")
    partes.append("</div>")
    return "".join(partes)


def mensaje_markdown(secciones):
    bloques = []
    for titulo, lineas in secciones:
        bloques.append(f"### {titulo}\n" + "\n".join(f"- {linea}" for linea in lineas))
    return "\n\n".join(bloques) + "\n"


# --------------------------------------------------------------------------
# Ejecución de las skills
# --------------------------------------------------------------------------
def _ejecutar(skill, script, argumentos, minutos):
    comando = [sys.executable, str(SKILLS / skill / "scripts" / script), *argumentos]
    try:
        resultado = subprocess.run(
            comando, cwd=DIR_DATOS, capture_output=True, text=True, timeout=minutos * 60
        )
    except subprocess.TimeoutExpired:
        print(f"[{skill}] superó los {minutos} minutos; se omite.")
        return False
    if resultado.returncode != 0:
        cola = (resultado.stdout + resultado.stderr).strip().splitlines()[-3:]
        print(f"[{skill}] falló: " + " | ".join(cola))
        return False
    return True


def _ultimo_json(carpeta, prefijo):
    candidatos = sorted(
        p for p in carpeta.glob(f"{prefijo}*.json") if not p.stem.endswith("history")
    )
    if not candidatos:
        return None
    try:
        return json.loads(candidatos[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _desactualizado(fecha_datos, hoy):
    try:
        return (hoy - date.fromisoformat(str(fecha_datos)[:10])).days > MAX_DIAS_DATOS
    except ValueError:
        return False


def leer_amplitud(carpeta, hoy):
    informe = _ultimo_json(carpeta, "market_breadth_2")
    if not informe:
        return None
    compuesto = informe.get("composite", {})
    fecha = (informe.get("metadata", {}).get("data_freshness") or {}).get("latest_date")
    return {
        "nombre": "Amplitud del mercado",
        "puntaje": compuesto.get("composite_score"),
        "zona": compuesto.get("zone"),
        "exposicion": compuesto.get("exposure_guidance"),
        "fecha_datos": fecha,
        "desactualizado": _desactualizado(fecha, hoy) if fecha else False,
    }


def leer_tendencia(carpeta, hoy):
    informe = _ultimo_json(carpeta, "uptrend_analysis_")
    if not informe:
        return None
    compuesto = informe.get("composite", {})
    fecha = informe.get("metadata", {}).get("latest_data_date")
    return {
        "nombre": "Acciones en tendencia alcista",
        "puntaje": compuesto.get("composite_score"),
        "zona": compuesto.get("zone"),
        "exposicion": compuesto.get("exposure_guidance"),
        "fecha_datos": fecha,
        "desactualizado": _desactualizado(fecha, hoy) if fecha else False,
    }


def leer_cripto(carpeta):
    informe = _ultimo_json(carpeta, "crypto_regime")
    if not informe:
        return None
    compuesto = informe.get("composite", {})
    if compuesto.get("zone") in (None, "UNKNOWN"):
        return None
    return {
        "nombre": "Régimen cripto",
        "puntaje": compuesto.get("score"),
        "zona": compuesto.get("zone"),
        "fecha_datos": str(informe.get("metadata", {}).get("as_of", ""))[:10] or None,
        "componentes": f"{compuesto.get('components_available')} de {compuesto.get('components_total')}",
    }


def recolectar(hoy, con_cripto=True):
    """Ejecuta las skills y devuelve (amplitud, tendencia, cripto); None si alguna falla."""
    carpeta = DIR_DATOS / "informes"
    carpeta.mkdir(parents=True, exist_ok=True)
    for viejo in carpeta.glob("*"):
        if viejo.is_file() and viejo.name != "market_breadth_history.json":
            viejo.unlink()
    salida = ["--output-dir", str(carpeta)]
    _ejecutar("market-breadth-analyzer", "market_breadth_analyzer.py", salida, 5)
    _ejecutar("uptrend-analyzer", "uptrend_analyzer.py", salida, 5)
    if con_cripto:
        cache = ["--cache-dir", str(DIR_DATOS / "cache_cripto"), "--quiet"]
        _ejecutar("crypto-regime-analyzer", "crypto_regime_analyzer.py", salida + cache, 20)
    return (
        leer_amplitud(carpeta, hoy),
        leer_tendencia(carpeta, hoy),
        leer_cripto(carpeta) if con_cripto else None,
    )


# --------------------------------------------------------------------------
# Estado y envío
# --------------------------------------------------------------------------
def cargar_estado(ruta):
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def guardar_estado(ruta, estado):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(estado, ensure_ascii=False, indent=2), encoding="utf-8")


def _env(nombre, defecto=""):
    return (os.environ.get(nombre) or "").strip() or defecto


def enviar_por_gmail(asunto, texto, cuerpo_html):
    usuario = _env("GMAIL_USER")
    clave = _env("GMAIL_APP_PASSWORD").replace(" ", "")
    if not (usuario and clave):
        return False
    msg = EmailMessage()
    msg["From"] = formataddr(("Señales de mercado", usuario))
    msg["To"] = _env("NOTIFY_EMAIL", usuario)
    msg["Subject"] = asunto
    msg.set_content(texto)
    msg.add_alternative(cuerpo_html, subtype="html")
    with smtplib.SMTP_SSL(_env("SMTP_HOST", "smtp.gmail.com"), int(_env("SMTP_PORT", "465")), timeout=30) as smtp:
        smtp.login(usuario, clave)
        smtp.send_message(msg)
    return True


def enviar_por_issue(asunto, markdown):
    """Abre un issue en el repositorio; GitHub lo notifica por correo al dueño."""
    token, repo = _env("GITHUB_TOKEN"), _env("GITHUB_REPOSITORY")
    if not (token and repo):
        return False
    import requests

    duenio = _env("GITHUB_REPOSITORY_OWNER")
    cuerpo = markdown + (f"\ncc @{duenio}\n" if duenio else "")
    respuesta = requests.post(
        f"https://api.github.com/repos/{repo}/issues",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json={"title": asunto, "body": cuerpo},
        timeout=30,
    )
    respuesta.raise_for_status()
    return True


def notificar(asunto, secciones):
    """Devuelve el canal usado: 'gmail', 'issue' o 'pantalla'."""
    try:
        if enviar_por_gmail(asunto, mensaje_texto(secciones), mensaje_html(secciones)):
            return "gmail"
    except (OSError, smtplib.SMTPException) as error:
        print(f"No se pudo enviar por Gmail ({error}); se intenta con un issue de GitHub.")
    if enviar_por_issue(asunto, mensaje_markdown(secciones)):
        return "issue"
    return "pantalla"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Señales de entrada y salida del mercado por correo")
    parser.add_argument("--resumen", action="store_true", help="enviar el resumen semanal aunque no haya cambios")
    parser.add_argument("--sin-enviar", action="store_true", help="no enviar ni guardar nada; solo mostrar")
    parser.add_argument("--sin-cripto", action="store_true", help="omitir la señal de cripto")
    args = parser.parse_args(argv)

    hoy = datetime.now(timezone.utc).date()
    ruta_estado = DIR_DATOS / "estado.json"
    DIR_DATOS.mkdir(parents=True, exist_ok=True)
    estado = cargar_estado(ruta_estado)

    amplitud, tendencia, cripto = recolectar(hoy, con_cripto=not args.sin_cripto)
    senales = {"eeuu": senal_eeuu(amplitud, tendencia, estado.get("eeuu"))}
    if not args.sin_cripto:
        senales["cripto"] = senal_cripto(cripto, estado.get("cripto"))
    if not any(senales.values()):
        print("Ninguna fuente de datos respondió; no se puede calcular la señal.")
        return 1

    cambios = detectar_cambios(senales, estado)
    secciones = secciones_mensaje(cambios, senales, estado, args.resumen)
    asunto = asunto_mensaje(cambios, senales, args.resumen)
    print(asunto + "\n\n" + mensaje_texto(secciones))

    if args.sin_enviar:
        return 0
    if cambios or args.resumen:
        print(f"Aviso enviado por: {notificar(asunto, secciones)}")
    else:
        print("Sin cambios de señal: no se envía aviso.")
    guardar_estado(ruta_estado, actualizar_estado(estado, senales, hoy.isoformat(), args.resumen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
