"""Agente de ayuno intermitente: te escribe al celular cuando abre y cierra tu ventana de comida.

Uso:
    python run_ayuno.py              # envía el aviso que corresponda ahora
    python run_ayuno.py --plan       # muestra los avisos de hoy y mañana (no envía nada)
    python run_ayuno.py --prueba     # envía un mensaje de prueba a tus canales
    python run_ayuno.py --chat-id    # te muestra tu TELEGRAM_CHAT_ID
    python run_ayuno.py --loop       # lo deja corriendo en tu PC (revisa cada minuto)
    python run_ayuno.py --esperar 14 # (GitHub Actions) espera hasta 14 min la hora exacta del próximo aviso
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, time as hora, timedelta

from ayuno import plan
from ayuno.agente import ejecutar
from ayuno.canales import ErrorCanal, chats_de_telegram, crear_canales, enviar_a_todos
from ayuno.config import DESCRIPCION_AVISOS, NOMBRES_DIAS, Config, ErrorConfig
from ayuno.mensajes import crear_redactor, texto_base, texto_prueba


def _salida_github(**valores):
    ruta = os.environ.get("GITHUB_OUTPUT")
    if ruta:
        with open(ruta, "a", encoding="utf-8") as f:
            for clave, valor in valores.items():
                f.write(f"{clave}={valor}\n")


def mostrar_plan(cfg):
    ahora = plan.ahora_utc()
    hoy = ahora.astimezone(cfg.tz).date()
    print(f"Plan {cfg.protocolo}: ventana de {cfg.ventana_inicio:%H:%M} a {cfg.ventana_fin:%H:%M} · {cfg.zona_horaria}")
    print("Días de ayuno:", ", ".join(NOMBRES_DIAS[d] for d in sorted(cfg.dias)))
    print("Canales:", ", ".join(cfg.canales_configurados) or "ninguno (configura Telegram, WhatsApp o ntfy)")
    if cfg.ia_configurada:
        print(f"Redacción: Claude ({cfg.llm_modelo}) reescribe cada mensaje al enviarlo; aquí ves la base.")
    print()
    desde = datetime.combine(hoy, hora(0), tzinfo=cfg.tz)
    for aviso in plan.calendario(cfg, desde, desde + timedelta(days=2)):
        local = aviso.cuando.astimezone(cfg.tz)
        marca = "✓" if aviso.cuando <= ahora else "·"
        print(f"{marca} {NOMBRES_DIAS[local.weekday()]:<9} {local:%d/%m %H:%M}  {DESCRIPCION_AVISOS[aviso.tipo]}")
        print(f"     {texto_base(aviso, cfg, aviso.cuando)}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Agente de ayuno intermitente")
    parser.add_argument("--plan", action="store_true", help="mostrar los avisos de hoy y mañana")
    parser.add_argument("--prueba", action="store_true", help="enviar un mensaje de prueba")
    parser.add_argument("--chat-id", action="store_true", help="mostrar tu TELEGRAM_CHAT_ID")
    parser.add_argument("--loop", action="store_true", help="seguir corriendo y revisar cada minuto")
    parser.add_argument("--esperar", type=int, default=0, metavar="MIN", help="esperar hasta MIN minutos al próximo aviso")
    parser.add_argument("--sin-ia", action="store_true", help="no usar Claude aunque haya API key")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        cfg = Config.desde_entorno()
    except ErrorConfig as e:
        print(f"Error de configuración: {e}", file=sys.stderr)
        return 2

    if args.chat_id:
        if not cfg.telegram_token:
            print("Primero define TELEGRAM_BOT_TOKEN (te lo da @BotFather).", file=sys.stderr)
            return 2
        try:
            chats = chats_de_telegram(cfg.telegram_token)
        except ErrorCanal as e:
            print(f"No pude consultar a Telegram: {e}", file=sys.stderr)
            return 1
        if not chats:
            print("Tu bot no tiene mensajes todavía: ábrelo en Telegram, envíale «hola» y vuelve a ejecutar esto.")
            return 1
        for chat_id, nombre in chats.items():
            print(f"TELEGRAM_CHAT_ID={chat_id}   ({nombre})")
        return 0

    if args.plan:
        return mostrar_plan(cfg)

    canales = crear_canales(cfg)
    if not canales:
        print(
            "No hay ningún canal configurado. Define TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID, "
            "WHATSAPP_TELEFONO + CALLMEBOT_APIKEY o NTFY_TEMA (mira el README).",
            file=sys.stderr,
        )
        return 2

    if args.prueba:
        ahora = plan.ahora_utc()
        ok = enviar_a_todos(canales, texto_prueba(cfg, plan.proximo(cfg, ahora), ahora))
        print("Prueba enviada por:", ", ".join(ok) if ok else "ningún canal (revisa los datos)")
        return 0 if ok else 1

    redactor = None if args.sin_ia else crear_redactor(cfg)
    # En GitHub Actions los registros de un repositorio público son visibles: no mostramos tus horarios.
    detalle = os.environ.get("GITHUB_ACTIONS") != "true"
    while True:
        resultado, estado = ejecutar(
            cfg, esperar_max_min=args.esperar, usar_ia=False, canales=canales, redactor=redactor
        )
        print(time.strftime("%Y-%m-%d %H:%M"), "·", resultado.texto(cfg, detalle), flush=True)
        _salida_github(cambio=str(estado.cambio).lower())
        if not args.loop:
            return 1 if resultado.fallidos else 0
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
