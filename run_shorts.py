"""Agente de YouTube Shorts desde la terminal, cron o GitHub Actions.

Ejemplos:
    python run_shorts.py auth              # autoriza tu cuenta de YouTube (una sola vez, en tu PC)
    python run_shorts.py                   # ciclo completo: explorar → cortar → subir
    python run_shorts.py --loop 180        # un ciclo cada 3 horas
    python run_shorts.py explorar          # solo mira qué está viral y qué seleccionaría
    python run_shorts.py cortar --video dQw4w9WgXcQ   # recorta un video que elijas tú
    python run_shorts.py subir --simular   # muestra qué subiría, sin subir nada
    python run_shorts.py estado            # resumen y últimas anotaciones
"""

import argparse
import logging
import sys
import time

from agentes_shorts import Config, YouTube, abrir_db, ejecutar_ciclo, seleccionar_manual

ACCIONES = ("ciclo", "explorar", "cortar", "subir", "auth", "estado")


def mostrar_estado(db, cfg):
    r = db.resumen()
    print(f"Videos: {r['videos'] or 'ninguno'}")
    print(f"Cortes: {r['cortes'] or 'ninguno'}")
    print(f"Cuota de la API usada hoy: {r['cuota_hoy']}/{cfg.cuota_diaria} · subidos hoy: {r['subidos_hoy']}/{cfg.max_subidas_dia}")
    print("\nCandidatos con mejor score:")
    for v in db.videos(estado=("descubierto", "seleccionado", "cortado"), limite=10):
        print(f"  {v['score']:3d}  {v['estado']:<12} {v['video_id']}  {v['titulo'][:60]}  ({v['canal']}, {v['vistas']:,} vistas)")
    print("\nÚltimas anotaciones:")
    for a in reversed(db.bitacora(12)):
        print(f"  {a['fecha'][:16]} [{a['agente']}] {a['mensaje'][:140]}")


def main():
    parser = argparse.ArgumentParser(description="Agente de YouTube Shorts: explora virales, recorta y publica")
    parser.add_argument("accion", nargs="?", default="ciclo", choices=ACCIONES, help="qué hacer (por defecto: ciclo)")
    parser.add_argument("--loop", type=int, metavar="MIN", help="repetir cada MIN minutos")
    parser.add_argument("--video", metavar="ID", help="con `cortar`: ID o URL de un video elegido por ti")
    parser.add_argument("--simular", action="store_true", help="no subir nada a YouTube (solo mostrar)")
    parser.add_argument("--sin-ia", action="store_true", help="no usar Claude aunque haya API key")
    parser.add_argument("--privacidad", choices=("private", "unlisted", "public"), help="sobrescribe SHORTS_PRIVACIDAD")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
    cfg = Config.desde_entorno()
    if args.privacidad:
        cfg.privacidad = args.privacidad
        cfg.normalizar()
    db = abrir_db(cfg)
    yt = YouTube(cfg, db)

    try:
        if args.accion == "auth":
            yt.credenciales(interactivo=True)
            print(f"✅ Cuenta autorizada. Token guardado en {cfg.token}")
            return 0
        if args.accion == "estado":
            mostrar_estado(db, cfg)
            return 0
        if cfg.licencia == "cualquiera":
            print("⚠️  SHORTS_LICENCIA=cualquiera: reutilizar videos ajenos sin permiso infringe derechos de autor "
                  "y las Condiciones de YouTube; espera reclamos de Content ID. Úsalo solo con contenido autorizado.")
        if not cfg.lectura_configurada and args.accion in ("ciclo", "explorar", "cortar"):
            print("YouTube no está configurado: define YOUTUBE_API_KEY o ejecuta `python run_shorts.py auth`.")
            if args.accion != "cortar" or not args.video:
                return 1

        etapas = {"ciclo": ("explorar", "cortar", "subir"), "explorar": ("explorar",), "cortar": ("cortar",), "subir": ("subir",)}[args.accion]
        if args.video:
            video_id = args.video.split("v=")[-1].split("/")[-1].split("&")[0].split("?")[0]
            v = seleccionar_manual(db, yt, cfg, video_id)
            print(f"Agregado «{v['titulo']}» ({v['canal']}, {v['vistas']:,} vistas, score {v['score']})")
        while True:
            resultado = ejecutar_ciclo(db, cfg, yt, etapas=etapas, simular=args.simular, usar_ia=not args.sin_ia)
            print(time.strftime("%Y-%m-%d %H:%M"), "·", resultado.texto(), flush=True)
            if args.accion == "explorar":
                for v in db.videos(estado=("seleccionado", "descubierto"), limite=8):
                    print(f"  {v['score']:3d}  {v['estado']:<12} https://youtu.be/{v['video_id']}  {v['titulo'][:60]}")
            for e in resultado.errores:
                print("  ⚠️ ", e)
            if not args.loop:
                return 1 if resultado.errores and not resultado.cortes and not (resultado.publicacion and resultado.publicacion.subidos) else 0
            time.sleep(args.loop * 60)
    finally:
        db.cerrar()


if __name__ == "__main__":
    sys.exit(main())
