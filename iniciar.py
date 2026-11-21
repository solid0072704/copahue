"""Inicia Tu Cartera en este computador.

Instala lo necesario, crea la cuenta de prueba y abre el navegador en http://localhost:8000.
Uso: doble clic en «Iniciar Tu Cartera» (Windows o Mac) o, en una terminal, `python iniciar.py`.
Con `python iniciar.py --red` también se puede abrir desde un celular conectado a la misma red Wi-Fi.
"""

import os
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

PUERTO = 8000


def ip_local():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no envía datos: solo elige la interfaz de red
            return s.getsockname()[0]
    except OSError:
        return ""


def main():
    os.chdir(Path(__file__).resolve().parent)
    if sys.version_info < (3, 11):
        print("Tu Cartera necesita Python 3.11 o superior: https://www.python.org/downloads/")
        input("Presiona Enter para salir.")
        sys.exit(1)
    print("Preparando Tu Cartera (la primera vez tarda un par de minutos)…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                           "-r", "requirements-tucartera.txt"])
    os.environ.setdefault("TUCARTERA_PRUEBA_EMAIL", "prueba@tucartera.cl")
    os.environ.setdefault("TUCARTERA_PRUEBA_CLAVE", "prueba1234")
    en_red = "--red" in sys.argv
    url = f"http://localhost:{PUERTO}"
    print(f"\n  Tu Cartera está en {url}")
    print(f"  Correo: {os.environ['TUCARTERA_PRUEBA_EMAIL']} · Contraseña: {os.environ['TUCARTERA_PRUEBA_CLAVE']}")
    if en_red and ip_local():
        print(f"  Desde tu celular (misma red Wi-Fi): http://{ip_local()}:{PUERTO}")
    print("  Para cerrarla, cierra esta ventana o presiona Ctrl+C.\n")
    threading.Timer(3, lambda: webbrowser.open(url)).start()
    try:
        subprocess.call([sys.executable, "-m", "tucartera", "servir",
                         "--host", "0.0.0.0" if en_red else "127.0.0.1", "--puerto", str(PUERTO)])
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
