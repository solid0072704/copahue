#!/usr/bin/env python
"""Administración de la academia: `python academia/manage.py <comando>` desde la raíz del repositorio."""
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))        # para importar el paquete `academia`
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "academia.plataforma.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
