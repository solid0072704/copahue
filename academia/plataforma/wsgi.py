"""Entrada para el servidor web: `gunicorn academia.plataforma.wsgi`."""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "academia.plataforma.settings")
application = get_wsgi_application()
