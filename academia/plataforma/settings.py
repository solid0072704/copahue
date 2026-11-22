"""Configuración de la academia.

Todo lo que cambia entre el computador de desarrollo y el servidor sale de variables de entorno (ver README):
sin ninguna, la academia corre en modo local con SQLite, correos por consola y la pasarela de prueba.
"""

import os
from pathlib import Path

AQUI = Path(__file__).resolve().parent.parent          # la carpeta academia/
try:                                                    # en un servidor propio, las variables pueden ir en academia/.env
    from dotenv import load_dotenv
    load_dotenv(AQUI / ".env")
except ImportError:
    pass
entorno = os.environ.get


def lista(nombre: str, por_defecto: str = "") -> list[str]:
    return [x.strip() for x in entorno(nombre, por_defecto).split(",") if x.strip()]


DEBUG = entorno("ACADEMIA_DEBUG", "1") == "1"
SECRET_KEY = entorno("ACADEMIA_SECRET_KEY", "solo-para-desarrollo-local")
if not DEBUG and SECRET_KEY == "solo-para-desarrollo-local":
    raise RuntimeError("Falta ACADEMIA_SECRET_KEY: en el servidor la clave secreta no puede ser la de desarrollo.")
ALLOWED_HOSTS = lista("ACADEMIA_HOSTS", "localhost,127.0.0.1,testserver")
CSRF_TRUSTED_ORIGINS = [f"https://{h}" for h in ALLOWED_HOSTS if "." in h and not h[0].isdigit()]
SITIO_URL = entorno("ACADEMIA_URL", "http://localhost:8000").rstrip("/")     # la dirección pública, para avisos de pago y correos

# Lo que se ve
SITIO_NOMBRE = entorno("ACADEMIA_NOMBRE", "Academia Rodrigo Barceló")
SITIO_BAJADA = "Finanzas y contabilidad con inteligencia artificial"

# Precios. Cada curso tiene su precio en dólares; cada alumno lo ve además en la moneda de su país. El cobro se hace
# en la moneda de las pasarelas, al tipo de cambio guardado, que se trae una vez al día de una fuente pública.
MONEDA = entorno("ACADEMIA_MONEDA", "CLP")             # la moneda en que cobran las pasarelas
CAMBIO_URL = entorno("ACADEMIA_CAMBIO_URL", "https://open.er-api.com/v6/latest/USD")
CAMBIO_AUTOMATICO = entorno("ACADEMIA_CAMBIO_AUTOMATICO", "1") == "1"     # con 0, solo `manage.py actualizar_monedas` o el panel

INSTALLED_APPS = [
    "django.contrib.admin", "django.contrib.auth", "django.contrib.contenttypes", "django.contrib.sessions",
    "django.contrib.messages", "whitenoise.runserver_nostatic", "django.contrib.staticfiles",
    "academia.cursos", "academia.pagos",        # antes que allauth: sus plantillas de cuenta reemplazan a las de la librería
    "allauth", "allauth.account", "allauth.socialaccount", "allauth.socialaccount.providers.google",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware", "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware", "django.middleware.locale.LocaleMiddleware",
    "django.middleware.common.CommonMiddleware", "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware", "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware", "allauth.account.middleware.AccountMiddleware",
]
ROOT_URLCONF = "academia.plataforma.urls"
WSGI_APPLICATION = "academia.plataforma.wsgi.application"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates", "DIRS": [], "APP_DIRS": True,
    "OPTIONS": {"context_processors": [
        "django.template.context_processors.request", "django.contrib.auth.context_processors.auth",
        "django.contrib.messages.context_processors.messages", "academia.cursos.contexto.sitio"]},
}]

# Base de datos: SQLite en local; en el servidor, la que diga DATABASE_URL (PostgreSQL)
if entorno("DATABASE_URL"):
    import dj_database_url
    DATABASES = {"default": dj_database_url.parse(entorno("DATABASE_URL"), conn_max_age=600)}
else:
    (AQUI / "datos").mkdir(exist_ok=True)          # la carpeta local de la base de datos y los medios; no va al repositorio
    DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": entorno("ACADEMIA_SQLITE", str(AQUI / "datos" / "academia.sqlite3"))}}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
FORMS_URLFIELD_ASSUME_HTTPS = True

# Cuentas: correo y contraseña, o Google
AUTHENTICATION_BACKENDS = ["django.contrib.auth.backends.ModelBackend", "allauth.account.auth_backends.AuthenticationBackend"]
AUTH_PASSWORD_VALIDATORS = [{"NAME": f"django.contrib.auth.password_validation.{n}"} for n in
                            ("UserAttributeSimilarityValidator", "MinimumLengthValidator", "CommonPasswordValidator", "NumericPasswordValidator")]
ACCOUNT_LOGIN_METHODS = {"email"}
ACCOUNT_SIGNUP_FIELDS = ["email*", "password1*", "password2*"]
ACCOUNT_SIGNUP_FORM_CLASS = "academia.cursos.formularios.Registro"       # agrega el nombre (va en el certificado) y el país (decide la moneda)
ACCOUNT_EMAIL_VERIFICATION = entorno("ACADEMIA_VERIFICAR_CORREO", "optional")   # «mandatory» cuando haya correo saliente configurado
ACCOUNT_USER_DISPLAY = lambda u: u.first_name or u.email                  # noqa: E731
ACCOUNT_SESSION_REMEMBER = True
ACCOUNT_LOGOUT_ON_GET = False
LOGIN_URL = "account_login"
LOGIN_REDIRECT_URL = "mi_aprendizaje"
ACCOUNT_LOGOUT_REDIRECT_URL = "catalogo"
# Google aparece como opción solo cuando están sus credenciales. Quien entra con Google y ya tenía cuenta con ese
# correo queda en la misma cuenta: Google entrega correos verificados.
GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET = entorno("GOOGLE_CLIENT_ID", ""), entorno("GOOGLE_CLIENT_SECRET", "")
CON_GOOGLE = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)
SOCIALACCOUNT_PROVIDERS = {"google": {"APPS": [{"client_id": GOOGLE_CLIENT_ID, "secret": GOOGLE_CLIENT_SECRET, "key": ""}] if CON_GOOGLE else [],
                                      "SCOPE": ["profile", "email"], "AUTH_PARAMS": {"access_type": "online"}}}
SOCIALACCOUNT_LOGIN_ON_GET = True
SOCIALACCOUNT_AUTO_SIGNUP = False         # quien llega por primera vez con Google también pasa por el registro: ahí indica su país
SOCIALACCOUNT_EMAIL_AUTHENTICATION = True
SOCIALACCOUNT_EMAIL_AUTHENTICATION_AUTO_CONNECT = True

# Correo saliente: por consola en local; con ACADEMIA_SMTP_* en el servidor
if entorno("ACADEMIA_SMTP_HOST"):
    EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
    EMAIL_HOST, EMAIL_PORT = entorno("ACADEMIA_SMTP_HOST"), int(entorno("ACADEMIA_SMTP_PORT", "587"))
    EMAIL_HOST_USER, EMAIL_HOST_PASSWORD = entorno("ACADEMIA_SMTP_USUARIO", ""), entorno("ACADEMIA_SMTP_CLAVE", "")
    EMAIL_USE_TLS = True
else:
    EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"
DEFAULT_FROM_EMAIL = entorno("ACADEMIA_REMITENTE", "Academia <no-responder@localhost>")

# Pagos. Cada pasarela se activa con sus credenciales; la de prueba, solo en desarrollo o si se pide a propósito.
FLOW_API_KEY, FLOW_SECRET_KEY = entorno("FLOW_API_KEY", ""), entorno("FLOW_SECRET_KEY", "")
FLOW_URL = entorno("FLOW_URL", "https://sandbox.flow.cl/api")             # producción: https://www.flow.cl/api
MERCADOPAGO_TOKEN, MERCADOPAGO_SECRETO = entorno("MERCADOPAGO_ACCESS_TOKEN", ""), entorno("MERCADOPAGO_WEBHOOK_SECRET", "")
PASARELAS = lista("ACADEMIA_PASARELAS") or ([p for p, activa in (("flow", FLOW_API_KEY and FLOW_SECRET_KEY), ("mercadopago", MERCADOPAGO_TOKEN)) if activa]
                                              + (["prueba"] if DEBUG else []))

# Idioma y hora
LANGUAGE_CODE, TIME_ZONE, USE_I18N, USE_TZ = "es", entorno("ACADEMIA_ZONA", "America/Santiago"), True, True

# Archivos. Los estáticos (hojas de estilo) los sirve la aplicación; los medios (videos, imágenes) van a disco en
# local y, con ACADEMIA_S3_BUCKET, a un almacenamiento compatible con S3 con enlaces firmados que caducan.
STATIC_URL, STATIC_ROOT = "/static/", AQUI / "datos" / "static"
STATIC_ROOT.mkdir(parents=True, exist_ok=True)
MEDIA_URL, MEDIA_ROOT = "/medios/", Path(entorno("ACADEMIA_MEDIOS", str(AQUI / "datos" / "medios")))
STORAGES = {"default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
            "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"}}
MEDIOS_FIRMADOS = bool(entorno("ACADEMIA_S3_BUCKET"))
if MEDIOS_FIRMADOS:
    STORAGES["default"] = {"BACKEND": "storages.backends.s3.S3Storage", "OPTIONS": {
        "bucket_name": entorno("ACADEMIA_S3_BUCKET"), "endpoint_url": entorno("ACADEMIA_S3_ENDPOINT") or None,
        "access_key": entorno("ACADEMIA_S3_CLAVE"), "secret_key": entorno("ACADEMIA_S3_SECRETO"), "region_name": entorno("ACADEMIA_S3_REGION", "auto"),
        "querystring_auth": True, "querystring_expire": 4 * 3600, "file_overwrite": False, "default_acl": None}}
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024

# Seguridad en el servidor
if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_SSL_REDIRECT = SESSION_COOKIE_SECURE = CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = 30 * 24 * 3600
