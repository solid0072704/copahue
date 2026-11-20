"""Inicio de sesión con Google y Microsoft (OpenID Connect, flujo de código de autorización).

El servidor canjea el código directamente con el proveedor por HTTPS, así que el id_token llega
desde la fuente (OIDC Core 3.1.3.7). Igual se validan emisor, audiencia, vencimiento y nonce.
Solo se pide nombre y correo: Tu Cartera no lee el correo ni los archivos de la cuenta.
"""

import base64
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

PROVEEDORES = {
    "google": {
        "nombre": "Google",
        "autorizar": "https://accounts.google.com/o/oauth2/v2/auth",
        "token": "https://oauth2.googleapis.com/token",
        "emisor": r"^(https://)?accounts\.google\.com$",
        "parametros": {"prompt": "select_account"},
    },
    "microsoft": {
        "nombre": "Microsoft",
        "autorizar": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "emisor": r"^https://login\.microsoftonline\.com/[0-9a-fA-F-]{36}/v2\.0$",
        "parametros": {"prompt": "select_account", "response_mode": "query"},
    },
}
ALCANCE = "openid email profile"


class ErrorOAuth(Exception):
    pass


def credenciales(cfg, proveedor):
    if proveedor == "google":
        return cfg.google_client_id, cfg.google_client_secret
    if proveedor == "microsoft":
        return cfg.microsoft_client_id, cfg.microsoft_client_secret
    return "", ""


def configurado(cfg, proveedor):
    cid, secreto = credenciales(cfg, proveedor)
    return bool(proveedor in PROVEEDORES and cid and secreto)


def url_autorizacion(cfg, proveedor, redirect_uri, state, nonce):
    p = PROVEEDORES[proveedor]
    cid, _ = credenciales(cfg, proveedor)
    parametros = {
        "client_id": cid, "redirect_uri": redirect_uri, "response_type": "code", "scope": ALCANCE,
        "state": state, "nonce": nonce, **p["parametros"],
    }
    return f"{p['autorizar']}?{urllib.parse.urlencode(parametros)}"


def _decodificar_jwt(token):
    try:
        carga = token.split(".")[1]
        carga += "=" * (-len(carga) % 4)
        return json.loads(base64.urlsafe_b64decode(carga))
    except (IndexError, ValueError) as e:
        raise ErrorOAuth("La respuesta del proveedor no es válida.") from e


def canjear_codigo(cfg, proveedor, codigo, redirect_uri):
    """Canjea el código por tokens y devuelve los datos (claims) del id_token."""
    cid, secreto = credenciales(cfg, proveedor)
    cuerpo = urllib.parse.urlencode({
        "code": codigo, "client_id": cid, "client_secret": secreto,
        "redirect_uri": redirect_uri, "grant_type": "authorization_code",
    }).encode()
    pedido = urllib.request.Request(PROVEEDORES[proveedor]["token"], data=cuerpo,
                                    headers={"Content-Type": "application/x-www-form-urlencoded",
                                             "Accept": "application/json"})
    try:
        with urllib.request.urlopen(pedido, timeout=15) as r:
            datos = json.loads(r.read())
    except urllib.error.HTTPError as e:
        log.warning("%s rechazó el canje del código: %s", proveedor, e.read()[:300])
        raise ErrorOAuth("No pudimos validar tu cuenta. Inténtalo de nuevo.") from e
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise ErrorOAuth("No pudimos comunicarnos con el proveedor. Inténtalo de nuevo.") from e
    if "id_token" not in datos:
        raise ErrorOAuth("El proveedor no entregó los datos de la cuenta.")
    return _decodificar_jwt(datos["id_token"])


def perfil(cfg, proveedor, claims, nonce, ahora=None):
    """Valida el id_token y devuelve {sub, email, email_verificado, nombre}."""
    ahora = ahora or time.time()
    cid, _ = credenciales(cfg, proveedor)
    audiencia = claims.get("aud")
    audiencias = audiencia if isinstance(audiencia, list) else [audiencia]
    if cid not in audiencias:
        raise ErrorOAuth("La cuenta no corresponde a esta aplicación.")
    if not re.match(PROVEEDORES[proveedor]["emisor"], str(claims.get("iss", ""))):
        raise ErrorOAuth("El emisor de la cuenta no es válido.")
    if float(claims.get("exp", 0)) < ahora - 60:
        raise ErrorOAuth("El acceso expiró. Inténtalo de nuevo.")
    if claims.get("nonce") != nonce:
        raise ErrorOAuth("La solicitud no es válida. Inténtalo de nuevo.")
    if not claims.get("sub"):
        raise ErrorOAuth("El proveedor no entregó un identificador de cuenta.")
    email = (claims.get("email") or claims.get("preferred_username") or "").strip().lower()
    if "@" not in email:
        raise ErrorOAuth("Tu cuenta no tiene un correo asociado.")
    # Google confirma la propiedad del correo; en Microsoft el correo puede no estar verificado.
    verificado = proveedor == "google" and claims.get("email_verified") in (True, "true")
    nombre = (claims.get("name") or " ".join(filter(None, [claims.get("given_name"), claims.get("family_name")]))
              or email.split("@")[0]).strip()
    return {"sub": str(claims["sub"]), "email": email, "email_verificado": verificado, "nombre": nombre[:120]}
