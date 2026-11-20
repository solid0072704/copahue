"""Contraseñas y sesiones."""

import hashlib
import hmac
import secrets

ITERACIONES = 240_000


def hash_clave(clave):
    sal = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", clave.encode(), bytes.fromhex(sal), ITERACIONES).hex()
    return f"pbkdf2_sha256${ITERACIONES}${sal}${digest}"


def verificar_clave(clave, guardado):
    try:
        _, iteraciones, sal, digest = guardado.split("$")
        calculado = hashlib.pbkdf2_hmac("sha256", clave.encode(), bytes.fromhex(sal), int(iteraciones)).hex()
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(calculado, digest)


def nuevo_token():
    return secrets.token_urlsafe(32)


def hash_token(token):
    return hashlib.sha256(token.encode()).hexdigest()
