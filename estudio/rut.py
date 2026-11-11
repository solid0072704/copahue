"""RUT chileno: validación, normalización y formato."""

import re


class RutInvalido(ValueError):
    pass


def digito_verificador(cuerpo):
    """Dígito verificador (módulo 11) para el cuerpo numérico de un RUT."""
    suma, factor = 0, 2
    for d in reversed(str(int(cuerpo))):
        suma += int(d) * factor
        factor = 2 if factor == 7 else factor + 1
    resto = 11 - suma % 11
    return {11: "0", 10: "K"}.get(resto, str(resto))


def _partes(rut):
    limpio = re.sub(r"[^0-9kK]", "", str(rut or "")).upper()
    if len(limpio) < 2 or not limpio[:-1].isdigit():
        return None
    return limpio[:-1].lstrip("0"), limpio[-1]


def validar(rut):
    partes = _partes(rut)
    if not partes or not partes[0]:
        return False
    cuerpo, dv = partes
    return 6 <= len(cuerpo) <= 8 and digito_verificador(cuerpo) == dv


def normalizar(rut):
    """Devuelve el RUT como '76123456-7' o lanza RutInvalido."""
    if not validar(rut):
        raise RutInvalido(f"El RUT «{rut}» no es válido (revisa el dígito verificador).")
    cuerpo, dv = _partes(rut)
    return f"{cuerpo}-{dv}"


def formatear(rut):
    """'76123456-7' → '76.123.456-7'. Si no es válido, devuelve el texto tal cual."""
    if not validar(rut):
        return str(rut or "")
    cuerpo, dv = _partes(rut)
    return f"{int(cuerpo):,}".replace(",", ".") + f"-{dv}"
