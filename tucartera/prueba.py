"""Cuenta de prueba persistente para que el dueño use Tu Cartera como un estudio real.

A diferencia del estudio de ejemplo de los visitantes, esta cuenta no se borra: los cambios quedan,
las automatizaciones corren solas cada cierto tiempo y el consumo de IA se mide de verdad.
"""

from . import demo
from .db import fila
from .seguridad import hash_clave


def asegurar_cuenta_prueba(conn, cfg, hoy, email, clave, nombre="", reiniciar=False):
    """Crea (o recrea) el estudio de prueba. Devuelve (estudio_id, creada)."""
    email = email.strip().lower()
    existente = fila(conn, "SELECT * FROM usuarios WHERE email = ?", email)
    if existente and not reiniciar:
        return existente["estudio_id"], False
    if existente:  # las sugerencias anotadas se conservan al reiniciar
        conn.execute("UPDATE comentarios SET estudio_id = 0 WHERE estudio_id = ?", (existente["estudio_id"],))
        demo.eliminar_estudio(conn, existente["estudio_id"])
    eid = demo.crear_demo(conn, cfg, hoy, uso_ilustrativo=False)
    conn.execute("UPDATE estudios SET demo = 0, nombre = 'Estudio de prueba', email = ? WHERE id = ?", (email, eid))
    socio = fila(conn, "SELECT id FROM usuarios WHERE estudio_id = ? AND rol = 'socio' ORDER BY id LIMIT 1", eid)
    cambios = {"email": email, "clave_hash": hash_clave(clave), "clave_definida": 1}
    if nombre:
        cambios["nombre"] = nombre.strip()
    sets = ", ".join(f"{k} = ?" for k in cambios)
    conn.execute(f"UPDATE usuarios SET {sets} WHERE id = ?", (*cambios.values(), socio["id"]))
    conn.execute("UPDATE comentarios SET estudio_id = ?, usuario_id = ? WHERE estudio_id = 0", (eid, socio["id"]))
    return eid, True
