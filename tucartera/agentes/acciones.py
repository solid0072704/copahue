"""Ejecuta una propuesta aprobada: envía el correo, crea la tarea y aplica sus efectos."""

from .. import calendario as cal
from .. import correo
from ..db import fila, insertar, json_cargar, json_texto, registrar_actividad


class AccionError(Exception):
    pass


def _aplicar_efectos(conn, estudio_id, efectos, hoy):
    for efecto in efectos or []:
        tipo = efecto.get("tipo")
        if tipo == "documentos_solicitados":
            for did in efecto.get("ids", []):
                conn.execute("""
                    UPDATE documentos SET estado = CASE WHEN estado = 'pendiente' THEN 'solicitado' ELSE estado END,
                           solicitado_el = ?, recordatorios = recordatorios + 1
                    WHERE id = ? AND estudio_id = ? AND estado != 'recibido'""", (hoy, did, estudio_id))
        elif tipo == "factura_recordada":
            conn.execute("UPDATE facturas SET recordatorios = recordatorios + 1 WHERE id = ? AND estudio_id = ?",
                         (efecto.get("id"), estudio_id))
        elif tipo == "obligacion_estado":
            conn.execute("UPDATE obligaciones SET estado = ? WHERE id = ? AND estudio_id = ? AND estado != 'presentada'",
                         (efecto.get("estado"), efecto.get("id"), estudio_id))


def ejecutar_propuesta(conn, cfg, estudio, propuesta, usuario=None, cambios=None):
    """Devuelve el texto del resultado. Lanza AccionError si no se pudo (la propuesta sigue pendiente)."""
    if propuesta["estado"] != "pendiente":
        raise AccionError("Esta propuesta ya fue resuelta.")
    datos = json_cargar(propuesta["datos"], {})
    for campo in ("para", "asunto", "cuerpo"):
        if cambios and cambios.get(campo) is not None:
            datos[campo] = str(cambios[campo]).strip()
    ahora = cal.ahora(cfg.tz)
    hoy = ahora[:10]
    resultados = []

    if propuesta["tipo"] == "correo":
        if not datos.get("para") or "@" not in datos["para"]:
            raise AccionError("El correo no tiene un destinatario válido.")
        if not datos.get("asunto") or not datos.get("cuerpo"):
            raise AccionError("Falta el asunto o el cuerpo del correo.")
        estado, detalle = correo.enviar(
            cfg, datos["para"], datos["asunto"], datos["cuerpo"],
            remitente_nombre=estudio["nombre"], responder_a=datos.get("responder_a", ""),
        )
        if estado == "error":
            raise AccionError(detalle)
        insertar(conn, "correos", {
            "estudio_id": estudio["id"], "cliente_id": propuesta["cliente_id"], "para": datos["para"],
            "asunto": datos["asunto"], "cuerpo": datos["cuerpo"], "estado": estado,
            "detalle": detalle, "fecha": ahora,
        })
        resultados.append(f"Correo enviado a {datos['para']}" if estado == "enviado"
                          else f"Correo registrado para {datos['para']} (envío simulado)")

    tarea = datos.get("tarea")
    if tarea:
        insertar(conn, "tareas", {
            "estudio_id": estudio["id"], "cliente_id": propuesta["cliente_id"],
            "titulo": tarea.get("titulo") or propuesta["titulo"], "detalle": tarea.get("detalle", ""),
            "vence": tarea.get("vence"), "responsable_id": tarea.get("responsable_id"),
            "origen": propuesta["agente"], "creada": ahora, "estado": "abierta",
        })
        responsable = fila(conn, "SELECT nombre FROM usuarios WHERE id = ?", tarea.get("responsable_id"))
        resultados.append(f"Tarea asignada a {responsable['nombre']}" if responsable else "Tarea creada")

    _aplicar_efectos(conn, estudio["id"], datos.get("efectos"), hoy)
    resultado = " · ".join(resultados) or "Listo"
    conn.execute("""
        UPDATE propuestas SET estado = 'aprobada', resuelta = ?, resuelta_por = ?, resultado = ?, datos = ?
        WHERE id = ?""", (ahora, usuario["id"] if usuario else None, resultado, json_texto(datos), propuesta["id"]))
    autor = f"usuario:{usuario['id']}" if usuario else f"agente:{propuesta['agente']}"
    registrar_actividad(conn, estudio["id"], ahora, autor, f"{propuesta['titulo']} — {resultado}", propuesta["cliente_id"])
    return resultado


def descartar_propuesta(conn, cfg, estudio, propuesta, usuario):
    if propuesta["estado"] != "pendiente":
        raise AccionError("Esta propuesta ya fue resuelta.")
    conn.execute("""
        UPDATE propuestas SET estado = 'descartada', resuelta = ?, resuelta_por = ?, resultado = 'Descartada'
        WHERE id = ?""", (cal.ahora(cfg.tz), usuario["id"], propuesta["id"]))
