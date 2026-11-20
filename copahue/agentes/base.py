"""Piezas comunes de las automatizaciones.

Cada automatización revisa la cartera y deja *propuestas*: un correo listo para enviar
o una tarea lista para asignar. En modo "revisar" esperan la aprobación de una persona
en la bandeja Para revisar; en modo "automático" se ejecutan solas.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from .. import calendario as cal
from ..db import filas, fila, insertar, json_cargar, json_texto


@dataclass
class Contexto:
    conn: object
    cfg: object
    estudio: dict
    hoy: date
    llm: object = None
    modo: str = "revisar"
    agente: str = ""
    creadas: int = 0
    actualizadas: int = 0
    ejecutadas: int = 0
    claves_vigentes: set = field(default_factory=set)
    _clientes: list = None

    @property
    def estudio_id(self):
        return self.estudio["id"]

    def ahora(self):
        return cal.ahora(self.cfg.tz)

    def clientes(self):
        if self._clientes is None:
            self._clientes = filas(self.conn, """
                SELECT c.*, u.nombre AS responsable_nombre, u.email AS responsable_email
                FROM clientes c LEFT JOIN usuarios u ON u.id = c.responsable_id
                WHERE c.estudio_id = ? AND c.estado != 'pausado'
                ORDER BY c.razon_social""", self.estudio_id)
            for c in self._clientes:
                c["servicios"] = json_cargar(c["servicios"], [])
        return self._clientes

    def socio(self):
        return fila(self.conn, """
            SELECT * FROM usuarios WHERE estudio_id = ? ORDER BY CASE rol WHEN 'socio' THEN 0 ELSE 1 END, id
            LIMIT 1""", self.estudio_id)


class Agente:
    clave = ""
    nombre = ""
    descripcion = ""
    detalle = ""
    icono = ""

    def ejecutar(self, ctx):  # pragma: no cover - interfaz
        raise NotImplementedError

    def resumen(self, ctx):
        partes = []
        if ctx.creadas:
            partes.append(f"{ctx.creadas} nueva{'s' if ctx.creadas != 1 else ''}")
        if ctx.ejecutadas:
            partes.append(f"{ctx.ejecutadas} ejecutada{'s' if ctx.ejecutadas != 1 else ''}")
        if ctx.actualizadas:
            partes.append(f"{ctx.actualizadas} actualizada{'s' if ctx.actualizadas != 1 else ''}")
        return ", ".join(partes) if partes else "Sin novedades"


def primer_nombre(nombre):
    return (nombre or "").split(" ")[0] or "Hola"


def firma(ctx, cliente):
    lineas = [cliente.get("responsable_nombre") or "", ctx.estudio["nombre"]]
    if ctx.estudio.get("firma"):
        lineas.append(ctx.estudio["firma"])
    return "\n".join(l for l in lineas if l)


def clp(monto):
    signo = "-" if monto < 0 else ""
    return f"{signo}${abs(int(monto)):,}".replace(",", ".")


def porcentaje(actual, base):
    if not base:
        return 0
    return round((actual - base) / base * 100)


def proponer(ctx, *, clave, tipo, titulo, datos, cliente_id=None, resumen="", motivos=(),
             prioridad=2, enfriamiento_dias=4):
    """Crea (o refresca) una propuesta. Evita repetir lo que ya se resolvió hace poco."""
    ctx.claves_vigentes.add(clave)
    previa = fila(ctx.conn, """
        SELECT * FROM propuestas WHERE estudio_id = ? AND clave = ? ORDER BY id DESC LIMIT 1""",
        ctx.estudio_id, clave)
    # La huella resume los datos de origen: si no cambian, no se reescribe la propuesta
    # (así se conserva una redacción pulida o editada).
    datos = {**datos, "_huella": json_texto(datos)}
    campos = {
        "titulo": titulo, "resumen": resumen, "motivos": json_texto(list(motivos)),
        "prioridad": prioridad, "cliente_id": cliente_id,
    }
    if previa and previa["estado"] == "pendiente":
        huella_previa = json_cargar(previa["datos"], {}).get("_huella")
        if huella_previa != datos["_huella"] or any(str(previa[k]) != str(v) for k, v in campos.items()):
            campos["datos"] = json_texto(datos)
            sets = ", ".join(f"{k} = ?" for k in campos)
            ctx.conn.execute(f"UPDATE propuestas SET {sets} WHERE id = ?", (*campos.values(), previa["id"]))
            ctx.actualizadas += 1
        return previa["id"]
    if previa and previa["resuelta"]:
        resuelta = datetime.fromisoformat(previa["resuelta"]).date()
        if (ctx.hoy - resuelta) < timedelta(days=enfriamiento_dias):
            return None
    if tipo == "correo" and ctx.llm is not None and ctx.modo != "automatico":
        pulido = ctx.llm.pulir_correo(datos.get("asunto", ""), datos.get("cuerpo", ""), list(motivos))
        if pulido:
            datos["cuerpo"] = pulido
            datos["redaccion_asistida"] = True
    campos["datos"] = json_texto(datos)
    pid = insertar(ctx.conn, "propuestas", {
        "estudio_id": ctx.estudio_id, "agente": ctx.agente, "tipo": tipo, "clave": clave,
        "creada": ctx.ahora(), "estado": "pendiente", **campos,
    })
    ctx.creadas += 1
    if ctx.modo == "automatico":
        from .acciones import AccionError, ejecutar_propuesta

        propuesta = fila(ctx.conn, "SELECT * FROM propuestas WHERE id = ?", pid)
        try:
            ejecutar_propuesta(ctx.conn, ctx.cfg, ctx.estudio, propuesta, usuario=None)
            ctx.ejecutadas += 1
            ctx.creadas -= 1
        except AccionError:
            pass  # queda pendiente para revisión manual
    return pid


def retirar_obsoletas(ctx):
    """Las propuestas pendientes que esta pasada ya no generó dejaron de ser necesarias."""
    pendientes = filas(ctx.conn, """
        SELECT id, clave FROM propuestas WHERE estudio_id = ? AND agente = ? AND estado = 'pendiente'""",
        ctx.estudio_id, ctx.agente)
    for p in pendientes:
        if p["clave"] not in ctx.claves_vigentes:
            ctx.conn.execute("""
                UPDATE propuestas SET estado = 'retirada', resuelta = ?, resultado = ?
                WHERE id = ?""", (ctx.ahora(), "Ya no era necesaria: la situación se resolvió.", p["id"]))
