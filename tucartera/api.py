"""API web de Tu Cartera y servidor de la aplicación."""

import logging
import mimetypes
import os
import re
import threading
import unicodedata
from urllib.parse import quote
import time as _time
from datetime import date, datetime, timedelta

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import agentes, asistente, demo, llm, oauth, planes
from . import calendario as cal
from .agentes.acciones import AccionError, descartar_propuesta, ejecutar_propuesta
from .agentes.documentos import asegurar_documentos
from .agentes.vencimientos import generar_obligaciones
from .config import cargar
from .db import conectar, fila, filas, inicializar, insertar, json_cargar, json_texto, registrar_actividad, valor
from .salud import evaluar_cartera
from .seguridad import hash_clave, hash_token, nuevo_token, verificar_clave

log = logging.getLogger(__name__)
mimetypes.add_type("application/manifest+json", ".webmanifest")
WEB = os.path.join(os.path.dirname(__file__), "web")
COOKIE = "tucartera_sesion"
COOKIE_OAUTH = "tucartera_oauth"
COLORES = ["#0a84ff", "#ff9f0a", "#30d158", "#ff375f", "#64d2ff", "#ac8e68", "#5e5ce6", "#ffd60a"]
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def orden(texto):
    """Clave de orden alfabético que ignora tildes (Óptica va con la O)."""
    return unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode().lower()


def crear_app(cfg=None):
    cfg = cfg or cargar()
    inicializar(cfg.db)
    claude = llm.crear(cfg)
    app = FastAPI(title="Tu Cartera", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.cfg = cfg

    def hoy():
        return date.fromisoformat(cfg.hoy_fijo) if cfg.hoy_fijo else datetime.now(cfg.tz).date()

    def ia_para(estudio):
        """En los estudios de ejemplo la IA queda apagada (salvo TUCARTERA_DEMO_IA) para no gastar tokens."""
        return None if estudio.get("demo") and not cfg.demo_ia else claude

    def resumen_ia(conn, estudio, usuario_id):
        if ia_para(estudio) is None:
            return {"activa": False}
        return {"activa": True, **planes.cupo(conn, estudio, usuario_id, hoy())}

    def db():
        with conectar(cfg.db) as conn:
            yield conn

    # La transacción se confirma antes de enviar la respuesta.
    DB = Depends(db, scope="function")

    # --- Seguridad ---------------------------------------------------------------------------
    @app.middleware("http")
    async def proteger(request: Request, call_next):
        # Las mutaciones exigen un encabezado propio: un formulario de otro sitio no puede enviarlo.
        if request.url.path.startswith("/api/") and request.method not in ("GET", "HEAD", "OPTIONS"):
            if request.headers.get("x-tucartera") != "1":
                return JSONResponse({"detail": "Solicitud no permitida."}, status_code=403)
        respuesta = await call_next(request)
        respuesta.headers["X-Content-Type-Options"] = "nosniff"
        respuesta.headers["Referrer-Policy"] = "same-origin"
        respuesta.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            respuesta.headers["Cache-Control"] = "no-store"
        return respuesta

    def sesion(request: Request, conn=DB):
        token = request.cookies.get(COOKIE)
        if not token:
            raise HTTPException(401, "Inicia sesión para continuar.")
        datos = fila(conn, """
            SELECT u.*, s.expira FROM sesiones s JOIN usuarios u ON u.id = s.usuario_id WHERE s.token_hash = ?""",
            hash_token(token))
        if not datos or datos["expira"] < cal.ahora(cfg.tz):
            raise HTTPException(401, "Tu sesión expiró. Inicia sesión nuevamente.")
        estudio = fila(conn, "SELECT * FROM estudios WHERE id = ?", datos["estudio_id"])
        return {"usuario": datos, "estudio": estudio, "conn": conn}

    def abrir_sesion(conn, respuesta, usuario_id):
        token = nuevo_token()
        expira = (datetime.now(cfg.tz) + timedelta(days=cfg.sesion_dias)).isoformat(timespec="seconds")
        conn.execute("INSERT INTO sesiones (token_hash, usuario_id, expira) VALUES (?, ?, ?)",
                     (hash_token(token), usuario_id, expira))
        respuesta.set_cookie(COOKIE, token, max_age=cfg.sesion_dias * 86400, httponly=True,
                             samesite="lax", secure=cfg.cookie_segura, path="/")

    # --- Autenticación -----------------------------------------------------------------------
    class Ingreso(BaseModel):
        email: str = Field(max_length=200)
        clave: str = Field(max_length=200)

    class Registro(BaseModel):
        estudio: str = Field(min_length=2, max_length=120)
        nombre: str = Field(min_length=2, max_length=120)
        email: str = Field(max_length=200)
        clave: str = Field(min_length=8, max_length=200)

    @app.post("/api/auth/ingresar")
    def ingresar(datos: Ingreso, respuesta: Response, conn=DB):
        usuario = fila(conn, "SELECT * FROM usuarios WHERE email = ?", datos.email.strip().lower())
        if not usuario or not verificar_clave(datos.clave, usuario["clave_hash"]):
            raise HTTPException(401, "El correo o la contraseña no coinciden.")
        abrir_sesion(conn, respuesta, usuario["id"])
        return {"ok": True}

    @app.post("/api/auth/registro")
    def registrar(datos: Registro, respuesta: Response, conn=DB):
        email = datos.email.strip().lower()
        if not EMAIL.match(email):
            raise HTTPException(422, "Revisa el correo electrónico.")
        if fila(conn, "SELECT id FROM usuarios WHERE email = ?", email):
            raise HTTPException(409, "Ya existe una cuenta con ese correo.")
        ahora = cal.ahora(cfg.tz)
        eid = insertar(conn, "estudios", {"nombre": datos.estudio.strip(), "email": email, "creado": ahora,
                                          **planes.iniciar_prueba(hoy())})
        uid = insertar(conn, "usuarios", {
            "estudio_id": eid, "nombre": datos.nombre.strip(), "email": email, "clave_hash": hash_clave(datos.clave),
            "rol": "socio", "cargo": "Socio", "color": COLORES[0], "creado": ahora,
        })
        agentes.asegurar_configuracion(conn, eid)
        abrir_sesion(conn, respuesta, uid)
        return {"ok": True}

    @app.post("/api/auth/demo")
    def entrar_demo(respuesta: Response, conn=DB):
        demo.limpiar_demos_antiguas(conn, hoy())
        eid = demo.crear_demo(conn, cfg, hoy())
        usuario = fila(conn, "SELECT id FROM usuarios WHERE estudio_id = ? AND rol = 'socio' ORDER BY id LIMIT 1", eid)
        abrir_sesion(conn, respuesta, usuario["id"])
        return {"ok": True}

    @app.post("/api/auth/salir")
    def salir(request: Request, respuesta: Response, conn=DB):
        token = request.cookies.get(COOKIE)
        if token:
            conn.execute("DELETE FROM sesiones WHERE token_hash = ?", (hash_token(token),))
        respuesta.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    # --- Google y Microsoft ------------------------------------------------------------------
    def usuario_de_cookie(request, conn):
        token = request.cookies.get(COOKIE)
        if not token:
            return None
        u = fila(conn, """SELECT u.* FROM sesiones s JOIN usuarios u ON u.id = s.usuario_id
                          WHERE s.token_hash = ? AND s.expira >= ?""", hash_token(token), cal.ahora(cfg.tz))
        return u

    def retorno_de(request, proveedor):
        base = cfg.url_publica or str(request.base_url).rstrip("/")
        return f"{base}/api/auth/{proveedor}/retorno"

    def volver(destino, aviso="", error=False):
        """Redirige a la app; el mensaje se muestra como aviso (o como error en la pantalla de ingreso)."""
        consulta = f"?{'error_acceso' if error else 'aviso'}={quote(aviso)}" if aviso else ""
        r = RedirectResponse(f"/{consulta}{destino}", status_code=303)
        r.delete_cookie(COOKIE_OAUTH, path="/api/auth")
        return r

    @app.get("/api/auth/proveedores")
    def proveedores():
        return {p: {"nombre": d["nombre"], "activo": oauth.configurado(cfg, p)} for p, d in oauth.PROVEEDORES.items()}

    @app.get("/api/auth/{proveedor}/iniciar")
    def oauth_iniciar(proveedor: str, request: Request, modo: str = "ingresar", conn=DB):
        if proveedor not in oauth.PROVEEDORES:
            raise HTTPException(404, "Proveedor desconocido.")
        nombre = oauth.PROVEEDORES[proveedor]["nombre"]
        conectar = modo == "conectar"
        if not oauth.configurado(cfg, proveedor):
            return volver("#/ajustes" if conectar else "", f"El acceso con {nombre} aún no está activado en este servidor.",
                          error=not conectar)
        usuario = usuario_de_cookie(request, conn) if conectar else None
        if conectar and not usuario:
            return volver("", "Inicia sesión para conectar tu cuenta.", error=True)
        if conectar and fila(conn, "SELECT demo FROM estudios WHERE id = ?", usuario["estudio_id"])["demo"]:
            return volver("#/ajustes", "En el estudio de ejemplo no se conectan cuentas reales. Crea tu cuenta para hacerlo.")
        state, nonce = nuevo_token(), nuevo_token()
        conn.execute("DELETE FROM oauth_estados WHERE expira < ?", (cal.ahora(cfg.tz),))
        conn.execute("INSERT INTO oauth_estados (state, proveedor, nonce, modo, usuario_id, expira) VALUES (?, ?, ?, ?, ?, ?)",
                     (state, proveedor, nonce, "conectar" if conectar else "ingresar", usuario["id"] if usuario else None,
                      (datetime.now(cfg.tz) + timedelta(minutes=10)).isoformat(timespec="seconds")))
        r = RedirectResponse(oauth.url_autorizacion(cfg, proveedor, retorno_de(request, proveedor), state, nonce), status_code=303)
        r.set_cookie(COOKIE_OAUTH, state, max_age=600, httponly=True, samesite="lax", secure=cfg.cookie_segura, path="/api/auth")
        return r

    @app.get("/api/auth/{proveedor}/retorno")
    def oauth_retorno(proveedor: str, request: Request, code: str = "", state: str = "", error: str = "", conn=DB):
        if proveedor not in oauth.PROVEEDORES:
            raise HTTPException(404, "Proveedor desconocido.")
        nombre = oauth.PROVEEDORES[proveedor]["nombre"]
        pendiente = fila(conn, "SELECT * FROM oauth_estados WHERE state = ? AND proveedor = ?", state, proveedor) if state else None
        if pendiente:
            conn.execute("DELETE FROM oauth_estados WHERE state = ?", (state,))
        conectar = bool(pendiente and pendiente["modo"] == "conectar")
        destino_error = "#/ajustes" if conectar else ""
        if error:
            return volver(destino_error, f"No se completó el acceso con {nombre}.", error=not conectar)
        if (not pendiente or pendiente["expira"] < cal.ahora(cfg.tz) or request.cookies.get(COOKIE_OAUTH) != state):
            return volver("", "La solicitud de acceso expiró o no es válida. Inténtalo de nuevo.", error=True)
        try:
            claims = oauth.canjear_codigo(cfg, proveedor, code, retorno_de(request, proveedor))
            datos = oauth.perfil(cfg, proveedor, claims, pendiente["nonce"])
        except oauth.ErrorOAuth as e:
            return volver(destino_error, str(e), error=not conectar)
        ahora = cal.ahora(cfg.tz)
        vinculada = fila(conn, "SELECT * FROM identidades WHERE proveedor = ? AND sub = ?", proveedor, datos["sub"])

        if conectar:
            if vinculada and vinculada["usuario_id"] != pendiente["usuario_id"]:
                return volver("#/ajustes", f"Esa cuenta de {nombre} ya está conectada a otro usuario.")
            if not vinculada:
                conn.execute("DELETE FROM identidades WHERE usuario_id = ? AND proveedor = ?", (pendiente["usuario_id"], proveedor))
                insertar(conn, "identidades", {"usuario_id": pendiente["usuario_id"], "proveedor": proveedor,
                                               "sub": datos["sub"], "email": datos["email"], "creado": ahora})
            return volver("#/ajustes", f"Cuenta de {nombre} conectada: {datos['email']}")

        usuario_id = vinculada["usuario_id"] if vinculada else None
        if not usuario_id:
            existente = fila(conn, "SELECT id FROM usuarios WHERE email = ?", datos["email"])
            if existente and not datos["email_verificado"]:
                return volver("", f"Ya existe una cuenta con {datos['email']}. Ingresa con tu contraseña y conecta "
                                  f"{nombre} desde Ajustes.", error=True)
            if existente:
                usuario_id = existente["id"]
            else:  # primera vez: se crea el estudio con su prueba gratis
                primer_nombre = datos["nombre"].split(" ")[0]
                eid = insertar(conn, "estudios", {"nombre": f"Estudio de {primer_nombre}", "email": datos["email"],
                                                  "creado": ahora, **planes.iniciar_prueba(hoy())})
                usuario_id = insertar(conn, "usuarios", {
                    "estudio_id": eid, "nombre": datos["nombre"], "email": datos["email"],
                    "clave_hash": hash_clave(nuevo_token()), "clave_definida": 0, "rol": "socio", "cargo": "Socio",
                    "color": COLORES[0], "creado": ahora,
                })
                agentes.asegurar_configuracion(conn, eid)
            insertar(conn, "identidades", {"usuario_id": usuario_id, "proveedor": proveedor, "sub": datos["sub"],
                                           "email": datos["email"], "creado": ahora})
        r = volver("#/hoy")
        abrir_sesion(conn, r, usuario_id)
        return r

    @app.post("/api/cuenta/identidades/{iid}/desconectar")
    def desconectar_identidad(iid: int, s=Depends(sesion)):
        conn, u = s["conn"], s["usuario"]
        ident = fila(conn, "SELECT * FROM identidades WHERE id = ? AND usuario_id = ?", iid, u["id"])
        if not ident:
            raise HTTPException(404, "No encontramos esa conexión.")
        otras = valor(conn, "SELECT COUNT(*) FROM identidades WHERE usuario_id = ? AND id != ?", u["id"], iid)
        if not u["clave_definida"] and not otras:
            raise HTTPException(409, "Es tu única forma de ingresar. Conecta otra cuenta antes de desconectar esta.")
        conn.execute("DELETE FROM identidades WHERE id = ?", (iid,))
        return {"ok": True}

    # --- Utilidades de presentación ----------------------------------------------------------
    def equipo_de(conn, eid):
        return {u["id"]: {"id": u["id"], "nombre": u["nombre"], "color": u["color"], "cargo": u["cargo"]}
                for u in filas(conn, "SELECT id, nombre, color, cargo FROM usuarios WHERE estudio_id = ?", eid)}

    def cliente_de(conn, eid, cid):
        cliente = fila(conn, "SELECT * FROM clientes WHERE id = ? AND estudio_id = ?", cid, eid)
        if not cliente:
            raise HTTPException(404, "No encontramos ese cliente.")
        cliente["servicios"] = json_cargar(cliente["servicios"], [])
        return cliente

    def obligacion_vista(o, hoy_):
        tipo = cal.TIPOS.get(o["tipo"], {})
        vista = {**o, "nombre": tipo.get("nombre", o["tipo"]), "descripcion": tipo.get("descripcion", ""),
                 "organismo": tipo.get("organismo", ""), "periodo_nombre": cal.nombre_periodo(o["periodo"]),
                 "atrasada": o["estado"] != "presentada" and o["vence"] < hoy_.isoformat()}
        return vista

    def propuesta_vista(p, clientes, equipo):
        datos = json_cargar(p["datos"], {})
        datos.pop("_huella", None)
        agente = agentes.POR_CLAVE.get(p["agente"])
        cliente = clientes.get(p["cliente_id"]) if p["cliente_id"] else None
        tarea = datos.get("tarea")
        if tarea and tarea.get("responsable_id") in equipo:
            tarea["responsable"] = equipo[tarea["responsable_id"]]
        return {
            "id": p["id"], "agente": p["agente"],
            "agente_nombre": agente.nombre if agente else "Instrucción",
            "agente_icono": agente.icono if agente else "comando",
            "tipo": p["tipo"], "titulo": p["titulo"], "resumen": p["resumen"],
            "motivos": json_cargar(p["motivos"], []), "prioridad": p["prioridad"], "datos": datos,
            "estado": p["estado"], "creada": p["creada"], "resuelta": p["resuelta"], "resultado": p["resultado"],
            "resuelta_por": equipo.get(p["resuelta_por"]),
            "cliente": {"id": cliente["id"], "nombre": cliente["nombre_corto"] or cliente["razon_social"],
                        "razon_social": cliente["razon_social"]} if cliente else None,
        }

    def mapa_clientes(conn, eid):
        return {c["id"]: c for c in filas(conn, "SELECT id, razon_social, nombre_corto FROM clientes WHERE estudio_id = ?", eid)}

    def autor_vista(autor, equipo):
        tipo, _, ident = autor.partition(":")
        if tipo == "usuario":
            persona = equipo.get(int(ident)) if ident.isdigit() else None
            return {"tipo": "persona", "nombre": persona["nombre"] if persona else "Alguien del equipo",
                    "color": persona["color"] if persona else "#8e8e93"}
        agente = agentes.POR_CLAVE.get(ident)
        return {"tipo": "automatizacion", "nombre": agente.nombre if agente else "Automatización",
                "icono": agente.icono if agente else "comando"}

    # --- Sesión y panel ----------------------------------------------------------------------
    @app.get("/api/sesion")
    def ver_sesion(s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        u = s["usuario"]
        return {
            "usuario": {k: u[k] for k in ("id", "nombre", "email", "rol", "cargo", "color")},
            "estudio": {k: s["estudio"][k] for k in ("id", "nombre", "rut", "email", "telefono", "firma", "demo")},
            "para_revisar": valor(conn, "SELECT COUNT(*) FROM propuestas WHERE estudio_id = ? AND estado = 'pendiente'", eid),
            "integraciones": {"correo": cfg.smtp_configurado, "redaccion": ia_para(s["estudio"]) is not None},
            "suscripcion": planes.suscripcion(s["estudio"], hoy()),
            "ia": resumen_ia(conn, s["estudio"], u["id"]),
            "hoy": hoy().isoformat(),
        }

    @app.get("/api/hoy")
    def panel(s=Depends(sesion)):
        conn, eid, u = s["conn"], s["estudio"]["id"], s["usuario"]
        h = hoy()
        equipo = equipo_de(conn, eid)
        clientes = mapa_clientes(conn, eid)
        salud = evaluar_cartera(conn, eid, h)
        hasta = h + timedelta(days=45)
        obligaciones = [obligacion_vista(o, h) for o in filas(conn, """
            SELECT o.*, c.razon_social, c.nombre_corto, c.responsable_id FROM obligaciones o
            JOIN clientes c ON c.id = o.cliente_id
            WHERE o.estudio_id = ? AND o.vence <= ? AND (o.estado != 'presentada' OR o.vence >= ?)
            ORDER BY o.vence, c.razon_social""", eid, hasta.isoformat(), h.isoformat())]
        dias = {}
        for o in obligaciones:
            clave = h.isoformat() if o["atrasada"] else o["vence"]
            dias.setdefault(clave, []).append(o)
        semana = sum(1 for o in obligaciones if o["estado"] != "presentada" and o["vence"] <= cal.sumar_habiles(h, 5).isoformat())
        agenda = []
        for d in sorted(dias)[:3]:
            fecha = date.fromisoformat(d)
            items = dias[d]
            agenda.append({
                "fecha": d, "etiqueta": "Hoy" if fecha == h else "Mañana" if fecha == h + timedelta(days=1)
                else cal.DIAS[fecha.weekday()].capitalize(),
                "detalle": cal.fecha_larga(fecha, con_dia=False),
                "total": len(items), "listas": sum(1 for o in items if o["estado"] == "presentada"),
                "items": items,
            })
        pendientes = filas(conn, """
            SELECT * FROM propuestas WHERE estudio_id = ? AND estado = 'pendiente'
            ORDER BY prioridad, creada DESC LIMIT 5""", eid)
        periodo = cal.periodo_cerrado(h)
        docs = fila(conn, """
            SELECT COUNT(*) AS total, SUM(estado = 'recibido') AS recibidos FROM documentos
            WHERE estudio_id = ? AND periodo = ?""", eid, periodo)
        cobro = fila(conn, """
            SELECT COALESCE(SUM(monto), 0) AS por_cobrar, COALESCE(SUM(CASE WHEN vence < ? THEN monto END), 0) AS vencido
            FROM facturas WHERE estudio_id = ? AND pagada_el IS NULL""", h.isoformat(), eid)
        riesgo = sorted(
            ({"cliente": {"id": cid, "nombre": clientes[cid]["nombre_corto"] or clientes[cid]["razon_social"]}, **v}
             for cid, v in salud.items() if v["nivel"] != "ok" and cid in clientes),
            key=lambda x: x["puntaje"])[:5]
        actividad = filas(conn, "SELECT * FROM actividad WHERE estudio_id = ? ORDER BY fecha DESC, id DESC LIMIT 8", eid)
        tareas = filas(conn, """
            SELECT t.*, c.nombre_corto, c.razon_social FROM tareas t LEFT JOIN clientes c ON c.id = t.cliente_id
            WHERE t.estudio_id = ? AND t.estado = 'abierta' AND t.responsable_id = ?
            ORDER BY COALESCE(t.vence, '9999'), t.id LIMIT 6""", eid, u["id"])
        hora = datetime.now(cfg.tz).hour
        return {
            "saludo": "Buenos días" if hora < 12 else "Buenas tardes" if hora < 20 else "Buenas noches",
            "fecha": f"{cal.fecha_larga(h)} de {h.year}",
            "periodo": cal.nombre_periodo(periodo),
            "kpis": {
                "para_revisar": valor(conn, "SELECT COUNT(*) FROM propuestas WHERE estudio_id = ? AND estado = 'pendiente'", eid),
                "vencen": semana,
                "atrasadas": sum(1 for o in obligaciones if o["atrasada"]),
                "documentos_total": docs["total"] or 0, "documentos_recibidos": docs["recibidos"] or 0,
                "por_cobrar": cobro["por_cobrar"], "vencido": cobro["vencido"],
                "clientes": len(salud), "en_riesgo": sum(1 for v in salud.values() if v["nivel"] == "riesgo"),
            },
            "agenda": agenda,
            "destacadas": [propuesta_vista(p, clientes, equipo) for p in pendientes],
            "riesgo": riesgo,
            "actividad": [{**a, "autor": autor_vista(a["autor"], equipo)} for a in actividad],
            "tareas": tareas,
            "automatizaciones": [{"clave": a["clave"], "ultima_ejecucion": a["ultima_ejecucion"],
                                  "ultimo_resultado": a["ultimo_resultado"]}
                                 for a in filas(conn, "SELECT * FROM automatizaciones WHERE estudio_id = ?", eid)],
        }

    # --- Clientes ----------------------------------------------------------------------------
    class ClienteDatos(BaseModel):
        razon_social: str = Field(min_length=2, max_length=160)
        nombre_corto: str = Field("", max_length=80)
        rut: str = Field("", max_length=20)
        giro: str = Field("", max_length=160)
        regimen: str = Field("14D3", max_length=10)
        servicios: list[str] = Field(default_factory=lambda: ["contabilidad", "iva"])
        trabajadores: int = Field(0, ge=0, le=100000)
        contacto_nombre: str = Field("", max_length=120)
        contacto_email: str = Field("", max_length=200)
        contacto_telefono: str = Field("", max_length=40)
        comuna: str = Field("", max_length=80)
        responsable_id: int | None = None
        honorario: int = Field(0, ge=0, le=1_000_000_000)
        notas: str = Field("", max_length=4000)

    class ClienteCambios(BaseModel):
        razon_social: str | None = Field(None, min_length=2, max_length=160)
        nombre_corto: str | None = Field(None, max_length=80)
        giro: str | None = Field(None, max_length=160)
        regimen: str | None = Field(None, max_length=10)
        servicios: list[str] | None = None
        trabajadores: int | None = Field(None, ge=0, le=100000)
        contacto_nombre: str | None = Field(None, max_length=120)
        contacto_email: str | None = Field(None, max_length=200)
        contacto_telefono: str | None = Field(None, max_length=40)
        comuna: str | None = Field(None, max_length=80)
        responsable_id: int | None = None
        honorario: int | None = Field(None, ge=0, le=1_000_000_000)
        estado: str | None = None
        notas: str | None = Field(None, max_length=4000)

    def validar_cliente(conn, eid, datos):
        if "regimen" in datos and datos["regimen"] not in cal.REGIMENES:
            raise HTTPException(422, "Régimen tributario no válido.")
        if "servicios" in datos:
            datos["servicios"] = [x for x in datos["servicios"] if x in cal.SERVICIOS]
        if datos.get("contacto_email") and not EMAIL.match(datos["contacto_email"].strip()):
            raise HTTPException(422, "Revisa el correo del contacto.")
        if datos.get("responsable_id") and not fila(conn, "SELECT id FROM usuarios WHERE id = ? AND estudio_id = ?",
                                                      datos["responsable_id"], eid):
            raise HTTPException(422, "Esa persona no pertenece al estudio.")
        if "estado" in datos and datos["estado"] not in ("activo", "incorporacion", "pausado"):
            raise HTTPException(422, "Estado no válido.")

    @app.get("/api/clientes")
    def listar_clientes(s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        h = hoy()
        equipo = equipo_de(conn, eid)
        salud = evaluar_cartera(conn, eid, h)
        periodo = cal.periodo_cerrado(h)
        proximas = {}
        for o in filas(conn, """
            SELECT cliente_id, tipo, periodo, vence, estado FROM obligaciones
            WHERE estudio_id = ? AND estado != 'presentada' ORDER BY vence""", eid):
            proximas.setdefault(o["cliente_id"], obligacion_vista(o, h))
        faltan = {d["cliente_id"]: d["n"] for d in filas(conn, """
            SELECT cliente_id, COUNT(*) AS n FROM documentos WHERE estudio_id = ? AND periodo = ? AND estado != 'recibido'
            GROUP BY cliente_id""", eid, periodo)}
        deuda = {f["cliente_id"]: f["total"] for f in filas(conn, """
            SELECT cliente_id, SUM(monto) AS total FROM facturas WHERE estudio_id = ? AND pagada_el IS NULL AND vence < ?
            GROUP BY cliente_id""", eid, h.isoformat())}
        resultado = []
        for c in filas(conn, "SELECT * FROM clientes WHERE estudio_id = ? ORDER BY razon_social", eid):
            c["servicios"] = json_cargar(c["servicios"], [])
            resultado.append({
                **{k: c[k] for k in ("id", "razon_social", "nombre_corto", "rut", "giro", "regimen", "servicios",
                                     "trabajadores", "honorario", "estado", "comuna", "contacto_nombre")},
                "responsable": equipo.get(c["responsable_id"]),
                "salud": salud.get(c["id"], {"puntaje": None, "nivel": "pausado", "motivos": []}),
                "proximo": proximas.get(c["id"]),
                "documentos_faltan": faltan.get(c["id"], 0),
                "deuda": deuda.get(c["id"], 0),
            })
        resultado.sort(key=lambda c: orden(c["razon_social"]))
        return {"clientes": resultado, "periodo": cal.nombre_periodo(periodo),
                "regimenes": cal.REGIMENES, "servicios": cal.SERVICIOS, "equipo": list(equipo.values())}

    @app.post("/api/clientes")
    def crear_cliente(datos: ClienteDatos, s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        maximo = planes.suscripcion(s["estudio"], hoy())["max_clientes"]
        if maximo is not None and valor(conn, "SELECT COUNT(*) FROM clientes WHERE estudio_id = ?", eid) >= maximo:
            raise HTTPException(402, f"Tu plan permite hasta {maximo} clientes. Cambia de plan para agregar más.")
        valores = datos.model_dump()
        validar_cliente(conn, eid, valores)
        h = hoy()
        valores["servicios"] = json_texto(valores["servicios"])
        valores["responsable_id"] = valores["responsable_id"] or s["usuario"]["id"]
        cid = insertar(conn, "clientes", {
            **valores, "estudio_id": eid, "estado": "activo", "cliente_desde": h.isoformat(),
            "honorario_desde": h.isoformat(), "creado": cal.ahora(cfg.tz),
        })
        cliente = cliente_de(conn, eid, cid)
        generar_obligaciones(conn, eid, cliente, h)
        asegurar_documentos(conn, eid, cliente, cal.periodo_cerrado(h))
        registrar_actividad(conn, eid, cal.ahora(cfg.tz), f"usuario:{s['usuario']['id']}",
                            f"Nuevo cliente: {cliente['razon_social']}", cid)
        return {"id": cid}

    @app.get("/api/clientes/{cid}")
    def ver_cliente(cid: int, s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        h = hoy()
        cliente = cliente_de(conn, eid, cid)
        equipo = equipo_de(conn, eid)
        clientes = mapa_clientes(conn, eid)
        periodo = cal.periodo_cerrado(h)
        obligaciones = [obligacion_vista(o, h) for o in filas(conn, """
            SELECT * FROM obligaciones WHERE cliente_id = ? ORDER BY vence DESC LIMIT 24""", cid)]
        return {
            "cliente": {**cliente, "responsable": equipo.get(cliente["responsable_id"]),
                        "regimen_nombre": cal.REGIMENES.get(cliente["regimen"], cliente["regimen"])},
            "salud": evaluar_cartera(conn, eid, h).get(cid, {"puntaje": None, "nivel": "pausado", "motivos": []}),
            "periodo": periodo, "periodo_nombre": cal.nombre_periodo(periodo),
            "obligaciones": obligaciones,
            "documentos": filas(conn, "SELECT * FROM documentos WHERE cliente_id = ? AND periodo = ? ORDER BY id", cid, periodo),
            "facturas": filas(conn, "SELECT * FROM facturas WHERE cliente_id = ? ORDER BY emitida DESC LIMIT 12", cid),
            "indicadores": filas(conn, """
                SELECT * FROM (SELECT * FROM indicadores WHERE cliente_id = ? ORDER BY periodo DESC LIMIT 12)
                ORDER BY periodo""", cid),
            "propuestas": [propuesta_vista(p, clientes, equipo) for p in filas(conn, """
                SELECT * FROM propuestas WHERE cliente_id = ? AND estudio_id = ? AND estado = 'pendiente'
                ORDER BY prioridad, creada DESC""", cid, eid)],
            "correos": filas(conn, "SELECT * FROM correos WHERE cliente_id = ? AND estudio_id = ? ORDER BY fecha DESC LIMIT 10", cid, eid),
            "tareas": filas(conn, """SELECT * FROM tareas WHERE cliente_id = ? AND estudio_id = ? AND estado = 'abierta'
                                     ORDER BY COALESCE(vence, '9999')""", cid, eid),
            "actividad": [{**a, "autor": autor_vista(a["autor"], equipo)} for a in filas(conn, """
                SELECT * FROM actividad WHERE cliente_id = ? AND estudio_id = ? ORDER BY fecha DESC LIMIT 12""", cid, eid)],
            "regimenes": cal.REGIMENES, "servicios": cal.SERVICIOS, "equipo": list(equipo.values()),
        }

    @app.patch("/api/clientes/{cid}")
    def editar_cliente(cid: int, cambios: ClienteCambios, s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        anterior = cliente_de(conn, eid, cid)
        datos = cambios.model_dump(exclude_unset=True)
        validar_cliente(conn, eid, datos)
        if "honorario" in datos and datos["honorario"] != anterior["honorario"]:
            datos["honorario_desde"] = hoy().isoformat()
        if "servicios" in datos:
            datos["servicios"] = json_texto(datos["servicios"])
        if datos:
            sets = ", ".join(f"{k} = ?" for k in datos)
            conn.execute(f"UPDATE clientes SET {sets} WHERE id = ? AND estudio_id = ?", (*datos.values(), cid, eid))
            cliente = cliente_de(conn, eid, cid)
            generar_obligaciones(conn, eid, cliente, hoy())
        return {"ok": True}

    # --- Vencimientos ------------------------------------------------------------------------
    class CambioEstado(BaseModel):
        estado: str

    @app.get("/api/vencimientos")
    def vencimientos(mes: str = "", s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        h = hoy()
        mes = mes if re.fullmatch(r"\d{4}-\d{2}", mes or "") else cal.periodo_de(h)
        inicio = f"{mes}-01"
        fin = f"{cal.mover_periodo(mes, 1)}-01"
        equipo = equipo_de(conn, eid)
        datos = filas(conn, """
            SELECT o.*, c.razon_social, c.nombre_corto, c.responsable_id FROM obligaciones o JOIN clientes c ON c.id = o.cliente_id
            WHERE o.estudio_id = ? AND ((o.vence >= ? AND o.vence < ?) OR (o.estado != 'presentada' AND o.vence < ?))
            ORDER BY o.vence, o.tipo, c.razon_social""", eid, inicio, fin, h.isoformat())
        vistas = []
        for o in datos:
            v = obligacion_vista(o, h)
            v["responsable"] = equipo.get(o["responsable_id"])
            vistas.append(v)
        anio, m = int(mes[:4]), int(mes[5:7])
        return {
            "mes": mes, "titulo": f"{cal.MESES[m - 1].capitalize()} {anio}", "hoy": h.isoformat(),
            "anterior": cal.mover_periodo(mes, -1), "siguiente": cal.mover_periodo(mes, 1),
            "obligaciones": vistas, "tipos": cal.TIPOS,
            "feriados": sorted(d.isoformat() for d in cal.FERIADOS if d.year == anio and d.month == m),
        }

    @app.patch("/api/obligaciones/{oid}")
    def cambiar_obligacion(oid: int, cambio: CambioEstado, s=Depends(sesion)):
        conn, eid, u = s["conn"], s["estudio"]["id"], s["usuario"]
        if cambio.estado not in ("pendiente", "preparacion", "revision", "presentada"):
            raise HTTPException(422, "Estado no válido.")
        o = fila(conn, "SELECT o.*, c.nombre_corto, c.razon_social FROM obligaciones o JOIN clientes c ON c.id = o.cliente_id "
                       "WHERE o.id = ? AND o.estudio_id = ?", oid, eid)
        if not o:
            raise HTTPException(404, "No encontramos esa obligación.")
        presentada = hoy().isoformat() if cambio.estado == "presentada" else None
        conn.execute("UPDATE obligaciones SET estado = ?, presentada_el = ? WHERE id = ?", (cambio.estado, presentada, oid))
        if cambio.estado == "presentada":
            registrar_actividad(conn, eid, cal.ahora(cfg.tz), f"usuario:{u['id']}",
                                f"{cal.TIPOS[o['tipo']]['nombre']} de {cal.nombre_periodo(o['periodo'])} presentado · "
                                f"{o['nombre_corto'] or o['razon_social']}", o["cliente_id"])
        return {"ok": True}

    # --- Documentos --------------------------------------------------------------------------
    @app.get("/api/documentos")
    def documentos(s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        h = hoy()
        periodo = cal.periodo_cerrado(h)
        equipo = equipo_de(conn, eid)
        grupos = {}
        for d in filas(conn, """
            SELECT d.*, c.razon_social, c.nombre_corto, c.responsable_id, c.contacto_nombre FROM documentos d
            JOIN clientes c ON c.id = d.cliente_id WHERE d.estudio_id = ? AND d.periodo = ?
            ORDER BY c.razon_social, d.id""", eid, periodo):
            g = grupos.setdefault(d["cliente_id"], {
                "cliente": {"id": d["cliente_id"], "nombre": d["nombre_corto"] or d["razon_social"],
                            "contacto": d["contacto_nombre"]},
                "responsable": equipo.get(d["responsable_id"]), "documentos": []})
            g["documentos"].append({k: d[k] for k in ("id", "tipo", "estado", "solicitado_el", "recibido_el", "recordatorios")})
        lista = list(grupos.values())
        for g in lista:
            g["recibidos"] = sum(1 for d in g["documentos"] if d["estado"] == "recibido")
            g["total"] = len(g["documentos"])
        lista.sort(key=lambda g: (g["recibidos"] == g["total"], -max(d["recordatorios"] for d in g["documentos"]),
                                  orden(g["cliente"]["nombre"])))
        vence = cal.vencimiento("F29", periodo)
        return {"periodo": periodo, "periodo_nombre": cal.nombre_periodo(periodo, True), "grupos": lista,
                "f29_vence": vence.isoformat(), "f29_vence_texto": cal.fecha_larga(vence)}

    @app.patch("/api/documentos/{did}")
    def cambiar_documento(did: int, cambio: CambioEstado, s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        if cambio.estado not in ("pendiente", "solicitado", "recibido"):
            raise HTTPException(422, "Estado no válido.")
        d = fila(conn, "SELECT * FROM documentos WHERE id = ? AND estudio_id = ?", did, eid)
        if not d:
            raise HTTPException(404, "No encontramos ese documento.")
        conn.execute("UPDATE documentos SET estado = ?, recibido_el = ? WHERE id = ?",
                     (cambio.estado, hoy().isoformat() if cambio.estado == "recibido" else None, did))
        return {"ok": True}

    # --- Cobranza ----------------------------------------------------------------------------
    class Pago(BaseModel):
        pagada: bool

    @app.get("/api/cobranza")
    def cobranza(s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        h = hoy()
        facturas = filas(conn, """
            SELECT f.*, c.razon_social, c.nombre_corto FROM facturas f JOIN clientes c ON c.id = f.cliente_id
            WHERE f.estudio_id = ? AND (f.pagada_el IS NULL OR f.emitida >= ?)
            ORDER BY (f.pagada_el IS NOT NULL), f.vence""", eid, (h - timedelta(days=75)).isoformat())
        for f in facturas:
            f["dias_atraso"] = (h - date.fromisoformat(f["vence"])).days if not f["pagada_el"] else 0
            f["estado"] = "pagada" if f["pagada_el"] else "vencida" if f["dias_atraso"] > 0 else "por_vencer"
            f["periodo_nombre"] = cal.nombre_periodo(f["periodo"], True)
        mes = cal.periodo_de(h)
        resumen = fila(conn, """
            SELECT COALESCE(SUM(CASE WHEN pagada_el IS NULL AND vence < ? THEN monto END), 0) AS vencido,
                   COALESCE(SUM(CASE WHEN pagada_el IS NULL AND vence >= ? THEN monto END), 0) AS por_vencer,
                   COALESCE(SUM(CASE WHEN substr(pagada_el, 1, 7) = ? THEN monto END), 0) AS cobrado_mes,
                   COALESCE(SUM(CASE WHEN substr(emitida, 1, 7) = ? THEN monto END), 0) AS emitido_mes
            FROM facturas WHERE estudio_id = ?""", h.isoformat(), h.isoformat(), mes, mes, eid)
        historico = filas(conn, """
            SELECT substr(emitida, 1, 7) AS mes, SUM(monto) AS emitido,
                   SUM(CASE WHEN pagada_el IS NOT NULL THEN monto ELSE 0 END) AS cobrado
            FROM facturas WHERE estudio_id = ? GROUP BY 1 ORDER BY 1 DESC LIMIT 6""", eid)
        for x in historico:
            x["nombre"] = cal.nombre_periodo(x["mes"])[:3]
        mensual = valor(conn, "SELECT COALESCE(SUM(honorario), 0) FROM clientes WHERE estudio_id = ? AND estado = 'activo'", eid)
        return {"facturas": facturas, "resumen": {**resumen, "cartera_mensual": mensual},
                "historico": list(reversed(historico))}

    @app.patch("/api/facturas/{fid}")
    def marcar_pago(fid: int, pago: Pago, s=Depends(sesion)):
        conn, eid, u = s["conn"], s["estudio"]["id"], s["usuario"]
        f = fila(conn, "SELECT f.*, c.nombre_corto, c.razon_social FROM facturas f JOIN clientes c ON c.id = f.cliente_id "
                       "WHERE f.id = ? AND f.estudio_id = ?", fid, eid)
        if not f:
            raise HTTPException(404, "No encontramos esa factura.")
        conn.execute("UPDATE facturas SET pagada_el = ? WHERE id = ?", (hoy().isoformat() if pago.pagada else None, fid))
        if pago.pagada:
            registrar_actividad(conn, eid, cal.ahora(cfg.tz), f"usuario:{u['id']}",
                                f"Pago registrado: factura N.º {f['folio']} · {f['nombre_corto'] or f['razon_social']}",
                                f["cliente_id"])
        return {"ok": True}

    # --- Para revisar ------------------------------------------------------------------------
    class Aprobacion(BaseModel):
        para: str | None = Field(None, max_length=200)
        asunto: str | None = Field(None, max_length=300)
        cuerpo: str | None = Field(None, max_length=20000)

    @app.get("/api/propuestas")
    def propuestas(estado: str = "pendiente", s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        equipo = equipo_de(conn, eid)
        clientes = mapa_clientes(conn, eid)
        if estado == "pendiente":
            datos = filas(conn, """SELECT * FROM propuestas WHERE estudio_id = ? AND estado = 'pendiente'
                                   ORDER BY prioridad, creada DESC, id DESC""", eid)
        else:
            datos = filas(conn, """SELECT * FROM propuestas WHERE estudio_id = ? AND estado != 'pendiente'
                                   ORDER BY resuelta DESC LIMIT 60""", eid)
        return {"propuestas": [propuesta_vista(p, clientes, equipo) for p in datos],
                "agentes": {a.clave: {"nombre": a.nombre, "icono": a.icono} for a in agentes.AGENTES}}

    def propuesta_de(conn, eid, pid):
        p = fila(conn, "SELECT * FROM propuestas WHERE id = ? AND estudio_id = ?", pid, eid)
        if not p:
            raise HTTPException(404, "No encontramos esa propuesta.")
        return p

    @app.post("/api/propuestas/{pid}/aprobar")
    def aprobar(pid: int, cambios: Aprobacion, s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        p = propuesta_de(conn, eid, pid)
        if cambios.para and not EMAIL.match(cambios.para.strip()):
            raise HTTPException(422, "Revisa el destinatario.")
        try:
            resultado = ejecutar_propuesta(conn, cfg, s["estudio"], p, usuario=s["usuario"], cambios=cambios.model_dump())
        except AccionError as e:
            raise HTTPException(409, str(e))
        return {"ok": True, "resultado": resultado}

    @app.post("/api/propuestas/{pid}/descartar")
    def descartar(pid: int, s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        try:
            descartar_propuesta(conn, cfg, s["estudio"], propuesta_de(conn, eid, pid), s["usuario"])
        except AccionError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    # --- Automatizaciones --------------------------------------------------------------------
    class AjusteAutomatizacion(BaseModel):
        activo: bool | None = None
        modo: str | None = None

    class Ejecucion(BaseModel):
        clave: str | None = None

    @app.get("/api/automatizaciones")
    def automatizaciones(s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        agentes.asegurar_configuracion(conn, eid)
        config = {a["clave"]: a for a in filas(conn, "SELECT * FROM automatizaciones WHERE estudio_id = ?", eid)}
        desde = (datetime.now(cfg.tz) - timedelta(days=30)).isoformat(timespec="seconds")
        conteo = {}
        for r in filas(conn, """
            SELECT agente, estado, COUNT(*) AS n FROM propuestas WHERE estudio_id = ? AND (estado = 'pendiente' OR creada >= ?)
            GROUP BY agente, estado""", eid, desde):
            conteo.setdefault(r["agente"], {})[r["estado"]] = r["n"]
        lista = []
        for a in agentes.AGENTES:
            c = config[a.clave]
            n = conteo.get(a.clave, {})
            lista.append({"clave": a.clave, "nombre": a.nombre, "descripcion": a.descripcion, "detalle": a.detalle,
                          "icono": a.icono, "activo": bool(c["activo"]), "modo": c["modo"],
                          "ultima_ejecucion": c["ultima_ejecucion"], "ultimo_resultado": c["ultimo_resultado"],
                          "pendientes": n.get("pendiente", 0), "aprobadas_30d": n.get("aprobada", 0),
                          "descartadas_30d": n.get("descartada", 0)})
        return {"automatizaciones": lista,
                "integraciones": {"correo": cfg.smtp_configurado, "redaccion": ia_para(s["estudio"]) is not None},
                "ia": resumen_ia(conn, s["estudio"], s["usuario"]["id"]),
                "vigente": planes.suscripcion(s["estudio"], hoy())["vigente"],
                "intervalo_min": cfg.intervalo_min}

    @app.patch("/api/automatizaciones/{clave}")
    def ajustar_automatizacion(clave: str, ajuste: AjusteAutomatizacion, s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        if clave not in agentes.POR_CLAVE:
            raise HTTPException(404, "No existe esa automatización.")
        agentes.asegurar_configuracion(conn, eid)
        if ajuste.modo is not None:
            if ajuste.modo not in agentes.MODOS:
                raise HTTPException(422, "Modo no válido.")
            conn.execute("UPDATE automatizaciones SET modo = ? WHERE estudio_id = ? AND clave = ?", (ajuste.modo, eid, clave))
        if ajuste.activo is not None:
            conn.execute("UPDATE automatizaciones SET activo = ? WHERE estudio_id = ? AND clave = ?",
                         (int(ajuste.activo), eid, clave))
        return {"ok": True}

    @app.post("/api/automatizaciones/ejecutar")
    def ejecutar_ahora(pedido: Ejecucion, s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        if pedido.clave and pedido.clave not in agentes.POR_CLAVE:
            raise HTTPException(404, "No existe esa automatización.")
        if not planes.suscripcion(s["estudio"], hoy())["vigente"]:
            raise HTTPException(402, "Tu suscripción no está activa. Las automatizaciones vuelven al renovar el plan.")
        resultados = agentes.ejecutar(conn, cfg, eid, hoy(), claves=[pedido.clave] if pedido.clave else None,
                                      llm=ia_para(s["estudio"]), forzar=bool(pedido.clave))
        return {"resultados": resultados}

    # --- Barra de instrucciones --------------------------------------------------------------
    class Instruccion(BaseModel):
        texto: str = Field(max_length=600)

    @app.post("/api/instruccion")
    def instruccion(pedido: Instruccion, s=Depends(sesion)):
        return asistente.responder(s["conn"], cfg, s["estudio"], s["usuario"], pedido.texto, hoy(), ia_para(s["estudio"]))

    # --- Tareas y actividad ------------------------------------------------------------------
    class TareaNueva(BaseModel):
        titulo: str = Field(min_length=2, max_length=200)
        detalle: str = Field("", max_length=4000)
        cliente_id: int | None = None
        responsable_id: int | None = None
        vence: str | None = None

    @app.get("/api/tareas")
    def tareas(s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        equipo = equipo_de(conn, eid)
        lista = filas(conn, """
            SELECT t.*, c.nombre_corto, c.razon_social FROM tareas t LEFT JOIN clientes c ON c.id = t.cliente_id
            WHERE t.estudio_id = ? AND (t.estado = 'abierta' OR t.hecha_el >= ?)
            ORDER BY t.estado, COALESCE(t.vence, '9999'), t.id""", eid, (hoy() - timedelta(days=7)).isoformat())
        for t in lista:
            t["responsable"] = equipo.get(t["responsable_id"])
            agente = agentes.POR_CLAVE.get(t["origen"])
            t["origen_nombre"] = agente.nombre if agente else ("Instrucción" if t["origen"] == "asistente" else "")
        return {"tareas": lista, "equipo": list(equipo.values())}

    @app.post("/api/tareas")
    def crear_tarea(datos: TareaNueva, s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        if datos.cliente_id:
            cliente_de(conn, eid, datos.cliente_id)
        responsable = datos.responsable_id or s["usuario"]["id"]
        if not fila(conn, "SELECT id FROM usuarios WHERE id = ? AND estudio_id = ?", responsable, eid):
            raise HTTPException(422, "Esa persona no pertenece al estudio.")
        tid = insertar(conn, "tareas", {
            "estudio_id": eid, "cliente_id": datos.cliente_id, "titulo": datos.titulo.strip(), "detalle": datos.detalle,
            "vence": datos.vence, "responsable_id": responsable, "estado": "abierta", "origen": "manual",
            "creada": cal.ahora(cfg.tz),
        })
        return {"id": tid}

    @app.patch("/api/tareas/{tid}")
    def cambiar_tarea(tid: int, cambio: CambioEstado, s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        if cambio.estado not in ("abierta", "hecha"):
            raise HTTPException(422, "Estado no válido.")
        if not fila(conn, "SELECT id FROM tareas WHERE id = ? AND estudio_id = ?", tid, eid):
            raise HTTPException(404, "No encontramos esa tarea.")
        conn.execute("UPDATE tareas SET estado = ?, hecha_el = ? WHERE id = ?",
                     (cambio.estado, hoy().isoformat() if cambio.estado == "hecha" else None, tid))
        return {"ok": True}

    @app.get("/api/actividad")
    def actividad(s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        equipo = equipo_de(conn, eid)
        return {"actividad": [{**a, "autor": autor_vista(a["autor"], equipo)} for a in filas(conn, """
            SELECT a.*, c.nombre_corto, c.razon_social FROM actividad a LEFT JOIN clientes c ON c.id = a.cliente_id
            WHERE a.estudio_id = ? ORDER BY a.fecha DESC, a.id DESC LIMIT 80""", eid)]}

    # --- Ajustes y equipo --------------------------------------------------------------------
    class AjustesEstudio(BaseModel):
        nombre: str | None = Field(None, min_length=2, max_length=120)
        rut: str | None = Field(None, max_length=20)
        email: str | None = Field(None, max_length=200)
        telefono: str | None = Field(None, max_length=40)
        firma: str | None = Field(None, max_length=300)

    class Integrante(BaseModel):
        nombre: str = Field(min_length=2, max_length=120)
        email: str = Field(max_length=200)
        cargo: str = Field("", max_length=80)
        clave: str = Field(min_length=8, max_length=200)

    @app.get("/api/ajustes")
    def ajustes(s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        return {
            "estudio": {k: s["estudio"][k] for k in ("id", "nombre", "rut", "email", "telefono", "firma", "demo")},
            "equipo": filas(conn, """SELECT u.id, u.nombre, u.email, u.cargo, u.rol, u.color,
                (SELECT COUNT(*) FROM clientes c WHERE c.responsable_id = u.id) AS clientes
                FROM usuarios u WHERE u.estudio_id = ? ORDER BY u.id""", eid),
            "identidades": filas(conn, "SELECT id, proveedor, email, creado FROM identidades WHERE usuario_id = ? ORDER BY id",
                                 s["usuario"]["id"]),
            "proveedores": {p: {"nombre": d["nombre"], "activo": oauth.configurado(cfg, p)} for p, d in oauth.PROVEEDORES.items()},
            "clave_definida": bool(s["usuario"]["clave_definida"]),
            "integraciones": {"correo": cfg.smtp_configurado, "redaccion": ia_para(s["estudio"]) is not None,
                              "remitente": cfg.smtp_remitente or cfg.smtp_usuario, "modelo": planes.MODELO},
        }

    @app.patch("/api/ajustes")
    def guardar_ajustes(datos: AjustesEstudio, s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        cambios = {k: v.strip() for k, v in datos.model_dump(exclude_unset=True).items() if v is not None}
        if cambios:
            sets = ", ".join(f"{k} = ?" for k in cambios)
            conn.execute(f"UPDATE estudios SET {sets} WHERE id = ?", (*cambios.values(), eid))
        return {"ok": True}

    @app.post("/api/equipo")
    def agregar_integrante(datos: Integrante, s=Depends(sesion)):
        conn, eid = s["conn"], s["estudio"]["id"]
        if s["usuario"]["rol"] != "socio":
            raise HTTPException(403, "Solo un socio puede agregar personas al equipo.")
        email = datos.email.strip().lower()
        if not EMAIL.match(email):
            raise HTTPException(422, "Revisa el correo electrónico.")
        if fila(conn, "SELECT id FROM usuarios WHERE email = ?", email):
            raise HTTPException(409, "Ya existe una cuenta con ese correo.")
        n = valor(conn, "SELECT COUNT(*) FROM usuarios WHERE estudio_id = ?", eid)
        maximo = planes.suscripcion(s["estudio"], hoy())["max_usuarios"]
        if maximo is not None and n >= maximo:
            raise HTTPException(402, f"Tu plan incluye hasta {maximo} usuarios. Cambia de plan para sumar más personas.")
        uid = insertar(conn, "usuarios", {
            "estudio_id": eid, "nombre": datos.nombre.strip(), "email": email, "clave_hash": hash_clave(datos.clave),
            "rol": "contador", "cargo": datos.cargo.strip(), "color": COLORES[n % len(COLORES)], "creado": cal.ahora(cfg.tz),
        })
        return {"id": uid}

    # --- Plan y uso ---------------------------------------------------------------------------
    @app.get("/api/plan")
    def ver_plan(s=Depends(sesion)):
        conn, estudio = s["conn"], s["estudio"]
        h = hoy()
        sus = planes.suscripcion(estudio, h)
        por_usuario, extras = planes.uso_del_mes(conn, estudio["id"], h)
        equipo = filas(conn, "SELECT id, nombre, color, cargo FROM usuarios WHERE estudio_id = ? ORDER BY id", estudio["id"])
        limite = sus["tokens_usuario"]
        excedido = sum(max(0, t - limite) for t in por_usuario.values())
        automatico = por_usuario.get(None, 0)
        return {
            "suscripcion": sus,
            "periodo_nombre": cal.nombre_periodo(planes.periodo(h), True),
            "ia_activa": ia_para(estudio) is not None,
            "uso": [{**u, "usado": por_usuario.get(u["id"], 0), "limite": limite} for u in equipo],
            "sin_responsable": automatico,
            "total": sum(por_usuario.values()),
            "cupo_total": limite * len(equipo) + extras,
            "extras": extras, "bolsa": max(0, extras - excedido),
            "usuarios": len(equipo),
            "clientes": valor(conn, "SELECT COUNT(*) FROM clientes WHERE estudio_id = ?", estudio["id"]),
            "total_mensual": planes.total_mensual(estudio, len(equipo), h),
            "planes": [{"clave": k, **v} for k, v in planes.PLANES.items()],
            "paquete": {"tokens": planes.PAQUETE_TOKENS, "precio": planes.PAQUETE_PRECIO},
            "equivalencias": {"instruccion": planes.TOKENS_POR_INSTRUCCION, "correo": planes.TOKENS_POR_CORREO},
            "prueba_dias": planes.PRUEBA_DIAS, "prueba_tokens": planes.PRUEBA_TOKENS_USUARIO,
            "ventas_email": cfg.ventas_email,
            "modelo": planes.MODELO,
        }

    # --- Aplicación web ----------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def inicio():
        return FileResponse(os.path.join(WEB, "index.html"), headers={"Cache-Control": "no-cache"})

    app.mount("/", StaticFiles(directory=WEB), name="web")

    if cfg.intervalo_min > 0:
        iniciar_programador(cfg, claude, hoy)
    return app


def iniciar_programador(cfg, claude, hoy):
    """Corre las automatizaciones de todos los estudios cada `intervalo_min` minutos."""

    def ciclo():
        while True:
            _time.sleep(cfg.intervalo_min * 60)
            try:
                ejecutar_todos(cfg, hoy(), claude)
            except Exception:  # el programador nunca debe caerse
                log.exception("Falló la ejecución programada")

    threading.Thread(target=ciclo, name="tucartera-programador", daemon=True).start()


def ejecutar_todos(cfg, hoy, claude=None):
    with conectar(cfg.db) as conn:
        estudios = filas(conn, "SELECT id, nombre FROM estudios WHERE demo = 0")
    resultados = {}
    for e in estudios:
        with conectar(cfg.db) as conn:
            resultados[e["nombre"]] = agentes.ejecutar(conn, cfg, e["id"], hoy, llm=claude)
    return resultados
