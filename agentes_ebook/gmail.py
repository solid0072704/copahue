"""Conexión con Gmail por IMAP/SMTP (con contraseña de aplicación).

- Lee las notificaciones de descarga que envía FormSubmit.
- Detecta si un lead te respondió o si ya le escribiste.
- Crea borradores en tu Gmail y envía correos.
"""

import email
import hashlib
import imaplib
import re
import smtplib
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import formataddr, getaddresses, parsedate_to_datetime
from html.parser import HTMLParser

from .db import iso

# Dominios reservados para ejemplos: nunca se les envía nada (datos demo).
DOMINIOS_SIMULADOS = ("example.com", "example.org", "example.net")

# Nombres de campo del formulario -> campo interno.
MAPA_CAMPOS = {
    "name": "nombre",
    "nombre": "nombre",
    "email": "email",
    "correo": "email",
    "telefono": "telefono",
    "phone": "telefono",
    "celular": "telefono",
    "whatsapp": "telefono",
    "cargo": "cargo",
    "puesto": "cargo",
    "empresa": "empresa",
    "company": "empresa",
    "motivo_de_descarga": "motivo",
    "motivo": "motivo",
}


def normalizar(texto):
    sin_tildes = unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", sin_tildes).strip().lower()


def es_destino_simulado(direccion):
    dominio = (direccion or "").rsplit("@", 1)[-1].lower()
    return dominio.endswith(".example") or dominio in DOMINIOS_SIMULADOS


# --------------------------------------------------------------------------
# Parseo de la notificación de FormSubmit
# --------------------------------------------------------------------------
class _TablaHTML(HTMLParser):
    """Extrae las filas de las tablas HTML como listas de textos."""

    def __init__(self):
        super().__init__()
        self.filas, self._fila, self._celda = [], None, None
        self.texto = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._fila = []
        elif tag in ("td", "th") and self._fila is not None:
            self._celda = []
        elif tag == "br" and self._celda is not None:
            self._celda.append("\n")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._fila is not None and self._celda is not None:
            self._fila.append("".join(self._celda).strip())
            self._celda = None
        elif tag == "tr" and self._fila is not None:
            self.filas.append(self._fila)
            self._fila = None

    def handle_data(self, data):
        if self._celda is not None:
            self._celda.append(data)
        self.texto.append(data)


def _campos_desde_pares(pares):
    datos, extra = {}, {}
    for clave, valor in pares:
        clave_n = normalizar(clave).replace(" ", "_")
        if not clave_n or (clave_n, normalizar(valor)) == ("name", "value"):
            continue  # fila de encabezado "Name | Value"
        destino = MAPA_CAMPOS.get(clave_n)
        if destino and not datos.get(destino):
            datos[destino] = valor.strip()
        elif not clave_n.startswith("_"):
            extra[clave.strip()] = valor.strip()
    datos["extra"] = extra
    return datos


def _fecha_formsubmit(texto):
    m = re.search(r"Submitted at (.+?\d{4} \d{1,2}:\d{2} [AP]M)", texto or "")
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%a, %b %d, %Y %I:%M %p").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parsear_formsubmit(html="", texto=""):
    """Convierte el cuerpo de una notificación de FormSubmit en un dict de campos."""
    pares, cuerpo_plano = [], texto or ""
    if html:
        p = _TablaHTML()
        p.feed(html)
        pares = [(f[0], f[1]) for f in p.filas if len(f) >= 2]
        cuerpo_plano = f"{cuerpo_plano} {' '.join(p.texto)}"
    if not pares and texto:
        # Versión texto: filas tipo "| clave | valor |" o "clave: valor"
        for linea in texto.splitlines():
            m = re.match(r"^\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*$", linea) or re.match(
                r"^\s*([\wÁÉÍÓÚáéíóúñÑ_ ]{2,40}):\s*(.+)$", linea
            )
            if m and not set(m.group(1)) <= {"-"}:
                pares.append((m.group(1), m.group(2)))
    datos = _campos_desde_pares(pares)
    datos["fecha_formulario"] = _fecha_formsubmit(cuerpo_plano)
    return datos


def _decodificar(valor):
    try:
        return str(make_header(decode_header(valor or "")))
    except Exception:
        return valor or ""


def _cuerpos(msg):
    html, texto = "", ""
    partes = msg.walk() if msg.is_multipart() else [msg]
    for parte in partes:
        tipo = parte.get_content_type()
        if parte.get_content_maintype() == "multipart" or parte.get("Content-Disposition", "").startswith("attachment"):
            continue
        try:
            contenido = parte.get_payload(decode=True)
            if contenido is None:
                continue
            contenido = contenido.decode(parte.get_content_charset() or "utf-8", errors="replace")
        except Exception:
            continue
        if tipo == "text/html" and not html:
            html = contenido
        elif tipo == "text/plain" and not texto:
            texto = contenido
    return html, texto


def registro_desde_mensaje(msg_bytes, cfg):
    """Convierte un correo crudo (bytes) en un registro de descarga, o None si no aplica."""
    msg = email.message_from_bytes(msg_bytes)
    asunto = _decodificar(msg.get("Subject"))
    if normalizar(cfg.descargas_asunto) not in normalizar(asunto):
        return None
    html, texto = _cuerpos(msg)
    datos = parsear_formsubmit(html, texto)
    correo = (datos.get("email") or "").strip().lower()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", correo) or correo in cfg.excluir_correos:
        return None
    fecha = datos.pop("fecha_formulario", None)
    if fecha is None:
        try:
            fecha = parsedate_to_datetime(msg.get("Date"))
        except Exception:
            fecha = datetime.now(timezone.utc)
    mensaje_id = (msg.get("Message-ID") or "").strip()
    if not mensaje_id:
        mensaje_id = "sha1:" + hashlib.sha1(msg_bytes).hexdigest()
    return {
        "mensaje_id": mensaje_id,
        "fecha": iso(fecha),
        "email": correo,
        "nombre": datos.get("nombre", ""),
        "telefono": datos.get("telefono", ""),
        "cargo": datos.get("cargo", ""),
        "empresa": datos.get("empresa", ""),
        "motivo": datos.get("motivo", ""),
        "fuente": "gmail",
        "extra": datos.get("extra", {}),
    }


# --------------------------------------------------------------------------
# IMAP
# --------------------------------------------------------------------------
def _fecha_imap(dt):
    return dt.strftime("%d-%b-%Y")


class ClienteGmail:
    def __init__(self, cfg):
        self.cfg = cfg
        self.imap = None
        self._carpetas = None

    def __enter__(self):
        self.imap = imaplib.IMAP4_SSL(self.cfg.imap_host)
        self.imap.login(self.cfg.gmail_usuario, self.cfg.gmail_app_password)
        return self

    def __exit__(self, *exc):
        try:
            self.imap.logout()
        except Exception:
            pass

    def carpeta(self, bandera, defecto="INBOX"):
        """Busca una carpeta por su bandera especial (\\All, \\Sent, \\Drafts), sin depender del idioma."""
        if self._carpetas is None:
            self._carpetas = {}
            estado, lineas = self.imap.list()
            for linea in lineas or []:
                linea = linea.decode(errors="replace")
                m = re.match(r'\((?P<flags>[^)]*)\) "(?P<sep>[^"]*)" (?P<nombre>.+)$', linea)
                if not m:
                    continue
                nombre = m.group("nombre").strip()
                for flag in m.group("flags").split():
                    self._carpetas[flag] = nombre
        return self._carpetas.get(bandera, defecto)

    def _seleccionar(self, bandera, defecto="INBOX"):
        nombre = self.carpeta(bandera, defecto)
        estado, _ = self.imap.select(nombre, readonly=True)
        if estado != "OK":
            self.imap.select(defecto, readonly=True)

    def _buscar(self, *criterios):
        estado, datos = self.imap.uid("SEARCH", None, *criterios)
        if estado != "OK" or not datos or not datos[0]:
            return []
        return datos[0].split()

    def leer_descargas(self, dias=None):
        """Devuelve los registros de descarga encontrados en los últimos `dias`."""
        dias = dias or self.cfg.descargas_dias
        desde = datetime.now(timezone.utc) - timedelta(days=dias)
        self._seleccionar("\\All")
        uids = self._buscar("FROM", f'"{self.cfg.descargas_remitente}"', "SINCE", _fecha_imap(desde))
        registros = []
        for i in range(0, len(uids), 50):
            lote = b",".join(uids[i : i + 50])
            estado, datos = self.imap.uid("FETCH", lote, "(BODY.PEEK[])")
            if estado != "OK":
                continue
            for item in datos:
                if isinstance(item, tuple) and len(item) == 2:
                    reg = registro_desde_mensaje(item[1], self.cfg)
                    if reg:
                        registros.append(reg)
        return registros

    def _fechas_de(self, uids):
        if not uids:
            return []
        estado, datos = self.imap.uid("FETCH", b",".join(uids[-20:]), "(INTERNALDATE)")
        fechas = []
        for item in datos or []:
            linea = item.decode() if isinstance(item, bytes) else ""
            tupla = imaplib.Internaldate2tuple(linea.encode()) if "INTERNALDATE" in linea else None
            if tupla:
                fechas.append(datetime.fromtimestamp(time.mktime(tupla), tz=timezone.utc))
        return fechas

    def ultima_respuesta(self, correos_desde):
        """{correo: fecha} del último correo recibido de cada lead después de `desde`."""
        self._seleccionar("\\All")
        resultado = {}
        for correo, desde in correos_desde.items():
            uids = self._buscar("FROM", f'"{correo}"', "SINCE", _fecha_imap(desde))
            fechas = [f for f in self._fechas_de(uids) if f >= desde]
            if fechas:
                resultado[correo] = max(fechas)
        return resultado

    def enviados_a(self, correos_desde):
        """{correo: [fechas]} de los correos que tú enviaste a cada lead después de `desde`."""
        self._seleccionar("\\Sent")
        resultado = {}
        for correo, desde in correos_desde.items():
            uids = self._buscar("TO", f'"{correo}"', "SINCE", _fecha_imap(desde))
            fechas = sorted(f for f in self._fechas_de(uids) if f >= desde)
            if fechas:
                resultado[correo] = fechas
        return resultado

    def crear_borrador(self, para, asunto, texto, html=None):
        if es_destino_simulado(para):
            return "simulado"
        msg = construir_mensaje(self.cfg, para, asunto, texto, html)
        carpeta = self.carpeta("\\Drafts", "[Gmail]/Drafts")
        estado, _ = self.imap.append(
            carpeta, "(\\Draft)", imaplib.Time2Internaldate(time.time()), msg.as_bytes()
        )
        return "ok" if estado == "OK" else "error"


# --------------------------------------------------------------------------
# SMTP
# --------------------------------------------------------------------------
def construir_mensaje(cfg, para, asunto, texto, html=None, cc=None):
    msg = EmailMessage()
    msg["From"] = formataddr((cfg.firma or cfg.empresa_nombre, cfg.gmail_usuario))
    msg["To"] = para
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg["Subject"] = asunto
    msg.set_content(texto)
    if html:
        msg.add_alternative(html, subtype="html")
    return msg


def enviar_correo(cfg, para, asunto, texto, html=None, cc=None):
    """Envía un correo. Devuelve 'ok', 'simulado' o lanza la excepción de SMTP."""
    destinatarios = [d for _, d in getaddresses([para] + list(cc or []))]
    if all(es_destino_simulado(d) for d in destinatarios):
        return "simulado"
    if not cfg.gmail_configurado:
        raise RuntimeError("Faltan GMAIL_USER / GMAIL_APP_PASSWORD para enviar correos.")
    msg = construir_mensaje(cfg, para, asunto, texto, html, cc)
    with smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_puerto, timeout=30) as smtp:
        smtp.login(cfg.gmail_usuario, cfg.gmail_app_password)
        smtp.send_message(msg)
    return "ok"
