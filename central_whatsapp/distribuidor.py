"""Distribuidor de leads de la central de WhatsApp.

Cuando el agente Breeze termina de calificar a un cliente en WhatsApp, un workflow de
HubSpot avisa a este servicio con el ID del contacto. El distribuidor:

1. Ignora avisos repetidos (HubSpot reintenta los webhooks).
2. Si el cliente ya tiene un negocio abierto del mismo proyecto con un vendedor activo,
   se lo devuelve a ese vendedor (no se duplican negocios ni se "roban" clientes).
3. Si es un lead nuevo, elige al vendedor del proyecto con menos carga hoy (según su peso),
   crea el negocio en el pipeline, una tarea con vencimiento según el SLA y, si
   corresponde, lo deja como propietario del contacto.
"""

import html
import logging
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

from .hubspot import ClienteHubSpot, ClienteSoloLectura, ErrorHubSpot
from .instalar import PROPIEDADES
from .notificar import Notificador
from .ruteo import cargar_tabla, elegir_vendedor

log = logging.getLogger(__name__)

PROPS_CONTACTO = (
    "firstname",
    "lastname",
    "phone",
    "mobilephone",
    "hs_whatsapp_phone_number",
    "email",
    "proyecto_de_interes",
    "hubspot_owner_id",
    "comuna",
    "central_plazo_compra",
    "central_forma_pago",
    "central_presupuesto",
    "central_quiere_visita",
    "central_resumen",
    "hs_chat_assistant_summary",
)

PROPS_NEGOCIO = (
    "dealname",
    "pipeline",
    "dealstage",
    "hubspot_owner_id",
    "proyecto_de_interes",
    "hs_is_closed",
    "createdate",
    "fuente",
    "central_estado",
    "central_asignado_en",
    "central_vendedores_previos",
    "central_reasignaciones",
    "central_tarea_id",
)

CALIFICACION = (
    ("central_plazo_compra", "Plazo de compra"),
    ("central_forma_pago", "Forma de pago"),
    ("central_presupuesto", "Presupuesto"),
    ("central_quiere_visita", "Visita al proyecto"),
    ("comuna", "Comuna"),
)

# Etiquetas legibles de las opciones de calificación ("1_3_meses" -> "En 1 a 3 meses")
ETIQUETAS = {
    d["name"]: {o["value"]: o["label"] for o in d.get("options", [])} for defs in PROPIEDADES.values() for d in defs
}

ETAPAS_CERRADAS = {"closedwon", "closedlost"}
ESTADOS_EN_SLA = {"asignado", "reasignado"}


def ahora_utc():
    return datetime.now(timezone.utc)


def ms(dt):
    return str(int(dt.timestamp() * 1000))


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_fecha(valor):
    """Fechas de HubSpot: ISO ('2026-11-24T15:15:28.083Z') o milisegundos."""
    if not valor:
        return None
    texto = str(valor).strip()
    try:
        if texto.isdigit():
            return datetime.fromtimestamp(int(texto) / 1000, tz=timezone.utc)
        dt = datetime.fromisoformat(texto.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def telefono_whatsapp(props):
    """Número en formato internacional sin '+', listo para wa.me (Chile por defecto)."""
    for campo in ("hs_whatsapp_phone_number", "mobilephone", "phone"):
        digitos = "".join(c for c in str(props.get(campo) or "") if c.isdigit())
        if not digitos:
            continue
        if len(digitos) == 9 and digitos.startswith("9"):
            digitos = "56" + digitos
        return digitos
    return ""


def nombre_contacto(props):
    nombre = f"{props.get('firstname') or ''} {props.get('lastname') or ''}".strip()
    if nombre:
        return nombre
    tel = telefono_whatsapp(props)
    return f"WhatsApp +{tel}" if tel else "Contacto sin nombre"


@dataclass
class Resultado:
    accion: str  # asignado | existente | duplicado | sin_vendedor
    contacto_id: str
    negocio_id: str = ""
    vendedor_id: str = ""
    vendedor_nombre: str = ""
    proyecto: str = ""
    origen: str = ""  # proyecto | zona | general | supervisor | negocio_existente
    tarea_id: str = ""
    detalle: str = ""

    def como_dict(self):
        return asdict(self)


class Distribuidor:
    def __init__(self, cfg, tabla, hubspot, notificador=None, reloj=ahora_utc):
        self.cfg = cfg
        self.tabla = tabla
        self.hs = hubspot
        self.notificador = notificador
        self.reloj = reloj
        self._candados = defaultdict(threading.Lock)
        self._candado_global = threading.Lock()

    # ------------------------------------------------------------ entrada
    def procesar(self, contacto_id, proyecto=None, zona=""):
        contacto_id = str(contacto_id).strip()
        with self._candado_global:
            candado = self._candados[contacto_id]
        with candado:  # dos webhooks del mismo contacto al mismo tiempo no crean dos negocios
            return self._procesar(contacto_id, proyecto, zona)

    def _procesar(self, contacto_id, proyecto_forzado, zona):
        ahora = self.reloj()
        p = self.hs.leer("contacts", contacto_id, PROPS_CONTACTO).get("properties", {})
        valor = (proyecto_forzado or p.get("proyecto_de_interes") or "").strip()
        proyecto = self.tabla.proyecto(valor)
        valor_crm = proyecto.nombre if proyecto else valor
        activos = self.hs.propietarios()
        negocios = self.negocios_del_contacto(contacto_id)
        mismos = sorted(
            (n for n in negocios if self._abierto(n) and self._mismo_proyecto(n, valor_crm)),
            key=lambda n: n["properties"].get("createdate") or "",
            reverse=True,
        )

        # 1) Aviso repetido: el negocio ya lo creó la central hace un momento
        for n in mismos:
            np_ = n["properties"]
            if np_.get("fuente") != self.cfg.fuente:
                continue
            asignado = parse_fecha(np_.get("central_asignado_en")) or parse_fecha(np_.get("createdate"))
            if asignado and ahora - asignado < timedelta(minutes=self.cfg.ventana_duplicado_min):
                dueño = np_.get("hubspot_owner_id", "")
                log.info("Contacto %s: aviso repetido, negocio %s ya asignado", contacto_id, n["id"])
                return Resultado(
                    "duplicado", contacto_id, n["id"], dueño, self._nombre(dueño, activos), valor_crm,
                    "negocio_existente", np_.get("central_tarea_id", ""), "Aviso repetido: no se hizo nada.",
                )

        # 2) Cliente que vuelve: se queda con su vendedor si sigue activo y disponible
        for n in mismos:
            dueño = n["properties"].get("hubspot_owner_id", "")
            if dueño and dueño in activos and self.tabla.disponible(dueño, ahora):
                return self._cliente_que_vuelve(contacto_id, p, n, valor_crm, activos, negocios, ahora)

        # 3) Lead nuevo (o negocio abierto cuyo vendedor ya no está): elegir vendedor
        candidatos, origen, _ = self.tabla.candidatos(valor_crm, zona)
        cargas_p, cargas_t = self.cargas([v.id for v in candidatos], valor_crm, ahora)
        elegido = elegir_vendedor(self.tabla, candidatos, ahora, cargas_p, cargas_t, activos)
        if elegido:
            vendedor_id = elegido.id
        else:
            vendedor_id = self.tabla.supervisor_de(proyecto)
            origen = "supervisor"
            if not vendedor_id or vendedor_id not in activos:
                log.warning("Contacto %s: no hay vendedores disponibles para %r", contacto_id, valor_crm)
                return Resultado(
                    "sin_vendedor", contacto_id, proyecto=valor_crm,
                    detalle="No hay vendedores disponibles ni supervisor activo. Revisa el archivo de ruteo.",
                )

        reutilizado = mismos[0] if mismos else None
        if reutilizado:
            negocio_id = reutilizado["id"]
            self._actualizar(
                "deals", negocio_id,
                {"hubspot_owner_id": vendedor_id, "central_estado": "asignado", "central_asignado_en": iso(ahora)},
                opcionales=("central_estado", "central_asignado_en"),
            )
        else:
            negocio_id = self._crear_negocio(contacto_id, p, valor_crm, proyecto, vendedor_id, ahora)

        tarea_id = self._crear_tarea(contacto_id, negocio_id, p, valor_crm, vendedor_id, ahora, "nuevo")
        self._actualizar("deals", negocio_id, {"central_tarea_id": tarea_id}, opcionales=("central_tarea_id",))
        self._actualizar_contacto(contacto_id, p, vendedor_id, negocios, activos, ahora)
        nombre_v = self._nombre(vendedor_id, activos)
        self._avisar(vendedor_id, activos, p, valor_crm, contacto_id, "nuevo")
        log.info("Contacto %s asignado a %s (negocio %s, origen %s)", contacto_id, vendedor_id, negocio_id, origen)
        return Resultado(
            "asignado", contacto_id, negocio_id, vendedor_id, nombre_v, valor_crm, origen, tarea_id,
            "Negocio existente reasignado." if reutilizado else "Negocio nuevo creado.",
        )

    # ----------------------------------------------------------- consultas
    def negocios_del_contacto(self, contacto_id):
        ids = self.hs.asociados("contacts", contacto_id, "deals")
        return self.hs.leer_lote("deals", ids, PROPS_NEGOCIO) if ids else []

    def cargas(self, ids, valor_proyecto, ahora):
        """Leads de hoy por vendedor: (en este proyecto, en total)."""
        if not ids:
            return {}, {}
        filtros = [
            {"propertyName": "createdate", "operator": "GTE", "value": ms(self.tabla.inicio_del_dia(ahora))},
            {"propertyName": "hubspot_owner_id", "operator": "IN", "values": list(ids)},
        ]
        if self.tabla.contar_carga == "central":
            filtros.append({"propertyName": "fuente", "operator": "EQ", "value": self.cfg.fuente})
        total, del_proyecto = Counter(), Counter()
        for n in self.hs.buscar("deals", [filtros], ("hubspot_owner_id", "proyecto_de_interes"), max_paginas=10):
            np_ = n.get("properties", {})
            dueño = str(np_.get("hubspot_owner_id") or "")
            total[dueño] += 1
            if valor_proyecto and self.tabla.mismo_proyecto(np_.get("proyecto_de_interes"), valor_proyecto):
                del_proyecto[dueño] += 1
        return (del_proyecto if valor_proyecto else total), total

    def _abierto(self, negocio):
        np_ = negocio.get("properties", {})
        return (
            np_.get("hs_is_closed") != "true"
            and np_.get("dealstage") not in ETAPAS_CERRADAS
            and (np_.get("pipeline") or self.tabla.pipeline) == self.tabla.pipeline
        )

    def _mismo_proyecto(self, negocio, valor_crm):
        np_ = negocio.get("properties", {})
        if not valor_crm:  # sin proyecto: solo cuenta un negocio de la central también sin proyecto
            return not np_.get("proyecto_de_interes") and np_.get("fuente") == self.cfg.fuente
        return self.tabla.mismo_proyecto(np_.get("proyecto_de_interes"), valor_crm)

    def _nombre(self, id_, activos):
        return (activos.get(str(id_), {}) or {}).get("nombre") or self.tabla.nombre(id_)

    # ------------------------------------------------------------ escritura
    def _actualizar(self, objeto, id_, props, opcionales=()):
        """PATCH tolerante: si faltan propiedades personalizadas (no se corrió `instalar`), reintenta sin ellas."""
        try:
            return self.hs.actualizar(objeto, id_, props)
        except ErrorHubSpot as e:
            resto = {k: v for k, v in props.items() if k not in opcionales}
            if e.status != 400 or not opcionales or resto == props:
                raise
            log.warning("HubSpot rechazó propiedades de la central en %s %s (¿falta `instalar`?): %s", objeto, id_, e.mensaje)
            return self.hs.actualizar(objeto, id_, resto) if resto else {}

    def _crear_negocio(self, contacto_id, p, valor_crm, proyecto, vendedor_id, ahora):
        props = {
            "dealname": nombre_contacto(p),
            "pipeline": self.tabla.pipeline,
            "dealstage": self.tabla.etapa_inicial,
            "hubspot_owner_id": vendedor_id,
            "fuente": self.cfg.fuente,
            "proyecto_de_interes": valor_crm or None,
            "zona_ubicacion_del_proyecto": (proyecto.zona if proyecto and proyecto.zona else None),
            "canal_de_contacto_principal": "Whatsapp",
            "central_estado": "asignado",
            "central_asignado_en": iso(ahora),
            "central_reasignaciones": "0",
        }
        try:
            return self.hs.crear("deals", props, [("contacts", contacto_id)])["id"]
        except ErrorHubSpot as e:
            if e.status != 400:
                raise
            log.warning("HubSpot rechazó alguna propiedad del negocio (%s); se crea con lo mínimo.", e.mensaje)
            minimo = {k: props[k] for k in ("dealname", "pipeline", "dealstage", "hubspot_owner_id", "fuente")}
            return self.hs.crear("deals", minimo, [("contacts", contacto_id)])["id"]

    def _crear_tarea(self, contacto_id, negocio_id, p, valor_crm, vendedor_id, ahora, motivo):
        vence = self.tabla.sumar_minutos_habiles(ahora, self.tabla.sla_minutos)
        prefijo = "Cliente volvió a escribir" if motivo == "vuelve" else "Lead WhatsApp"
        props = {
            "hs_task_subject": f"{prefijo} · {nombre_contacto(p)} · {valor_crm or 'proyecto por definir'}",
            "hs_task_body": self._cuerpo_tarea(p, valor_crm, contacto_id, motivo),
            "hs_timestamp": iso(vence),
            "hubspot_owner_id": vendedor_id,
            "hs_task_status": "NOT_STARTED",
            "hs_task_priority": "HIGH",
            "hs_task_type": "TODO",
        }
        return self.hs.crear("tasks", props, [("contacts", contacto_id), ("deals", negocio_id)])["id"]

    def _cuerpo_tarea(self, p, valor_crm, contacto_id, motivo):
        e = html.escape
        tel = telefono_whatsapp(p)
        if motivo == "vuelve":
            titulo = "Tu cliente volvió a escribir a la central de WhatsApp."
        else:
            titulo = "Lead nuevo de la central de WhatsApp (lo atendió el agente Breeze)."
        filas = [f"<li><strong>Proyecto:</strong> {e(valor_crm or 'por definir: pregúntale qué proyecto le interesa')}</li>"]
        if tel:
            filas.append(f'<li><strong>WhatsApp:</strong> +{tel} · <a href="https://wa.me/{tel}">abrir chat</a></li>')
        for campo, etiqueta in CALIFICACION:
            if p.get(campo):
                valor = ETIQUETAS.get(campo, {}).get(str(p[campo]), str(p[campo]))
                filas.append(f"<li><strong>{etiqueta}:</strong> {e(valor)}</li>")
        resumen = p.get("central_resumen") or p.get("hs_chat_assistant_summary")
        partes = [f"<p><strong>{titulo}</strong></p>", "<ul>" + "".join(filas) + "</ul>"]
        if resumen:
            partes.append(f"<p><strong>Resumen de la conversación:</strong> {e(str(resumen))}</p>")
        url = self.hs.url_registro("contacts", contacto_id)
        if url:
            partes.append(f'<p><a href="{url}">Ver el contacto y la conversación en HubSpot</a></p>')
        if motivo != "vuelve":
            partes.append(
                f"<p>⏱️ Contáctalo y marca esta tarea como <strong>completada</strong> dentro de "
                f"{self.tabla.sla_minutos} minutos hábiles. Si no, el lead pasa a otro vendedor del proyecto.</p>"
            )
        return "".join(partes)

    def _cliente_que_vuelve(self, contacto_id, p, negocio, valor_crm, activos, negocios, ahora):
        np_ = negocio["properties"]
        vendedor_id = np_["hubspot_owner_id"]
        tarea_id = self._crear_tarea(contacto_id, negocio["id"], p, valor_crm, vendedor_id, ahora, "vuelve")
        cambios = {"central_tarea_id": tarea_id}
        if (np_.get("central_estado") or "") not in ESTADOS_EN_SLA:  # si ya está en SLA, se mantiene
            cambios["central_estado"] = "existente"
        self._actualizar("deals", negocio["id"], cambios, opcionales=tuple(cambios))
        self._actualizar_contacto(contacto_id, p, vendedor_id, negocios, activos, ahora)
        self._avisar(vendedor_id, activos, p, valor_crm, contacto_id, "vuelve")
        log.info("Contacto %s volvió: negocio %s sigue con %s", contacto_id, negocio["id"], vendedor_id)
        return Resultado(
            "existente", contacto_id, negocio["id"], vendedor_id, self._nombre(vendedor_id, activos), valor_crm,
            "negocio_existente", tarea_id, "El cliente ya tenía un negocio abierto: se avisó a su vendedor.",
        )

    def _actualizar_contacto(self, contacto_id, p, vendedor_id, negocios, activos, ahora):
        props = {"central_ultimo_ruteo": iso(ahora)}
        actual = str(p.get("hubspot_owner_id") or "")
        politica = self.cfg.propietario_contacto
        if politica == "siempre":
            cambiar = True
        elif politica == "si_libre":
            tiene_negocio = any(
                self._abierto(n) and n["properties"].get("hubspot_owner_id") == actual for n in negocios
            )
            cambiar = not actual or actual not in activos or not tiene_negocio
        else:
            cambiar = False
        if cambiar and actual != vendedor_id:
            props["hubspot_owner_id"] = vendedor_id
        self._actualizar("contacts", contacto_id, props, opcionales=("central_ultimo_ruteo",))

    def _avisar(self, vendedor_id, activos, p, valor_crm, contacto_id, motivo):
        if not self.notificador or not self.notificador.activo:
            return
        correo = (activos.get(str(vendedor_id), {}) or {}).get("email")
        if not correo:
            return
        asunto = ("Tu cliente volvió a escribir" if motivo == "vuelve" else "Nuevo lead de WhatsApp") + f" · {valor_crm or 'sin proyecto'}"
        tel = telefono_whatsapp(p)
        lineas = [
            f"Cliente: {nombre_contacto(p)}",
            f"Proyecto: {valor_crm or 'por definir'}",
            f"WhatsApp: https://wa.me/{tel}" if tel else "",
            f"Ficha en HubSpot: {self.hs.url_registro('contacts', contacto_id)}",
            "",
            f"Tienes {self.tabla.sla_minutos} minutos hábiles para contactarlo y completar la tarea en HubSpot.",
        ]
        self.notificador.enviar(correo, asunto, "\n".join(x for x in lineas if x is not None))


def construir(cfg, simular=None):
    """Arma el distribuidor real (HubSpot + archivo de ruteo + aviso por correo opcional)."""
    tabla = cargar_tabla(cfg)
    hs = ClienteHubSpot(cfg.hubspot_token, cfg.hubspot_portal_id)
    if cfg.simular if simular is None else simular:
        hs = ClienteSoloLectura(hs)
    return Distribuidor(cfg, tabla, hs, Notificador(cfg))
