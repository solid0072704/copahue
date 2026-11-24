"""Reglas de ruteo: qué vendedores atienden cada proyecto, horarios, ausencias y SLA.

Se cargan desde un YAML (ver ruteo.ejemplo.yaml). La elección del vendedor es
"round-robin ponderado por carga": entre los vendedores del proyecto disponibles,
gana el que tiene menos leads del proyecto hoy en relación con su peso.
"""

import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import yaml

DIAS = {"lunes": 0, "martes": 1, "miercoles": 2, "jueves": 3, "viernes": 4, "sabado": 5, "domingo": 6}
PALABRAS_VACIAS = {"raices", "de", "del", "los", "las", "la", "el"}


class ErrorRuteo(ValueError):
    pass


def normalizar(texto):
    """'Raíces del Río Paloma' y 'Raices de Rio Paloma' -> 'rio paloma'."""
    s = unicodedata.normalize("NFKD", str(texto or "")).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(p for p in s.split() if p not in PALABRAS_VACIAS)


def _sin_tildes(texto):
    return unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode().lower().strip()


def _fecha(valor):
    if isinstance(valor, datetime):
        return valor.date()
    if isinstance(valor, date):
        return valor
    return date.fromisoformat(str(valor).strip())


def _hora(texto):
    h, m = str(texto).strip().split(":")
    return time(int(h), int(m))


def _parse_intervalos(texto):
    texto = str(texto or "").strip().lower()
    if not texto or texto in ("cerrado", "no", "-"):
        return []
    intervalos = []
    for parte in texto.split(","):
        inicio, fin = parte.split("-")
        a, b = _hora(inicio), _hora(fin)
        if b <= a:
            raise ErrorRuteo(f"Intervalo de horario inválido: {parte!r}")
        intervalos.append((a, b))
    return intervalos


def _parse_dias(clave):
    clave = _sin_tildes(clave)
    dias = []
    for parte in clave.split(","):
        parte = parte.strip()
        if "-" in parte:
            a, b = (DIAS[x.strip()] for x in parte.split("-"))
            dias.extend(range(a, b + 1))
        else:
            dias.append(DIAS[parte])
    return dias


def _parse_horario(datos):
    """{'lunes-viernes': '09:00-19:00', 'sabado': '10:00-14:00'} -> {0: [(9:00, 19:00)], ...}.

    Sin horario (None) significa 24/7.
    """
    if not datos:
        return None
    horario = {d: [] for d in range(7)}
    try:
        for clave, intervalos in datos.items():
            for dia in _parse_dias(clave):
                horario[dia] = _parse_intervalos(intervalos)
    except (KeyError, ValueError) as e:
        raise ErrorRuteo(f"Horario inválido: {e}") from e
    return horario


@dataclass
class Vendedor:
    id: str
    nombre: str = ""
    peso: int = 1


@dataclass
class Proyecto:
    nombre: str  # valor interno de la propiedad "Proyecto de Interés"
    zona: str = ""
    vendedores: list = field(default_factory=list)
    supervisor: str = ""
    alias: list = field(default_factory=list)


def _vendedores(lista, catalogo):
    resultado = []
    for item in lista or []:
        if isinstance(item, dict):
            id_ = str(item.get("id", "")).strip()
            peso = int(item.get("peso", 1))
            nombre = str(item.get("nombre", "") or "")
        else:
            id_, peso, nombre = str(item).strip(), 1, ""
        if not id_:
            continue
        if peso < 1:
            continue  # peso 0 = pausado en este proyecto
        nombre = nombre or catalogo.get(id_, {}).get("nombre", "")
        resultado.append(Vendedor(id_, nombre, peso))
    return resultado


def _union(listas):
    vistos, resultado = set(), []
    for lista in listas:
        for v in lista:
            if v.id not in vistos:
                vistos.add(v.id)
                resultado.append(Vendedor(v.id, v.nombre, 1))
    return resultado


class TablaRuteo:
    def __init__(self, datos):
        datos = datos or {}
        if not isinstance(datos, dict):
            raise ErrorRuteo("El archivo de ruteo debe ser un diccionario YAML.")
        self.zona_horaria = datos.get("zona_horaria", "America/Santiago")
        self.tz = ZoneInfo(self.zona_horaria)
        self.pipeline = str(datos.get("pipeline", "default"))
        self.etapa_inicial = str(datos.get("etapa_inicial", "appointmentscheduled"))
        self.contar_carga = str(datos.get("contar_carga", "todos")).lower()  # todos | central
        sla = datos.get("sla") or {}
        self.sla_minutos = int(sla.get("minutos", 15))
        self.max_reasignaciones = int(sla.get("max_reasignaciones", 2))
        self.supervisor = str(sla.get("supervisor") or "")
        self.horario = _parse_horario(datos.get("horario"))
        self.feriados = {_fecha(f) for f in datos.get("feriados") or []}

        # Catálogo de vendedores: nombre, ausencias, tope diario y horario propio (todo opcional)
        self.catalogo = {}
        for id_, info in (datos.get("vendedores") or {}).items():
            info = dict(info or {})
            if info.get("horario"):
                info["horario"] = _parse_horario(info["horario"])
            for clave in ("ausente_desde", "ausente_hasta"):
                if info.get(clave):
                    info[clave] = _fecha(info[clave])
            self.catalogo[str(id_)] = info

        self.proyectos = {}
        self._alias = {}
        for nombre, info in (datos.get("proyectos") or {}).items():
            info = info or {}
            p = Proyecto(
                nombre=str(nombre),
                zona=str(info.get("zona", "") or ""),
                vendedores=_vendedores(info.get("vendedores"), self.catalogo),
                supervisor=str(info.get("supervisor", "") or ""),
                alias=[str(a) for a in info.get("alias") or []],
            )
            self.proyectos[normalizar(p.nombre)] = p
            for a in p.alias:
                self._alias[normalizar(a)] = p

        # Zonas: explícitas o, si no se definen, la unión de los vendedores de sus proyectos
        zonas = datos.get("zonas")
        if zonas:
            self.zonas = {str(z): _vendedores(v, self.catalogo) for z, v in zonas.items()}
        else:
            agrupado = {}
            for p in self.proyectos.values():
                if p.zona:
                    agrupado.setdefault(p.zona, []).append(p.vendedores)
            self.zonas = {z: _union(listas) for z, listas in agrupado.items()}

        pool = datos.get("pool_general", "todos")
        if pool in (None, "todos", "") or (isinstance(pool, str) and pool.lower() == "todos"):
            self.pool_general = _union(p.vendedores for p in self.proyectos.values())
        else:
            self.pool_general = _vendedores(pool, self.catalogo)

    # ----------------------------------------------------------------- carga
    @classmethod
    def desde_texto(cls, texto):
        try:
            return cls(yaml.safe_load(texto) or {})
        except yaml.YAMLError as e:
            raise ErrorRuteo(f"El YAML de ruteo no es válido: {e}") from e

    @classmethod
    def desde_archivo(cls, ruta):
        with open(ruta, encoding="utf-8") as f:
            return cls.desde_texto(f.read())

    # -------------------------------------------------------------- consulta
    def proyecto(self, valor):
        clave = normalizar(valor)
        if not clave:
            return None
        return self.proyectos.get(clave) or self._alias.get(clave)

    def mismo_proyecto(self, a, b):
        pa, pb = self.proyecto(a), self.proyecto(b)
        if pa and pb:
            return pa is pb
        return bool(normalizar(a)) and normalizar(a) == normalizar(b)

    def candidatos(self, valor_proyecto, zona=""):
        """Devuelve (vendedores, origen, proyecto). origen: 'proyecto' | 'zona' | 'general'."""
        p = self.proyecto(valor_proyecto)
        if p and p.vendedores:
            return p.vendedores, "proyecto", p
        z = (p.zona if p else "") or zona
        if z and self.zonas.get(z):
            return self.zonas[z], "zona", p
        return self.pool_general, "general", p

    def supervisor_de(self, proyecto=None):
        return (proyecto.supervisor if proyecto and proyecto.supervisor else "") or self.supervisor

    def nombre(self, id_):
        id_ = str(id_)
        if self.catalogo.get(id_, {}).get("nombre"):
            return self.catalogo[id_]["nombre"]
        for p in self.proyectos.values():
            for v in p.vendedores:
                if v.id == id_ and v.nombre:
                    return v.nombre
        return ""

    def todos_los_vendedores(self):
        return _union([p.vendedores for p in self.proyectos.values()] + [self.pool_general])

    def disponible(self, id_, ahora):
        info = self.catalogo.get(str(id_), {})
        if info.get("ausente") is True:
            return False
        hoy = ahora.astimezone(self.tz).date()
        desde, hasta = info.get("ausente_desde"), info.get("ausente_hasta")
        if hasta and hoy <= hasta and (not desde or hoy >= desde):
            return False
        return True

    def max_diario(self, id_):
        valor = self.catalogo.get(str(id_), {}).get("max_diario")
        return int(valor) if valor else None

    def _horario_de(self, id_=None):
        if id_ is not None and self.catalogo.get(str(id_), {}).get("horario"):
            return self.catalogo[str(id_)]["horario"]
        return self.horario

    # ------------------------------------------------------- tiempo hábil
    def _intervalos(self, dia, horario):
        if dia in self.feriados:
            return []
        if horario is None:  # 24/7
            inicio = datetime.combine(dia, time(0), tzinfo=self.tz)
            return [(inicio, datetime.combine(dia + timedelta(days=1), time(0), tzinfo=self.tz))]
        return [
            (datetime.combine(dia, a, tzinfo=self.tz), datetime.combine(dia, b, tzinfo=self.tz))
            for a, b in horario.get(dia.weekday(), [])
        ]

    def en_turno(self, id_, ahora):
        horario = self._horario_de(id_)
        local = ahora.astimezone(self.tz)
        return any(a <= local < b for a, b in self._intervalos(local.date(), horario))

    def en_horario(self, ahora):
        local = ahora.astimezone(self.tz)
        return any(a <= local < b for a, b in self._intervalos(local.date(), self.horario))

    def minutos_habiles(self, desde, hasta):
        """Minutos dentro del horario de atención entre dos instantes."""
        if hasta <= desde:
            return 0.0
        desde_utc, hasta_utc = desde.astimezone(timezone.utc), hasta.astimezone(timezone.utc)
        dia, ultimo = desde.astimezone(self.tz).date(), hasta.astimezone(self.tz).date()
        total = 0.0
        for _ in range(400):
            if dia > ultimo:
                break
            for a, b in self._intervalos(dia, self.horario):
                ini = max(a.astimezone(timezone.utc), desde_utc)
                fin = min(b.astimezone(timezone.utc), hasta_utc)
                if fin > ini:
                    total += (fin - ini).total_seconds() / 60
            dia += timedelta(days=1)
        return total

    def sumar_minutos_habiles(self, desde, minutos):
        """Instante en que se cumplen `minutos` hábiles contados desde `desde`."""
        desde_utc = desde.astimezone(timezone.utc)
        restante = float(minutos)
        dia = desde.astimezone(self.tz).date()
        for _ in range(400):
            for a, b in self._intervalos(dia, self.horario):
                ini = max(a.astimezone(timezone.utc), desde_utc)
                fin = b.astimezone(timezone.utc)
                if fin <= ini:
                    continue
                disponible = (fin - ini).total_seconds() / 60
                if disponible >= restante:
                    return ini + timedelta(minutes=restante)
                restante -= disponible
            dia += timedelta(days=1)
        return desde_utc + timedelta(minutes=minutos)

    def inicio_del_dia(self, ahora):
        local = ahora.astimezone(self.tz)
        return datetime.combine(local.date(), time(0), tzinfo=self.tz)

    # ------------------------------------------------------------ revisión
    def validar(self, propietarios=None):
        """Lista de advertencias (vendedores inactivos, proyectos sin vendedores, etc.)."""
        avisos = []
        if not self.proyectos:
            avisos.append("No hay proyectos definidos.")
        for p in self.proyectos.values():
            if not p.vendedores:
                avisos.append(f"El proyecto «{p.nombre}» no tiene vendedores: sus leads irán a la zona o al pool general.")
        if not self.supervisor:
            avisos.append("Falta sla.supervisor: los leads que nadie atienda no se escalarán a nadie.")
        if propietarios is not None:
            ids = {v.id for v in self.todos_los_vendedores()}
            ids.update(p.supervisor for p in self.proyectos.values() if p.supervisor)
            if self.supervisor:
                ids.add(self.supervisor)
            for id_ in sorted(ids):
                if id_ not in propietarios:
                    nombre = self.nombre(id_) or "sin nombre"
                    avisos.append(f"El propietario {id_} ({nombre}) no existe o está inactivo en HubSpot.")
        return avisos


def elegir_vendedor(tabla, candidatos, ahora, cargas_proyecto, cargas_total, activos=None, excluir=()):
    """Elige al vendedor con menor carga relativa a su peso entre los disponibles.

    - Descarta excluidos, inactivos en HubSpot, ausentes y quienes llegaron a su tope diario.
    - Prefiere a quienes están en su horario; si nadie lo está, considera a todos los disponibles.
    """
    excluir = {str(x) for x in excluir if x}
    aptos = []
    for v in candidatos:
        if v.id in excluir:
            continue
        if activos is not None and v.id not in activos:
            continue
        if not tabla.disponible(v.id, ahora):
            continue
        tope = tabla.max_diario(v.id)
        if tope and cargas_total.get(v.id, 0) >= tope:
            continue
        aptos.append(v)
    if not aptos:
        return None
    en_turno = [v for v in aptos if tabla.en_turno(v.id, ahora)]
    pool = en_turno or aptos
    return min(
        pool,
        key=lambda v: ((cargas_proyecto.get(v.id, 0) + 1) / max(v.peso, 1), cargas_total.get(v.id, 0), -v.peso, v.id),
    )


def cargar_tabla(cfg):
    """Carga la tabla desde CENTRAL_RUTEO_YAML (contenido) o desde CENTRAL_RUTEO_ARCHIVO."""
    if cfg.ruteo_yaml:
        return TablaRuteo.desde_texto(cfg.ruteo_yaml)
    if os.path.exists(cfg.ruteo_archivo):
        return TablaRuteo.desde_archivo(cfg.ruteo_archivo)
    raise ErrorRuteo(
        f"No encontré el archivo de ruteo {cfg.ruteo_archivo!r}. Genéralo con "
        "`python run_central.py sugerir-ruteo` o copia central_whatsapp/ruteo.ejemplo.yaml."
    )
