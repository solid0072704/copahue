"""Estado de la cartera (efectivo, posiciones, operaciones, curva de patrimonio) guardado en JSON."""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def hoy_iso(fecha=None):
    if fecha is None:
        return datetime.now(timezone.utc).date().isoformat()
    return str(fecha)[:10]


@dataclass
class Cartera:
    modo: str = "simulado"
    capital_inicial: float = 10_000.0
    efectivo: float = 10_000.0
    posiciones: dict = field(default_factory=dict)   # {ticker: {cantidad, precio_promedio, fecha}}
    operaciones: list = field(default_factory=list)  # historial de compras/ventas
    curva: list = field(default_factory=list)        # [{fecha, patrimonio, efectivo}]
    maximo: float = 10_000.0
    ultimo_rebalanceo: str = ""
    refugio_hasta: str = ""                          # en efectivo hasta esta fecha tras una caída fuerte
    pendiente: dict = field(default_factory=dict)    # órdenes decididas al cierre, se ejecutan a la apertura siguiente
    ultima_fecha: str = ""                           # última rueda procesada
    ultima_señal: dict = field(default_factory=dict)
    creada: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # ---- valoración ----
    def valor_posiciones(self, precios):
        total = 0.0
        for t, pos in self.posiciones.items():
            precio = precios.get(t)
            if precio is None:
                precio = pos["precio_promedio"]
            total += pos["cantidad"] * float(precio)
        return total

    def patrimonio(self, precios):
        return self.efectivo + self.valor_posiciones(precios)

    def ganancia(self, precios):
        return self.patrimonio(precios) - self.capital_inicial

    def pesos(self, precios):
        patrimonio = self.patrimonio(precios)
        if patrimonio <= 0:
            return {}
        return {t: pos["cantidad"] * float(precios.get(t, pos["precio_promedio"])) / patrimonio for t, pos in self.posiciones.items()}

    # ---- operaciones ----
    def comprar(self, ticker, importe, precio, fecha, comision_pct=0.0, motivo=""):
        """Compra por un importe en USD (acepta fracciones de acción). Devuelve la operación o None."""
        importe = min(float(importe), self.efectivo)
        if importe < 1 or precio <= 0:
            return None
        comision = importe * comision_pct
        cantidad = (importe - comision) / precio
        pos = self.posiciones.get(ticker, {"cantidad": 0.0, "precio_promedio": 0.0, "fecha": hoy_iso(fecha)})
        total_anterior = pos["cantidad"] * pos["precio_promedio"]
        pos["cantidad"] = round(pos["cantidad"] + cantidad, 6)
        pos["precio_promedio"] = round((total_anterior + cantidad * precio) / pos["cantidad"], 4)
        if not pos.get("fecha"):
            pos["fecha"] = hoy_iso(fecha)
        self.posiciones[ticker] = pos
        self.efectivo -= importe
        return self._registrar(fecha, ticker, "compra", cantidad, precio, importe, comision, motivo)

    def vender(self, ticker, cantidad, precio, fecha, comision_pct=0.0, motivo=""):
        pos = self.posiciones.get(ticker)
        if not pos or precio <= 0:
            return None
        cantidad = min(float(cantidad), pos["cantidad"])
        if cantidad <= 0:
            return None
        bruto = cantidad * precio
        comision = bruto * comision_pct
        self.efectivo += bruto - comision
        pos["cantidad"] = round(pos["cantidad"] - cantidad, 6)
        if pos["cantidad"] <= 1e-6:
            del self.posiciones[ticker]
        op = self._registrar(fecha, ticker, "venta", cantidad, precio, bruto, comision, motivo)
        op["resultado"] = round((precio - pos["precio_promedio"]) * cantidad - comision, 2)
        return op

    def vender_todo(self, ticker, precio, fecha, comision_pct=0.0, motivo=""):
        pos = self.posiciones.get(ticker)
        return self.vender(ticker, pos["cantidad"], precio, fecha, comision_pct, motivo) if pos else None

    def _registrar(self, fecha, ticker, lado, cantidad, precio, importe, comision, motivo):
        op = {
            "fecha": hoy_iso(fecha),
            "ticker": ticker,
            "lado": lado,
            "cantidad": round(cantidad, 6),
            "precio": round(float(precio), 4),
            "importe": round(float(importe), 2),
            "comision": round(float(comision), 4),
            "motivo": motivo,
        }
        self.operaciones.append(op)
        return op

    def anotar_curva(self, fecha, precios):
        patrimonio = self.patrimonio(precios)
        self.maximo = max(self.maximo, patrimonio)
        punto = {"fecha": hoy_iso(fecha), "patrimonio": round(patrimonio, 2), "efectivo": round(self.efectivo, 2)}
        if self.curva and self.curva[-1]["fecha"] == punto["fecha"]:
            self.curva[-1] = punto
        else:
            self.curva.append(punto)
        return patrimonio

    # ---- persistencia ----
    def guardar(self, ruta):
        os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
        tmp = ruta + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=1)
        os.replace(tmp, ruta)

    @classmethod
    def cargar(cls, ruta):
        with open(ruta, encoding="utf-8") as f:
            datos = json.load(f)
        campos = {k: v for k, v in datos.items() if k in cls.__dataclass_fields__}
        return cls(**campos)

    @classmethod
    def abrir(cls, cfg):
        """Carga la cartera del disco o crea una nueva con el capital inicial configurado."""
        if os.path.exists(cfg.ruta_cartera):
            return cls.cargar(cfg.ruta_cartera)
        return cls(modo=cfg.modo, capital_inicial=cfg.capital_inicial, efectivo=cfg.capital_inicial, maximo=cfg.capital_inicial)
