"""Conexión con el broker que custodia tu dinero y ejecuta las órdenes.

- BrokerSimulado: cartera de práctica guardada en SQLite. Depositas dinero ficticio y las órdenes
  se ejecutan al último precio de mercado (con comisión). Ideal para probar antes de arriesgar.
- BrokerAlpaca: Alpaca Markets (acciones y ETFs de EE.UU., acepta fracciones de acción). Con
  ALPACA_PAPER=true usa su cuenta de práctica; con dinero real tu plata queda en Alpaca, nunca aquí.
"""

import requests


class ErrorBroker(Exception):
    pass


def id_cliente(prop):
    """Identificador único de la orden en el broker: reenviarla nunca crea una orden duplicada."""
    return f"copahue-{prop['plan_id']}-{prop['id']}"


class BrokerSimulado:
    nombre = "simulado"
    acepta_depositos = True

    def __init__(self, db, mercado, comision_pct=0.1):
        self.db = db
        self.mercado = mercado
        self.comision = comision_pct / 100

    def cuenta(self):
        return {
            "efectivo": float(self.db.get_meta("sim_efectivo", 0) or 0),
            "posiciones": {
                s: {"cantidad": p["cantidad"], "costo_promedio": p["costo_promedio"]}
                for s, p in self.db.sim_posiciones().items()
            },
        }

    def depositar(self, monto, nota="Depósito"):
        if monto <= 0:
            raise ErrorBroker("El monto debe ser mayor a cero.")
        self.db.set_meta("sim_efectivo", self.cuenta()["efectivo"] + monto)
        self.db.registrar_deposito(monto, nota)

    def retirar(self, monto):
        efectivo = self.cuenta()["efectivo"]
        if monto <= 0 or monto > efectivo + 1e-9:
            raise ErrorBroker(f"Solo hay US$ {efectivo:,.2f} en efectivo para retirar.")
        self.db.set_meta("sim_efectivo", efectivo - monto)
        self.db.registrar_deposito(-monto, "Retiro")

    def ejecutar(self, prop):
        precio = self.mercado.precio(prop["simbolo"])
        cuenta = self.cuenta()
        efectivo = cuenta["efectivo"]
        pos = cuenta["posiciones"].get(prop["simbolo"], {"cantidad": 0.0, "costo_promedio": 0.0})
        cantidad = prop["cantidad"]
        bruto = cantidad * precio
        comision = bruto * self.comision
        if prop["lado"] == "compra":
            if bruto + comision > efectivo + 1e-6:
                raise ErrorBroker(f"Efectivo insuficiente: la compra cuesta US$ {bruto + comision:,.2f} y hay US$ {efectivo:,.2f}.")
            nueva = pos["cantidad"] + cantidad
            costo = (pos["cantidad"] * pos["costo_promedio"] + bruto + comision) / nueva
            self.db.sim_guardar_posicion(prop["simbolo"], nueva, costo)
            self.db.set_meta("sim_efectivo", efectivo - bruto - comision)
        else:
            if cantidad > pos["cantidad"] + 1e-9:
                raise ErrorBroker(f"Solo tienes {pos['cantidad']:.4f} {prop['simbolo']} para vender.")
            self.db.sim_guardar_posicion(prop["simbolo"], pos["cantidad"] - cantidad, pos["costo_promedio"])
            self.db.set_meta("sim_efectivo", efectivo + bruto - comision)
        return {"estado": "ejecutada", "precio": precio, "cantidad": cantidad, "id_broker": f"sim-{prop['id']}"}

    def estado_orden(self, prop):
        return {"estado": prop["estado"]}


class BrokerAlpaca:
    nombre = "alpaca"
    acepta_depositos = False
    URL_PAPER = "https://paper-api.alpaca.markets"
    URL_REAL = "https://api.alpaca.markets"

    def __init__(self, cfg, sesion=None):
        listo, motivo = cfg.broker_listo
        if not listo:
            raise ErrorBroker(motivo)
        self.base = self.URL_PAPER if cfg.alpaca_paper else self.URL_REAL
        self.sesion = sesion or requests.Session()
        self.sesion.headers.update(
            {"APCA-API-KEY-ID": cfg.alpaca_api_key, "APCA-API-SECRET-KEY": cfg.alpaca_secret_key}
        )

    def _pedir(self, metodo, ruta, **kwargs):
        try:
            r = self.sesion.request(metodo, self.base + ruta, timeout=30, **kwargs)
        except requests.RequestException as e:
            raise ErrorBroker(f"Sin conexión con Alpaca: {e}") from e
        if r.status_code >= 400:
            try:
                mensaje = r.json().get("message", r.text)
            except ValueError:
                mensaje = r.text
            raise ErrorBroker(f"Alpaca respondió {r.status_code}: {mensaje}")
        return r.json()

    def cuenta(self):
        cuenta = self._pedir("GET", "/v2/account")
        if cuenta.get("trading_blocked") or cuenta.get("account_blocked"):
            raise ErrorBroker("Tu cuenta de Alpaca está bloqueada para operar.")
        return {
            "efectivo": float(cuenta["cash"]),
            "posiciones": {
                p["symbol"]: {"cantidad": float(p["qty"]), "costo_promedio": float(p["avg_entry_price"])}
                for p in self._pedir("GET", "/v2/positions")
                if float(p["qty"]) > 0
            },
        }

    def depositar(self, monto, nota=""):
        raise ErrorBroker("Con Alpaca los depósitos se hacen desde su app o web; el agente verá el saldo solo.")

    def _traducir(self, orden):
        estado = orden.get("status")
        llenado = float(orden.get("filled_qty") or 0)
        precio = float(orden["filled_avg_price"]) if orden.get("filled_avg_price") else None
        if estado == "filled" or (estado in ("canceled", "expired", "done_for_day") and llenado > 0):
            return {"estado": "ejecutada", "precio": precio, "cantidad": llenado, "id_broker": orden["id"]}
        if estado in ("canceled", "expired", "rejected", "suspended"):
            return {"estado": "fallida", "error": f"Alpaca dejó la orden como '{estado}'.", "id_broker": orden["id"]}
        return {"estado": "enviada", "id_broker": orden["id"]}

    def ejecutar(self, prop):
        cuerpo = {
            "symbol": prop["simbolo"],
            "qty": f"{prop['cantidad']:.6f}".rstrip("0").rstrip("."),
            "side": "buy" if prop["lado"] == "compra" else "sell",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": id_cliente(prop),
        }
        try:
            orden = self._pedir("POST", "/v2/orders", json=cuerpo)
        except ErrorBroker:
            # Si la orden ya había llegado (p. ej. se cortó la conexión), la recuperamos en vez de duplicarla.
            existente = self._orden_existente(prop)
            if not existente:
                raise
            orden = existente
        return self._traducir(orden)

    def _orden_existente(self, prop):
        try:
            return self._pedir("GET", "/v2/orders:by_client_order_id", params={"client_order_id": id_cliente(prop)})
        except ErrorBroker:
            return None

    def estado_orden(self, prop):
        orden = self._orden_existente(prop)
        return self._traducir(orden) if orden else {"estado": prop["estado"]}


def crear_broker(cfg, db, mercado):
    if cfg.broker == "alpaca":
        return BrokerAlpaca(cfg)
    return BrokerSimulado(db, mercado, cfg.comision_simulada_pct)
