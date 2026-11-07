"""Broker Alpaca (cuenta de práctica por defecto; real solo con confirmación explícita).

Documentación: https://docs.alpaca.markets/reference . Las órdenes son "a mercado" y, al
enviarse tras el cierre, se ejecutan en la apertura siguiente, igual que en el backtest.
"""

import logging

import requests

log = logging.getLogger("trading.broker")
URL_PAPEL = "https://paper-api.alpaca.markets"
URL_REAL = "https://api.alpaca.markets"


class BrokerAlpaca:
    def __init__(self, cfg, sesion=None):
        if not cfg.alpaca_configurado:
            raise RuntimeError("Faltan ALPACA_API_KEY / ALPACA_SECRET_KEY.")
        self.cfg = cfg
        self.real = cfg.dinero_real
        self.base = URL_REAL if self.real else URL_PAPEL
        self.sesion = sesion or requests.Session()
        self.sesion.headers.update(
            {"APCA-API-KEY-ID": cfg.alpaca_key, "APCA-API-SECRET-KEY": cfg.alpaca_secret, "Accept": "application/json"}
        )

    @property
    def nombre(self):
        return "Alpaca (dinero REAL)" if self.real else "Alpaca (cuenta de práctica)"

    def _get(self, ruta, **params):
        r = self.sesion.get(self.base + ruta, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, ruta, cuerpo):
        r = self.sesion.post(self.base + ruta, json=cuerpo, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca rechazó la orden ({r.status_code}): {r.text[:300]}")
        return r.json()

    def _delete(self, ruta):
        r = self.sesion.delete(self.base + ruta, timeout=30)
        if r.status_code >= 400 and r.status_code != 404:
            raise RuntimeError(f"Alpaca devolvió {r.status_code}: {r.text[:300]}")
        return r.json() if r.text else {}

    def cuenta(self):
        return self._get("/v2/account")

    def reloj(self):
        return self._get("/v2/clock")

    def posiciones(self):
        return {
            p["symbol"]: {
                "cantidad": float(p["qty"]),
                "precio_promedio": float(p["avg_entry_price"]),
                "precio_actual": float(p.get("current_price") or 0),
                "fecha": "",
            }
            for p in self._get("/v2/positions")
        }

    def ordenes_abiertas(self):
        return self._get("/v2/orders", status="open")

    def comprar(self, ticker, importe):
        return self._post("/v2/orders", {"symbol": ticker, "notional": f"{importe:.2f}", "side": "buy", "type": "market", "time_in_force": "day"})

    def vender(self, ticker, cantidad):
        return self._post("/v2/orders", {"symbol": ticker, "qty": f"{cantidad:.6f}".rstrip("0").rstrip("."), "side": "sell", "type": "market", "time_in_force": "day"})

    def cerrar_posicion(self, ticker):
        return self._delete(f"/v2/positions/{ticker}")

    def sincronizar(self, cartera):
        """Copia a la cartera local el efectivo y las posiciones reales de la cuenta."""
        cuenta = self.cuenta()
        posiciones = self.posiciones()
        for t, pos in posiciones.items():
            pos["fecha"] = cartera.posiciones.get(t, {}).get("fecha", "")
        cartera.efectivo = float(cuenta["cash"])
        cartera.posiciones = posiciones
        cartera.modo = self.cfg.modo
        patrimonio = float(cuenta.get("equity") or cuenta.get("portfolio_value") or cartera.efectivo)
        if not cartera.curva:
            cartera.capital_inicial = patrimonio
            cartera.maximo = patrimonio
        cartera.maximo = max(cartera.maximo, patrimonio)
        return patrimonio

    def enviar(self, cartera, pendiente, precios):
        """Convierte las órdenes decididas por el motor en órdenes de mercado. Devuelve lo enviado."""
        if not pendiente:
            return []
        enviadas = []
        fecha = pendiente.get("fecha", "")
        for t in pendiente.get("vender", []):
            if t in cartera.posiciones:
                self.cerrar_posicion(t)
                enviadas.append(self._anotar(cartera, fecha, t, "venta", cartera.posiciones[t]["cantidad"], precios.get(t, 0), pendiente.get("motivo_venta", "venta")))
        objetivo = pendiente.get("objetivo")
        if objetivo is not None:
            patrimonio = cartera.patrimonio(precios)
            for t, pos in list(cartera.posiciones.items()):
                if t in pendiente.get("vender", []):
                    continue
                valor = pos["cantidad"] * precios.get(t, pos["precio_promedio"])
                exceso = valor - objetivo.get(t, 0.0) * patrimonio
                if objetivo.get(t, 0.0) == 0:
                    self.cerrar_posicion(t)
                    enviadas.append(self._anotar(cartera, fecha, t, "venta", pos["cantidad"], precios.get(t, 0), "rebalanceo: sale de la cartera"))
                elif exceso > max(25.0, 0.01 * patrimonio):
                    cantidad = exceso / precios[t]
                    self.vender(t, cantidad)
                    enviadas.append(self._anotar(cartera, fecha, t, "venta", cantidad, precios[t], "rebalanceo: reduce peso"))
            for t, peso in objetivo.items():
                valor = cartera.posiciones.get(t, {"cantidad": 0})["cantidad"] * precios.get(t, 0)
                falta = peso * patrimonio - valor
                if falta > max(25.0, 0.01 * patrimonio):
                    self.comprar(t, falta)
                    enviadas.append(self._anotar(cartera, fecha, t, "compra", falta / precios[t] if precios.get(t) else 0, precios.get(t, 0), "rebalanceo: entra o sube peso"))
        return enviadas

    @staticmethod
    def _anotar(cartera, fecha, ticker, lado, cantidad, precio, motivo):
        op = {
            "fecha": fecha,
            "ticker": ticker,
            "lado": lado,
            "cantidad": round(float(cantidad), 6),
            "precio": round(float(precio), 4),
            "importe": round(float(cantidad) * float(precio), 2),
            "comision": 0.0,
            "motivo": motivo + " (orden enviada al broker; se ejecuta en la apertura)",
        }
        cartera.operaciones.append(op)
        return op
