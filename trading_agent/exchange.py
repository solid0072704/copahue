"""Exchange adapters: a deterministic paper exchange and a ccxt wrapper.

Every adapter exposes the same small surface. Only `place` changes the world,
and only the risk layer's approved quantity ever reaches it (see loop.py).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from trading_agent.schema import AccountState, BookSnapshot, Candle, Fill, OrderIntent, Side, Trade

log = logging.getLogger(__name__)

TIMEFRAMES = {60: "1m", 180: "3m", 300: "5m", 900: "15m", 1800: "30m", 3600: "1h", 14_400: "4h", 86_400: "1d"}


class ExchangeError(RuntimeError):
    pass


class Exchange(Protocol):
    name: str

    def place(self, intent: OrderIntent, now_ts: int) -> Fill: ...
    def account(self, symbol: str, now_ts: int, mark: float) -> AccountState: ...
    def position_qty(self, symbol: str) -> float: ...
    def gross_other_notional(self, symbol: str, marks: dict[str, float]) -> float: ...


@dataclass
class _Position:
    qty: float = 0.0
    avg_entry: float = 0.0
    realized: float = 0.0


def _utc_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")


class PaperExchange:
    """Fills market orders against the latest book, walking levels, plus slippage and taker fees."""

    name = "paper"

    def __init__(self, starting_equity: float, taker_fee_bps: float = 5.0, slippage_bps: float = 2.0):
        if starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        self.cash = starting_equity
        self.taker_fee_bps = taker_fee_bps
        self.slippage_bps = slippage_bps
        self.positions: dict[str, _Position] = {}
        self.books: dict[str, BookSnapshot] = {}
        self.marks: dict[str, float] = {}
        self.fills: list[Fill] = []
        self.peak_equity = starting_equity
        self.day_key = ""
        self.day_start_equity = starting_equity
        self.realized_today: dict[str, float] = {}
        self.fees_paid = 0.0

    # -- market data pushed by the loop ------------------------------------
    def set_book(self, symbol: str, book: BookSnapshot) -> None:
        self.books[symbol] = book
        self.marks[symbol] = book.mid

    def set_mark(self, symbol: str, price: float) -> None:
        self.marks[symbol] = price

    # -- accounting ----------------------------------------------------------
    def equity(self) -> float:
        value = self.cash
        for sym, pos in self.positions.items():
            mark = self.marks.get(sym, pos.avg_entry)
            value += pos.qty * mark
        return value

    def _roll_day(self, now_ts: int) -> None:
        day = _utc_day(now_ts)
        if day != self.day_key:
            self.day_key = day
            self.day_start_equity = self.equity()
            self.realized_today = {}

    def position_qty(self, symbol: str) -> float:
        return self.positions.get(symbol, _Position()).qty

    def gross_other_notional(self, symbol: str, marks: Optional[dict[str, float]] = None) -> float:
        marks = marks or self.marks
        total = 0.0
        for sym, pos in self.positions.items():
            if sym != symbol and pos.qty:
                total += abs(pos.qty) * marks.get(sym, pos.avg_entry)
        return total

    def account(self, symbol: str, now_ts: int, mark: Optional[float] = None) -> AccountState:
        if mark is not None:
            self.marks[symbol] = mark
        self._roll_day(now_ts)
        eq = self.equity()
        self.peak_equity = max(self.peak_equity, eq)
        pos = self.positions.get(symbol, _Position())
        return AccountState(
            ts=now_ts, equity=eq, position_qty=pos.qty, avg_entry=pos.avg_entry,
            peak_equity=self.peak_equity, day_start_equity=self.day_start_equity,
            realized_pnl_today=self.realized_today.get(symbol, 0.0),
        )

    # -- execution -----------------------------------------------------------
    def _walk_book(self, symbol: str, side: Side, qty: float) -> float:
        book = self.books.get(symbol)
        if book is None:
            mark = self.marks.get(symbol)
            if mark is None:
                raise ExchangeError(f"no market data for {symbol}")
            return mark
        levels = book.asks if side == Side.buy else book.bids
        remaining = qty
        cost = 0.0
        last_price = levels[0][0]
        for price, size in levels:
            take = min(remaining, size)
            cost += take * price
            remaining -= take
            last_price = price
            if remaining <= 1e-15:
                break
        if remaining > 1e-15:   # beyond visible depth: pay an extra 0.5 % for the rest
            penalty = 1.005 if side == Side.buy else 0.995
            cost += remaining * last_price * penalty
        return cost / qty

    def place(self, intent: OrderIntent, now_ts: int) -> Fill:
        if not (math.isfinite(intent.qty) and intent.qty > 0):
            raise ExchangeError("invalid quantity")
        self._roll_day(now_ts)
        base = self._walk_book(intent.symbol, intent.side, intent.qty)
        slip = self.slippage_bps / 1e4
        price = base * (1 + slip) if intent.side == Side.buy else base * (1 - slip)
        notional = price * intent.qty
        fee = notional * self.taker_fee_bps / 1e4
        pos = self.positions.setdefault(intent.symbol, _Position())
        signed = intent.qty if intent.side == Side.buy else -intent.qty
        new_qty = pos.qty + signed
        if pos.qty == 0 or (pos.qty > 0) == (signed > 0):   # opening or adding
            total_cost = pos.avg_entry * abs(pos.qty) + price * abs(signed)
            pos.avg_entry = total_cost / abs(new_qty) if new_qty else 0.0
        else:                                               # reducing or flipping
            closed = min(abs(pos.qty), abs(signed))
            pnl = (price - pos.avg_entry) * closed * (1 if pos.qty > 0 else -1)
            pos.realized += pnl
            self.realized_today[intent.symbol] = self.realized_today.get(intent.symbol, 0.0) + pnl
            if abs(signed) > abs(pos.qty):                  # flipped through zero
                pos.avg_entry = price
            elif new_qty == 0:
                pos.avg_entry = 0.0
        pos.qty = new_qty
        self.cash -= signed * price + fee
        self.fees_paid += fee
        mid = self.marks.get(intent.symbol, price)
        fill = Fill(
            ts=now_ts, symbol=intent.symbol, side=intent.side, qty=intent.qty, price=price, fee=fee,
            intent_id=intent.intent_id, order_id=f"paper-{intent.intent_id}",
            slippage_bps=abs(price / mid - 1.0) * 1e4 if mid else 0.0,
        )
        self.fills.append(fill)
        return fill


class CcxtExchange:
    """Thin wrapper over ccxt. Market data is public; orders need keys and TRADER_LIVE=true.

    Positions are tracked locally from fills (spot-style). Phase 5 of the build
    plan reconciles them with the exchange's own balances before going live.
    """

    name = "ccxt"

    def __init__(self, exchange_id: str, api_key: str = "", api_secret: str = "", *, sandbox: bool = True,
                 live_confirmed: bool = False, starting_equity: float = 0.0, taker_fee_bps: float = 5.0):
        import ccxt  # lazy

        if not hasattr(ccxt, exchange_id):
            raise ExchangeError(f"unknown ccxt exchange: {exchange_id}")
        self.exchange_id = exchange_id
        self.client = getattr(ccxt, exchange_id)({"apiKey": api_key, "secret": api_secret, "enableRateLimit": True})
        if sandbox:
            try:
                self.client.set_sandbox_mode(True)
            except Exception as exc:  # not every exchange has a sandbox
                log.warning("sandbox mode unavailable on %s: %s", exchange_id, exc)
        self.live_confirmed = live_confirmed
        self.sandbox = sandbox
        self.paper = PaperExchange(starting_equity or 1.0, taker_fee_bps=taker_fee_bps, slippage_bps=0.0)
        self.fills: list[Fill] = self.paper.fills

    # -- market data ---------------------------------------------------------
    def fetch_book(self, symbol: str, depth: int = 10) -> BookSnapshot:
        ob = self.client.fetch_order_book(symbol, limit=max(depth, 5))
        ts = int(ob.get("timestamp") or time.time() * 1000)
        bids = tuple((float(p), float(s)) for p, s in ob["bids"][:depth])
        asks = tuple((float(p), float(s)) for p, s in ob["asks"][:depth])
        book = BookSnapshot(ts=ts, bids=bids, asks=asks)
        self.paper.set_book(symbol, book)
        return book

    def fetch_trades(self, symbol: str, since_ts: Optional[int] = None, limit: int = 500) -> list[Trade]:
        raw = self.client.fetch_trades(symbol, since=since_ts, limit=limit)
        out = []
        for t in raw:
            side = Side.buy if t.get("side") == "buy" else Side.sell
            out.append(Trade(ts=int(t["timestamp"]), price=float(t["price"]), size=float(t["amount"]), side=side))
        out.sort(key=lambda t: t.ts)
        return out

    def fetch_candles(self, symbol: str, candle_seconds: int, limit: int = 200, now_ts: Optional[int] = None) -> list[Candle]:
        tf = TIMEFRAMES.get(candle_seconds)
        if tf is None:
            raise ExchangeError(f"unsupported candle_seconds {candle_seconds}")
        rows = self.client.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        now = now_ts or int(time.time() * 1000)
        out = []
        for open_ts, o, h, l, c, v in rows:
            close_ts = int(open_ts) + candle_seconds * 1000
            if close_ts > now:
                continue   # still open: nothing from it is known yet
            out.append(Candle(ts=close_ts, open=float(o), high=float(h), low=float(l), close=float(c), volume=float(v)))
        return out

    # -- accounting ----------------------------------------------------------
    def sync_equity(self, quote: str = "USDT") -> float:
        bal = self.client.fetch_balance()
        total = bal.get("total", {}).get(quote)
        if total is not None:
            self.paper.cash = float(total) - sum(p.qty * self.paper.marks.get(s, p.avg_entry) for s, p in self.paper.positions.items())
        return self.paper.equity()

    def account(self, symbol: str, now_ts: int, mark: Optional[float] = None) -> AccountState:
        return self.paper.account(symbol, now_ts, mark)

    def position_qty(self, symbol: str) -> float:
        return self.paper.position_qty(symbol)

    def gross_other_notional(self, symbol: str, marks: Optional[dict[str, float]] = None) -> float:
        return self.paper.gross_other_notional(symbol, marks)

    # -- execution -----------------------------------------------------------
    def place(self, intent: OrderIntent, now_ts: int) -> Fill:
        if not self.live_confirmed:
            raise ExchangeError("live orders require TRADER_LIVE=true (and an armed kill switch)")
        params: dict[str, Any] = {}
        if intent.reduce_only:
            params["reduceOnly"] = True
        order = self.client.create_order(intent.symbol, "market", intent.side.value, intent.qty, None, params)
        filled = float(order.get("filled") or intent.qty)
        price = float(order.get("average") or order.get("price") or intent.ref_price)
        fee_info = order.get("fee") or {}
        fee = float(fee_info.get("cost") or 0.0)
        # mirror the fill into the local ledger so account() stays consistent
        fill = self.paper.place(intent.with_qty(filled), now_ts)
        fill = Fill(ts=fill.ts, symbol=fill.symbol, side=fill.side, qty=filled, price=price, fee=fee or fill.fee,
                    intent_id=intent.intent_id, order_id=str(order.get("id", "")),
                    slippage_bps=abs(price / intent.ref_price - 1.0) * 1e4)
        self.fills[-1] = fill
        return fill
