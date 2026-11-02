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


@dataclass
class ReconciliationReport:
    exchange_positions: dict[str, float]
    local_positions: dict[str, float]
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches


class CcxtExchange:
    """Thin wrapper over ccxt for market data and, in live mode, orders.

    market_type "spot": long only. market_type "swap": USDT-margined perpetuals
    (Binance: exchange id `binanceusdm`, symbols like `UNI/USDT:USDT`), which is
    what the short setups need. Leverage is pinned (default 1x) and margin is
    isolated; sizing is already capped by RiskLimits, so leverage only changes the
    margin posted, never the notional.

    Positions are mirrored into a local PaperExchange ledger from our own fills.
    `reconcile` compares that ledger with the venue; live start refuses to run on a
    mismatch unless the operator adopts the venue's positions.
    """

    name = "ccxt"

    def __init__(self, exchange_id: str, api_key: str = "", api_secret: str = "", *, market_type: str = "spot",
                 sandbox: bool = True, live_confirmed: bool = False, starting_equity: float = 0.0,
                 taker_fee_bps: float = 5.0, leverage: int = 1, client: Any = None):
        self.market_type = "swap" if market_type == "swap" else "spot"
        if self.market_type == "swap" and exchange_id == "binance":
            exchange_id = "binanceusdm"
        self.exchange_id = exchange_id
        if client is None:
            import ccxt  # lazy

            if not hasattr(ccxt, exchange_id):
                raise ExchangeError(f"unknown ccxt exchange: {exchange_id}")
            client = getattr(ccxt, exchange_id)({
                "apiKey": api_key, "secret": api_secret, "enableRateLimit": True,
                "options": {"defaultType": self.market_type},
            })
            if sandbox:
                try:
                    client.set_sandbox_mode(True)
                except Exception as exc:  # not every exchange has a sandbox
                    log.warning("sandbox mode unavailable on %s: %s", exchange_id, exc)
        self.client = client
        self.live_confirmed = live_confirmed
        self.sandbox = sandbox
        self.leverage = max(1, int(leverage))
        self.paper = PaperExchange(starting_equity or 1.0, taker_fee_bps=taker_fee_bps, slippage_bps=0.0)
        self.fills: list[Fill] = self.paper.fills
        self._markets_loaded = False
        self._equity_synced = False
        self._last_sync_ts = 0
        self.sync_interval_ms = 3_600_000     # live: re-read venue equity hourly (fees, funding, manual moves)

    # -- symbols -------------------------------------------------------------
    def market_symbol(self, symbol: str) -> str:
        """Universe symbol (UNI/USDT) -> venue symbol (UNI/USDT:USDT on perpetuals)."""
        if self.market_type == "swap" and ":" not in symbol and "/" in symbol:
            return f"{symbol}:{symbol.split('/')[1]}"
        return symbol

    def universe_symbol(self, market_symbol: str) -> str:
        return market_symbol.split(":")[0]

    def load_markets(self) -> None:
        if not self._markets_loaded:
            self.client.load_markets()
            self._markets_loaded = True

    def check_markets(self, symbols: list[str]) -> list[dict[str, Any]]:
        """One row per symbol: found, taker fee, min notional, min amount."""
        self.load_markets()
        rows = []
        for symbol in symbols:
            ms = self.market_symbol(symbol)
            m = (getattr(self.client, "markets", None) or {}).get(ms)
            if not m:
                rows.append({"symbol": symbol, "market_symbol": ms, "found": False})
                continue
            limits = m.get("limits") or {}
            rows.append({
                "symbol": symbol, "market_symbol": ms, "found": True,
                "type": "swap" if m.get("swap") else ("spot" if m.get("spot") else m.get("type")),
                "taker_bps": round(float(m.get("taker") or 0.0) * 1e4, 2),
                "min_cost": (limits.get("cost") or {}).get("min"),
                "min_amount": (limits.get("amount") or {}).get("min"),
                "contract_size": m.get("contractSize") or 1,
            })
        return rows

    def configure_markets(self, symbols: list[str], margin_mode: str = "isolated") -> list[dict[str, Any]]:
        """Perpetuals only: isolated margin and pinned leverage per symbol (best effort, logged)."""
        results = []
        if self.market_type != "swap":
            return results
        for symbol in symbols:
            ms = self.market_symbol(symbol)
            row = {"symbol": symbol, "margin_mode": None, "leverage": None}
            try:
                self.client.set_margin_mode(margin_mode, ms)
                row["margin_mode"] = margin_mode
            except Exception as exc:   # "No need to change margin type" is the usual answer
                row["margin_mode"] = f"unchanged ({str(exc)[:80]})"
            try:
                self.client.set_leverage(self.leverage, ms)
                row["leverage"] = self.leverage
            except Exception as exc:
                row["leverage"] = f"error ({str(exc)[:80]})"
            results.append(row)
        return results

    # -- market data ---------------------------------------------------------
    def fetch_book(self, symbol: str, depth: int = 10) -> BookSnapshot:
        ob = self.client.fetch_order_book(self.market_symbol(symbol), limit=max(depth, 5))
        ts = int(ob.get("timestamp") or time.time() * 1000)
        bids = tuple((float(p), float(s)) for p, s in ob["bids"][:depth])
        asks = tuple((float(p), float(s)) for p, s in ob["asks"][:depth])
        book = BookSnapshot(ts=ts, bids=bids, asks=asks)
        self.paper.set_book(symbol, book)
        return book

    def fetch_trades(self, symbol: str, since_ts: Optional[int] = None, limit: int = 500) -> list[Trade]:
        raw = self.client.fetch_trades(self.market_symbol(symbol), since=since_ts, limit=limit)
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
        rows = self.client.fetch_ohlcv(self.market_symbol(symbol), timeframe=tf, limit=limit)
        now = now_ts or int(time.time() * 1000)
        out = []
        for open_ts, o, h, l, c, v in rows:
            close_ts = int(open_ts) + candle_seconds * 1000
            if close_ts > now:
                continue   # still open: nothing from it is known yet
            out.append(Candle(ts=close_ts, open=float(o), high=float(h), low=float(l), close=float(c), volume=float(v)))
        return out

    # -- venue truth: positions and equity ------------------------------------
    def fetch_positions_map(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        """{universe symbol: {qty (signed), entry, unrealized}} as the venue reports it."""
        out: dict[str, dict[str, float]] = {s: {"qty": 0.0, "entry": 0.0, "unrealized": 0.0} for s in symbols}
        if self.market_type == "swap":
            wanted = {self.market_symbol(s): s for s in symbols}
            for p in self.client.fetch_positions(list(wanted)) or []:
                symbol = wanted.get(p.get("symbol"))
                if symbol is None:
                    continue
                contracts = float(p.get("contracts") or 0.0)
                size = contracts * float(p.get("contractSize") or 1.0)
                side = str(p.get("side") or "").lower()
                qty = -size if side == "short" else size
                out[symbol] = {"qty": qty, "entry": float(p.get("entryPrice") or 0.0),
                               "unrealized": float(p.get("unrealizedPnl") or 0.0)}
            return out
        balance = self.client.fetch_balance()
        for symbol in symbols:
            base = symbol.split("/")[0]
            qty = float((balance.get(base) or {}).get("total") or 0.0)
            out[symbol] = {"qty": qty, "entry": 0.0, "unrealized": 0.0}
        return out

    def reconcile(self, symbols: list[str], tolerance: float = 1e-6) -> ReconciliationReport:
        venue = self.fetch_positions_map(symbols)
        local = {s: self.paper.position_qty(s) for s in symbols}
        mismatches = []
        for s in symbols:
            vq, lq = venue[s]["qty"], local[s]
            if abs(vq - lq) > max(tolerance, 1e-6 * max(abs(vq), abs(lq))):
                mismatches.append({"symbol": s, "venue_qty": vq, "local_qty": lq, "venue_entry": venue[s]["entry"]})
        return ReconciliationReport({s: venue[s]["qty"] for s in symbols}, local, mismatches)

    def adopt_positions(self, symbols: list[str]) -> ReconciliationReport:
        """Operator action: make the venue's positions the local truth."""
        venue = self.fetch_positions_map(symbols)
        for s in symbols:
            pos = self.paper.positions.setdefault(s, _Position())
            pos.qty = venue[s]["qty"]
            pos.avg_entry = venue[s]["entry"] or self.paper.marks.get(s, 0.0)
        return self.reconcile(symbols)

    def equity_from_exchange(self, quote: str = "USDT", symbols: Optional[list[str]] = None) -> float:
        balance = self.client.fetch_balance()
        if self.market_type == "swap":
            info = balance.get("info") or {}
            for key in ("totalMarginBalance",):          # Binance USDT-M account: wallet + unrealized
                if info.get(key) is not None:
                    return float(info[key])
            for asset in info.get("assets") or []:
                if asset.get("asset") == quote and asset.get("marginBalance") is not None:
                    return float(asset["marginBalance"])
            total = float((balance.get("total") or {}).get(quote) or 0.0)
            unrealized = sum(p["unrealized"] for p in self.fetch_positions_map(symbols or []).values())
            return total + unrealized
        total = float((balance.get("total") or {}).get(quote) or 0.0)
        for symbol in symbols or []:
            base = symbol.split("/")[0]
            qty = float((balance.get(base) or {}).get("total") or 0.0)
            total += qty * self.paper.marks.get(symbol, 0.0)
        return total

    def sync_equity(self, quote: str = "USDT", symbols: Optional[list[str]] = None) -> float:
        """Make the local ledger's equity equal the venue's (captures fees, funding, manual moves)."""
        equity = self.equity_from_exchange(quote, symbols)
        if equity <= 0:
            raise ExchangeError("venue reports non-positive equity")
        held = sum(p.qty * self.paper.marks.get(s, p.avg_entry) for s, p in self.paper.positions.items())
        self.paper.cash = equity - held
        if not self._equity_synced:
            self.paper.peak_equity = equity
            self.paper.day_start_equity = equity
            self._equity_synced = True
        self.paper.peak_equity = max(self.paper.peak_equity, equity)
        return equity

    def account(self, symbol: str, now_ts: int, mark: Optional[float] = None) -> AccountState:
        if self.live_confirmed and self._equity_synced and now_ts - self._last_sync_ts >= self.sync_interval_ms:
            try:
                self.sync_equity(symbols=list(self.paper.positions) or [symbol])
            except Exception as exc:   # a venue hiccup must not stop the loop; the ledger stays as it was
                log.warning("periodic equity sync failed: %s", exc)
            self._last_sync_ts = now_ts
        return self.paper.account(symbol, now_ts, mark)

    def position_qty(self, symbol: str) -> float:
        return self.paper.position_qty(symbol)

    def gross_other_notional(self, symbol: str, marks: Optional[dict[str, float]] = None) -> float:
        return self.paper.gross_other_notional(symbol, marks)

    # -- execution -----------------------------------------------------------
    def place(self, intent: OrderIntent, now_ts: int) -> Fill:
        if not self.live_confirmed:
            raise ExchangeError("live orders require TRADER_LIVE=true (and an armed kill switch)")
        self.load_markets()
        ms = self.market_symbol(intent.symbol)
        market = (getattr(self.client, "markets", None) or {}).get(ms) or {}
        amount = float(self.client.amount_to_precision(ms, intent.qty)) if hasattr(self.client, "amount_to_precision") else intent.qty
        limits = market.get("limits") or {}
        min_amount = (limits.get("amount") or {}).get("min")
        min_cost = (limits.get("cost") or {}).get("min")
        if amount <= 0 or (min_amount and amount < float(min_amount)):
            raise ExchangeError(f"amount {amount} below the venue minimum {min_amount} for {ms}")
        if min_cost and amount * intent.ref_price < float(min_cost) and not intent.reduce_only:
            raise ExchangeError(f"notional {amount * intent.ref_price:.2f} below the venue minimum {min_cost} for {ms}")
        params: dict[str, Any] = {"reduceOnly": True} if intent.reduce_only else {}
        order = self.client.create_order(ms, "market", intent.side.value, amount, None, params)
        filled = float(order.get("filled") or 0.0)
        if filled <= 0:
            raise ExchangeError(f"order {order.get('id')} reported no fill")
        price = float(order.get("average") or order.get("price") or intent.ref_price)
        fee_info = order.get("fee") or {}
        fee = float(fee_info.get("cost") or 0.0)
        # mirror the fill into the local ledger so account() stays consistent
        mirrored = self.paper.place(intent.with_qty(filled), now_ts)
        fill = Fill(ts=mirrored.ts, symbol=intent.symbol, side=intent.side, qty=filled, price=price,
                    fee=fee or mirrored.fee, intent_id=intent.intent_id, order_id=str(order.get("id", "")),
                    slippage_bps=abs(price / intent.ref_price - 1.0) * 1e4 if intent.ref_price else 0.0)
        self.fills[-1] = fill
        return fill

    def flatten_all(self, symbols: list[str], now_ts: Optional[int] = None) -> list[Fill]:
        """Emergency: close every venue position with reduce-only market orders."""
        now_ts = now_ts or int(time.time() * 1000)
        fills = []
        for symbol, pos in self.fetch_positions_map(symbols).items():
            qty = pos["qty"]
            if abs(qty) <= 0:
                continue
            mark = self.fetch_book(symbol).mid
            side = Side.sell if qty > 0 else Side.buy
            intent = OrderIntent(symbol=symbol, side=side, qty=abs(qty), ref_price=mark, reason="flatten_all",
                                 reduce_only=True, intent_id=f"flatten-{now_ts}-{symbol}")
            local = self.paper.positions.setdefault(symbol, _Position())
            if abs(local.qty - qty) > 1e-9:          # adopt first so the mirror ledger does not go negative
                local.qty, local.avg_entry = qty, pos["entry"] or mark
            fills.append(self.place(intent, now_ts))
        return fills
