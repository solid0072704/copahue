"""Deterministic state engine: order book + trades + candles + account -> one
compact snapshot under 400 tokens, with strict causal timestamps.

Rules:
  * Every event carries an epoch-ms timestamp. Streams must arrive in order;
    an older event than the last one of its stream raises CausalityError.
  * `snapshot(as_of_ts)` only reads events with ts <= as_of_ts. Nothing from the
    future can leak into a field, which makes replay and live byte-identical.
  * Every number is computed here in code: mid, spread, imbalance, realized
    volatility, inventory, drawdown and limit usage. The model only judges.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Optional, Sequence

from trading_agent.config import RiskLimits
from trading_agent.risk import compute_usage, risk_state_from_usage
from trading_agent.schema import (
    AccountState,
    BookSnapshot,
    Candle,
    Direction,
    MarketSnapshot,
    SetupSpec,
    Side,
    Trade,
)

SECONDS_PER_YEAR = 365 * 86_400


class CausalityError(ValueError):
    """An event arrived out of order or a snapshot would read the future."""


class InsufficientData(ValueError):
    """Not enough history for a snapshot yet."""


# ---------------------------------------------------------------------------
# Pure functions (unit-tested individually)
# ---------------------------------------------------------------------------
def log_returns(closes: Sequence[float]) -> list[float]:
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0 and closes[i] > 0]


def realized_vol_pct(closes: Sequence[float], candle_seconds: int, window: int = 48) -> float:
    """Annualised realized volatility (percent) from the last `window` candle log returns."""
    rets = log_returns(list(closes)[-(window + 1):])
    if len(rets) < 2:
        return 0.0
    sd = statistics.stdev(rets)
    return sd * math.sqrt(SECONDS_PER_YEAR / candle_seconds) * 100.0


def atr_pct(candles: Sequence[Candle], n: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    recent = list(candles)[-(n + 1):]
    trs = []
    for prev, cur in zip(recent[:-1], recent[1:]):
        trs.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
    if not trs or recent[-1].close <= 0:
        return 0.0
    return (sum(trs) / len(trs)) / recent[-1].close * 100.0


def pct_return(closes: Sequence[float], k: int) -> float:
    if len(closes) <= k or closes[-1 - k] <= 0:
        return 0.0
    return (closes[-1] / closes[-1 - k] - 1.0) * 100.0


def range_position(candles: Sequence[Candle], n: int = 48) -> float:
    recent = list(candles)[-n:]
    if not recent:
        return 0.5
    hi = max(c.high for c in recent)
    lo = min(c.low for c in recent)
    if hi - lo <= 0:
        return 0.5
    return min(1.0, max(0.0, (recent[-1].close - lo) / (hi - lo)))


def volume_zscore(volumes: Sequence[float], n: int = 48) -> float:
    vols = list(volumes)
    hist = vols[-(n + 1):-1]
    if len(hist) < 5:
        return 0.0
    mean = statistics.fmean(hist)
    sd = statistics.stdev(hist)
    if sd <= 0:
        return 0.0
    return (vols[-1] - mean) / sd


def book_imbalance(book: BookSnapshot, depth: int = 10) -> float:
    bid_depth = sum(size for _, size in book.bids[:depth])
    ask_depth = sum(size for _, size in book.asks[:depth])
    total = bid_depth + ask_depth
    if total <= 0:
        return 0.0
    return (bid_depth - ask_depth) / total


def microprice_dev_bps(book: BookSnapshot) -> float:
    bid, bid_size = book.bids[0]
    ask, ask_size = book.asks[0]
    if bid_size + ask_size <= 0:
        return 0.0
    micro = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
    return (micro - book.mid) / book.mid * 1e4


def spread_bps(book: BookSnapshot) -> float:
    return (book.best_ask - book.best_bid) / book.mid * 1e4


def signed_flow(trades: Sequence[Trade]) -> float:
    buy = sum(t.size for t in trades if t.side == Side.buy)
    sell = sum(t.size for t in trades if t.side == Side.sell)
    total = buy + sell
    if total <= 0:
        return 0.0
    return (buy - sell) / total


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class StateEngine:
    def __init__(
        self,
        symbol: str,
        candle_seconds: int,
        limits: RiskLimits,
        *,
        book_depth: int = 10,
        flow_window_ms: Optional[int] = None,
        max_candles: int = 256,
        max_trades: int = 4000,
        setup: Optional[SetupSpec] = None,
        max_snapshot_tokens: int = 400,
    ):
        if candle_seconds <= 0:
            raise ValueError("candle_seconds must be positive")
        self.symbol = symbol
        self.candle_seconds = candle_seconds
        self.limits = limits
        self.book_depth = book_depth
        self.flow_window_ms = flow_window_ms or candle_seconds * 1000
        self.setup = setup
        self.max_snapshot_tokens = max_snapshot_tokens
        self._candles: Deque[Candle] = deque(maxlen=max_candles)
        self._books: Deque[BookSnapshot] = deque(maxlen=max_candles)
        self._trades: Deque[Trade] = deque(maxlen=max_trades)
        self._accounts: Deque[AccountState] = deque(maxlen=64)
        self._last_ts = {"candle": -1, "book": -1, "trade": -1, "account": -1}

    # -- ingestion (strictly causal) ----------------------------------------
    def _accept(self, stream: str, ts: int, strict: bool) -> None:
        last = self._last_ts[stream]
        if ts < last or (strict and ts == last):
            raise CausalityError(f"{stream} event at {ts} arrived after {last}")
        self._last_ts[stream] = ts

    def ingest_candle(self, candle: Candle) -> None:
        self._accept("candle", candle.ts, strict=True)
        self._candles.append(candle)

    def ingest_book(self, book: BookSnapshot) -> None:
        self._accept("book", book.ts, strict=False)
        self._books.append(book)

    def ingest_trade(self, trade: Trade) -> None:
        self._accept("trade", trade.ts, strict=False)
        self._trades.append(trade)

    def ingest_account(self, account: AccountState) -> None:
        self._accept("account", account.ts, strict=False)
        self._accounts.append(account)

    @property
    def candles(self) -> list[Candle]:
        return list(self._candles)

    def last_candle(self) -> Optional[Candle]:
        return self._candles[-1] if self._candles else None

    def last_book(self) -> Optional[BookSnapshot]:
        return self._books[-1] if self._books else None

    # -- snapshot ------------------------------------------------------------
    def snapshot(self, as_of_ts: int) -> MarketSnapshot:
        candles = [c for c in self._candles if c.ts <= as_of_ts]
        book = next((b for b in reversed(self._books) if b.ts <= as_of_ts), None)
        account = next((a for a in reversed(self._accounts) if a.ts <= as_of_ts), None)
        if not candles:
            raise InsufficientData("no closed candle at or before as_of_ts")
        if book is None:
            raise InsufficientData("no order book at or before as_of_ts")
        if account is None:
            raise InsufficientData("no account state at or before as_of_ts")

        window_start = as_of_ts - self.flow_window_ms
        trades = [t for t in self._trades if window_start < t.ts <= as_of_ts]
        last_trade_ts = next((t.ts for t in reversed(self._trades) if t.ts <= as_of_ts), None)

        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        mid = book.mid
        usage = compute_usage(account, mid, self.limits)
        risk_state = risk_state_from_usage(usage, self.limits)
        upnl = 0.0
        if abs(account.position_qty) > 0 and account.avg_entry > 0:
            upnl = (mid / account.avg_entry - 1.0) * 100.0 * (1.0 if account.position_qty > 0 else -1.0)

        when = datetime.fromtimestamp(as_of_ts / 1000.0, tz=timezone.utc)
        setup = self.setup
        snap = MarketSnapshot(
            ts=as_of_ts,
            symbol=self.symbol,
            candle_seconds=self.candle_seconds,
            mid=mid,
            spread_bps=spread_bps(book),
            imbalance=book_imbalance(book, self.book_depth),
            microprice_dev_bps=microprice_dev_bps(book),
            book_age_ms=as_of_ts - book.ts,
            ret_1_bps=pct_return(closes, 1) * 100.0,
            ret_12_pct=pct_return(closes, 12),
            ret_48_pct=pct_return(closes, 48),
            realized_vol_pct=realized_vol_pct(closes, self.candle_seconds),
            atr_pct=atr_pct(candles),
            range_pos_48=range_position(candles),
            volume_z=volume_zscore(volumes),
            candles_seen=len(candles),
            flow=signed_flow(trades),
            trade_rate_per_min=len(trades) / (self.flow_window_ms / 60_000.0),
            trades_age_ms=(as_of_ts - last_trade_ts) if last_trade_ts is not None else -1,
            position_qty=account.position_qty,
            position_frac=usage.position_frac,
            unrealized_pnl_pct=upnl,
            equity=account.equity,
            drawdown=usage.drawdown,
            daily_pnl_pct=usage.daily_pnl_pct * 100.0,
            dd_usage=usage.dd_usage,
            daily_usage=usage.daily_usage,
            pos_usage=usage.pos_usage,
            risk_state=risk_state,
            hour_utc=when.hour,
            weekday=when.weekday(),
            setup_direction=setup.direction if setup else Direction.neutral,
            setup_reward_risk=setup.reward_risk if setup else 1.5,
            setup_invalidated=setup.is_invalidated(mid) if setup else False,
            setup_thesis=setup.thesis if setup else "",
            setup_regime_fit=tuple(setup.regime_fit) if setup else (),
        )
        if snap.approx_tokens() > self.max_snapshot_tokens:
            snap = snap.model_copy(update={"setup_thesis": ""})
            if snap.approx_tokens() > self.max_snapshot_tokens:
                raise ValueError(f"snapshot exceeds {self.max_snapshot_tokens} tokens: {snap.approx_tokens()}")
        return snap
