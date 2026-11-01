"""Market data feeds. Each yields CandleTick objects: one closed candle per
symbol with the order book at the close and the trades during the candle.

SyntheticFeed is seeded and fully deterministic (tests, backtests). ReplayFeed
reads CSV candles. CcxtFeed polls a live exchange once per candle.
"""

from __future__ import annotations

import csv
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Protocol

from trading_agent.schema import BookSnapshot, Candle, Side, Trade

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandleTick:
    symbol: str
    candle: Candle
    book: BookSnapshot
    trades: tuple[Trade, ...] = ()

    @property
    def ts(self) -> int:
        return self.candle.ts


class Feed(Protocol):
    name: str

    def ticks(self) -> Iterator[CandleTick]: ...


def synth_book(ts: int, mid: float, spread_bps: float = 2.0, depth: int = 10, skew: float = 0.0,
               base_size: float = 1.0) -> BookSnapshot:
    """A plausible book around `mid`. `skew` in [-1, 1] tilts depth to bids (+) or asks (-)."""
    half = mid * spread_bps / 2e4
    bids = tuple((mid - half - mid * 1e-4 * k, base_size * (1 + k) * (1 + 0.5 * skew)) for k in range(depth))
    asks = tuple((mid + half + mid * 1e-4 * k, base_size * (1 + k) * (1 - 0.5 * skew)) for k in range(depth))
    return BookSnapshot(ts=ts, bids=bids, asks=asks)


@dataclass
class SyntheticRegime:
    drift_per_candle: float = 0.0     # log drift
    vol_per_candle: float = 0.004     # log stdev
    spread_bps: float = 2.0
    flow_bias: float = 0.0            # aggressor buy share - 0.5


class SyntheticFeed:
    """Seeded geometric random walk with regime segments; deterministic for a given seed."""

    name = "synthetic"

    def __init__(self, symbols: list[str], candle_seconds: int, n_candles: int, *, seed: int = 0,
                 start_ts: int = 1_760_000_000_000, start_prices: Optional[dict[str, float]] = None,
                 regimes: Optional[list[tuple[int, SyntheticRegime]]] = None, trades_per_candle: int = 20):
        self.symbols = list(symbols)
        self.candle_seconds = candle_seconds
        self.n_candles = n_candles
        self.seed = seed
        self.start_ts = start_ts
        self.start_prices = start_prices or {s: 100.0 * (i + 1) for i, s in enumerate(self.symbols)}
        self.regimes = regimes or [(n_candles, SyntheticRegime())]
        self.trades_per_candle = trades_per_candle

    def _regime_at(self, i: int) -> SyntheticRegime:
        acc = 0
        for length, regime in self.regimes:
            acc += length
            if i < acc:
                return regime
        return self.regimes[-1][1]

    def ticks(self) -> Iterator[CandleTick]:
        rng = random.Random(self.seed)
        prices = dict(self.start_prices)
        step = self.candle_seconds * 1000
        for i in range(self.n_candles):
            regime = self._regime_at(i)
            close_ts = self.start_ts + (i + 1) * step
            for symbol in self.symbols:
                o = prices[symbol]
                r = rng.gauss(regime.drift_per_candle, regime.vol_per_candle)
                c = o * math.exp(r)
                wick = abs(rng.gauss(0, regime.vol_per_candle / 2))
                h = max(o, c) * (1 + wick)
                l = min(o, c) * (1 - wick)
                v = max(0.1, rng.lognormvariate(2.0, 0.5))
                prices[symbol] = c
                candle = Candle(ts=close_ts, open=o, high=h, low=l, close=c, volume=v)
                trades = []
                for j in range(self.trades_per_candle):
                    t_ts = close_ts - step + int((j + 1) * step / (self.trades_per_candle + 1))
                    side = Side.buy if rng.random() < 0.5 + regime.flow_bias else Side.sell
                    trades.append(Trade(ts=t_ts, price=o + (c - o) * (j + 1) / self.trades_per_candle,
                                        size=max(0.001, rng.lognormvariate(-2.0, 0.8)), side=side))
                book = synth_book(close_ts, c, regime.spread_bps, skew=rng.uniform(-0.3, 0.3) + regime.flow_bias)
                yield CandleTick(symbol=symbol, candle=candle, book=book, trades=tuple(trades))


class ReplayFeed:
    """CSV candles: ts,open,high,low,close,volume (ts = close time in ms, or open time with open_time=True)."""

    name = "replay"

    def __init__(self, files: dict[str, str], candle_seconds: int, *, open_time: bool = False, spread_bps: float = 2.0):
        self.files = dict(files)
        self.candle_seconds = candle_seconds
        self.open_time = open_time
        self.spread_bps = spread_bps

    def _load(self, symbol: str, path: str) -> list[Candle]:
        out = []
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                ts = int(float(row["ts"]))
                if self.open_time:
                    ts += self.candle_seconds * 1000
                out.append(Candle(ts=ts, open=float(row["open"]), high=float(row["high"]), low=float(row["low"]),
                                  close=float(row["close"]), volume=float(row.get("volume", 0) or 0)))
        out.sort(key=lambda c: c.ts)
        return out

    def ticks(self) -> Iterator[CandleTick]:
        series = {s: self._load(s, p) for s, p in self.files.items()}
        merged = sorted(((c.ts, s, c) for s, cs in series.items() for c in cs), key=lambda x: (x[0], x[1]))
        for ts, symbol, candle in merged:
            yield CandleTick(symbol=symbol, candle=candle, book=synth_book(ts, candle.close, self.spread_bps), trades=())


class CcxtFeed:
    """Polls a live exchange once per closed candle. Blocks between candles."""

    name = "ccxt"

    def __init__(self, exchange, symbols: list[str], candle_seconds: int, *, book_depth: int = 10,
                 warmup: int = 200, sleep_fn=time.sleep, now_fn=lambda: int(time.time() * 1000), max_ticks: Optional[int] = None):
        self.exchange = exchange
        self.symbols = list(symbols)
        self.candle_seconds = candle_seconds
        self.book_depth = book_depth
        self.warmup = warmup
        self.sleep = sleep_fn
        self.now = now_fn
        self.max_ticks = max_ticks
        self.last_close: dict[str, int] = {}

    def ticks(self) -> Iterator[CandleTick]:
        step = self.candle_seconds * 1000
        emitted = 0
        # warm-up: historical closed candles so the state engine has context
        for symbol in self.symbols:
            for candle in self.exchange.fetch_candles(symbol, self.candle_seconds, limit=self.warmup, now_ts=self.now()):
                # historical candles carry a synthetic book at the close: no real book exists for the past
                yield CandleTick(symbol=symbol, candle=candle, book=synth_book(candle.ts, candle.close), trades=())
                self.last_close[symbol] = candle.ts
        while self.max_ticks is None or emitted < self.max_ticks:
            now = self.now()
            next_close = (now // step + 1) * step
            self.sleep(max(0.0, (next_close - now) / 1000.0 + 2.0))   # 2 s grace for the exchange to close the candle
            now = self.now()
            for symbol in self.symbols:
                candles = self.exchange.fetch_candles(symbol, self.candle_seconds, limit=3, now_ts=now)
                new = [c for c in candles if c.ts > self.last_close.get(symbol, 0)]
                if not new:
                    continue
                book = self.exchange.fetch_book(symbol, self.book_depth)
                trades = tuple(t for t in self.exchange.fetch_trades(symbol, since_ts=new[-1].ts - step) if t.ts <= new[-1].ts)
                for c in new[:-1]:
                    yield CandleTick(symbol=symbol, candle=c, book=synth_book(c.ts, c.close), trades=())
                # the live book is fetched just after the close, so its ts is later than the candle's:
                # the loop snapshots as of the latest timestamp it holds, never earlier.
                yield CandleTick(symbol=symbol, candle=new[-1], book=book, trades=trades)
                self.last_close[symbol] = new[-1].ts
                emitted += 1
