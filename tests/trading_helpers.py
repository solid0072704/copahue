"""Shared builders for the trading agent tests (no network, no keys)."""

from __future__ import annotations

import math
import random

from trading_agent.config import PolicyParams, RiskLimits, TraderConfig
from trading_agent.schema import (
    AccountState,
    BookSnapshot,
    Candle,
    Direction,
    JevDecision,
    Regime,
    RiskState,
    SetupSpec,
    Side,
    Trade,
)
from trading_agent.state_engine import StateEngine

T0 = 1_760_000_000_000
STEP = 300_000


def make_config(**overrides) -> TraderConfig:
    cfg = TraderConfig(symbols=["BTC/USDT"], candle_seconds=300, starting_equity=10_000.0,
                       journal_path=":memory:", kill_switch_path="", heartbeat_path="", universe_path="")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_book(ts: int, mid: float, spread_bps: float = 2.0, skew: float = 0.0) -> BookSnapshot:
    half = mid * spread_bps / 2e4
    bids = tuple((mid - half - mid * 1e-4 * k, (1 + k) * (1 + skew)) for k in range(10))
    asks = tuple((mid + half + mid * 1e-4 * k, (1 + k) * (1 - skew)) for k in range(10))
    return BookSnapshot(ts=ts, bids=bids, asks=asks)


def make_candles(n: int, start: float = 100.0, drift: float = 0.0, vol: float = 0.004, seed: int = 1, t0: int = T0):
    rng = random.Random(seed)
    px = start
    out = []
    for i in range(n):
        o = px
        px = o * math.exp(rng.gauss(drift, vol))
        h = max(o, px) * 1.0005
        l = min(o, px) * 0.9995
        out.append(Candle(ts=t0 + (i + 1) * STEP, open=o, high=h, low=l, close=px, volume=10 + rng.random()))
    return out


def make_engine(n: int = 80, *, drift: float = 0.0, vol: float = 0.004, seed: int = 1, setup: SetupSpec | None = None,
                limits: RiskLimits | None = None, account: AccountState | None = None, flow_side: Side | None = None):
    limits = limits or RiskLimits()
    eng = StateEngine("BTC/USDT", 300, limits, setup=setup)
    eng.ingest_account(AccountState(ts=T0, equity=10_000.0))
    candles = make_candles(n, drift=drift, vol=vol, seed=seed)
    rng = random.Random(seed + 1)
    for c in candles:
        for j in range(10):
            side = flow_side or (Side.buy if rng.random() < 0.5 else Side.sell)
            eng.ingest_trade(Trade(ts=c.ts - STEP + (j + 1) * 25_000, price=c.close, size=0.1, side=side))
        eng.ingest_candle(c)
        eng.ingest_book(make_book(c.ts, c.close))
    last = candles[-1]
    eng.ingest_account(account or AccountState(ts=last.ts, equity=10_000.0))
    return eng, candles


def decision(**overrides) -> JevDecision:
    base = dict(regime=Regime.trending, direction=Direction.long, toxic_flow=False, setup_quality=3,
                risk_state=RiskState.safe, confidence=0.9, source="test")
    base.update(overrides)
    return JevDecision(**base)


class ScriptedReflex:
    """Returns the same decision every call (or a per-call sequence)."""

    name = "scripted"

    def __init__(self, decisions):
        self.decisions = list(decisions) if isinstance(decisions, (list, tuple)) else [decisions]
        self.calls = 0

    def judge(self, snapshot):
        d = self.decisions[min(self.calls, len(self.decisions) - 1)]
        self.calls += 1
        return d


class FailingReflex:
    name = "failing"

    def judge(self, snapshot):
        from trading_agent.reflex import ReflexError

        raise ReflexError("boom")
