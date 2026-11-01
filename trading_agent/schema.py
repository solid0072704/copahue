"""Typed contracts shared by every layer: market events, the compact snapshot,
the Jev decision schema, order intents and fills.

The Jev schema is the *only* thing the reflex model sees and returns. Every
threshold that turns a decision into an order lives in code (policy.py,
risk.py), never here.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations (the typed choices Jev answers with)
# ---------------------------------------------------------------------------
class Regime(str, Enum):
    trending = "trending"
    mean_reverting = "mean_reverting"
    high_vol = "high_vol"
    crisis = "crisis"


class Direction(str, Enum):
    long = "long"
    short = "short"
    neutral = "neutral"


class RiskState(str, Enum):
    safe = "safe"
    near_limit = "near_limit"
    reduce = "reduce"


RISK_ORDER = {RiskState.safe: 0, RiskState.near_limit: 1, RiskState.reduce: 2}


def worse_risk_state(a: RiskState, b: RiskState) -> RiskState:
    """The more conservative of two risk states. The model may only make it worse."""
    return a if RISK_ORDER[a] >= RISK_ORDER[b] else b


class Side(str, Enum):
    buy = "buy"
    sell = "sell"


# ---------------------------------------------------------------------------
# Jev question set: evaluated in parallel in ONE call. Names are the keys of
# the `answers` object in the response. Keep in sync with JevDecision.
# ---------------------------------------------------------------------------
JEV_QUESTIONS: dict[str, dict[str, Any]] = {
    "regime": {
        "type": "choice",
        "instructions": "Which regime best describes this market over the next 12 candles?",
        "criteria": {
            "trending": "Directional drift dominates noise; pullbacks are shallow.",
            "mean_reverting": "Moves fade; price oscillates inside a range.",
            "high_vol": "Volatility elevated vs history; direction unclear; gaps and wicks.",
            "crisis": "Disorderly: spreads blown out, depth gone, forced flows, cascading liquidations.",
        },
    },
    "direction": {
        "type": "choice",
        "instructions": "Which direction has positive expectancy after costs over the next 12 candles?",
        "criteria": {
            "long": "Price more likely up than down net of spread and fees.",
            "short": "Price more likely down than up net of spread and fees.",
            "neutral": "No edge after costs; stay flat.",
        },
    },
    "toxic_flow": {
        "type": "noul",
        "instructions": "Recent order flow is informed or toxic: entering now trades against better-informed participants.",
    },
    "setup_quality": {
        "type": "score",
        "instructions": "Quality of the setup described in `setup` given this state.",
        "criteria": [
            "0: no setup, thesis invalidated or wrong regime",
            "1: weak alignment, poor location or stale catalyst",
            "2: good alignment of thesis, regime and location",
            "3: excellent: thesis, regime, flow and location all agree",
        ],
    },
    "risk_state": {
        "type": "choice",
        "instructions": "Given inventory, drawdown and limit usage, how should risk be treated?",
        "criteria": {
            "safe": "Limits far away; normal sizing is appropriate.",
            "near_limit": "Some limit usage is high; only small or no new risk.",
            "reduce": "Exposure should be cut; no new risk.",
        },
    },
}


class JevDecision(BaseModel):
    """One typed judgement for one symbol on one candle.

    `confidence` is the calibrated probability that `direction` is right over the
    horizon. It is what policy.py gates on (> 0.80) and sizes with (Kelly).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    regime: Regime
    direction: Direction
    toxic_flow: bool
    setup_quality: int = Field(ge=0, le=3)
    risk_state: RiskState
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, dict[str, float]] = Field(default_factory=dict)
    source: str = "unknown"      # jev | claude | rules | recorded
    model: str = ""
    latency_ms: float = 0.0

    @field_validator("probabilities")
    @classmethod
    def _probs_in_range(cls, v: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        for q, dist in v.items():
            for opt, p in dist.items():
                if not (0.0 <= p <= 1.0) or math.isnan(p):
                    raise ValueError(f"probability out of range for {q}/{opt}: {p}")
        return v

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Strict JSON schema (used for the Claude structured-output fallback)."""
        return {
            "type": "object",
            "properties": {
                "regime": {"type": "string", "enum": [r.value for r in Regime]},
                "direction": {"type": "string", "enum": [d.value for d in Direction]},
                "toxic_flow": {"type": "boolean"},
                "setup_quality": {"type": "integer", "minimum": 0, "maximum": 3},
                "risk_state": {"type": "string", "enum": [r.value for r in RiskState]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["regime", "direction", "toxic_flow", "setup_quality", "risk_state", "confidence"],
            "additionalProperties": False,
        }

    @classmethod
    def safe_hold(cls, source: str = "fallback", reason_model: str = "") -> "JevDecision":
        """What the loop uses when the reflex layer fails: never fires."""
        return cls(
            regime=Regime.high_vol,
            direction=Direction.neutral,
            toxic_flow=True,
            setup_quality=0,
            risk_state=RiskState.near_limit,
            confidence=0.0,
            source=source,
            model=reason_model,
        )


# ---------------------------------------------------------------------------
# Market events (immutable, timestamped in epoch milliseconds, UTC)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candle:
    ts: int            # close time of the candle (ms). Nothing from after ts is known.
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if not (self.low <= min(self.open, self.close) and self.high >= max(self.open, self.close)):
            raise ValueError(f"inconsistent candle at {self.ts}")
        if self.volume < 0:
            raise ValueError("negative volume")


@dataclass(frozen=True)
class BookSnapshot:
    ts: int
    bids: tuple[tuple[float, float], ...]   # (price, size) best first (descending)
    asks: tuple[tuple[float, float], ...]   # (price, size) best first (ascending)

    def __post_init__(self) -> None:
        if not self.bids or not self.asks:
            raise ValueError("empty book side")
        if self.bids[0][0] >= self.asks[0][0]:
            raise ValueError("crossed book")
        for i in range(1, len(self.bids)):
            if self.bids[i][0] > self.bids[i - 1][0]:
                raise ValueError("bids not descending")
        for i in range(1, len(self.asks)):
            if self.asks[i][0] < self.asks[i - 1][0]:
                raise ValueError("asks not ascending")

    @property
    def best_bid(self) -> float:
        return self.bids[0][0]

    @property
    def best_ask(self) -> float:
        return self.asks[0][0]

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0


@dataclass(frozen=True)
class Trade:
    ts: int
    price: float
    size: float
    side: Side          # aggressor side


@dataclass(frozen=True)
class AccountState:
    ts: int
    equity: float
    position_qty: float = 0.0       # signed base quantity for the symbol
    avg_entry: float = 0.0
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    realized_pnl_today: float = 0.0

    def __post_init__(self) -> None:
        if self.equity <= 0:
            raise ValueError("equity must be positive")
        object.__setattr__(self, "peak_equity", max(self.peak_equity, self.equity) if self.peak_equity else self.equity)
        object.__setattr__(self, "day_start_equity", self.day_start_equity or self.equity)


# ---------------------------------------------------------------------------
# Setup spec (compiled from research finalists) as the model sees it
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SetupSpec:
    symbol: str
    direction: Direction                 # thesis direction (long/short); neutral = no bias
    thesis: str = ""
    reward_risk: float = 1.5             # b in the Kelly formula
    regime_fit: tuple[str, ...] = ("trending",)
    invalidate_above: Optional[float] = None   # price level that kills a short thesis
    invalidate_below: Optional[float] = None   # price level that kills a long thesis
    unlock_risk_90d: str = "not_sourced"
    data_quality: str = "C"

    def is_invalidated(self, price: float) -> bool:
        if self.invalidate_above is not None and price >= self.invalidate_above:
            return True
        if self.invalidate_below is not None and price <= self.invalidate_below:
            return True
        return False


# ---------------------------------------------------------------------------
# The compact snapshot: the ONLY thing Jev sees. Under 400 tokens.
# ---------------------------------------------------------------------------
class MarketSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ts: int
    symbol: str
    candle_seconds: int
    # price / book
    mid: float
    spread_bps: float
    imbalance: float            # (bid depth - ask depth) / total, -1..1
    microprice_dev_bps: float   # (microprice - mid) / mid in bps
    book_age_ms: int
    # candles
    ret_1_bps: float
    ret_12_pct: float
    ret_48_pct: float
    realized_vol_pct: float     # annualised, from candle log returns
    atr_pct: float              # ATR(14) / mid
    range_pos_48: float         # 0..1 within the 48-candle range
    volume_z: float
    candles_seen: int
    # trades
    flow: float                 # signed aggressor volume share in the flow window, -1..1
    trade_rate_per_min: float
    trades_age_ms: int
    # inventory and risk (computed in code, model may only tighten)
    position_qty: float
    position_frac: float        # |position notional| / equity
    unrealized_pnl_pct: float
    equity: float
    drawdown: float             # 1 - equity / peak
    daily_pnl_pct: float
    dd_usage: float             # drawdown / max_drawdown
    daily_usage: float          # daily loss / max_daily_loss (0 when profitable)
    pos_usage: float            # position_frac / max_position_frac
    risk_state: RiskState
    # calendar
    hour_utc: int
    weekday: int
    # setup
    setup_direction: Direction = Direction.neutral
    setup_reward_risk: float = 1.5
    setup_invalidated: bool = False
    setup_thesis: str = ""
    setup_regime_fit: tuple[str, ...] = ()

    def compact(self) -> dict[str, Any]:
        """Short keys and rounded numbers: deterministic and small."""
        return {
            "t": self.ts,
            "sym": self.symbol,
            "tf": self.candle_seconds,
            "px": {"mid": _r(self.mid, 6), "spr": _r(self.spread_bps, 2), "imb": _r(self.imbalance, 3),
                    "micro": _r(self.microprice_dev_bps, 2), "age": self.book_age_ms},
            "c": {"r1": _r(self.ret_1_bps, 1), "r12": _r(self.ret_12_pct, 2), "r48": _r(self.ret_48_pct, 2),
                   "rv": _r(self.realized_vol_pct, 1), "atr": _r(self.atr_pct, 3), "rng": _r(self.range_pos_48, 2),
                   "vz": _r(self.volume_z, 2), "n": self.candles_seen},
            "f": {"flow": _r(self.flow, 3), "rate": _r(self.trade_rate_per_min, 1), "age": self.trades_age_ms},
            "inv": {"qty": _r(self.position_qty, 6), "frac": _r(self.position_frac, 3), "upnl": _r(self.unrealized_pnl_pct, 2)},
            "risk": {"dd": _r(self.drawdown, 4), "day": _r(self.daily_pnl_pct, 3), "u_dd": _r(self.dd_usage, 2),
                      "u_day": _r(self.daily_usage, 2), "u_pos": _r(self.pos_usage, 2), "state": self.risk_state.value},
            "cal": {"h": self.hour_utc, "wd": self.weekday},
            "setup": {"dir": self.setup_direction.value, "rr": _r(self.setup_reward_risk, 2),
                       "inv": self.setup_invalidated, "fit": list(self.setup_regime_fit), "th": self.setup_thesis[:120]},
        }

    def compact_json(self) -> str:
        return json.dumps(self.compact(), separators=(",", ":"), sort_keys=True)

    def approx_tokens(self) -> int:
        return approx_tokens(self.compact_json())


def approx_tokens(text: str) -> int:
    """Conservative token estimate (3 chars per token) for the budget check."""
    return math.ceil(len(text) / 3.0)


def _r(x: float, nd: int) -> float:
    if x is None or math.isnan(x) or math.isinf(x):
        return 0.0
    return round(float(x), nd)


# ---------------------------------------------------------------------------
# Orders and fills
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Side
    qty: float
    ref_price: float
    reason: str = ""
    reduce_only: bool = False
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    decision_id: str = ""
    intent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def signed_qty(self) -> float:
        return self.qty if self.side == Side.buy else -self.qty

    def with_qty(self, qty: float) -> "OrderIntent":
        return OrderIntent(
            symbol=self.symbol, side=self.side, qty=qty, ref_price=self.ref_price, reason=self.reason,
            reduce_only=self.reduce_only, stop_price=self.stop_price, take_profit=self.take_profit,
            decision_id=self.decision_id, intent_id=self.intent_id,
        )


@dataclass(frozen=True)
class Fill:
    ts: int
    symbol: str
    side: Side
    qty: float
    price: float
    fee: float
    intent_id: str
    order_id: str = ""
    slippage_bps: float = 0.0

    @property
    def notional(self) -> float:
        return self.qty * self.price
