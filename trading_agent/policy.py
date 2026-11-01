"""Deterministic policy: (snapshot, decision) -> hold | order | flatten | escalate.

Every threshold is a PolicyParams field owned by the operator. The reflex model
supplies judgements; this module decides whether they become an order and how
big it is (fractional Kelly, hard-capped at quarter Kelly). The risk layer then
has the final word on the order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from trading_agent.config import PolicyParams, RiskLimits
from trading_agent.schema import (
    Direction,
    JevDecision,
    MarketSnapshot,
    OrderIntent,
    Regime,
    RiskState,
    Side,
    worse_risk_state,
)


class Outcome(str, Enum):
    hold = "hold"
    order = "order"
    flatten = "flatten"
    escalate = "escalate"


@dataclass(frozen=True)
class PolicyResult:
    outcome: Outcome
    reason: str
    intent: Optional[OrderIntent] = None
    effective_risk_state: RiskState = RiskState.safe
    kelly_full: float = 0.0
    size_frac: float = 0.0
    gates: tuple[tuple[str, bool], ...] = field(default_factory=tuple)

    @property
    def failed_gates(self) -> list[str]:
        return [name for name, ok in self.gates if not ok]


def kelly_full(p: float, b: float) -> float:
    """Full Kelly fraction for win probability p and reward/risk b: f* = p - (1 - p) / b."""
    if b <= 0 or not (0.0 <= p <= 1.0):
        return 0.0
    return max(0.0, p - (1.0 - p) / b)


def size_fraction(p: float, b: float, params: PolicyParams) -> tuple[float, float]:
    """(full Kelly, fraction of equity to risk) with the quarter-Kelly hard cap."""
    f_star = kelly_full(p, b)
    frac = min(params.kelly_fraction, params.kelly_cap, 0.25) * f_star
    return f_star, frac


class Policy:
    def __init__(self, params: PolicyParams, limits: RiskLimits):
        params.validate()
        limits.validate()
        self.params = params
        self.limits = limits

    def decide(
        self,
        snap: MarketSnapshot,
        decision: JevDecision,
        *,
        previous_regime: Optional[Regime] = None,
        entries_paused: bool = False,
    ) -> PolicyResult:
        p = self.params
        eff_risk = worse_risk_state(snap.risk_state, decision.risk_state)   # model may only tighten
        has_position = abs(snap.position_qty) > 0
        pos_dir = Direction.long if snap.position_qty > 0 else Direction.short if snap.position_qty < 0 else Direction.neutral

        # 1. escalate to the brain: deep re-read, no new risk this candle -----
        # A directional call the reflex is unsure about goes to the brain. A neutral verdict is
        # "no edge", not doubt: it holds. While entries are already paused by a previous
        # escalation, low confidence holds instead of calling the brain every candle.
        if (decision.direction != Direction.neutral and decision.confidence < p.escalate_below_confidence
                and not entries_paused):
            return PolicyResult(Outcome.escalate, f"confidence {decision.confidence:.2f} below {p.escalate_below_confidence}", effective_risk_state=eff_risk)
        if decision.regime == Regime.crisis and previous_regime != Regime.crisis:
            return PolicyResult(Outcome.escalate, "regime flipped to crisis", effective_risk_state=eff_risk)

        # 2. manage an open position --------------------------------------------
        if has_position:
            flatten_reason = None
            if snap.setup_invalidated:
                flatten_reason = "setup invalidated"
            elif eff_risk == RiskState.reduce:
                flatten_reason = "risk state reduce"
            elif decision.regime == Regime.crisis:
                flatten_reason = "crisis regime"
            elif decision.direction != Direction.neutral and decision.direction != pos_dir and decision.confidence > p.min_confidence:
                flatten_reason = f"direction flipped to {decision.direction.value} with confidence {decision.confidence:.2f}"
            if flatten_reason:
                side = Side.sell if snap.position_qty > 0 else Side.buy
                intent = OrderIntent(
                    symbol=snap.symbol, side=side, qty=abs(snap.position_qty), ref_price=snap.mid,
                    reason=f"flatten: {flatten_reason}", reduce_only=True,
                )
                return PolicyResult(Outcome.flatten, flatten_reason, intent=intent, effective_risk_state=eff_risk)

        # 3. entry gates (all in code) --------------------------------------------
        gates = (
            ("entries_not_paused", not entries_paused),
            ("flat", not has_position),                                   # no pyramiding
            ("setup_quality>=min", decision.setup_quality >= p.min_setup_quality),
            ("confidence>min", decision.confidence > p.min_confidence),
            ("risk_state_safe", eff_risk == RiskState.safe),
            ("direction_not_neutral", decision.direction != Direction.neutral),
            ("flow_not_toxic", not decision.toxic_flow),
            ("regime_not_crisis", decision.regime != Regime.crisis),
            ("setup_not_invalidated", not snap.setup_invalidated),
            ("direction_matches_setup", snap.setup_direction == Direction.neutral or decision.direction == snap.setup_direction),
            ("regime_fits_setup", snap.setup_direction == Direction.neutral or not snap.setup_regime_fit
             or decision.regime.value in snap.setup_regime_fit),
            ("book_fresh", snap.book_age_ms <= snap.candle_seconds * 1000),
        )
        failed = [name for name, ok in gates if not ok]
        if failed:
            return PolicyResult(Outcome.hold, "gate failed: " + ", ".join(failed), effective_risk_state=eff_risk, gates=gates)

        # 4. size with fractional Kelly, capped at quarter Kelly -----------------
        b = snap.setup_reward_risk if snap.setup_reward_risk > 0 else p.default_reward_risk
        f_star, frac = size_fraction(decision.confidence, b, p)
        frac = min(frac, self.limits.max_position_frac)
        if frac <= 0:
            return PolicyResult(Outcome.hold, "kelly fraction is zero", effective_risk_state=eff_risk, gates=gates, kelly_full=f_star)
        qty = frac * snap.equity / snap.mid
        atr_abs = snap.mid * snap.atr_pct / 100.0
        stop_dist = max(atr_abs * p.stop_atr_mult, snap.mid * 0.001)
        if decision.direction == Direction.long:
            side, stop, take = Side.buy, snap.mid - stop_dist, snap.mid + stop_dist * b
        else:
            side, stop, take = Side.sell, snap.mid + stop_dist, snap.mid - stop_dist * b
        intent = OrderIntent(
            symbol=snap.symbol, side=side, qty=qty, ref_price=snap.mid,
            reason=f"entry {decision.direction.value} q={decision.setup_quality} p={decision.confidence:.2f} b={b:.2f} f*={f_star:.3f} frac={frac:.4f}",
            stop_price=stop, take_profit=take,
        )
        return PolicyResult(Outcome.order, "all gates passed", intent=intent, effective_risk_state=eff_risk,
                            kelly_full=f_star, size_frac=frac, gates=gates)
