"""Deterministic risk layer. No model output reaches this module.

Three hard rules, checked before EVERY order:
  1. The kill switch must be armed (self-tested) and not triggered.
  2. Max drawdown (15 % by default) triggers the kill switch; only reduce-only
     orders pass afterwards.
  3. Max daily loss, max position, max gross exposure and the order rate limit
     either block the order or clip its size.

The policy may ask; the risk layer decides. See docs/trading/ARCHITECTURE.md.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

from trading_agent.config import RiskLimits
from trading_agent.schema import AccountState, OrderIntent, RiskState

log = logging.getLogger(__name__)

PRICE_STALENESS_LIMIT = 0.02   # reference price must be within 2 % of the live mid
QTY_EPS = 1e-12


@dataclass(frozen=True)
class RiskUsage:
    drawdown: float
    daily_pnl_pct: float
    position_frac: float
    gross_frac: float
    dd_usage: float
    daily_usage: float
    pos_usage: float
    gross_usage: float

    @property
    def max_usage(self) -> float:
        return max(self.dd_usage, self.daily_usage, self.pos_usage, self.gross_usage)


def compute_usage(account: AccountState, mid: float, limits: RiskLimits, gross_other_notional: float = 0.0) -> RiskUsage:
    """How much of each hard limit is used right now (1.0 = at the limit)."""
    if mid <= 0 or not math.isfinite(mid):
        raise ValueError("mid must be a positive finite price")
    drawdown = max(0.0, 1.0 - account.equity / account.peak_equity)
    daily = account.equity / account.day_start_equity - 1.0
    pos_notional = abs(account.position_qty) * mid
    position_frac = pos_notional / account.equity
    gross_frac = (pos_notional + max(0.0, gross_other_notional)) / account.equity
    return RiskUsage(
        drawdown=drawdown,
        daily_pnl_pct=daily,
        position_frac=position_frac,
        gross_frac=gross_frac,
        dd_usage=drawdown / limits.max_drawdown,
        daily_usage=max(0.0, -daily) / limits.max_daily_loss,
        pos_usage=position_frac / limits.max_position_frac,
        gross_usage=gross_frac / limits.max_gross_frac,
    )


def risk_state_from_usage(usage: RiskUsage, limits: RiskLimits) -> RiskState:
    m = usage.max_usage
    if m >= 1.0:
        return RiskState.reduce
    if m >= limits.near_limit_ratio:
        return RiskState.near_limit
    return RiskState.safe


class KillSwitch:
    """Armed means the mechanism is live and self-tested; triggered means pulled.

    The operator pulls it by creating the file at `path` (for example
    ``touch data/KILL_SWITCH``). Code pulls it on a hard breach. Re-arming is an
    explicit operator action (`reset`), never done by the loop.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._armed = False
        self._triggered = False
        self.reason = ""
        self.triggered_at: Optional[float] = None

    def arm(self) -> None:
        if self.path:
            directory = os.path.dirname(self.path) or "."
            os.makedirs(directory, exist_ok=True)
            probe = os.path.join(directory, ".kill_switch_probe")
            with open(probe, "w", encoding="utf-8") as fh:   # self-test: we must be able to write the trigger
                fh.write(str(time.time()))
            os.remove(probe)
            if os.path.exists(self.path):
                self._triggered = True
                self.reason = self.reason or "trigger file present at start"
        self._armed = True

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def triggered(self) -> bool:
        if not self._triggered and self.path and os.path.exists(self.path):
            self._triggered = True
            self.reason = "operator trigger file"
            self.triggered_at = time.time()
        return self._triggered

    def trigger(self, reason: str) -> None:
        if not self._triggered:
            self._triggered = True
            self.reason = reason
            self.triggered_at = time.time()
            log.error("KILL SWITCH TRIGGERED: %s", reason)
        if self.path:
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {reason}\n")
            except OSError as exc:  # pragma: no cover - best effort
                log.error("could not persist kill switch file: %s", exc)

    def check(self) -> tuple[bool, str]:
        if not self._armed:
            return False, "kill switch not armed"
        if self.triggered:
            return False, f"kill switch triggered: {self.reason}"
        return True, "ok"

    def reset(self) -> None:
        """Operator only. Clears the trigger file and the flag; does not arm."""
        if self.path and os.path.exists(self.path):
            os.remove(self.path)
        self._triggered = False
        self.reason = ""
        self.triggered_at = None


@dataclass(frozen=True)
class RiskVerdict:
    allowed: bool
    reason: str
    qty: float
    reduce_only: bool = False
    usage: Optional[RiskUsage] = None
    kill_switch_fired: bool = False

    @property
    def blocked(self) -> bool:
        return not self.allowed


def _reduces_exposure(before: float, signed_qty: float) -> bool:
    """True when the order trades against the open position.

    Such an order is always clipped to |position| by the caller: the risk layer
    never lets one order flip a position through zero (flatten first, re-enter
    on a later candle through the normal gates).
    """
    return abs(before) > QTY_EPS and signed_qty * before < 0


class RiskLayer:
    def __init__(self, limits: RiskLimits, kill_switch: KillSwitch):
        limits.validate()
        self.limits = limits
        self.kill_switch = kill_switch

    # -- read-only helpers ---------------------------------------------------
    def usage(self, account: AccountState, mid: float, gross_other_notional: float = 0.0) -> RiskUsage:
        return compute_usage(account, mid, self.limits, gross_other_notional)

    def risk_state(self, account: AccountState, mid: float, gross_other_notional: float = 0.0) -> RiskState:
        return risk_state_from_usage(self.usage(account, mid, gross_other_notional), self.limits)

    # -- the gate ------------------------------------------------------------
    def evaluate(
        self,
        intent: OrderIntent,
        account: AccountState,
        mid: float,
        *,
        orders_last_hour: int = 0,
        gross_other_notional: float = 0.0,
    ) -> RiskVerdict:
        limits = self.limits
        ok, why = self.kill_switch.check()
        triggered = not ok and self.kill_switch.armed

        # 0. sanity ----------------------------------------------------------
        if not (math.isfinite(intent.qty) and intent.qty > 0):
            return RiskVerdict(False, "invalid quantity", 0.0)
        if not (math.isfinite(mid) and mid > 0):
            return RiskVerdict(False, "invalid mid price", 0.0)
        if not (math.isfinite(intent.ref_price) and intent.ref_price > 0) or abs(intent.ref_price / mid - 1.0) > PRICE_STALENESS_LIMIT:
            return RiskVerdict(False, "reference price stale vs mid", 0.0)

        usage = compute_usage(account, mid, limits, gross_other_notional)
        before = account.position_qty
        signed = intent.signed_qty()
        reduces = _reduces_exposure(before, signed)

        # 1. kill switch -----------------------------------------------------
        if not self.kill_switch.armed:
            return RiskVerdict(False, why, 0.0, usage=usage)
        fired = False
        if usage.drawdown >= limits.max_drawdown and not self.kill_switch.triggered:
            self.kill_switch.trigger(f"max drawdown breached: {usage.drawdown:.4f} >= {limits.max_drawdown}")
            fired = True
            triggered = True
        if triggered:
            if intent.reduce_only and reduces:
                qty = min(intent.qty, abs(before))
                return RiskVerdict(True, "reduce-only while kill switch triggered", qty, reduce_only=True, usage=usage, kill_switch_fired=fired)
            return RiskVerdict(False, f"kill switch triggered: {self.kill_switch.reason}", 0.0, usage=usage, kill_switch_fired=fired)

        # 2. reduce-only situations -----------------------------------------
        if reduces:
            qty = min(intent.qty, abs(before))   # never flip through zero on a reduce
            return RiskVerdict(True, "reduces exposure", qty, reduce_only=True, usage=usage)
        if intent.reduce_only:
            return RiskVerdict(False, "reduce_only order does not reduce exposure", 0.0, usage=usage)
        if usage.daily_usage >= 1.0:
            return RiskVerdict(False, f"daily loss limit reached: {usage.daily_pnl_pct:.4f}", 0.0, usage=usage)
        if risk_state_from_usage(usage, limits) == RiskState.reduce:
            return RiskVerdict(False, "risk state is reduce: no new risk", 0.0, usage=usage)
        if orders_last_hour >= limits.max_orders_per_hour:
            return RiskVerdict(False, "order rate limit reached", 0.0, usage=usage)

        # 3. size caps -------------------------------------------------------
        equity = account.equity
        max_order_qty = limits.max_order_notional_frac * equity / mid
        max_pos_qty = limits.max_position_frac * equity / mid
        max_gross_qty = max(0.0, limits.max_gross_frac * equity - max(0.0, gross_other_notional)) / mid
        # the order adds to (or opens) a position: room so that |before| + qty <= cap
        qty = min(intent.qty, max_order_qty, max_pos_qty - abs(before), max_gross_qty - abs(before))
        if qty <= QTY_EPS:
            return RiskVerdict(False, "no room under position or gross limits", 0.0, usage=usage)
        reason = "ok" if qty >= intent.qty - QTY_EPS else f"clipped from {intent.qty:.8f} to {qty:.8f}"
        return RiskVerdict(True, reason, qty, usage=usage)
