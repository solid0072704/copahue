"""The live loop: one closed candle in, at most one order out.

    feed tick -> state engine (code) -> snapshot (< 400 tokens)
             -> kill switch check (code)
             -> reflex judge (Jev)              ~ one typed call per candle
             -> policy (code: gates + Kelly)    -> hold | order | flatten | escalate
             -> brain (Opus 5.5) only on escalate
             -> risk layer (code)               -> allow | clip | block
             -> exchange                        -> fill
             -> journal (everything)

Any failure in the model layer degrades to HOLD. Any hard breach degrades to
FLATTEN and the kill switch stays triggered until an operator resets it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from trading_agent.brain import NullBrain
from trading_agent.config import TraderConfig
from trading_agent.exchange import ExchangeError
from trading_agent.feed import CandleTick
from trading_agent.journal import Journal
from trading_agent.policy import Outcome, Policy, PolicyResult
from trading_agent.reflex import ReflexError
from trading_agent.risk import KillSwitch, RiskLayer
from trading_agent.schema import (
    AccountState,
    Fill,
    JevDecision,
    MarketSnapshot,
    OrderIntent,
    Regime,
    SetupSpec,
    Side,
)
from trading_agent.state_engine import CausalityError, InsufficientData, StateEngine
from trading_agent.universe import setup_for

log = logging.getLogger(__name__)


@dataclass
class SymbolContext:
    engine: StateEngine
    candle_index: int = 0
    previous_regime: Optional[Regime] = None
    paused_until: int = 0


@dataclass
class LoopStats:
    ticks: int = 0
    warmup_skipped: int = 0
    decisions: int = 0
    holds: int = 0
    orders_requested: int = 0
    orders_blocked: int = 0
    orders_clipped: int = 0
    fills: int = 0
    flattens: int = 0
    escalations: int = 0
    reflex_failures: int = 0
    causality_drops: int = 0
    kill_switch_blocks: int = 0
    max_reflex_latency_ms: float = 0.0
    last_equity: float = 0.0
    peak_equity: float = 0.0
    max_drawdown: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class TickResult:
    symbol: str
    ts: int
    skipped: bool = False
    snapshot: Optional[MarketSnapshot] = None
    decision: Optional[JevDecision] = None
    policy: Optional[PolicyResult] = None
    fill: Optional[Fill] = None
    note: str = ""


class LiveLoop:
    def __init__(
        self,
        cfg: TraderConfig,
        feed: Any,
        exchange: Any,
        reflex: Any,
        brain: Any = None,
        journal: Optional[Journal] = None,
        universe: Optional[dict[str, SetupSpec]] = None,
        kill_switch: Optional[KillSwitch] = None,
        *,
        warmup_candles: int = 50,
        on_tick: Optional[Callable[[TickResult], None]] = None,
    ):
        self.cfg = cfg
        self.feed = feed
        self.exchange = exchange
        self.reflex = reflex
        self.brain = brain or NullBrain()
        self.journal = journal or Journal(":memory:")
        self.universe = universe or {}
        self.kill_switch = kill_switch or KillSwitch(cfg.kill_switch_path)
        self.risk = RiskLayer(cfg.risk, self.kill_switch)
        self.policy = Policy(cfg.policy, cfg.risk)
        self.warmup_candles = warmup_candles
        self.on_tick = on_tick
        self.stats = LoopStats()
        self.contexts: dict[str, SymbolContext] = {}
        self._started = False

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self.kill_switch.arm()
        self._started = True
        ok, why = self.kill_switch.check()
        if not ok:
            log.error("starting with kill switch not clear: %s", why)

    def run(self, max_ticks: Optional[int] = None) -> LoopStats:
        if not self._started:
            self.start()
        for tick in self.feed.ticks():
            result = self.process(tick)
            if self.on_tick:
                self.on_tick(result)
            if max_ticks is not None and self.stats.ticks >= max_ticks:
                break
        return self.stats

    # -- helpers ---------------------------------------------------------------
    def _ctx(self, symbol: str) -> SymbolContext:
        ctx = self.contexts.get(symbol)
        if ctx is None:
            engine = StateEngine(symbol, self.cfg.candle_seconds, self.cfg.risk, book_depth=self.cfg.book_depth,
                                 setup=setup_for(symbol, self.universe), max_snapshot_tokens=self.cfg.policy.max_snapshot_tokens)
            ctx = SymbolContext(engine=engine)
            self.contexts[symbol] = ctx
        return ctx

    def _heartbeat(self, ts: int) -> None:
        path = self.cfg.heartbeat_path
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f"{int(time.time())} {ts}\n")
        except OSError:  # pragma: no cover
            pass

    def _flatten_intent(self, account: AccountState, snap: MarketSnapshot, reason: str, decision_id: str = "") -> Optional[OrderIntent]:
        if abs(account.position_qty) <= 0:
            return None
        side = Side.sell if account.position_qty > 0 else Side.buy
        return OrderIntent(symbol=snap.symbol, side=side, qty=abs(account.position_qty), ref_price=snap.mid,
                           reason=f"flatten: {reason}", reduce_only=True, decision_id=decision_id,
                           intent_id=f"{decision_id or 'ks'}-{snap.ts}-{side.value}")

    # -- the per-candle pipeline ------------------------------------------------
    def process(self, tick: CandleTick) -> TickResult:
        cfg = self.cfg
        symbol = tick.symbol
        ctx = self._ctx(symbol)
        eng = ctx.engine
        self.stats.ticks += 1

        # 1. ingest (trades precede the close; drop anything out of order) -----
        for trade in tick.trades:
            if trade.ts > tick.candle.ts:
                continue
            try:
                eng.ingest_trade(trade)
            except CausalityError:
                self.stats.causality_drops += 1
        try:
            eng.ingest_candle(tick.candle)
            eng.ingest_book(tick.book)
        except CausalityError as exc:
            self.stats.causality_drops += 1
            self.journal.record_event(tick.ts, "warning", f"{symbol} causality drop: {exc}")
            return TickResult(symbol, tick.ts, skipped=True, note=str(exc))
        as_of = max(tick.candle.ts, tick.book.ts)
        if hasattr(self.exchange, "set_book"):
            self.exchange.set_book(symbol, tick.book)
        account = self.exchange.account(symbol, as_of, tick.book.mid)
        eng.ingest_account(account)
        ctx.candle_index += 1

        usage = self.risk.usage(account, tick.book.mid, self.exchange.gross_other_notional(symbol))
        self.journal.record_equity(as_of, symbol, account.equity, usage.drawdown, usage.daily_pnl_pct, account.position_qty)
        self.stats.last_equity = account.equity
        self.stats.peak_equity = max(self.stats.peak_equity, account.equity)
        self.stats.max_drawdown = max(self.stats.max_drawdown, usage.drawdown)
        self.journal.resolve_outcomes(symbol, as_of, tick.book.mid, cfg.policy.outcome_move_bps)
        self._heartbeat(as_of)

        if ctx.candle_index < self.warmup_candles:
            self.stats.warmup_skipped += 1
            return TickResult(symbol, as_of, skipped=True, note="warmup")

        try:
            snap = eng.snapshot(as_of)
        except InsufficientData as exc:
            return TickResult(symbol, as_of, skipped=True, note=str(exc))

        # 2. kill switch before anything else -----------------------------------
        ok, why = self.kill_switch.check()
        if not ok:
            self.stats.kill_switch_blocks += 1
            intent = self._flatten_intent(account, snap, why)
            fill = self._execute(intent, account, snap, as_of) if intent else None
            return TickResult(symbol, as_of, snapshot=snap, fill=fill, note=why)

        # 3. reflex: one typed call ------------------------------------------------
        try:
            decision = self.reflex.judge(snap)
        except ReflexError as exc:
            self.stats.reflex_failures += 1
            self.journal.record_event(as_of, "warning", f"{symbol} reflex failed: {exc}")
            decision = JevDecision.safe_hold(source="fallback", reason_model=str(exc)[:60])
        self.stats.max_reflex_latency_ms = max(self.stats.max_reflex_latency_ms, decision.latency_ms)
        if decision.latency_ms > cfg.jev_timeout_ms:
            log.warning("%s reflex latency %.0f ms above budget", symbol, decision.latency_ms)

        # 4. policy (code) ----------------------------------------------------------
        entries_paused = ctx.candle_index < ctx.paused_until
        result = self.policy.decide(snap, decision, previous_regime=ctx.previous_regime, entries_paused=entries_paused)
        decision_id = hashlib.sha1(f"{symbol}:{as_of}".encode()).hexdigest()[:16]   # deterministic: replays match
        horizon_ts = as_of + cfg.policy.horizon_candles * cfg.candle_seconds * 1000
        self.stats.decisions += 1

        intent = result.intent
        if result.outcome == Outcome.escalate:
            self.stats.escalations += 1
            verdict = self.brain.deep_reread(snap, decision, self.journal.recent_decisions(symbol, 24), result.reason)
            self.journal.record_escalation(as_of, symbol, result.reason, verdict.action, verdict.regime.value,
                                           verdict.pause_candles, verdict.rationale, verdict.source)
            if verdict.action == "pause":
                ctx.paused_until = ctx.candle_index + max(1, verdict.pause_candles)
            elif verdict.action == "flatten":
                intent = self._flatten_intent(account, snap, f"brain: {verdict.rationale[:80]}", decision_id)
            ctx.previous_regime = verdict.regime
        else:
            ctx.previous_regime = decision.regime
        if result.outcome == Outcome.hold:
            self.stats.holds += 1
        elif result.outcome == Outcome.flatten:
            self.stats.flattens += 1

        self.journal.record_decision(decision_id, snap, decision, result.outcome.value, result.reason, result.size_frac, horizon_ts)

        # 5. risk layer + exchange ----------------------------------------------------
        fill = None
        if intent is not None:
            intent = dataclasses.replace(intent, decision_id=decision_id, intent_id=f"{decision_id}-{intent.side.value}")
            fill = self._execute(intent, account, snap, as_of)
        return TickResult(symbol, as_of, snapshot=snap, decision=decision, policy=result, fill=fill)

    def _execute(self, intent: OrderIntent, account: AccountState, snap: MarketSnapshot, as_of: int) -> Optional[Fill]:
        self.stats.orders_requested += 1
        verdict = self.risk.evaluate(
            intent, account, snap.mid,
            orders_last_hour=self.journal.orders_last_hour(intent.symbol, as_of),
            gross_other_notional=self.exchange.gross_other_notional(intent.symbol),
        )
        self.journal.record_order(intent, as_of, verdict.allowed, verdict.qty, verdict.reason)
        if verdict.kill_switch_fired:
            self.journal.record_event(as_of, "error", f"kill switch fired: {self.kill_switch.reason}")
        if not verdict.allowed:
            self.stats.orders_blocked += 1
            log.info("%s order blocked: %s", intent.symbol, verdict.reason)
            return None
        if verdict.qty < intent.qty:
            self.stats.orders_clipped += 1
        final = intent.with_qty(verdict.qty)
        if verdict.reduce_only and not final.reduce_only:
            final = dataclasses.replace(final, reduce_only=True)
        try:
            fill = self.exchange.place(final, as_of)
        except ExchangeError as exc:
            self.journal.record_event(as_of, "error", f"{intent.symbol} order failed: {exc}")
            return None
        self.journal.record_fill(fill)
        self.stats.fills += 1
        return fill
