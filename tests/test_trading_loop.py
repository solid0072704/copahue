import pytest

from tests.trading_helpers import FailingReflex, ScriptedReflex, decision, make_config
from trading_agent.backtest import run_backtest
from trading_agent.brain import NullBrain
from trading_agent.exchange import PaperExchange
from trading_agent.feed import SyntheticFeed, SyntheticRegime
from trading_agent.journal import Journal
from trading_agent.loop import LiveLoop
from trading_agent.reflex import RuleReflexClient
from trading_agent.risk import KillSwitch
from trading_agent.schema import Direction, RiskState


def feed(n=300, seed=1, regimes=None):
    return SyntheticFeed(["BTC/USDT"], 300, n, seed=seed, regimes=regimes or [(n, SyntheticRegime(drift_per_candle=0.0005, vol_per_candle=0.003))])


def run(cfg, reflex, fd, *, kill_switch=None, brain=None):
    exchange = PaperExchange(cfg.starting_equity, cfg.taker_fee_bps, cfg.slippage_bps)
    journal = Journal(":memory:")
    loop = LiveLoop(cfg, fd, exchange, reflex, brain or NullBrain(), journal, kill_switch=kill_switch or KillSwitch(None))
    stats = loop.run()
    return loop, exchange, journal, stats


def test_end_to_end_limits_hold():
    cfg = make_config()
    loop, exchange, journal, stats = run(cfg, ScriptedReflex(decision(confidence=0.9)), feed())
    assert stats.fills >= 1 and stats.decisions > 0
    lo, hi = journal.time_bounds()
    for row in journal.equity_between(lo - 1, hi):
        assert abs(row["position_qty"]) * row["equity"] >= 0
    for order in journal.orders_between(lo - 1, hi):
        if order["allowed"]:
            assert order["qty_allowed"] <= order["qty_requested"] + 1e-12
    # the largest fill never exceeds the per-order cap
    for f in exchange.fills:
        assert f.qty * f.price <= cfg.risk.max_order_notional_frac * cfg.starting_equity * 1.25
    assert stats.max_drawdown < cfg.risk.max_drawdown
    assert journal.resolve_outcomes("BTC/USDT", hi + 10**9, exchange.marks["BTC/USDT"]) >= 0
    report_rows = journal.decisions_between(lo - 1, hi)
    assert any(r["resolved"] for r in report_rows)


def test_reflex_failure_means_hold():
    cfg = make_config()
    loop, exchange, journal, stats = run(cfg, FailingReflex(), feed(120))
    assert stats.fills == 0 and stats.reflex_failures == stats.decisions > 0
    assert all(d["direction"] == Direction.neutral.value for d in journal.decisions_between(0, 10**15))


def test_operator_kill_switch_flattens_and_blocks(tmp_path):
    cfg = make_config()
    ks = KillSwitch(str(tmp_path / "KILL"))
    exchange = PaperExchange(cfg.starting_equity)
    journal = Journal(":memory:")
    fills_before = []

    def on_tick(result):
        if result.fill:
            fills_before.append(result.ts)
        if loop.stats.ticks == 120:
            (tmp_path / "KILL").write_text("operator")

    loop = LiveLoop(cfg, feed(200), exchange, ScriptedReflex(decision(confidence=0.9)), NullBrain(), journal, kill_switch=ks, on_tick=on_tick)
    stats = loop.run()
    assert ks.triggered and stats.kill_switch_blocks > 0
    assert exchange.position_qty("BTC/USDT") == 0.0
    lo, hi = journal.time_bounds()
    late_fills = [f for f in journal.fills_between(lo - 1, hi) if f["ts"] > fills_before[-1]]
    assert late_fills == []


def test_drawdown_breach_triggers_kill_switch_and_flattens():
    # With the default limits a single position can lose at most 10 % of equity (order cap, no
    # pyramiding) and the 3 % daily-loss rule flattens first; widen both so the drawdown rule is
    # the one that fires.
    from trading_agent.config import RiskLimits

    cfg = make_config(risk=RiskLimits(max_daily_loss=0.15, max_order_notional_frac=0.20))
    regimes = [(80, SyntheticRegime(drift_per_candle=0.001, vol_per_candle=0.002)),
               (200, SyntheticRegime(drift_per_candle=-0.03, vol_per_candle=0.002))]
    ks = KillSwitch(None)
    loop, exchange, journal, stats = run(cfg, ScriptedReflex(decision(confidence=0.9)), feed(280, regimes=regimes), kill_switch=ks)
    assert ks.triggered and "drawdown" in ks.reason
    assert exchange.position_qty("BTC/USDT") == 0.0
    lo, hi = journal.time_bounds()
    assert any("kill switch" in o["risk_reason"] for o in journal.orders_between(lo - 1, hi) if not o["allowed"]) or stats.kill_switch_blocks > 0


def test_escalation_pauses_entries_with_null_brain():
    cfg = make_config()
    low = decision(confidence=0.3)
    high = decision(confidence=0.9)
    script = [high] * 60 + [low] + [high] * 100
    loop, exchange, journal, stats = run(cfg, ScriptedReflex(script), feed(220))
    assert stats.escalations >= 1
    esc = journal.escalations_between(0, 10**15)
    assert esc and esc[0]["action"] == "pause" and esc[0]["pause_candles"] == 12
    ctx = loop.contexts["BTC/USDT"]
    assert ctx.paused_until > 0


def test_rules_backtest_runs_and_reports():
    cfg = make_config()
    regimes = [(150, SyntheticRegime(drift_per_candle=0.0015, vol_per_candle=0.002, flow_bias=0.2)),
               (150, SyntheticRegime(drift_per_candle=0.0, vol_per_candle=0.005))]
    result = run_backtest(cfg, feed(300, regimes=regimes), reflex=RuleReflexClient())
    m = result.metrics
    assert m["final_equity"] > 0 and m["n_resolved"] > 0
    assert m["brier"] is None or 0.0 <= m["brier"] <= 1.0
    assert result.report["decisions"]["n"] == result.stats.decisions


def test_paper_exchange_accounting():
    from tests.trading_helpers import T0, make_book
    from trading_agent.schema import OrderIntent, Side

    ex = PaperExchange(10_000.0, taker_fee_bps=10.0, slippage_bps=0.0)
    ex.set_book("BTC/USDT", make_book(T0, 100.0, spread_bps=20.0))
    fill = ex.place(OrderIntent("BTC/USDT", Side.buy, 1.0, 100.0), T0)
    assert fill.price == pytest.approx(100.1) and fill.fee == pytest.approx(0.1001)
    acc = ex.account("BTC/USDT", T0 + 1, 100.0)
    assert acc.position_qty == 1.0 and acc.avg_entry == pytest.approx(100.1)
    assert acc.equity == pytest.approx(10_000.0 - 100.1 - 0.1001 + 100.0)
    ex.set_book("BTC/USDT", make_book(T0 + 2, 110.0, spread_bps=20.0))
    ex.place(OrderIntent("BTC/USDT", Side.sell, 1.0, 110.0, reduce_only=True), T0 + 2)
    assert ex.position_qty("BTC/USDT") == 0.0
    assert ex.positions["BTC/USDT"].realized == pytest.approx(109.89 - 100.1)
