import inspect

import pytest

from tests.trading_helpers import T0
from trading_agent.config import RiskLimits
from trading_agent.risk import KillSwitch, RiskLayer, compute_usage, risk_state_from_usage
from trading_agent.schema import AccountState, OrderIntent, RiskState, Side

MID = 100.0


def armed(tmp_path=None):
    ks = KillSwitch(str(tmp_path / "KILL") if tmp_path else None)
    ks.arm()
    return ks


def intent(side=Side.buy, qty=1.0, ref=MID, reduce_only=False):
    return OrderIntent("BTC/USDT", side, qty, ref, reduce_only=reduce_only)


def test_risk_layer_does_not_take_model_input():
    params = inspect.signature(RiskLayer.evaluate).parameters
    assert "decision" not in params and "snapshot" not in params


def test_not_armed_blocks_everything():
    layer = RiskLayer(RiskLimits(), KillSwitch(None))
    v = layer.evaluate(intent(), AccountState(T0, 10_000.0), MID)
    assert v.blocked and "not armed" in v.reason


def test_operator_trigger_file(tmp_path):
    ks = armed(tmp_path)
    layer = RiskLayer(RiskLimits(), ks)
    (tmp_path / "KILL").write_text("stop")
    v = layer.evaluate(intent(), AccountState(T0, 10_000.0), MID)
    assert v.blocked and "triggered" in v.reason
    # reduce-only still passes so the loop can flatten
    acc = AccountState(T0, 10_000.0, position_qty=2.0, avg_entry=MID)
    v = layer.evaluate(intent(Side.sell, 5.0, reduce_only=True), acc, MID)
    assert v.allowed and v.reduce_only and v.qty == 2.0


def test_drawdown_breach_fires_kill_switch():
    ks = armed()
    layer = RiskLayer(RiskLimits(max_drawdown=0.15), ks)
    acc = AccountState(T0, 8_400.0, position_qty=5.0, avg_entry=MID, peak_equity=10_000.0, day_start_equity=10_000.0)
    v = layer.evaluate(intent(), acc, MID)
    assert v.blocked and v.kill_switch_fired and ks.triggered
    v2 = layer.evaluate(intent(Side.sell, 5.0, reduce_only=True), acc, MID)
    assert v2.allowed and v2.reduce_only
    v3 = layer.evaluate(intent(Side.buy, 1.0, reduce_only=True), acc, MID)   # "reduce only" that adds: blocked
    assert v3.blocked


def test_daily_loss_blocks_new_risk_but_allows_reduce():
    layer = RiskLayer(RiskLimits(max_daily_loss=0.03), armed())
    acc = AccountState(T0, 9_650.0, position_qty=1.0, avg_entry=MID, peak_equity=10_000.0, day_start_equity=10_000.0)
    assert layer.evaluate(intent(), acc, MID).blocked
    assert layer.evaluate(intent(Side.sell, 1.0), acc, MID).allowed


def test_order_and_position_caps():
    limits = RiskLimits(max_order_notional_frac=0.10, max_position_frac=0.20, max_gross_frac=0.60)
    layer = RiskLayer(limits, armed())
    acc = AccountState(T0, 10_000.0)
    v = layer.evaluate(intent(qty=50.0), acc, MID)          # 5000 notional requested
    assert v.allowed and v.qty == pytest.approx(10.0)         # 10 % of equity = 1000 / 100
    acc2 = AccountState(T0, 10_000.0, position_qty=15.0, avg_entry=MID)   # already 15 % of equity
    v2 = layer.evaluate(intent(qty=10.0), acc2, MID)
    assert v2.allowed and v2.qty == pytest.approx(5.0)        # room to the 20 % cap
    acc3 = AccountState(T0, 10_000.0, position_qty=20.0, avg_entry=MID)
    assert layer.evaluate(intent(qty=1.0), acc3, MID).blocked


def test_gross_cap_counts_other_symbols():
    layer = RiskLayer(RiskLimits(max_gross_frac=0.30, max_position_frac=0.20, max_order_notional_frac=0.10), armed())
    acc = AccountState(T0, 10_000.0)
    v = layer.evaluate(intent(qty=10.0), acc, MID, gross_other_notional=2_500.0)
    assert v.allowed and v.qty == pytest.approx(5.0)          # 3000 - 2500 = 500 notional of room


def test_rate_limit_and_stale_price():
    layer = RiskLayer(RiskLimits(max_orders_per_hour=3), armed())
    acc = AccountState(T0, 10_000.0)
    assert layer.evaluate(intent(), acc, MID, orders_last_hour=3).blocked
    assert layer.evaluate(intent(ref=MID * 1.05), acc, MID).blocked
    assert layer.evaluate(intent(qty=float("nan")), acc, MID).blocked


def test_reduce_never_flips_through_zero():
    layer = RiskLayer(RiskLimits(), armed())
    acc = AccountState(T0, 10_000.0, position_qty=2.0, avg_entry=MID)
    v = layer.evaluate(intent(Side.sell, 3.0), acc, MID)      # would flip to -1: treated as a reduce of 2
    assert v.allowed and v.qty == pytest.approx(2.0) and v.reduce_only


def test_usage_and_state():
    u = compute_usage(AccountState(T0, 9_000.0, position_qty=10.0, avg_entry=MID, peak_equity=10_000.0, day_start_equity=9_100.0), MID, RiskLimits())
    assert u.drawdown == pytest.approx(0.10) and u.dd_usage == pytest.approx(0.10 / 0.15)
    assert u.position_frac == pytest.approx(1000 / 9000)
    assert risk_state_from_usage(u, RiskLimits()) == RiskState.safe
    u2 = compute_usage(AccountState(T0, 8_400.0, peak_equity=10_000.0, day_start_equity=10_000.0), MID, RiskLimits())
    assert risk_state_from_usage(u2, RiskLimits()) == RiskState.reduce


def test_limits_validation():
    with pytest.raises(ValueError):
        RiskLimits(max_drawdown=0.9).validate()
    with pytest.raises(ValueError):
        RiskLimits(max_daily_loss=0.5).validate()
