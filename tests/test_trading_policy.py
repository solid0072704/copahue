import pytest

from tests.trading_helpers import T0, decision, make_engine
from trading_agent.config import PolicyParams, RiskLimits
from trading_agent.policy import Outcome, Policy, kelly_full, size_fraction
from trading_agent.schema import AccountState, Direction, Regime, RiskState, SetupSpec, Side


@pytest.fixture
def policy():
    return Policy(PolicyParams(), RiskLimits())


def snap(**kw):
    account = kw.pop("account", None)
    setup = kw.pop("setup", None)
    eng, _ = make_engine(drift=0.002, vol=0.002, account=account, setup=setup)
    return eng.snapshot(eng.last_candle().ts)


def test_kelly_math_and_quarter_cap():
    assert kelly_full(0.8, 1.5) == pytest.approx(0.8 - 0.2 / 1.5)
    assert kelly_full(0.3, 1.0) == 0.0
    f_star, frac = size_fraction(0.9, 2.0, PolicyParams())
    assert frac == pytest.approx(0.25 * f_star)
    f_star2, frac2 = size_fraction(0.9, 2.0, PolicyParams(kelly_fraction=0.25, kelly_cap=0.25))
    assert frac2 <= 0.25 * f_star2 + 1e-12
    with pytest.raises(ValueError):
        PolicyParams(kelly_fraction=0.5, kelly_cap=0.5).validate()


def test_fires_only_above_thresholds(policy):
    s = snap()
    assert policy.decide(s, decision(confidence=0.80)).outcome == Outcome.hold      # strict >
    r = policy.decide(s, decision(confidence=0.81, setup_quality=2))
    assert r.outcome == Outcome.order and r.intent.side == Side.buy
    assert policy.decide(s, decision(setup_quality=1)).outcome == Outcome.hold
    assert policy.decide(s, decision(toxic_flow=True)).outcome == Outcome.hold
    assert policy.decide(s, decision(direction=Direction.neutral)).outcome == Outcome.hold
    assert policy.decide(s, decision(risk_state=RiskState.near_limit)).outcome == Outcome.hold   # model may tighten


def test_code_risk_state_wins_over_model(policy):
    acc = AccountState(ts=T0 + 80 * 300_000, equity=8_800.0, peak_equity=10_000.0, day_start_equity=8_900.0)
    s = snap(account=acc)
    assert s.risk_state == RiskState.near_limit
    r = policy.decide(s, decision(risk_state=RiskState.safe))
    assert r.outcome == Outcome.hold and r.effective_risk_state == RiskState.near_limit


def test_escalations(policy):
    s = snap()
    assert policy.decide(s, decision(confidence=0.59)).outcome == Outcome.escalate
    assert policy.decide(s, decision(confidence=0.59, direction=Direction.neutral)).outcome == Outcome.hold   # no edge, not doubt
    assert policy.decide(s, decision(confidence=0.59), entries_paused=True).outcome == Outcome.hold          # no brain call while paused
    assert policy.decide(s, decision(regime=Regime.crisis)).outcome == Outcome.escalate
    assert policy.decide(s, decision(regime=Regime.crisis), previous_regime=Regime.crisis).outcome == Outcome.hold


def test_sizing_and_stops(policy):
    s = snap()
    r = policy.decide(s, decision(confidence=0.9))
    b = s.setup_reward_risk
    f_star = 0.9 - 0.1 / b
    assert r.size_frac == pytest.approx(min(0.25 * f_star, RiskLimits().max_position_frac))
    assert r.intent.qty == pytest.approx(r.size_frac * s.equity / s.mid)
    assert r.intent.stop_price < s.mid < r.intent.take_profit
    assert (r.intent.take_profit - s.mid) == pytest.approx((s.mid - r.intent.stop_price) * b)
    rs = policy.decide(s, decision(direction=Direction.short, confidence=0.9))
    assert rs.intent.side == Side.sell and rs.intent.stop_price > s.mid > rs.intent.take_profit


def test_setup_direction_must_match(policy):
    s = snap(setup=SetupSpec("BTC/USDT", Direction.short))
    assert policy.decide(s, decision(direction=Direction.long)).outcome == Outcome.hold
    assert policy.decide(s, decision(direction=Direction.short)).outcome == Outcome.order


def test_position_management(policy):
    acc = AccountState(ts=T0 + 80 * 300_000, equity=10_000.0, position_qty=5.0, avg_entry=100.0)
    s = snap(account=acc)
    assert policy.decide(s, decision()).outcome == Outcome.hold                      # no pyramiding
    r = policy.decide(s, decision(direction=Direction.short, confidence=0.9))
    assert r.outcome == Outcome.flatten and r.intent.reduce_only and r.intent.qty == 5.0 and r.intent.side == Side.sell
    assert policy.decide(s, decision(direction=Direction.short, confidence=0.7)).outcome == Outcome.hold
    assert policy.decide(s, decision(risk_state=RiskState.reduce)).outcome == Outcome.flatten
    s2 = snap(account=acc, setup=SetupSpec("BTC/USDT", Direction.long, invalidate_below=1e9))
    assert policy.decide(s2, decision()).outcome == Outcome.flatten


def test_entries_paused(policy):
    r = policy.decide(snap(), decision(), entries_paused=True)
    assert r.outcome == Outcome.hold and "entries_not_paused" in r.failed_gates


def test_regime_must_fit_setup(policy):
    s = snap(setup=SetupSpec("BTC/USDT", Direction.long, regime_fit=("mean_reverting",)))
    assert "regime_fits_setup" in policy.decide(s, decision(regime=Regime.trending)).failed_gates
    assert policy.decide(s, decision(regime=Regime.mean_reverting)).outcome == Outcome.order
    neutral = snap()                                        # no thesis: any regime but crisis may fire
    assert policy.decide(neutral, decision(regime=Regime.high_vol)).outcome == Outcome.order
