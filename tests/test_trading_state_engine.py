import pytest

from tests.trading_helpers import STEP, T0, make_book, make_candles, make_engine
from trading_agent.config import RiskLimits
from trading_agent.schema import AccountState, BookSnapshot, Candle, Direction, RiskState, SetupSpec, Side, Trade
from trading_agent.state_engine import (
    CausalityError,
    InsufficientData,
    StateEngine,
    atr_pct,
    book_imbalance,
    microprice_dev_bps,
    pct_return,
    realized_vol_pct,
    signed_flow,
)


def test_determinism_same_inputs_same_snapshot():
    e1, _ = make_engine(seed=3)
    e2, _ = make_engine(seed=3)
    ts = e1.last_candle().ts
    assert e1.snapshot(ts).compact_json() == e2.snapshot(ts).compact_json()


def test_out_of_order_candle_raises():
    eng, candles = make_engine(n=10)
    with pytest.raises(CausalityError):
        eng.ingest_candle(candles[-1])                    # same ts again
    with pytest.raises(CausalityError):
        eng.ingest_candle(Candle(ts=candles[0].ts, open=1, high=2, low=0.5, close=1, volume=1))
    with pytest.raises(CausalityError):
        eng.ingest_trade(Trade(ts=T0, price=1, size=1, side=Side.buy))


def test_snapshot_never_reads_the_future():
    eng, candles = make_engine(n=60)
    as_of = candles[39].ts
    snap = eng.snapshot(as_of)
    assert snap.candles_seen == 40
    assert snap.ts == as_of
    assert snap.mid == pytest.approx(make_book(as_of, candles[39].close).mid)
    # a book that arrives later than as_of is ignored even if it is the newest
    later = eng.snapshot(candles[-1].ts)
    assert later.candles_seen == 60 and later.mid != snap.mid


def test_insufficient_data():
    eng = StateEngine("BTC/USDT", 300, RiskLimits())
    with pytest.raises(InsufficientData):
        eng.snapshot(T0)
    eng.ingest_candle(Candle(ts=T0 + STEP, open=1, high=1.1, low=0.9, close=1, volume=1))
    with pytest.raises(InsufficientData):            # no book yet
        eng.snapshot(T0 + STEP)


def test_pure_metrics():
    assert realized_vol_pct([100.0] * 30, 300) == 0.0
    assert pct_return([100.0, 110.0], 1) == pytest.approx(10.0)
    assert pct_return([100.0], 1) == 0.0
    c = [Candle(ts=T0 + STEP, open=100, high=102, low=98, close=100, volume=1),
         Candle(ts=T0 + 2 * STEP, open=100, high=103, low=99, close=101, volume=1)]
    assert atr_pct(c, 14) == pytest.approx(4.0 / 101 * 100)   # TR = max(4, 3, 1)
    book = BookSnapshot(ts=T0, bids=((99.0, 3.0), (98.0, 1.0)), asks=((101.0, 1.0), (102.0, 1.0)))
    assert book_imbalance(book, 10) == pytest.approx((4 - 2) / 6)
    assert microprice_dev_bps(book) > 0                       # heavier bid pushes microprice up
    trades = [Trade(T0, 1, 3, Side.buy), Trade(T0, 1, 1, Side.sell)]
    assert signed_flow(trades) == pytest.approx(0.5)
    assert signed_flow([]) == 0.0


def test_risk_state_is_computed_in_code():
    limits = RiskLimits()
    acc = AccountState(ts=T0 + 80 * STEP, equity=8_800.0, peak_equity=10_000.0, day_start_equity=8_900.0)
    eng, _ = make_engine(n=80, account=acc, limits=limits)
    snap = eng.snapshot(eng.last_candle().ts)
    assert snap.drawdown == pytest.approx(0.12)
    assert snap.dd_usage == pytest.approx(0.8)
    assert snap.risk_state == RiskState.near_limit
    assert snap.daily_pnl_pct == pytest.approx((8_800 / 8_900 - 1) * 100)


def test_setup_and_token_guard():
    setup = SetupSpec("BTC/USDT", Direction.long, thesis="x" * 2000, reward_risk=2.0, invalidate_below=50.0)
    eng, _ = make_engine(setup=setup)
    snap = eng.snapshot(eng.last_candle().ts)
    assert snap.approx_tokens() < 400
    assert snap.setup_direction == Direction.long and snap.setup_reward_risk == 2.0
    assert snap.setup_invalidated is False
    eng2, _ = make_engine(setup=SetupSpec("BTC/USDT", Direction.long, invalidate_below=10_000.0))
    assert eng2.snapshot(eng2.last_candle().ts).setup_invalidated is True


def test_flow_reflects_aggressor_side():
    eng, _ = make_engine(flow_side=Side.sell)
    assert eng.snapshot(eng.last_candle().ts).flow == -1.0
