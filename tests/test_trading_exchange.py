import time

import pytest

from trading_agent import watchdog
from trading_agent.exchange import CcxtExchange, ExchangeError
from trading_agent.schema import OrderIntent, Side

T0 = 1_760_000_000_000


class FakeCcxt:
    """Just enough of ccxt's surface for the adapter logic (no network)."""

    def __init__(self):
        self.markets = {
            "UNI/USDT:USDT": {"swap": True, "spot": False, "taker": 0.0005, "contractSize": 1,
                              "limits": {"cost": {"min": 5.0}, "amount": {"min": 0.1}}},
            "UNI/USDT": {"swap": False, "spot": True, "taker": 0.001,
                         "limits": {"cost": {"min": 5.0}, "amount": {"min": 0.01}}},
        }
        self.positions = []
        self.balance = {"USDT": {"total": 1000.0}, "UNI": {"total": 3.0}, "total": {"USDT": 1000.0, "UNI": 3.0},
                        "info": {"totalMarginBalance": "1010.5"}}
        self.orders = []
        self.leverage_calls = []
        self.margin_calls = []
        self.sandbox = None
        self.loaded = 0

    def load_markets(self):
        self.loaded += 1
        return self.markets

    def set_sandbox_mode(self, flag):
        self.sandbox = flag

    def set_leverage(self, leverage, symbol):
        self.leverage_calls.append((leverage, symbol))

    def set_margin_mode(self, mode, symbol):
        self.margin_calls.append((mode, symbol))
        raise Exception("No need to change margin type")

    def fetch_positions(self, symbols=None):
        return [p for p in self.positions if symbols is None or p["symbol"] in symbols]

    def fetch_balance(self):
        return self.balance

    def amount_to_precision(self, symbol, amount):
        return f"{amount:.3f}"

    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        self.orders.append({"symbol": symbol, "type": type_, "side": side, "amount": amount, "params": params or {}})
        return {"id": f"o{len(self.orders)}", "filled": amount, "average": 10.0, "fee": {"cost": 0.005}}

    def fetch_order_book(self, symbol, limit=10):
        return {"timestamp": T0, "bids": [[9.99 - i * 0.01, 100] for i in range(5)], "asks": [[10.01 + i * 0.01, 100] for i in range(5)]}

    def fetch_ohlcv(self, symbol, timeframe, limit):
        return [[T0 + i * 300_000, 10, 10.1, 9.9, 10, 1] for i in range(limit)]

    def fetch_trades(self, symbol, since=None, limit=500):
        return []


def swap_exchange(live=False):
    fake = FakeCcxt()
    ex = CcxtExchange("binance", "k", "s", market_type="swap", live_confirmed=live, starting_equity=1000.0, client=fake)
    return ex, fake


def test_swap_symbol_mapping_and_exchange_id():
    ex, _ = swap_exchange()
    assert ex.exchange_id == "binanceusdm"
    assert ex.market_symbol("UNI/USDT") == "UNI/USDT:USDT" and ex.universe_symbol("UNI/USDT:USDT") == "UNI/USDT"
    spot = CcxtExchange("binance", market_type="spot", client=FakeCcxt())
    assert spot.exchange_id == "binance" and spot.market_symbol("UNI/USDT") == "UNI/USDT"


def test_check_and_configure_markets():
    ex, fake = swap_exchange()
    rows = ex.check_markets(["UNI/USDT", "XYZ/USDT"])
    assert rows[0]["found"] and rows[0]["type"] == "swap" and rows[0]["taker_bps"] == 5.0 and rows[0]["min_cost"] == 5.0
    assert rows[1]["found"] is False
    cfgd = ex.configure_markets(["UNI/USDT"])
    assert fake.leverage_calls == [(1, "UNI/USDT:USDT")]
    assert cfgd[0]["leverage"] == 1 and "unchanged" in cfgd[0]["margin_mode"]


def test_reconcile_adopt_and_sync_equity():
    ex, fake = swap_exchange()
    fake.positions = [{"symbol": "UNI/USDT:USDT", "contracts": 2.0, "side": "short", "entryPrice": 11.0, "unrealizedPnl": 2.0}]
    report = ex.reconcile(["UNI/USDT"])
    assert not report.ok and report.mismatches[0]["venue_qty"] == -2.0 and report.mismatches[0]["local_qty"] == 0.0
    ex.fetch_book("UNI/USDT")                     # marks
    report = ex.adopt_positions(["UNI/USDT"])
    assert report.ok and ex.position_qty("UNI/USDT") == -2.0 and ex.paper.positions["UNI/USDT"].avg_entry == 11.0
    equity = ex.sync_equity(symbols=["UNI/USDT"])
    assert equity == pytest.approx(1010.5)
    acc = ex.account("UNI/USDT", T0 + 1)
    assert acc.equity == pytest.approx(1010.5) and acc.position_qty == -2.0 and acc.peak_equity == pytest.approx(1010.5)


def test_spot_equity_counts_holdings():
    fake = FakeCcxt()
    ex = CcxtExchange("binance", market_type="spot", client=fake)
    ex.fetch_book("UNI/USDT")
    assert ex.equity_from_exchange(symbols=["UNI/USDT"]) == pytest.approx(1000.0 + 3.0 * 10.0)
    assert ex.fetch_positions_map(["UNI/USDT"])["UNI/USDT"]["qty"] == 3.0


def test_place_requires_live_and_respects_minimums():
    ex, fake = swap_exchange(live=False)
    ex.fetch_book("UNI/USDT")
    with pytest.raises(ExchangeError):
        ex.place(OrderIntent("UNI/USDT", Side.buy, 1.0, 10.0), T0)
    ex, fake = swap_exchange(live=True)
    ex.fetch_book("UNI/USDT")
    with pytest.raises(ExchangeError):                      # 0.3 * 10 = 3 < min cost 5
        ex.place(OrderIntent("UNI/USDT", Side.buy, 0.3, 10.0), T0)
    fill = ex.place(OrderIntent("UNI/USDT", Side.buy, 1.0, 10.0), T0)
    assert fake.orders[-1] == {"symbol": "UNI/USDT:USDT", "type": "market", "side": "buy", "amount": 1.0, "params": {}}
    assert fill.price == 10.0 and fill.fee == 0.005 and ex.position_qty("UNI/USDT") == 1.0
    ex.place(OrderIntent("UNI/USDT", Side.sell, 1.0, 10.0, reduce_only=True), T0 + 1)
    assert fake.orders[-1]["params"] == {"reduceOnly": True} and ex.position_qty("UNI/USDT") == 0.0


def test_flatten_all_closes_venue_positions():
    ex, fake = swap_exchange(live=True)
    fake.positions = [{"symbol": "UNI/USDT:USDT", "contracts": 2.0, "side": "short", "entryPrice": 11.0, "unrealizedPnl": 0.0}]
    fills = ex.flatten_all(["UNI/USDT"], now_ts=T0)
    assert len(fills) == 1 and fills[0].side == Side.buy and fills[0].qty == 2.0
    assert fake.orders[-1]["params"] == {"reduceOnly": True} and ex.position_qty("UNI/USDT") == 0.0


def test_fetch_candles_drops_the_open_candle():
    ex, _ = swap_exchange()
    candles = ex.fetch_candles("UNI/USDT", 300, limit=5, now_ts=T0 + 4 * 300_000 + 100)
    assert len(candles) == 4 and candles[-1].ts == T0 + 4 * 300_000


def test_watchdog(tmp_path):
    hb = tmp_path / "hb"
    ks = tmp_path / "KILL"
    now = time.time()
    hb.write_text(f"{int(now)} 0\n")
    ok, msg = watchdog.check(str(hb), str(ks), max_age_s=720, now=now + 60)
    assert ok and not ks.exists()
    ok, msg = watchdog.check(str(hb), str(ks), max_age_s=720, now=now + 1000)
    assert not ok and ks.exists() and "old" in msg
    ks.unlink()
    ok, msg = watchdog.check(str(tmp_path / "missing"), str(ks), max_age_s=720, now=now)
    assert not ok and ks.exists() and "missing" in msg


def test_periodic_equity_sync_in_live_mode():
    ex, fake = swap_exchange(live=True)
    ex.fetch_book("UNI/USDT")
    ex.sync_equity(symbols=["UNI/USDT"])
    ex._last_sync_ts = T0
    fake.balance["info"]["totalMarginBalance"] = "990.0"        # funding paid on the venue
    assert ex.account("UNI/USDT", T0 + 1_000).equity == pytest.approx(1010.5)      # not yet
    assert ex.account("UNI/USDT", T0 + 3_600_001).equity == pytest.approx(990.0)   # hourly re-sync
    fake.balance = {}                                                                # venue hiccup: ledger untouched
    ex._last_sync_ts = T0
    assert ex.account("UNI/USDT", T0 + 7_200_002).equity == pytest.approx(990.0)
