"""Command line for the trading agent.

    python run_trader.py --mode backtest --synthetic --candles 1500 --seed 7
    python run_trader.py --mode backtest --replay BTC/USDT=data/btc_5m.csv
    python run_trader.py --check-jev                        # one real Jev call, prints the parsed decision
    python run_trader.py --check-exchange                   # markets, fees, balance and positions on the venue
    python run_trader.py --mode paper                       # live data, paper fills
    python run_trader.py --mode live [--adopt-positions]    # needs TRADER_LIVE=true and keys
    python run_trader.py --mode once                        # one candle (cron)
    python run_trader.py --mode review [--date YYYY-MM-DD]  # nightly review
    python run_trader.py --kill "reason"                    # pull the kill switch
    python run_trader.py --flatten                          # close every venue position now (live keys)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from trading_agent.backtest import run_backtest
from trading_agent.brain import create_brain
from trading_agent.config import TraderConfig
from trading_agent.exchange import CcxtExchange, PaperExchange
from trading_agent.feed import CcxtFeed, ReplayFeed, SyntheticFeed, SyntheticRegime
from trading_agent.journal import Journal
from trading_agent.loop import LiveLoop
from trading_agent.reflex import JevClient, ReflexError, create_reflex
from trading_agent.review import NightlyReview
from trading_agent.risk import KillSwitch
from trading_agent.universe import load_universe


def _synthetic(cfg: TraderConfig, candles: int, seed: int) -> SyntheticFeed:
    third = max(1, candles // 3)
    regimes = [
        (third, SyntheticRegime(drift_per_candle=0.0006, vol_per_candle=0.003, flow_bias=0.1)),
        (third, SyntheticRegime(drift_per_candle=0.0, vol_per_candle=0.006, flow_bias=0.0)),
        (candles, SyntheticRegime(drift_per_candle=-0.0008, vol_per_candle=0.009, spread_bps=6.0, flow_bias=-0.15)),
    ]
    return SyntheticFeed(cfg.symbols, cfg.candle_seconds, candles, seed=seed, regimes=regimes)


def _venue(cfg: TraderConfig, live: bool) -> CcxtExchange:
    return CcxtExchange(cfg.exchange_id, cfg.exchange_api_key, cfg.exchange_api_secret, market_type=cfg.exchange_market,
                        sandbox=cfg.exchange_sandbox, live_confirmed=live, starting_equity=cfg.starting_equity,
                        taker_fee_bps=cfg.taker_fee_bps, leverage=cfg.exchange_leverage)


def check_jev(cfg: TraderConfig) -> int:
    """Send one real snapshot to Jev and show what comes back."""
    from trading_agent.state_engine import StateEngine
    from trading_agent.schema import AccountState

    if not cfg.jev_api_key:
        print("JEV_API_KEY is empty", file=sys.stderr)
        return 2
    feed = SyntheticFeed(["BTC/USDT"], cfg.candle_seconds, 80, seed=1)
    engine = StateEngine("BTC/USDT", cfg.candle_seconds, cfg.risk)
    engine.ingest_account(AccountState(ts=feed.start_ts, equity=cfg.starting_equity))
    last = None
    for tick in feed.ticks():
        for t in tick.trades:
            engine.ingest_trade(t)
        engine.ingest_candle(tick.candle)
        engine.ingest_book(tick.book)
        last = tick
    snap = engine.snapshot(last.ts)
    print(f"state: {snap.approx_tokens()} tokens\n{snap.compact_json()}\n")
    client = JevClient(cfg.jev_api_url, cfg.jev_api_key, cfg.jev_model, cfg.jev_timeout_ms)
    try:
        decision = client.judge(snap)
    except ReflexError as exc:
        print(f"parse or transport problem: {exc}\nraw response for diagnosis:")
        import requests

        resp = requests.post(cfg.jev_api_url, json={"model": cfg.jev_model, "state": snap.compact_json(),
                                                    "questions": client.questions},
                             headers={"Authorization": f"Bearer {cfg.jev_api_key}"}, timeout=cfg.jev_timeout_ms / 1000.0)
        print(resp.status_code, resp.text[:3000])
        return 1
    print(decision.model_dump_json(indent=1))
    print(f"latency: {client.last_latency_ms:.0f} ms (budget {cfg.jev_timeout_ms} ms)")
    return 0


def check_exchange(cfg: TraderConfig) -> int:
    venue = _venue(cfg, live=False)
    print(f"exchange: {venue.exchange_id} market={cfg.exchange_market} sandbox={cfg.exchange_sandbox}")
    rows = venue.check_markets(cfg.symbols)
    for row in rows:
        print(json.dumps(row))
    missing = [r["symbol"] for r in rows if not r["found"]]
    for symbol in cfg.symbols:
        if symbol not in missing:
            book = venue.fetch_book(symbol, cfg.book_depth)
            print(f"{symbol}: mid {book.mid} spread {(book.best_ask - book.best_bid) / book.mid * 1e4:.2f} bps")
    if cfg.exchange_api_key:
        equity = venue.equity_from_exchange(symbols=cfg.symbols)
        print(f"equity on venue: {equity:.2f} USDT")
        print("positions:", json.dumps(venue.fetch_positions_map(cfg.symbols)))
    else:
        print("no EXCHANGE_API_KEY: balance and positions not checked")
    if missing:
        print(f"MISSING markets: {missing}", file=sys.stderr)
        return 1
    return 0


def prepare_live(cfg: TraderConfig, adopt: bool) -> CcxtExchange:
    """Phase 6 start sequence: markets, leverage, marks, reconciliation, equity."""
    venue = _venue(cfg, live=True)
    rows = venue.check_markets(cfg.symbols)
    missing = [r["symbol"] for r in rows if not r["found"]]
    if missing:
        raise SystemExit(f"refusing live start: markets missing on the venue: {missing}")
    for row in venue.configure_markets(cfg.symbols):
        print("market config:", json.dumps(row))
    for symbol in cfg.symbols:
        venue.fetch_book(symbol, cfg.book_depth)           # marks for the ledger
    report = venue.reconcile(cfg.symbols)
    if not report.ok:
        if not adopt:
            print("refusing live start: venue positions differ from the local ledger", file=sys.stderr)
            for m in report.mismatches:
                print(json.dumps(m), file=sys.stderr)
            print("re-run with --adopt-positions to take the venue's positions as the truth", file=sys.stderr)
            raise SystemExit(3)
        report = venue.adopt_positions(cfg.symbols)
        print("adopted venue positions:", json.dumps(report.exchange_positions))
    equity = venue.sync_equity(symbols=cfg.symbols)
    print(f"live start: equity {equity:.2f} USDT, positions {json.dumps(report.exchange_positions)}")
    return venue


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Autonomous crypto trading agent")
    parser.add_argument("--mode", choices=["paper", "live", "backtest", "review", "once"], default=None)
    parser.add_argument("--synthetic", action="store_true", help="backtest on a seeded synthetic feed")
    parser.add_argument("--candles", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay", action="append", default=[], help="SYMBOL=path.csv (ts,open,high,low,close,volume)")
    parser.add_argument("--open-time", action="store_true", help="replay CSV ts is the candle open time")
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--kill", default=None, help="pull the kill switch with a reason and exit")
    parser.add_argument("--flatten", action="store_true", help="close every venue position now and pull the kill switch")
    parser.add_argument("--check-jev", action="store_true")
    parser.add_argument("--check-exchange", action="store_true")
    parser.add_argument("--adopt-positions", action="store_true", help="live: accept the venue's positions as the local truth")
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = TraderConfig.from_env()
    if args.mode:
        cfg.mode = args.mode

    if args.kill:
        ks = KillSwitch(cfg.kill_switch_path)
        ks.arm()
        ks.trigger(args.kill)
        print(f"kill switch pulled: {args.kill} ({cfg.kill_switch_path})")
        return 0
    if args.flatten:
        if not (cfg.live_confirmed and cfg.exchange_api_key):
            print("--flatten needs TRADER_LIVE=true and exchange keys", file=sys.stderr)
            return 2
        ks = KillSwitch(cfg.kill_switch_path)
        ks.arm()
        ks.trigger("operator flatten")
        venue = _venue(cfg, live=True)
        for fill in venue.flatten_all(cfg.symbols):
            print(f"flattened {fill.symbol} {fill.side.value} {fill.qty} @ {fill.price}")
        return 0
    if args.check_jev:
        return check_jev(cfg)
    if args.check_exchange:
        return check_exchange(cfg)

    universe = load_universe(cfg.universe_path)

    if cfg.mode == "backtest":
        if args.replay:
            files = dict(item.split("=", 1) for item in args.replay)
            feed = ReplayFeed(files, cfg.candle_seconds, open_time=args.open_time)
        else:
            feed = _synthetic(cfg, args.candles, args.seed)
        reflex = create_reflex(cfg) if cfg.reflex_backend != "auto" else None
        result = run_backtest(cfg, feed, reflex=reflex, universe=universe, max_ticks=args.max_ticks)
        if args.json:
            print(json.dumps(result.to_dict(), default=str))
        else:
            print(json.dumps(result.metrics, indent=1, default=str))
            print(json.dumps(result.stats.to_dict(), indent=1))
        return 0

    if cfg.mode == "review":
        journal = Journal(cfg.journal_path)
        review = NightlyReview(cfg, journal, create_brain(cfg))
        report, proposal, markdown = review.run(date=args.date)
        print(markdown)
        return 0

    # paper / live / once: real market data through ccxt
    if cfg.mode == "live":
        if not cfg.live_confirmed:
            print("refusing live mode: set TRADER_LIVE=true after the Phase 5 checklist", file=sys.stderr)
            return 2
        venue = prepare_live(cfg, args.adopt_positions)
        exchange = venue
    else:
        venue = _venue(cfg, live=False)
        exchange = PaperExchange(cfg.starting_equity, cfg.taker_fee_bps, cfg.slippage_bps)
    feed = CcxtFeed(venue, cfg.symbols, cfg.candle_seconds, book_depth=cfg.book_depth,
                    max_ticks=len(cfg.symbols) if cfg.mode == "once" else args.max_ticks)
    journal = Journal(cfg.journal_path)
    loop = LiveLoop(cfg, feed, exchange, create_reflex(cfg), create_brain(cfg), journal, universe=universe)
    loop.start()
    try:
        stats = loop.run()
    except KeyboardInterrupt:
        stats = loop.stats
        print("interrupted; open positions are left as they are (use --flatten or --kill)")
    print(json.dumps(stats.to_dict(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
