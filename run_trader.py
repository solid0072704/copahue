"""Command line for the trading agent.

    python run_trader.py --mode backtest --synthetic --candles 1500 --seed 7
    python run_trader.py --mode backtest --replay BTC/USDT=data/btc_5m.csv
    python run_trader.py --mode paper                      # live data, paper fills
    python run_trader.py --mode live                       # needs TRADER_LIVE=true and keys
    python run_trader.py --mode once                       # one candle (cron)
    python run_trader.py --mode review [--date YYYY-MM-DD] # nightly review
    python run_trader.py --kill "reason"                   # pull the kill switch
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
from trading_agent.reflex import create_reflex
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
    venue = CcxtExchange(cfg.exchange_id, cfg.exchange_api_key, cfg.exchange_api_secret, sandbox=cfg.exchange_sandbox,
                         live_confirmed=(cfg.mode == "live" and cfg.live_confirmed), starting_equity=cfg.starting_equity,
                         taker_fee_bps=cfg.taker_fee_bps)
    if cfg.mode == "live":
        if not cfg.live_confirmed:
            print("refusing live mode: set TRADER_LIVE=true after the Phase 5 checklist", file=sys.stderr)
            return 2
        exchange = venue
    else:
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
        print("interrupted; open positions are left as they are (use --kill to flatten on next candle)")
    print(json.dumps(stats.to_dict(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
